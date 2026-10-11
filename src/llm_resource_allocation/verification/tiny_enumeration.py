"""Independent exhaustive oracle for the supported tiny ``P_DM`` problem.

The instance has one query type, one model, one GPU tier, and one or more
TP/PP configurations. Choose deployment ``q``, admission ``z``, configuration
selectors ``w``, and served fraction ``x`` to minimize the versioned C1-C5 cost
subject to unmet-demand, resource, budget, and SLO constraints.

Enumerate binary ``q,z,w``. Once fixed, they determine ``y=sum(n*m*w)``
(v2 Eq. 8e); demand balance gives ``u=1-x`` and exact products give ``v=x*w``.
Only one continuous variable, ``x``, remains. Resource/SLO rows have the form
``fixed+slope*x <= limit`` with non-negative slopes, so they bound ``x`` from
above or check a fixed term. Together with ``1-zeta <= x <= z`` and ``[0,1]``,
their intersection is an interval. The objective is affine on that interval,
so its minimum is attained at an endpoint (D-0026, D-0028).

Construct and independently validate the endpoints, then retain the cheapest
states and their ties. The oracle uses no optimizer and provides only tiny
synthetic-case evidence.
"""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from numbers import Real
from typing import Protocol, Sequence

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.domain.units import GFLOPS_PER_TFLOP
from llm_resource_allocation.math.latency import compute_configuration_delay_seconds
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.validation.objective_comparison import (
    DEFAULT_OBJECTIVE_COMPARISON_ABSOLUTE_TOLERANCE,
)
from llm_resource_allocation.validation.placement import validate_placement


ModelTierKey = tuple[str, str]
RouteKey = tuple[str, str, str]
ConfigurationKey = tuple[str, str, int, int]
AuxiliaryProductKey = tuple[str, str, str, int, int]


class _IndexSetsView(Protocol):
    """Expose ordered query, model, and tier IDs."""
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    tier_ids: tuple[str, ...]


class _VariableLatticesView(Protocol):
    """Expose ordered keys for ``q,y``, ``x,z``, ``w``, and ``v``."""
    model_tier_keys: tuple[ModelTierKey, ...]
    route_keys: tuple[RouteKey, ...]
    configuration_keys: tuple[ConfigurationKey, ...]
    auxiliary_product_keys: tuple[AuxiliaryProductKey, ...]


class PdmProblemDataView(Protocol):
    """Accept instance data and ordered keys without importing optimization."""

    instance: ProblemInstance
    index_sets: _IndexSetsView
    variable_lattices: _VariableLatticesView


class TinyEnumerationError(RuntimeError):
    """Identify an explicit tiny-oracle refusal."""


class TinyEnumerationLimitError(TinyEnumerationError):
    """Reject a raw binary search space above the configured limit."""

    def __init__(self, search_space_size: int, maximum: int) -> None:
        """Retain the raw count and configured maximum."""
        self.search_space_size = search_space_size
        self.maximum = maximum
        super().__init__(
            "tiny enumeration search space "
            f"{search_space_size} exceeds configured maximum {maximum}"
        )


class UnsupportedTinyOracleInstanceError(TinyEnumerationError):
    """Reject dimensions outside the analytical 1x1x1 scope (D-0026)."""


@dataclass(frozen=True, slots=True)
class TinyEnumerationConfig:
    """Bound raw enumeration and absolute objective ties (D-0028)."""

    max_discrete_combinations: int = 4096
    objective_tolerance: float = DEFAULT_OBJECTIVE_COMPARISON_ABSOLUTE_TOLERANCE

    def __post_init__(self) -> None:
        """Require a positive integer limit and a finite non-negative tolerance."""
        if (
            isinstance(self.max_discrete_combinations, bool)
            or not isinstance(self.max_discrete_combinations, int)
            or self.max_discrete_combinations <= 0
        ):
            raise ValueError("max_discrete_combinations must be a positive integer")
        if (
            isinstance(self.objective_tolerance, bool)
            or not isinstance(self.objective_tolerance, Real)
            or not isfinite(float(self.objective_tolerance))
            or self.objective_tolerance < 0.0
        ):
            raise ValueError(
                "objective_tolerance must be a finite non-negative real number"
            )


@dataclass(frozen=True, slots=True)
class TinyCandidateDiagnostic:
    """Record a pruned ``q,z,w`` candidate, its ``x`` interval, and outcome."""

    ordinal: int
    deployment_indicator: int
    admission_indicator: int
    configuration_selectors: tuple[int, ...]
    selected_configuration: ConfigurationKey | None
    routing_lower_bound: float
    routing_upper_bound: float
    selected_routing_fraction: float | None
    objective_value: float | None
    endpoint_tie: bool
    rejection_reason: str | None
    violated_families: tuple[str, ...]

    @property
    def is_feasible(self) -> bool:
        """Return whether endpoint validation supplied a routing fraction."""

        return self.selected_routing_fraction is not None


@dataclass(frozen=True, slots=True)
class TinyEnumerationResult:
    """Store counts, diagnostics, ordered optima, and the first tie's objective."""

    search_space_size: int
    best_objective: float | None
    optimal_states: tuple[PlacementDecisionState, ...]
    diagnostics: tuple[TinyCandidateDiagnostic, ...]

    def __post_init__(self) -> None:
        """Own the ordered candidate diagnostics and optimal states."""

        object.__setattr__(self, "optimal_states", tuple(self.optimal_states))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def candidates_examined(self) -> int:
        """Count the discrete candidates retained in the diagnostics."""

        return len(self.diagnostics)

    @property
    def feasible_candidates(self) -> int:
        """Count candidates with an independently feasible routing endpoint."""

        return sum(diagnostic.is_feasible for diagnostic in self.diagnostics)

    @property
    def best_state(self) -> PlacementDecisionState | None:
        """Return the first optimum in the enumeration's stable tie order."""

        return self.optimal_states[0] if self.optimal_states else None

    @property
    def tie_count(self) -> int:
        """Count retained optimal routing endpoints."""

        return len(self.optimal_states)

    @property
    def is_unique_optimum(self) -> bool:
        """Return whether exactly one optimal state was retained."""

        return self.tie_count == 1


@dataclass(frozen=True, slots=True)
class _RoutingEndpoint:
    """Store one independently validated endpoint and its objective."""

    objective_value: float
    routing_fraction: float
    state: PlacementDecisionState


def _minimum_ties(
    endpoints: Sequence[_RoutingEndpoint], objective_tolerance: float
) -> tuple[_RoutingEndpoint, ...]:
    """Retain endpoints within tolerance of the minimum, in input order."""

    if not endpoints:
        return ()
    minimum = min(endpoint.objective_value for endpoint in endpoints)
    return tuple(
        endpoint
        for endpoint in endpoints
        if abs(endpoint.objective_value - minimum) <= objective_tolerance
    )


def estimate_tiny_discrete_search_space(problem_data: PdmProblemDataView) -> int:
    """Return raw binary count ``2^(|q|+|z|+|w|)`` before pruning."""

    lattices = problem_data.variable_lattices
    binary_count = (
        len(lattices.model_tier_keys)
        + len(lattices.route_keys)
        + len(lattices.configuration_keys)
    )
    return 2**binary_count


def _require_supported_instance(problem_data: PdmProblemDataView) -> None:
    """Require one query/model/tier and at least one TP/PP configuration."""

    index_sets = problem_data.index_sets
    lattices = problem_data.variable_lattices
    counts = (
        len(index_sets.query_ids),
        len(index_sets.model_ids),
        len(index_sets.tier_ids),
        len(lattices.model_tier_keys),
        len(lattices.route_keys),
    )
    if counts != (1, 1, 1, 1, 1):
        raise UnsupportedTinyOracleInstanceError(
            "tiny enumeration supports exactly one query, one model, and one "
            f"GPU tier; observed counts {counts[:3]}"
        )
    if not lattices.configuration_keys:
        raise UnsupportedTinyOracleInstanceError(
            "tiny enumeration requires at least one tier-valid TP/PP configuration"
        )


def _tighten_upper_bound(
    lower: float,
    upper: float,
    *,
    fixed: float,
    slope: float,
    limit: float,
    label: str,
) -> tuple[float, str | None]:
    """Intersect ``fixed+slope*x <= limit``; reject negative slopes."""

    if slope < 0.0:
        return upper, f"{label} has unsupported negative routing coefficient"
    if slope == 0.0:
        if fixed <= limit:
            return upper, None
        return upper, f"{label} fixed term exceeds its limit"
    tightened = min(upper, (limit - fixed) / slope)
    if tightened < lower:
        return tightened, f"{label} makes the analytical routing interval empty"
    return tightened, None


def _routing_interval(
    instance: ProblemInstance,
    configuration_key: ConfigurationKey | None,
    *,
    deployment_indicator: int,
    admission_indicator: int,
    gpu_count: int,
) -> tuple[float, float, str | None]:
    """Intersect demand/routing bounds and affine resource/SLO bounds on ``x``.

    Start with ``max(0,1-zeta) <= x <= min(1,z)`` (v2 Eqs. 8b, 8d, 8k), then
    intersect rows ``(label,fixed,slope,limit)`` for ``fixed+slope*x <= limit``.
    Keep exact bounds, the selected unit convention, and per-query storage closure.
    """

    query = instance.queries[0]
    model = instance.models[0]
    tier = instance.gpu_tiers[0]
    triple = (query.id, model.id, tier.id)
    lower = max(0.0, 1.0 - query.max_unmet_fraction)
    upper = min(1.0, float(admission_indicator))
    if upper < lower:
        return lower, upper, "demand balance/unmet cap conflicts with x <= z"

    if configuration_key is None:
        delay_slope = 0.0
        memory_fixed = 0.0
        memory_slope = 0.0
    else:
        _, _, tensor_degree, pipeline_depth = configuration_key
        config = next(
            item
            for item in tier.parallelism_configs
            if item.tensor_parallel_degree == tensor_degree
            and item.pipeline_parallel_depth == pipeline_depth
        )
        delay_slope = compute_configuration_delay_seconds(
            instance.compute_delay_seconds_per_token[triple],
            instance.communication_delay_seconds_per_token[triple],
            query,
            config,
        )
        memory_fixed = model.weight_gb / config.gpu_count
        memory_slope = (
            model.kv_cache_gb_per_token
            / config.gpu_count
            * query.total_tokens
            * instance.kv_cache_residency_seconds[triple]
        )
        if instance.uses_coherent_units:
            memory_slope *= query.arrival_rate_per_hour / instance.seconds_per_hour

    storage_slope = (
        query.storage_gb_per_token
        * query.total_tokens
        * query.arrival_rate_per_hour
    )
    data_storage_cost_slope = (
        instance.horizon_hours
        * instance.storage_price_usd_per_gb_hour
        * query.storage_gb_per_token
        * query.total_tokens
        * query.arrival_rate_per_hour
    )
    if instance.uses_coherent_units:
        storage_slope *= instance.data_storage_duration_hours
        data_storage_cost_slope *= instance.data_storage_duration_hours

    constraints = (
        ("delay Eq. (8i)", 0.0, delay_slope, query.delay_slo_seconds),
        (
            "error Eq. (8j)",
            0.0,
            tier.quantization_error_multiplier
            * instance.base_error_rate[(query.id, model.id)],
            query.error_slo,
        ),
        (
            "compute Eq. (8g)",
            0.0,
            instance.compute_gflop_per_token[triple]
            * query.total_tokens
            * query.arrival_rate_per_hour
            / GFLOPS_PER_TFLOP,
            instance.compute_utilization
            * instance.seconds_per_hour
            * tier.compute_throughput_tflops
            * gpu_count,
        ),
        (
            "memory Eq. (8f)/D-0002",
            memory_fixed,
            memory_slope,
            tier.memory_capacity_gb * deployment_indicator,
        ),
        (
            "storage Eq. (8h)/D-0016",
            model.weight_gb * admission_indicator,
            storage_slope,
            instance.storage_capacity_gb,
        ),
        (
            "budget Eq. (8c)",
            instance.horizon_hours
            * (
                tier.hourly_rental_usd * gpu_count
                + instance.storage_price_usd_per_gb_hour
                * model.weight_gb
                * admission_indicator
            ),
            data_storage_cost_slope,
            instance.cost_budget_usd,
        ),
    )
    for label, fixed, slope, limit in constraints:
        upper, reason = _tighten_upper_bound(
            lower,
            upper,
            fixed=fixed,
            slope=slope,
            limit=limit,
            label=label,
        )
        if reason is not None:
            return lower, upper, reason
    return lower, upper, None


def _build_state(
    problem_data: PdmProblemDataView,
    *,
    deployment_indicator: int,
    admission_indicator: int,
    selector_bits: tuple[int, ...],
    gpu_count: int,
    routing_fraction: float,
) -> PlacementDecisionState:
    """Build plain ``q,z,w,y,x,u=1-x,v=x*w`` oracle data (D-0026)."""

    lattices = problem_data.variable_lattices
    pair = lattices.model_tier_keys[0]
    route = lattices.route_keys[0]
    query_id = problem_data.index_sets.query_ids[0]
    selectors = dict(zip(lattices.configuration_keys, selector_bits, strict=True))
    routing = RoutingDecisionState(
        routing_fractions={route: routing_fraction},
        admission_indicators={route: admission_indicator},
        deployment_indicators={pair: deployment_indicator},
        unmet_fractions={query_id: 1.0 - routing_fraction},
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors=selectors,
        gpu_counts={pair: gpu_count},
    )
    auxiliary_products = {
        key: routing_fraction
        * selectors[(key[1], key[2], key[3], key[4])]
        for key in lattices.auxiliary_product_keys
    }
    return PlacementDecisionState(
        parallelism_decisions=parallelism,
        auxiliary_products=auxiliary_products,
    )


def _evaluate_routing_endpoints(
    problem_data: PdmProblemDataView,
    paper_version: PaperVersion,
    *,
    deployment_indicator: int,
    admission_indicator: int,
    selector_bits: tuple[int, ...],
    gpu_count: int,
    lower: float,
    upper: float,
) -> tuple[tuple[_RoutingEndpoint, ...], tuple[str, ...]]:
    """Validate distinct endpoints lower-first; retain ordered failure families."""

    endpoints = (lower,) if lower == upper else (lower, upper)
    endpoint_records: list[_RoutingEndpoint] = []
    invalid_families: list[str] = []
    for endpoint in endpoints:
        state = _build_state(
            problem_data,
            deployment_indicator=deployment_indicator,
            admission_indicator=admission_indicator,
            selector_bits=selector_bits,
            gpu_count=gpu_count,
            routing_fraction=endpoint,
        )
        report = validate_placement(
            problem_data.instance,
            state,
            paper_version,
        )
        if report.is_feasible and report.objective_breakdown is not None:
            endpoint_records.append(
                _RoutingEndpoint(report.objective_breakdown.total, endpoint, state)
            )
        else:
            invalid_families.extend(
                family.value for family in report.violated_families
            )
    return tuple(endpoint_records), tuple(dict.fromkeys(invalid_families))


def _evaluate_discrete_candidate(
    problem_data: PdmProblemDataView,
    paper_version: PaperVersion,
    bits: tuple[int, ...],
    ordinal: int,
    objective_tolerance: float,
) -> tuple[TinyCandidateDiagnostic, tuple[_RoutingEndpoint, ...]]:
    """Evaluate one pruned ``(q,z,w...)`` tuple; retain lower-first endpoint ties."""

    lattices = problem_data.variable_lattices
    deployment, admission, *raw_selectors = bits
    selector_bits = tuple(raw_selectors)
    selected_configuration = next(
        (
            key
            for key, bit in zip(
                lattices.configuration_keys, selector_bits, strict=True
            )
            if bit
        ),
        None,
    )
    # sum(w)=q permits at most one active configuration; y=n*m for that choice.
    gpu_count = (
        0
        if selected_configuration is None
        else selected_configuration[2] * selected_configuration[3]
    )
    lower, upper, rejection = _routing_interval(
        problem_data.instance,
        selected_configuration,
        deployment_indicator=deployment,
        admission_indicator=admission,
        gpu_count=gpu_count,
    )
    tied_endpoints: tuple[_RoutingEndpoint, ...] = ()
    violated_families: tuple[str, ...] = ()
    if rejection is None:
        endpoint_records, rejected_families = _evaluate_routing_endpoints(
            problem_data,
            paper_version,
            deployment_indicator=deployment,
            admission_indicator=admission,
            selector_bits=selector_bits,
            gpu_count=gpu_count,
            lower=lower,
            upper=upper,
        )
        tied_endpoints = _minimum_ties(endpoint_records, objective_tolerance)
        if not tied_endpoints:
            rejection = "independent validator rejected both endpoints"
            violated_families = rejected_families

    best_endpoint = tied_endpoints[0] if tied_endpoints else None
    diagnostic = TinyCandidateDiagnostic(
        ordinal=ordinal,
        deployment_indicator=deployment,
        admission_indicator=admission,
        configuration_selectors=selector_bits,
        selected_configuration=selected_configuration,
        routing_lower_bound=lower,
        routing_upper_bound=upper,
        selected_routing_fraction=(
            None if best_endpoint is None else best_endpoint.routing_fraction
        ),
        objective_value=None if best_endpoint is None else best_endpoint.objective_value,
        endpoint_tie=len(tied_endpoints) > 1,
        rejection_reason=rejection,
        violated_families=violated_families,
    )
    return diagnostic, tied_endpoints


def enumerate_tiny_pdm(
    problem_data: PdmProblemDataView,
    paper_version: PaperVersion = PaperVersion.V2,
    config: TinyEnumerationConfig | None = None,
) -> TinyEnumerationResult:
    """Find the tiny-instance optimum through exact pruning and endpoint checks.

    Prune by ``sum(w)=q`` and ``z<=q`` (v2 Eq. 8d). Scan configurations in index
    order, validate endpoints, and retain ordered minimum ties (D-0026, D-0028).
    """

    selected_config = config or TinyEnumerationConfig()
    if not isinstance(selected_config, TinyEnumerationConfig):
        raise TypeError("config must be TinyEnumerationConfig or None")
    if not isinstance(paper_version, PaperVersion):
        raise ValueError("paper_version must be PaperVersion.V1 or PaperVersion.V2")
    if problem_data.instance.uses_coherent_units and paper_version is PaperVersion.V1:
        raise ValueError("coherent units require PaperVersion.V2; use literal_paper for v1")

    search_space_size = estimate_tiny_discrete_search_space(problem_data)
    if search_space_size > selected_config.max_discrete_combinations:
        raise TinyEnumerationLimitError(
            search_space_size,
            selected_config.max_discrete_combinations,
        )
    _require_supported_instance(problem_data)

    n_selectors = len(problem_data.variable_lattices.configuration_keys)
    diagnostics: list[TinyCandidateDiagnostic] = []
    feasible_endpoints: list[_RoutingEndpoint] = []

    raw_states = product((0, 1), repeat=2 + n_selectors)
    structurally_admissible = (
        bits
        for bits in raw_states
        if sum(bits[2:]) == bits[0] and bits[1] <= bits[0]
    )

    def semantic_order(bits: tuple[int, ...]) -> tuple[int, int, int]:
        """Order by ``q,z``, then the selected configuration's lattice index."""

        selected_index = next(
            (index for index, bit in enumerate(bits[2:]) if bit),
            -1,
        )
        return bits[0], bits[1], selected_index

    for ordinal, bits in enumerate(sorted(structurally_admissible, key=semantic_order)):
        diagnostic, candidate_records = _evaluate_discrete_candidate(
            problem_data,
            paper_version,
            bits,
            ordinal,
            selected_config.objective_tolerance,
        )
        diagnostics.append(diagnostic)
        feasible_endpoints.extend(candidate_records)

    tied_endpoints = _minimum_ties(feasible_endpoints, selected_config.objective_tolerance)
    best_endpoint = tied_endpoints[0] if tied_endpoints else None
    return TinyEnumerationResult(
        search_space_size=search_space_size,
        best_objective=None if best_endpoint is None else best_endpoint.objective_value,
        optimal_states=tuple(endpoint.state for endpoint in tied_endpoints),
        diagnostics=tuple(diagnostics),
    )
