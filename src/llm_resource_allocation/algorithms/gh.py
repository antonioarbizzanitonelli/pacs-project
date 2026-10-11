"""Compose GH, materialize its decisions, and independently validate (Algorithm 1)."""

from dataclasses import dataclass
from enum import Enum
from math import fsum
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.algorithms.gh_phase1 import (
    GHPhase1Config,
    GHPhase1Result,
    _run_gh_phase1,
)
from llm_resource_allocation.algorithms.gh_phase2 import (
    GHPhase2State,
    _validate_phase2_state,
    _initialize_gh_phase2_state,
)
from llm_resource_allocation.algorithms.gh_phase2_run import (
    GHPhase2RunResult,
    _run_gh_phase2,
)
from llm_resource_allocation.domain import (
    ParallelismConfig,
    ProblemInstance,
    validate_problem_instance,
)
from llm_resource_allocation.math.objective import (
    ObjectiveBreakdown,
    PaperVersion,
    compute_data_storage_cost,
    compute_gpu_rental_cost,
    compute_model_storage_cost,
)
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.optimization.problem_data import (
    PdmProblemData,
    PdmVariableCounts,
)
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    ValidationTolerances,
    validate_placement,
)


ModelTierKey = tuple[str, str]


class GHCompletionStatus(str, Enum):
    """Distinguish construction completion from scientific feasibility."""

    ALGORITHM_1_COMPLETED = "algorithm_1_completed"


@dataclass(frozen=True, slots=True)
class GHRunIdentity:
    """Snapshot the instance identity, semantic order, and paper version."""

    instance_id: str
    paper_version: PaperVersion
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    tier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze semantic identity sequences for direct construction too."""

        object.__setattr__(self, "query_ids", tuple(self.query_ids))
        object.__setattr__(self, "model_ids", tuple(self.model_ids))
        object.__setattr__(self, "tier_ids", tuple(self.tier_ids))


@dataclass(frozen=True, slots=True)
class GHCanonicalBinaryCounts:
    """Count active binary values after dense canonical materialization."""

    admission_indicators_equal_one: int
    deployment_indicators_equal_one: int
    configuration_selectors_equal_one: int


@dataclass(frozen=True, slots=True)
class GHCanonicalizationReport:
    """Record exact conversion from sparse GH state to dense ``P_DM`` decisions."""

    source_state: GHPhase2State
    variable_counts: PdmVariableCounts
    selected_configuration_by_pair: Mapping[ModelTierKey, ParallelismConfig]
    route_totals_by_query: Mapping[str, float]
    mapping_inconsistencies: tuple[str, ...]
    placement_decisions: PlacementDecisionState
    completed: bool

    @property
    def positive_route_count(self) -> int:
        return sum(
            value > 0.0
            for value in self.placement_decisions.routing_fractions.values()
        )

    @property
    def positive_auxiliary_count(self) -> int:
        return sum(
            value > 0.0
            for value in self.placement_decisions.auxiliary_products.values()
        )

    @property
    def binary_counts(self) -> GHCanonicalBinaryCounts:
        placement = self.placement_decisions
        return GHCanonicalBinaryCounts(
            admission_indicators_equal_one=sum(
                value == 1 for value in placement.admission_indicators.values()
            ),
            deployment_indicators_equal_one=sum(
                value == 1 for value in placement.deployment_indicators.values()
            ),
            configuration_selectors_equal_one=sum(
                value == 1 for value in placement.configuration_selectors.values()
            ),
        )

    def __post_init__(self) -> None:
        """Freeze report-owned mappings and diagnostic sequences."""

        object.__setattr__(
            self,
            "selected_configuration_by_pair",
            MappingProxyType(dict(self.selected_configuration_by_pair)),
        )
        object.__setattr__(
            self,
            "route_totals_by_query",
            MappingProxyType(dict(self.route_totals_by_query)),
        )
        object.__setattr__(
            self, "mapping_inconsistencies", tuple(self.mapping_inconsistencies)
        )


@dataclass(frozen=True, slots=True)
class GHBudgetConsistencyAudit:
    """Compare carried Phase-2 budget use with authoritative C1+C2+C3."""

    carried_budget_cost_usd: float
    gpu_rental_cost: float
    model_storage_cost: float
    data_storage_cost: float

    @property
    def recomputed_budget_cost_usd(self) -> float:
        """Return the authoritative C1+C2+C3 cost."""

        return self.gpu_rental_cost + self.model_storage_cost + self.data_storage_cost

    @property
    def absolute_difference_usd(self) -> float:
        """Return the absolute gap from the carried budget use."""

        return abs(self.carried_budget_cost_usd - self.recomputed_budget_cost_usd)

    @property
    def is_exactly_consistent(self) -> bool:
        """Return exact agreement without introducing a hidden tolerance."""

        return self.absolute_difference_usd == 0.0


@dataclass(frozen=True, slots=True)
class GHDemandConsistencyAudit:
    """Record the final demand-balance evidence for one query (v2 Eq. 8b)."""

    query_id: str
    routed_fraction: float
    unmet_fraction: float

    @property
    def lhs(self) -> float:
        """Return the routed and unmet fractions summed accurately."""

        return fsum((self.routed_fraction, self.unmet_fraction))

    @property
    def signed_residual(self) -> float:
        """Return the demand-balance difference from one."""

        return self.lhs - 1.0

    @property
    def absolute_residual(self) -> float:
        """Return the magnitude without changing the signed evidence."""

        return abs(self.signed_residual)


@dataclass(frozen=True, slots=True)
class GHResult:
    """Return complete Algorithm-1 evidence and independent validation."""

    identity: GHRunIdentity
    phase1_config: GHPhase1Config
    phase1_result: GHPhase1Result
    initialized_phase2_state: GHPhase2State
    phase2_result: GHPhase2RunResult
    canonicalization: GHCanonicalizationReport
    validation_report: PlacementValidationReport
    budget_consistency: GHBudgetConsistencyAudit
    demand_consistency: tuple[GHDemandConsistencyAudit, ...]
    completion_status: GHCompletionStatus

    def __post_init__(self) -> None:
        """Freeze final evidence owned by the GH producer."""

        object.__setattr__(
            self, "demand_consistency", tuple(self.demand_consistency)
        )

    @property
    def literal_final_state(self) -> GHPhase2State:
        """Return the literal final state retained by Phase 2."""

        return self.phase2_result.final_state

    @property
    def objective_breakdown(self) -> ObjectiveBreakdown | None:
        """Return the independently computed objective, when available."""

        return self.validation_report.objective_breakdown

    @property
    def residual_demand_by_query(self) -> Mapping[str, float]:
        """Return residual demand retained by Phase 2."""

        return self.phase2_result.residual_demand_by_query

    @property
    def placement_decisions(self) -> PlacementDecisionState:
        """Return the exact canonical decision state retained by the report."""

        return self.canonicalization.placement_decisions

    @property
    def algorithm_completed(self) -> bool:
        """Report only whether the printed construction reached line 23."""

        return self.completion_status is GHCompletionStatus.ALGORITHM_1_COMPLETED

    @property
    def canonicalization_completed(self) -> bool:
        """Report whether every authoritative lattice value was materialized."""

        return self.canonicalization.completed

    @property
    def independently_feasible(self) -> bool:
        """Forward the unchanged complete-placement validator verdict."""

        return self.validation_report.is_feasible


def canonicalize_gh_phase2_state(
    instance: ProblemInstance,
    state: GHPhase2State,
    *,
    expected_state_order: tuple[str, ...] | None = None,
) -> GHCanonicalizationReport:
    """Materialize final GH state on the authoritative ``P_DM`` lattice.

    Copy sparse routes to ``x`` with absent entries zero and copy ``u`` directly.
    Set ``z=1[x>0]``; final deployments define ``q,w,y`` and ``v=x*w`` (D-0041).
    Values are never clipped, normalized, recomputed from demand balance, or
    repaired. Unsupported configurations are excluded by the tier lattice.
    """

    problem_data = PdmProblemData(instance)
    _validate_phase2_state(
        instance, state, expected_state_order=expected_state_order
    )
    return _canonicalize_gh_phase2_state(instance, state, problem_data)


def run_gh(
    instance: ProblemInstance,
    config: GHPhase1Config,
    *,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
    query_order_override: tuple[str, ...] | None = None,
) -> GHResult:
    """Run both GH phases, materialize decisions, and independently validate once.

    Retain construction completion and scientific feasibility separately, with
    budget/demand audits and no repair of the literal final state (D-0041).
    """

    validate_problem_instance(instance)
    if not isinstance(paper_version, PaperVersion):
        raise ValueError("paper_version must be PaperVersion.V1 or PaperVersion.V2")
    phase1_result = _run_gh_phase1(instance, config)
    return _run_gh_with_phase1(
        instance,
        config,
        phase1_result,
        paper_version=paper_version,
        validation_tolerances=validation_tolerances,
        query_order_override=query_order_override,
    )


def _canonicalize_gh_phase2_state(
    instance: ProblemInstance,
    state: GHPhase2State,
    problem_data: PdmProblemData,
) -> GHCanonicalizationReport:
    """Materialize a checked GH state using the same instance's prepared lattice."""

    lattices = problem_data.variable_lattices
    deployment_by_pair = {
        deployment.pair_id: deployment for deployment in state.deployments
    }

    x = {
        key: state.routing_fraction_by_candidate.get(key, 0.0)
        for key in lattices.route_keys
    }
    z = {key: 1 if x[key] > 0.0 else 0 for key in lattices.route_keys}
    q = {
        key: 1 if key in deployment_by_pair else 0
        for key in lattices.model_tier_keys
    }
    u = {
        query_id: state.unmet_fraction_by_query[query_id]
        for query_id in lattices.unmet_keys
    }
    w = {
        key: int(
            key[:2] in deployment_by_pair
            and deployment_by_pair[key[:2]].configuration
            == ParallelismConfig(key[2], key[3])
        )
        for key in lattices.configuration_keys
    }
    y = {
        key: (
            deployment_by_pair[key].gpu_count if key in deployment_by_pair else 0
        )
        for key in lattices.model_tier_keys
    }
    v = {
        key: x[key[:3]] * w[(key[1], key[2], key[3], key[4])]
        for key in lattices.auxiliary_product_keys
    }

    routing = RoutingDecisionState(
        routing_fractions=x,
        admission_indicators=z,
        deployment_indicators=q,
        unmet_fractions=u,
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors=w,
        gpu_counts=y,
    )
    placement = PlacementDecisionState(
        parallelism_decisions=parallelism,
        auxiliary_products=v,
    )
    route_totals = {
        query_id: fsum(
            x[key] for key in lattices.route_keys if key[0] == query_id
        )
        for query_id in lattices.unmet_keys
    }
    selected_configurations = {
        pair: deployment_by_pair[pair].configuration
        for pair in lattices.model_tier_keys
        if pair in deployment_by_pair
    }
    return GHCanonicalizationReport(
        source_state=state,
        variable_counts=problem_data.variable_counts,
        selected_configuration_by_pair=selected_configurations,
        route_totals_by_query=route_totals,
        mapping_inconsistencies=(),
        placement_decisions=placement,
        completed=True,
    )


def _audit_budget(
    instance: ProblemInstance,
    state: GHPhase2State,
    placement: PlacementDecisionState,
    objective: ObjectiveBreakdown | None,
) -> GHBudgetConsistencyAudit:
    """Reuse validated C1-C3, with a domain-invalid report fallback."""

    if objective is None:
        c1 = compute_gpu_rental_cost(instance, placement.gpu_counts)
        c2 = compute_model_storage_cost(instance, placement.admission_indicators)
        c3 = compute_data_storage_cost(instance, placement.routing_fractions)
    else:
        c1 = objective.gpu_rental_cost
        c2 = objective.model_storage_cost
        c3 = objective.data_storage_cost
    return GHBudgetConsistencyAudit(
        carried_budget_cost_usd=state.accumulated_budget_cost_usd,
        gpu_rental_cost=c1,
        model_storage_cost=c2,
        data_storage_cost=c3,
    )


def _audit_demand(
    canonicalization: GHCanonicalizationReport,
) -> tuple[GHDemandConsistencyAudit, ...]:
    """Retain exact final ``sum x``, direct ``u``, LHS, and Eq. (8b) residual."""

    placement = canonicalization.placement_decisions
    return tuple(
        GHDemandConsistencyAudit(
            query_id=query_id,
            routed_fraction=routed,
            unmet_fraction=placement.unmet_fractions[query_id],
        )
        for query_id, routed in canonicalization.route_totals_by_query.items()
    )


def _run_gh_with_phase1(
    instance: ProblemInstance,
    config: GHPhase1Config,
    phase1_result: GHPhase1Result,
    *,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
    query_order_override: tuple[str, ...] | None = None,
    problem_data: PdmProblemData | None = None,
) -> GHResult:
    """Complete GH from one prepared Phase-1 result on the same instance."""

    phase2_initial = _initialize_gh_phase2_state(
        instance,
        phase1_result,
        query_order_override=query_order_override,
    )
    phase2_result = _run_gh_phase2(instance, phase2_initial)
    final_state = phase2_result.final_state
    if problem_data is None:
        problem_data = PdmProblemData(instance)
    canonicalization = _canonicalize_gh_phase2_state(
        instance, final_state, problem_data
    )
    validation = validate_placement(
        instance,
        canonicalization.placement_decisions,
        paper_version,
        validation_tolerances,
    )
    return GHResult(
        identity=GHRunIdentity(
            instance_id=instance.id,
            paper_version=paper_version,
            query_ids=instance.query_ids,
            model_ids=instance.model_ids,
            tier_ids=instance.gpu_tier_ids,
        ),
        phase1_config=config,
        phase1_result=phase1_result,
        initialized_phase2_state=phase2_initial,
        phase2_result=phase2_result,
        canonicalization=canonicalization,
        validation_report=validation,
        budget_consistency=_audit_budget(
            instance, final_state, canonicalization.placement_decisions,
            validation.objective_breakdown,
        ),
        demand_consistency=_audit_demand(canonicalization),
        completion_status=GHCompletionStatus.ALGORITHM_1_COMPLETED,
    )
