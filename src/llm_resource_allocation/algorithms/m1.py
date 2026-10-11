"""Select TP/PP configurations with M1 weight/delay checks (v2 Eq. 9)."""

from dataclasses import dataclass

from llm_resource_allocation.domain import (
    FoundationModel,
    GpuTier,
    ParallelismConfig,
    ProblemInstance,
    QueryType,
    validate_problem_instance,
)
from llm_resource_allocation.math.latency import compute_configuration_delay_seconds

from ._mechanism_support import _entity_by_id


@dataclass(frozen=True, slots=True)
class M1ConfigurationDiagnostic:
    """Record M1 weight and delay checks for one configuration (v2 Eq. 9).

    ``weight_shard_gb = B_j/(n*m)`` and ``memory_capacity_gb = C_gpu[k]``.
    Delay and SLO values are in seconds; the weight check excludes routing/KV-cache.
    """

    configuration: ParallelismConfig
    weight_shard_gb: float
    memory_capacity_gb: float
    fixed_configuration_delay_seconds: float
    delay_slo_seconds: float

    @property
    def weight_condition_passes(self) -> bool:
        return self.weight_shard_gb <= self.memory_capacity_gb

    @property
    def delay_condition_passes(self) -> bool:
        return self.fixed_configuration_delay_seconds <= self.delay_slo_seconds

    @property
    def is_feasible(self) -> bool:
        """Return whether this same configuration passes both M1 conditions."""

        return self.weight_condition_passes and self.delay_condition_passes


@dataclass(frozen=True, slots=True)
class M1SelectionResult:
    """Store the M1 selection and all diagnostics in tier-semantic order.

    ``selected_configuration=None`` marks an infeasible candidate (D-0032).
    """

    query_id: str
    model_id: str
    tier_id: str
    selected_configuration: ParallelismConfig | None
    configuration_diagnostics: tuple[M1ConfigurationDiagnostic, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "configuration_diagnostics", tuple(self.configuration_diagnostics)
        )

    @property
    def is_feasible(self) -> bool:
        """Return whether Eq. (9) selected a feasible configuration."""

        return self.selected_configuration is not None

    @property
    def selected_diagnostic(self) -> M1ConfigurationDiagnostic | None:
        """Return the complete evidence for the selected configuration."""

        if self.selected_configuration is None:
            return None
        return next(
            (
                diagnostic
                for diagnostic in self.configuration_diagnostics
                if diagnostic.configuration == self.selected_configuration
            ),
            None,
        )

    @property
    def selected_weight_shard_gb(self) -> float | None:
        diagnostic = self.selected_diagnostic
        return None if diagnostic is None else diagnostic.weight_shard_gb

    @property
    def selected_fixed_configuration_delay_seconds(self) -> float | None:
        diagnostic = self.selected_diagnostic
        return (
            None if diagnostic is None else diagnostic.fixed_configuration_delay_seconds
        )


def select_minimum_feasible_parallelism(
    instance: ProblemInstance,
    query_id: str,
    model_id: str,
    tier_id: str,
) -> M1SelectionResult:
    """Select the minimum ``n*m`` satisfying M1 for ``(i,j,k)`` (v2 Eq. 9).

    The checks are ``B_j/(n*m) <= C_gpu[k]`` in GB and ``D_cfg <= Delta_i``
    in seconds. Delay uses ``compute_configuration_delay_seconds``; the printed
    weight term omits ``nu_k``. Only ``tier.parallelism_configs`` are evaluated.
    Exact GPU-count ties retain tier order; no feasible shape returns ``None``
    with every diagnostic retained (D-0032).
    """

    validate_problem_instance(instance)
    return _select_minimum_feasible_parallelism(instance, query_id, model_id, tier_id)


def _evaluate_m1_configuration(
    instance: ProblemInstance,
    query: QueryType,
    model: FoundationModel,
    tier: GpuTier,
    configuration: ParallelismConfig,
) -> M1ConfigurationDiagnostic:
    """Evaluate the two literal Paper v2 Eq. (9) inequalities once."""

    coefficient_key = (query.id, model.id, tier.id)
    weight_shard_gb = model.weight_gb / configuration.gpu_count
    delay_seconds = compute_configuration_delay_seconds(
        instance.compute_delay_seconds_per_token[coefficient_key],
        instance.communication_delay_seconds_per_token[coefficient_key],
        query,
        configuration,
    )
    return M1ConfigurationDiagnostic(
        configuration=configuration,
        weight_shard_gb=weight_shard_gb,
        memory_capacity_gb=tier.memory_capacity_gb,
        fixed_configuration_delay_seconds=delay_seconds,
        delay_slo_seconds=query.delay_slo_seconds,
    )


def _select_minimum_feasible_parallelism(
    instance: ProblemInstance,
    query_id: str,
    model_id: str,
    tier_id: str,
) -> M1SelectionResult:
    """Select M1 on an already validated instance; retain ID checks."""

    query = _entity_by_id(instance.queries, query_id, "query")
    model = _entity_by_id(instance.models, model_id, "model")
    tier = _entity_by_id(instance.gpu_tiers, tier_id, "GPU-tier")

    diagnostics = tuple(
        _evaluate_m1_configuration(instance, query, model, tier, configuration)
        for configuration in tier.parallelism_configs
    )

    selected: M1ConfigurationDiagnostic | None = None
    for diagnostic in diagnostics:
        if not diagnostic.is_feasible:
            continue
        # Strict improvement preserves the first tier-semantic configuration
        # when several feasible choices use the same minimum number of GPUs.
        if (
            selected is None
            or diagnostic.configuration.gpu_count
            < selected.configuration.gpu_count
        ):
            selected = diagnostic

    return M1SelectionResult(
        query_id=query.id,
        model_id=model.id,
        tier_id=tier.id,
        selected_configuration=(
            None if selected is None else selected.configuration
        ),
        configuration_diagnostics=diagnostics,
    )
