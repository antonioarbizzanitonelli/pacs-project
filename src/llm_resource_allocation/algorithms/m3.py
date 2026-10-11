"""Select M3 parallelism upgrades within the residual rental budget (v2 Eq. 12)."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from llm_resource_allocation.domain import (
    GpuTier,
    ParallelismConfig,
    ProblemInstance,
    QueryType,
    validate_problem_instance,
)
from llm_resource_allocation.math.latency import compute_configuration_delay_seconds

from ._mechanism_support import _entity_by_id


class M3NoUpgradeReason(str, Enum):
    """Explain why Paper v2 M3 did not select a parallelism upgrade."""

    INACTIVE_PAIR = "inactive_pair"
    CURRENT_DELAY_WITHIN_SLO = "current_delay_within_slo"
    NO_HIGHER_GPU_CONFIGURATION = "no_higher_gpu_configuration"
    NO_DELAY_FEASIBLE_CONFIGURATION = "no_delay_feasible_configuration"
    INSUFFICIENT_REMAINING_BUDGET = "insufficient_remaining_budget"


@dataclass(frozen=True, slots=True)
class M3EvaluationState:
    """Store active configuration and remaining M3 rental budget (v2 Eq. 12).

    ``current_configuration=None`` marks an inactive pair; otherwise ``y=n*m``.
    The budget is in USD and may be negative; no clipping is applied (D-0034).
    """

    current_configuration: ParallelismConfig | None
    remaining_budget_usd: float

    def __post_init__(self) -> None:
        """Require a finite budget while preserving zero and negative values."""

        if (
            self.current_configuration is not None
            and not isinstance(self.current_configuration, ParallelismConfig)
        ):
            raise ValueError(
                "current_configuration must be ParallelismConfig or None"
            )
        budget = self.remaining_budget_usd
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            raise ValueError("remaining_budget_usd must be a real number")
        if not isfinite(budget):
            raise ValueError("remaining_budget_usd must be finite")

    @property
    def is_active(self) -> bool:
        """Return whether the model-tier pair already has a configuration."""

        return self.current_configuration is not None

    @property
    def current_gpu_count(self) -> int:
        """Return linked ``y[j,k]=n*m``, or zero for an inactive pair."""

        if self.current_configuration is None:
            return 0
        return self.current_configuration.gpu_count


@dataclass(frozen=True, slots=True)
class M3ConfigurationDiagnostic:
    """Retain all Eq. (12) checks for one tier-valid configuration."""

    current_configuration: ParallelismConfig
    candidate_configuration: ParallelismConfig
    incremental_gpu_cost_usd: float
    remaining_budget_usd: float
    candidate_delay_seconds: float
    delay_slo_seconds: float

    @property
    def current_gpu_count(self) -> int:
        return self.current_configuration.gpu_count

    @property
    def candidate_gpu_count(self) -> int:
        return self.candidate_configuration.gpu_count

    @property
    def higher_gpu_count_passes(self) -> bool:
        return self.candidate_gpu_count > self.current_gpu_count

    @property
    def incremental_gpu_count(self) -> int:
        return max(self.candidate_gpu_count - self.current_gpu_count, 0)

    @property
    def budget_condition_passes(self) -> bool:
        return self.incremental_gpu_cost_usd <= self.remaining_budget_usd

    @property
    def delay_condition_passes(self) -> bool:
        return self.candidate_delay_seconds <= self.delay_slo_seconds

    @property
    def is_feasible_upgrade(self) -> bool:
        """Return whether the higher-GPU, delay, and budget checks all pass."""

        return (
            self.higher_gpu_count_passes
            and self.delay_condition_passes
            and self.budget_condition_passes
        )


@dataclass(frozen=True, slots=True)
class M3UpgradeResult:
    """Return one minimum-GPU M3 upgrade or explicit no-upgrade evidence."""

    query_id: str
    model_id: str
    tier_id: str
    current_configuration: ParallelismConfig | None
    current_delay_seconds: float | None
    delay_slo_seconds: float
    remaining_budget_usd: float
    configuration_diagnostics: tuple[M3ConfigurationDiagnostic, ...]
    selected_configuration: ParallelismConfig | None
    no_upgrade_reasons: tuple[M3NoUpgradeReason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "configuration_diagnostics", tuple(self.configuration_diagnostics)
        )
        object.__setattr__(self, "no_upgrade_reasons", tuple(self.no_upgrade_reasons))

    @property
    def current_gpu_count(self) -> int:
        return (
            0
            if self.current_configuration is None
            else self.current_configuration.gpu_count
        )

    @property
    def is_applicable(self) -> bool:
        """Return whether M3's active-pair, excessive-delay premise holds."""

        return (
            self.current_configuration is not None
            and self.current_delay_seconds is not None
            and self.current_delay_seconds > self.delay_slo_seconds
        )

    @property
    def has_upgrade(self) -> bool:
        """Return whether Eq. (12) selected a feasible configuration with more GPUs."""

        return self.selected_configuration is not None

    @property
    def selected_diagnostic(self) -> M3ConfigurationDiagnostic | None:
        """Return complete evidence for the selected configuration."""

        if self.selected_configuration is None:
            return None
        return next(
            (
                diagnostic
                for diagnostic in self.configuration_diagnostics
                if diagnostic.candidate_configuration == self.selected_configuration
            ),
            None,
        )


def select_m3_parallelism_upgrade(
    instance: ProblemInstance,
    query_id: str,
    model_id: str,
    tier_id: str,
    state: M3EvaluationState,
) -> M3UpgradeResult:
    """Select the minimum-GPU M3 upgrade on an active pair (v2 Eq. 12).

    M3 applies when the current delay exceeds ``Delta_i``. Eligible shapes satisfy
    ``n*m > y[j,k]``, ``D_cfg <= Delta_i``, and
    ``DeltaT*p_gpu[k]*(n*m-y[j,k]) <= state.remaining_budget_usd`` (D-0034).
    Delay is in seconds and rental cost is in USD; only incremental GPU rental
    is charged. All tier-valid shapes retain diagnostics in tier order, and
    exact GPU-count ties retain the first feasible shape.
    """

    validate_problem_instance(instance)
    return _select_m3_parallelism_upgrade(instance, query_id, model_id, tier_id, state)


def _evaluate_m3_configuration(
    instance: ProblemInstance,
    query: QueryType,
    tier: GpuTier,
    state: M3EvaluationState,
    candidate_configuration: ParallelismConfig,
    coefficient_key: tuple[str, str, str],
) -> M3ConfigurationDiagnostic:
    """Evaluate one tier-valid configuration against the M3 conditions."""

    current_configuration = state.current_configuration
    candidate_gpu_count = candidate_configuration.gpu_count
    incremental_gpu_count = max(
        candidate_gpu_count - state.current_gpu_count,
        0,
    )
    incremental_gpu_cost_usd = (
        instance.horizon_hours
        * tier.hourly_rental_usd
        * incremental_gpu_count
    )
    candidate_delay_seconds = compute_configuration_delay_seconds(
        instance.compute_delay_seconds_per_token[coefficient_key],
        instance.communication_delay_seconds_per_token[coefficient_key],
        query,
        candidate_configuration,
    )
    return M3ConfigurationDiagnostic(
        current_configuration=current_configuration,
        candidate_configuration=candidate_configuration,
        incremental_gpu_cost_usd=incremental_gpu_cost_usd,
        remaining_budget_usd=state.remaining_budget_usd,
        candidate_delay_seconds=candidate_delay_seconds,
        delay_slo_seconds=query.delay_slo_seconds,
    )


def _select_m3_parallelism_upgrade(
    instance: ProblemInstance,
    query_id: str,
    model_id: str,
    tier_id: str,
    state: M3EvaluationState,
) -> M3UpgradeResult:
    """Select M3 on a validated instance and immutable evaluation state."""

    query = _entity_by_id(instance.queries, query_id, "query")
    model = _entity_by_id(instance.models, model_id, "model")
    tier = _entity_by_id(instance.gpu_tiers, tier_id, "GPU-tier")

    if not state.is_active:
        return M3UpgradeResult(
            query_id=query.id,
            model_id=model.id,
            tier_id=tier.id,
            current_configuration=None,
            current_delay_seconds=None,
            delay_slo_seconds=query.delay_slo_seconds,
            remaining_budget_usd=state.remaining_budget_usd,
            configuration_diagnostics=(),
            selected_configuration=None,
            no_upgrade_reasons=(M3NoUpgradeReason.INACTIVE_PAIR,),
        )

    current_configuration = state.current_configuration
    if current_configuration not in tier.parallelism_configs:
        raise ValueError("current configuration is not valid for its GPU tier")

    coefficient_key = (query.id, model.id, tier.id)
    current_delay_seconds = compute_configuration_delay_seconds(
        instance.compute_delay_seconds_per_token[coefficient_key],
        instance.communication_delay_seconds_per_token[coefficient_key],
        query,
        current_configuration,
    )
    if current_delay_seconds <= query.delay_slo_seconds:
        return M3UpgradeResult(
            query_id=query.id,
            model_id=model.id,
            tier_id=tier.id,
            current_configuration=current_configuration,
            current_delay_seconds=current_delay_seconds,
            delay_slo_seconds=query.delay_slo_seconds,
            remaining_budget_usd=state.remaining_budget_usd,
            configuration_diagnostics=(),
            selected_configuration=None,
            no_upgrade_reasons=(
                M3NoUpgradeReason.CURRENT_DELAY_WITHIN_SLO,
            ),
        )

    diagnostics = tuple(
        _evaluate_m3_configuration(
            instance,
            query,
            tier,
            state,
            candidate_configuration,
            coefficient_key,
        )
        for candidate_configuration in tier.parallelism_configs
    )

    selected: M3ConfigurationDiagnostic | None = None
    for diagnostic in diagnostics:
        if not diagnostic.is_feasible_upgrade:
            continue
        if (
            selected is None
            or diagnostic.candidate_gpu_count < selected.candidate_gpu_count
        ):
            selected = diagnostic

    no_upgrade_reasons: tuple[M3NoUpgradeReason, ...] = ()
    if selected is None:
        higher = tuple(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.higher_gpu_count_passes
        )
        if not higher:
            no_upgrade_reasons = (
                M3NoUpgradeReason.NO_HIGHER_GPU_CONFIGURATION,
            )
        elif not any(diagnostic.delay_condition_passes for diagnostic in higher):
            no_upgrade_reasons = (
                M3NoUpgradeReason.NO_DELAY_FEASIBLE_CONFIGURATION,
            )
        else:
            no_upgrade_reasons = (
                M3NoUpgradeReason.INSUFFICIENT_REMAINING_BUDGET,
            )

    return M3UpgradeResult(
        query_id=query.id,
        model_id=model.id,
        tier_id=tier.id,
        current_configuration=current_configuration,
        current_delay_seconds=current_delay_seconds,
        delay_slo_seconds=query.delay_slo_seconds,
        remaining_budget_usd=state.remaining_budget_usd,
        configuration_diagnostics=diagnostics,
        selected_configuration=(
            None if selected is None else selected.candidate_configuration
        ),
        no_upgrade_reasons=no_upgrade_reasons,
    )
