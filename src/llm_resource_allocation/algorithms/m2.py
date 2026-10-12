"""Evaluate M2 marginal cost, effective coverage, and ranking (v2 Eqs. 10-11)."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol

from llm_resource_allocation.domain import (
    FoundationModel,
    GpuTier,
    ParallelismConfig,
    ProblemInstance,
    QueryType,
    validate_problem_instance,
)
from llm_resource_allocation.math.error import compute_effective_error
from llm_resource_allocation.domain.units import MILLISECONDS_PER_SECOND

from ._mechanism_support import _entity_by_id


class M2ConfigurationEvidence(Protocol):
    """Supply read-only candidate identity, configuration, and delay to M2."""

    @property
    def query_id(self) -> str:
        """Return the candidate's query identifier."""

        ...

    @property
    def model_id(self) -> str:
        """Return the candidate's model identifier."""

        ...

    @property
    def tier_id(self) -> str:
        """Return the candidate's GPU-tier identifier."""

        ...

    @property
    def selected_configuration(self) -> ParallelismConfig | None:
        """Return the selected configuration, or None if unavailable."""

        ...

    @property
    def selected_fixed_configuration_delay_seconds(self) -> float | None:
        """Return the stored configuration delay in seconds, or None."""

        ...

    @property
    def is_feasible(self) -> bool:
        """Return whether configuration and delay evidence are available."""

        ...


class M2CoverageLimiter(str, Enum):
    """Name one printed cap attaining M2's effective-coverage minimum."""

    REMAINING_DEMAND = "remaining_demand"
    ERROR_BUDGET = "error_budget"
    DELAY_BUDGET = "delay_budget"


class M2UnrankableReason(str, Enum):
    """Explain why Algorithm 1 must discard an M2 candidate before sorting."""

    NO_FEASIBLE_CONFIGURATION = "no_feasible_configuration"
    NO_REMAINING_DEMAND = "no_remaining_demand"
    ERROR_BUDGET_EXHAUSTED = "error_budget_exhausted"
    ERROR_BUDGET_EXCEEDED = "error_budget_exceeded"
    DELAY_BUDGET_EXHAUSTED = "delay_budget_exhausted"
    DELAY_BUDGET_EXCEEDED = "delay_budget_exceeded"


@dataclass(frozen=True, slots=True)
class M2EvaluationInputs:
    """Store dynamic inputs for M2 cost and coverage (v2 Eqs. 10-11).

    ``remaining_demand``, ``cumulative_error``, ``cumulative_delay_seconds``, and
    ``current_gpu_count`` map to ``r_tilde_i``, ``E_used_i``, ``D_used_i``,
    and ``y[j,k]``. Static coefficients come from ``ProblemInstance``.
    """

    remaining_demand: float
    cumulative_error: float
    cumulative_delay_seconds: float
    current_gpu_count: int

    def __post_init__(self) -> None:
        """Reject invalid dynamic inputs instead of clipping or repairing it."""

        for name, value in (
            ("remaining_demand", self.remaining_demand),
            ("cumulative_error", self.cumulative_error),
            ("cumulative_delay_seconds", self.cumulative_delay_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.remaining_demand > 1.0:
            raise ValueError("remaining_demand must be <= 1")
        if (
            isinstance(self.current_gpu_count, bool)
            or not isinstance(self.current_gpu_count, int)
            or self.current_gpu_count < 0
        ):
            raise ValueError("current_gpu_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class M2MarginalCostBreakdown:
    """Break down GPU, storage, and delay costs in Paper v2 Eq. (10)."""

    current_gpu_count: int
    required_gpu_count: int
    incremental_gpu_cost: float
    model_and_data_storage_cost: float
    delay_penalty_cost: float

    @property
    def incremental_gpu_count(self) -> int:
        return max(self.required_gpu_count - self.current_gpu_count, 0)

    @property
    def marginal_cost(self) -> float:
        return (
            self.incremental_gpu_cost
            + self.model_and_data_storage_cost
            + self.delay_penalty_cost
        )


@dataclass(frozen=True, slots=True)
class M2EffectiveCoverageDiagnostic:
    """Record Eq. (11) limits and non-negative allocatable coverage.

    A ``None`` limit means zero consumption with a non-negative residual budget.
    Negative printed ratios remain visible; allocatable coverage is at least zero.
    """

    remaining_demand: float
    error_slo: float
    cumulative_error: float
    effective_error: float
    delay_slo_seconds: float
    cumulative_delay_seconds: float
    fixed_configuration_delay_seconds: float

    @property
    def residual_error_budget(self) -> float:
        return self.error_slo - self.cumulative_error

    @property
    def error_coverage_limit(self) -> float | None:
        return _paper_coverage_limit(self.residual_error_budget, self.effective_error)

    @property
    def residual_delay_budget_seconds(self) -> float:
        return self.delay_slo_seconds - self.cumulative_delay_seconds

    @property
    def delay_coverage_limit(self) -> float | None:
        return _paper_coverage_limit(
            self.residual_delay_budget_seconds,
            self.fixed_configuration_delay_seconds,
        )

    def _finite_limits(self) -> tuple[tuple[M2CoverageLimiter, float], ...]:
        limits = [(M2CoverageLimiter.REMAINING_DEMAND, self.remaining_demand)]
        error_limit = self.error_coverage_limit
        delay_limit = self.delay_coverage_limit
        if error_limit is not None:
            limits.append((M2CoverageLimiter.ERROR_BUDGET, error_limit))
        if delay_limit is not None:
            limits.append((M2CoverageLimiter.DELAY_BUDGET, delay_limit))
        return tuple(limits)

    @property
    def effective_coverage(self) -> float:
        return max(0.0, min(value for _, value in self._finite_limits()))

    @property
    def limiting_factors(self) -> tuple[M2CoverageLimiter, ...]:
        limits = self._finite_limits()
        minimum = min(value for _, value in limits)
        return tuple(limiter for limiter, value in limits if value == minimum)

    @property
    def covers_remaining_demand(self) -> bool:
        """Return whether this candidate covers all remaining demand."""

        return self.effective_coverage == self.remaining_demand


@dataclass(frozen=True, slots=True)
class M2CandidateEvaluation:
    """Store configuration, cost, coverage, and rank evidence for one candidate."""

    query_id: str
    model_id: str
    tier_id: str
    selected_configuration: ParallelismConfig | None
    cost_breakdown: M2MarginalCostBreakdown | None
    coverage_diagnostic: M2EffectiveCoverageDiagnostic | None

    @property
    def pi(self) -> int | None:
        if self.coverage_diagnostic is None:
            return None
        return int(
            self.coverage_diagnostic.effective_coverage
            < self.coverage_diagnostic.remaining_demand
        )

    @property
    def kappa(self) -> float | None:
        if self.cost_breakdown is None or self.coverage_diagnostic is None:
            return None
        coverage = self.coverage_diagnostic.effective_coverage
        return self.cost_breakdown.marginal_cost / coverage if coverage > 0.0 else None

    @property
    def unrankable_reasons(self) -> tuple[M2UnrankableReason, ...]:
        if self.coverage_diagnostic is None:
            return (M2UnrankableReason.NO_FEASIBLE_CONFIGURATION,)
        return _m2_unrankable_reasons(self.coverage_diagnostic)

    @property
    def is_rankable(self) -> bool:
        """Return whether Algorithm 1 keeps this positive-coverage candidate."""

        return self.kappa is not None

    @property
    def rank_key(self) -> tuple[int, float] | None:
        """Return exact Paper v2 ``(pi, kappa)`` or ``None`` if discarded."""

        if self.pi is None or self.kappa is None:
            return None
        return (self.pi, self.kappa)


def compute_m2_marginal_cost(
    instance: ProblemInstance,
    configuration_evidence: M2ConfigurationEvidence,
    evaluation_inputs: M2EvaluationInputs,
) -> M2MarginalCostBreakdown:
    """Return M2 marginal cost from selected configuration/delay (v2 Eq. 10).

    ``c = DeltaT*(p_gpu*max(n*m-y,0) + p_s*(B_j+theta_i*r_i*lambda_i))
         + rho_i*D_cfg``.
    ``DeltaT`` is ``instance.horizon_hours``; ``y`` is ``evaluation_inputs.current_gpu_count``.
    Only the GPU term uses the positive part. Coherent units multiply the data
    rate by its configured residency and charge delay in milliseconds for all
    arrivals over the horizon. Literal mode retains the printed arithmetic.
    """

    query, model, tier = _validate_m2_candidate_identity(instance, configuration_evidence)
    configuration, delay_seconds = _require_feasible_configuration_evidence(
        tier, configuration_evidence
    )
    return _compute_m2_marginal_cost(
        instance,
        query,
        model,
        tier,
        configuration,
        delay_seconds,
        evaluation_inputs,
    )


def compute_m2_effective_coverage(
    instance: ProblemInstance,
    configuration_evidence: M2ConfigurationEvidence,
    evaluation_inputs: M2EvaluationInputs,
) -> M2EffectiveCoverageDiagnostic:
    """Return M2 effective coverage from selected delay evidence (v2 Eq. 11).

    ``x_bar = min(r_tilde_i, (epsilon_i-E_used_i)/e_bar,
                 (Delta_i-D_used_i)/D_cfg)``.
    Delay quantities are in seconds. Zero consumption is non-binding only when
    the residual budget is non-negative; allocatable coverage is at least zero.
    Memory, capacity, and rental budget are later commit checks.
    """

    query, model, tier = _validate_m2_candidate_identity(instance, configuration_evidence)
    _, delay_seconds = _require_feasible_configuration_evidence(
        tier, configuration_evidence
    )
    return _compute_m2_effective_coverage(
        instance,
        query,
        model,
        tier,
        delay_seconds,
        evaluation_inputs,
    )


def evaluate_m2_candidate(
    instance: ProblemInstance,
    configuration_evidence: M2ConfigurationEvidence,
    evaluation_inputs: M2EvaluationInputs,
) -> M2CandidateEvaluation:
    """Evaluate one configuration-screened M2 candidate (v2 Eqs. 10-11).

    ``pi = 1[x_bar < r_tilde_i]`` and, for positive coverage, ``kappa=c/x_bar``.
    Missing configurations or non-positive coverage remain diagnostic evidence
    with ``kappa=None`` and are excluded from ranking (Algorithm 1).
    """

    validate_problem_instance(instance)
    _, _, tier = _resolve_m2_candidate_identity(instance, configuration_evidence)
    if configuration_evidence.is_feasible:
        _require_feasible_configuration_evidence(tier, configuration_evidence)
    return _evaluate_m2_candidate(instance, configuration_evidence, evaluation_inputs)


def rank_m2_candidates(
    evaluations: tuple[M2CandidateEvaluation, ...],
) -> tuple[M2CandidateEvaluation, ...]:
    """Return rankable candidates in ascending ``(pi,kappa)`` order.

    Exact ties retain supplied semantic order; unrankable candidates are omitted
    without tolerance or secondary ID ordering (Algorithm 1).
    """

    rankable = tuple(
        evaluation for evaluation in evaluations if evaluation.is_rankable
    )
    return tuple(sorted(rankable, key=lambda evaluation: evaluation.rank_key))


def _validate_m2_candidate_identity(
    instance: ProblemInstance,
    selection_evidence: M2ConfigurationEvidence,
) -> tuple[QueryType, FoundationModel, GpuTier]:
    """Validate the instance and resolve one candidate's selection evidence."""

    validate_problem_instance(instance)
    return _resolve_m2_candidate_identity(instance, selection_evidence)


def _resolve_m2_candidate_identity(
    instance: ProblemInstance,
    selection_evidence: M2ConfigurationEvidence,
) -> tuple[QueryType, FoundationModel, GpuTier]:
    """Resolve candidate IDs within an already validated instance."""

    query = _entity_by_id(instance.queries, selection_evidence.query_id, "query")
    model = _entity_by_id(instance.models, selection_evidence.model_id, "model")
    tier = _entity_by_id(
        instance.gpu_tiers, selection_evidence.tier_id, "GPU-tier"
    )
    return query, model, tier


def _require_feasible_configuration_evidence(
    tier: GpuTier,
    selection_evidence: M2ConfigurationEvidence,
) -> tuple[ParallelismConfig, float]:
    """Return selected evidence without recomputing its configuration delay."""

    configuration = selection_evidence.selected_configuration
    delay_seconds = selection_evidence.selected_fixed_configuration_delay_seconds
    if configuration is None:
        raise ValueError("M2 requires feasible configuration evidence")
    if configuration not in tier.parallelism_configs:
        raise ValueError("selected configuration is not valid for its GPU tier")
    if delay_seconds is None or not isfinite(delay_seconds) or delay_seconds < 0.0:
        raise ValueError("selected delay evidence must be finite and non-negative")
    return configuration, delay_seconds


def _compute_m2_marginal_cost(
    instance: ProblemInstance,
    query: QueryType,
    model: FoundationModel,
    tier: GpuTier,
    configuration: ParallelismConfig,
    fixed_configuration_delay_seconds: float,
    evaluation_inputs: M2EvaluationInputs,
) -> M2MarginalCostBreakdown:
    """Evaluate Eq. (10) under the instance's explicit unit convention."""

    required_gpu_count = configuration.gpu_count
    incremental_gpu_count = max(
        required_gpu_count - evaluation_inputs.current_gpu_count,
        0,
    )
    incremental_gpu_cost = (
        instance.horizon_hours
        * tier.hourly_rental_usd
        * incremental_gpu_count
    )
    data_storage = (
        query.storage_gb_per_token
        * query.total_tokens
        * query.arrival_rate_per_hour
    )
    if instance.uses_coherent_units:
        data_storage *= instance.data_storage_duration_hours
    model_and_data_storage_cost = (
        instance.horizon_hours
        * instance.storage_price_usd_per_gb_hour
        * (
            model.weight_gb
            + data_storage
        )
    )
    delay_penalty_cost = (
        query.delay_penalty_usd_per_millisecond_query
        * fixed_configuration_delay_seconds
    )
    if instance.uses_coherent_units:
        delay_penalty_cost = (
            delay_penalty_cost
            * MILLISECONDS_PER_SECOND
            * query.arrival_rate_per_hour
            * instance.horizon_hours
        )
    return M2MarginalCostBreakdown(
        current_gpu_count=evaluation_inputs.current_gpu_count,
        required_gpu_count=required_gpu_count,
        incremental_gpu_cost=incremental_gpu_cost,
        model_and_data_storage_cost=model_and_data_storage_cost,
        delay_penalty_cost=delay_penalty_cost,
    )


def _paper_coverage_limit(
    residual_budget: float,
    unit_consumption: float,
) -> float | None:
    """Return ``residual_budget/unit_consumption`` with Eq. (11) edge handling.

    For zero consumption, return ``None`` if the residual is non-negative,
    otherwise zero: no future consumption can repair an exceeded budget.
    """

    if unit_consumption > 0.0:
        return residual_budget / unit_consumption
    if residual_budget >= 0.0:
        return None
    return 0.0


def _compute_m2_effective_coverage(
    instance: ProblemInstance,
    query: QueryType,
    model: FoundationModel,
    tier: GpuTier,
    fixed_configuration_delay_seconds: float,
    evaluation_inputs: M2EvaluationInputs,
) -> M2EffectiveCoverageDiagnostic:
    """Evaluate Paper v2 Eq. (11) and retain every limiting factor."""

    effective_error = compute_effective_error(
        instance.base_error_rate[(query.id, model.id)],
        tier.quantization_error_multiplier,
    )
    return M2EffectiveCoverageDiagnostic(
        remaining_demand=evaluation_inputs.remaining_demand,
        error_slo=query.error_slo,
        cumulative_error=evaluation_inputs.cumulative_error,
        effective_error=effective_error,
        delay_slo_seconds=query.delay_slo_seconds,
        cumulative_delay_seconds=evaluation_inputs.cumulative_delay_seconds,
        fixed_configuration_delay_seconds=fixed_configuration_delay_seconds,
    )


def _m2_unrankable_reasons(
    coverage_diagnostic: M2EffectiveCoverageDiagnostic,
) -> tuple[M2UnrankableReason, ...]:
    """Derive all reasons for Algorithm 1's non-positive-coverage discard."""

    reasons: list[M2UnrankableReason] = []
    if coverage_diagnostic.remaining_demand == 0.0:
        reasons.append(M2UnrankableReason.NO_REMAINING_DEMAND)
    if coverage_diagnostic.error_coverage_limit is not None:
        if (
            coverage_diagnostic.residual_error_budget == 0.0
            and coverage_diagnostic.effective_error > 0.0
        ):
            reasons.append(M2UnrankableReason.ERROR_BUDGET_EXHAUSTED)
        elif coverage_diagnostic.residual_error_budget < 0.0:
            reasons.append(M2UnrankableReason.ERROR_BUDGET_EXCEEDED)
    if coverage_diagnostic.delay_coverage_limit is not None:
        if (
            coverage_diagnostic.residual_delay_budget_seconds == 0.0
            and coverage_diagnostic.fixed_configuration_delay_seconds > 0.0
        ):
            reasons.append(M2UnrankableReason.DELAY_BUDGET_EXHAUSTED)
        elif coverage_diagnostic.residual_delay_budget_seconds < 0.0:
            reasons.append(M2UnrankableReason.DELAY_BUDGET_EXCEEDED)
    return tuple(reasons)


def _evaluate_m2_candidate(
    instance: ProblemInstance,
    configuration_evidence: M2ConfigurationEvidence,
    evaluation_inputs: M2EvaluationInputs,
) -> M2CandidateEvaluation:
    """Evaluate M2 using a validated instance and matching configuration evidence."""

    query, model, tier = _resolve_m2_candidate_identity(instance, configuration_evidence)
    if not configuration_evidence.is_feasible:
        return M2CandidateEvaluation(
            query_id=query.id,
            model_id=model.id,
            tier_id=tier.id,
            selected_configuration=None,
            cost_breakdown=None,
            coverage_diagnostic=None,
        )

    configuration = configuration_evidence.selected_configuration
    delay_seconds = configuration_evidence.selected_fixed_configuration_delay_seconds
    cost_breakdown = _compute_m2_marginal_cost(
        instance,
        query,
        model,
        tier,
        configuration,
        delay_seconds,
        evaluation_inputs,
    )
    coverage_diagnostic = _compute_m2_effective_coverage(
        instance,
        query,
        model,
        tier,
        delay_seconds,
        evaluation_inputs,
    )
    return M2CandidateEvaluation(
        query_id=query.id,
        model_id=model.id,
        tier_id=tier.id,
        selected_configuration=configuration,
        cost_breakdown=cost_breakdown,
        coverage_diagnostic=coverage_diagnostic,
    )
