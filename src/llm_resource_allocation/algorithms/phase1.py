"""Evaluate GH Phase-1 coverage and activation score (v2 Eqs. 13-14)."""

from dataclasses import dataclass
from enum import Enum
from math import inf

from llm_resource_allocation.algorithms.m1 import (
    M1ConfigurationDiagnostic,
    M1SelectionResult,
)
from llm_resource_allocation.domain import (
    GpuTier,
    ParallelismConfig,
    ProblemInstance,
    validate_problem_instance,
)
from llm_resource_allocation.math.error import compute_effective_error

from ._mechanism_support import _entity_by_id


class Phase1QueryExclusionReason(str, Enum):
    """Explain exclusion from the literal Paper v2 Eq. (13) set."""

    M1_INFEASIBLE = "m1_infeasible"
    ERROR_SLO_EXCEEDED = "error_slo_exceeded"


class Phase1NonRankableReason(str, Enum):
    """Explain why a static Phase-1 candidate has no selection score."""

    EMPTY_COVERAGE = "empty_coverage"
    NO_COMMON_CONFIGURATION_AT_EQ14_GPU_COUNT = (
        "no_common_configuration_at_eq14_gpu_count"
    )


@dataclass(frozen=True, slots=True)
class Phase1QueryCoverageDiagnostic:
    """Retain all Eq. (13) evidence for one currently uncovered query."""

    query_id: str
    m1_selection: M1SelectionResult
    effective_error: float
    error_slo: float

    @property
    def error_condition_passes(self) -> bool:
        return self.effective_error <= self.error_slo

    @property
    def exclusion_reasons(self) -> tuple[Phase1QueryExclusionReason, ...]:
        reasons: list[Phase1QueryExclusionReason] = []
        if not self.m1_selection.is_feasible:
            reasons.append(Phase1QueryExclusionReason.M1_INFEASIBLE)
        if not self.error_condition_passes:
            reasons.append(Phase1QueryExclusionReason.ERROR_SLO_EXCEEDED)
        return tuple(reasons)

    @property
    def is_covered(self) -> bool:
        """Return whether the query belongs to the literal coverage set."""

        return not self.exclusion_reasons


@dataclass(frozen=True, slots=True)
class Phase1CommonConfigurationQueryDiagnostic:
    """Expose one covered query's M1 checks at a proposed common shape."""

    query_id: str
    m1_configuration_diagnostic: M1ConfigurationDiagnostic

    @property
    def is_compatible(self) -> bool:
        """Return whether the proposed shape passes both existing M1 checks."""

        return self.m1_configuration_diagnostic.is_feasible


@dataclass(frozen=True, slots=True)
class Phase1CommonConfigurationDiagnostic:
    """Check one tier-valid shape at the Eq. (14) GPU count."""

    configuration: ParallelismConfig
    required_gpu_count: int
    query_diagnostics: tuple[Phase1CommonConfigurationQueryDiagnostic, ...]

    @property
    def gpu_count_matches_eq14(self) -> bool:
        return self.configuration.gpu_count == self.required_gpu_count

    @property
    def supports_all_covered_queries(self) -> bool:
        """Return whether this exact Eq. (14)-sized shape serves all members."""

        return self.gpu_count_matches_eq14 and all(
            diagnostic.is_compatible for diagnostic in self.query_diagnostics
        )


@dataclass(frozen=True, slots=True)
class Phase1CoverageResult:
    """Store the Eq. (13) coverage set and common-configuration evidence."""

    model_id: str
    tier_id: str
    uncovered_query_ids: tuple[str, ...]
    query_diagnostics: tuple[Phase1QueryCoverageDiagnostic, ...]
    common_configuration_diagnostics: tuple[
        Phase1CommonConfigurationDiagnostic, ...
    ]

    def __post_init__(self) -> None:
        object.__setattr__(self, "uncovered_query_ids", tuple(self.uncovered_query_ids))
        object.__setattr__(self, "query_diagnostics", tuple(self.query_diagnostics))
        object.__setattr__(
            self,
            "common_configuration_diagnostics",
            tuple(self.common_configuration_diagnostics),
        )

    @property
    def covered_query_ids(self) -> tuple[str, ...]:
        return tuple(item.query_id for item in self.query_diagnostics if item.is_covered)

    @property
    def excluded_query_ids(self) -> tuple[str, ...]:
        return tuple(item.query_id for item in self.query_diagnostics if not item.is_covered)

    @property
    def required_gpu_count(self) -> int | None:
        counts = tuple(
            item.m1_selection.selected_configuration.gpu_count
            for item in self.query_diagnostics
            if item.is_covered and item.m1_selection.selected_configuration is not None
        )
        return max(counts) if counts else None

    @property
    def common_configuration(self) -> ParallelismConfig | None:
        return next(
            (
                item.configuration
                for item in self.common_configuration_diagnostics
                if item.supports_all_covered_queries
            ),
            None,
        )

    @property
    def is_empty(self) -> bool:
        """Return whether Eq. (13) contains no currently uncovered query."""

        return not self.covered_query_ids


@dataclass(frozen=True, slots=True)
class Phase1ActivationScore:
    """Record Eq. (14) rental cost in USD and the ``|F|/Cost`` ratio."""

    coverage_count: int
    required_gpu_count: int | None
    activation_cost_usd: float | None
    non_rankable_reasons: tuple[Phase1NonRankableReason, ...]

    @property
    def coverage_per_cost(self) -> float | None:
        if self.non_rankable_reasons or self.activation_cost_usd is None:
            return None
        return (
            inf
            if self.activation_cost_usd == 0.0
            else self.coverage_count / self.activation_cost_usd
        )

    @property
    def is_rankable(self) -> bool:
        """Return whether Algorithm 1 may compare this candidate's ratio."""

        return self.coverage_per_cost is not None


@dataclass(frozen=True, slots=True)
class Phase1CandidateEvaluation:
    """Compose immutable coverage and activation evidence for one ``(j,k)``."""

    coverage: Phase1CoverageResult
    activation: Phase1ActivationScore

    @property
    def model_id(self) -> str:
        """Return the candidate model identifier."""

        return self.coverage.model_id

    @property
    def tier_id(self) -> str:
        """Return the candidate tier identifier."""

        return self.coverage.tier_id

    @property
    def is_rankable(self) -> bool:
        """Return whether the candidate participates in the static ranking."""

        return self.activation.is_rankable


def compute_phase1_pair_coverage(
    instance: ProblemInstance,
    model_id: str,
    tier_id: str,
    uncovered_query_ids: tuple[str, ...],
    m1_selections: tuple[M1SelectionResult, ...],
) -> Phase1CoverageResult:
    """Compute Eq. (13) coverage and common TP/PP diagnostics for one model-tier pair.

    Reuse M1 evidence and check each query's error SLO, preserving query order.
    The result selects the first tier-ordered configuration at the Eq. (14)
    GPU count that passes all covered queries' M1 checks.
    Keep the coverage set unchanged if no common configuration exists.
    """

    validate_problem_instance(instance)
    uncovered = _validate_uncovered_query_ids(instance, uncovered_query_ids)
    tier = _entity_by_id(instance.gpu_tiers, tier_id, "GPU-tier")
    _validate_and_index_m1_evidence(tier, model_id, uncovered, m1_selections)
    return _compute_phase1_pair_coverage(instance, model_id, tier_id, uncovered_query_ids, m1_selections)


def compute_phase1_activation_score(
    instance: ProblemInstance,
    coverage: Phase1CoverageResult,
) -> Phase1ActivationScore:
    """Return ``Cost = DeltaT*p_gpu[k]*max_i(n_i*m_i)`` and ``|F|/Cost``.

    This is Eq. (14) rental cost in USD. ``DeltaT`` is ``instance.horizon_hours``;
    ``p_gpu`` is ``tier.hourly_rental_usd``. Empty coverage or a missing
    common shape is non-rankable; nonempty zero-cost coverage has infinite ratio.
    """

    validate_problem_instance(instance)
    if tuple(item.query_id for item in coverage.query_diagnostics) != coverage.uncovered_query_ids:
        raise ValueError("query diagnostics must follow uncovered query order")
    if not coverage.is_empty and coverage.required_gpu_count is None:
        raise ValueError("nonempty Phase-1 coverage requires a GPU count")
    return _compute_phase1_activation_score(instance, coverage)


def evaluate_phase1_candidate(
    instance: ProblemInstance,
    model_id: str,
    tier_id: str,
    uncovered_query_ids: tuple[str, ...],
    m1_selections: tuple[M1SelectionResult, ...],
) -> Phase1CandidateEvaluation:
    """Compose coverage and activation scoring for one static ``(j,k)`` pair."""

    validate_problem_instance(instance)
    uncovered = _validate_uncovered_query_ids(instance, uncovered_query_ids)
    tier = _entity_by_id(instance.gpu_tiers, tier_id, "GPU-tier")
    _validate_and_index_m1_evidence(tier, model_id, uncovered, m1_selections)
    return _evaluate_phase1_candidate(instance, model_id, tier_id, uncovered_query_ids, m1_selections)


def rank_phase1_candidates(
    evaluations: tuple[Phase1CandidateEvaluation, ...],
) -> tuple[Phase1CandidateEvaluation, ...]:
    """Return rankable candidates by descending ``|F|/Cost`` (Algorithm 1).

    Exact ties retain supplied semantic order.
    """

    rankable = tuple(evaluation for evaluation in evaluations if evaluation.is_rankable)
    return tuple(
        sorted(
            rankable,
            key=lambda evaluation: evaluation.activation.coverage_per_cost,
            reverse=True,
        )
    )


def _validate_uncovered_query_ids(
    instance: ProblemInstance,
    uncovered_query_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Require an immutable, unique, known query sequence in caller order."""

    if not isinstance(uncovered_query_ids, tuple):
        raise ValueError("uncovered_query_ids must be an ordered tuple")
    known = set(instance.query_ids)
    seen: set[str] = set()
    for query_id in uncovered_query_ids:
        if query_id not in known:
            raise ValueError(f"unknown uncovered query ID: {query_id!r}")
        if query_id in seen:
            raise ValueError(f"duplicate uncovered query ID: {query_id!r}")
        seen.add(query_id)
    return uncovered_query_ids


def _validate_and_index_m1_evidence(
    tier: GpuTier,
    model_id: str,
    uncovered_query_ids: tuple[str, ...],
    m1_selections: tuple[M1SelectionResult, ...],
) -> dict[str, M1SelectionResult]:
    """Validate one complete M1 result per uncovered query without recomputing M1."""

    if not isinstance(m1_selections, tuple):
        raise ValueError("m1_selections must be an ordered tuple")
    expected_configurations = tier.parallelism_configs
    expected_queries = set(uncovered_query_ids)
    indexed: dict[str, M1SelectionResult] = {}
    for selection in m1_selections:
        if selection.model_id != model_id or selection.tier_id != tier.id:
            raise ValueError("M1 evidence belongs to a different Phase-1 candidate")
        if selection.query_id not in expected_queries:
            raise ValueError(
                f"M1 evidence has unexpected query ID: {selection.query_id!r}"
            )
        if selection.query_id in indexed:
            raise ValueError(
                f"duplicate M1 evidence for query ID: {selection.query_id!r}"
            )
        diagnosed_configurations = tuple(
            diagnostic.configuration
            for diagnostic in selection.configuration_diagnostics
        )
        if diagnosed_configurations != expected_configurations:
            raise ValueError(
                "M1 diagnostics do not match the selected tier's configuration order"
            )
        if selection.selected_configuration is not None:
            matching = tuple(
                diagnostic
                for diagnostic in selection.configuration_diagnostics
                if diagnostic.configuration == selection.selected_configuration
            )
            if len(matching) != 1 or not matching[0].is_feasible:
                raise ValueError("M1 selected configuration lacks feasible evidence")
        indexed[selection.query_id] = selection
    missing = tuple(
        query_id for query_id in uncovered_query_ids if query_id not in indexed
    )
    if missing:
        raise ValueError(f"missing M1 evidence for uncovered queries: {missing!r}")
    return indexed


def _m1_diagnostic_for_configuration(
    selection: M1SelectionResult,
    configuration: ParallelismConfig,
) -> M1ConfigurationDiagnostic:
    """Return already-computed M1 evidence for one exact tier-valid shape."""

    return next(
        diagnostic
        for diagnostic in selection.configuration_diagnostics
        if diagnostic.configuration == configuration
    )


def _compute_phase1_pair_coverage(
    instance: ProblemInstance,
    model_id: str,
    tier_id: str,
    uncovered_query_ids: tuple[str, ...],
    m1_selections: tuple[M1SelectionResult, ...],
) -> Phase1CoverageResult:
    """Reuse matching M1 results on a validated instance to derive pair coverage."""

    model = _entity_by_id(instance.models, model_id, "model")
    tier = _entity_by_id(instance.gpu_tiers, tier_id, "GPU-tier")
    uncovered = uncovered_query_ids
    evidence_by_query = {selection.query_id: selection for selection in m1_selections}

    diagnostics: list[Phase1QueryCoverageDiagnostic] = []
    for query_id in uncovered:
        query = _entity_by_id(instance.queries, query_id, "query")
        selection = evidence_by_query[query_id]
        effective_error = compute_effective_error(
            instance.base_error_rate[(query.id, model.id)],
            tier.quantization_error_multiplier,
        )
        diagnostics.append(
            Phase1QueryCoverageDiagnostic(
                query_id=query.id,
                m1_selection=selection,
                effective_error=effective_error,
                error_slo=query.error_slo,
            )
        )

    query_diagnostics = tuple(diagnostics)
    covered = tuple(
        diagnostic.query_id
        for diagnostic in query_diagnostics
        if diagnostic.is_covered
    )
    if not covered:
        return Phase1CoverageResult(
            model_id=model.id,
            tier_id=tier.id,
            uncovered_query_ids=uncovered,
            query_diagnostics=query_diagnostics,
            common_configuration_diagnostics=(),
        )

    required_gpu_count = max(
        evidence_by_query[query_id].selected_configuration.gpu_count
        for query_id in covered
        if evidence_by_query[query_id].selected_configuration is not None
    )
    common_diagnostics = tuple(
        Phase1CommonConfigurationDiagnostic(
            configuration=configuration,
            required_gpu_count=required_gpu_count,
            query_diagnostics=tuple(
                Phase1CommonConfigurationQueryDiagnostic(
                    query_id=query_id,
                    m1_configuration_diagnostic=(
                        _m1_diagnostic_for_configuration(
                            evidence_by_query[query_id], configuration
                        )
                    ),
                )
                for query_id in covered
            ),
        )
        for configuration in tier.parallelism_configs
    )
    return Phase1CoverageResult(
        model_id=model.id,
        tier_id=tier.id,
        uncovered_query_ids=uncovered,
        query_diagnostics=query_diagnostics,
        common_configuration_diagnostics=common_diagnostics,
    )


def _compute_phase1_activation_score(
    instance: ProblemInstance,
    coverage: Phase1CoverageResult,
) -> Phase1ActivationScore:
    """Derive rental cost from checked coverage on a validated instance."""

    tier = _entity_by_id(instance.gpu_tiers, coverage.tier_id, "GPU-tier")
    _entity_by_id(instance.models, coverage.model_id, "model")
    coverage_count = len(coverage.covered_query_ids)
    if coverage_count == 0:
        return Phase1ActivationScore(
            coverage_count=0,
            required_gpu_count=None,
            activation_cost_usd=None,
            non_rankable_reasons=(Phase1NonRankableReason.EMPTY_COVERAGE,),
        )

    required_gpu_count = coverage.required_gpu_count
    activation_cost = (
        instance.horizon_hours
        * tier.hourly_rental_usd
        * required_gpu_count
    )
    if coverage.common_configuration is None:
        return Phase1ActivationScore(
            coverage_count=coverage_count,
            required_gpu_count=required_gpu_count,
            activation_cost_usd=activation_cost,
            non_rankable_reasons=(
                Phase1NonRankableReason.NO_COMMON_CONFIGURATION_AT_EQ14_GPU_COUNT,
            ),
        )
    return Phase1ActivationScore(
        coverage_count=coverage_count,
        required_gpu_count=coverage.common_configuration.gpu_count,
        activation_cost_usd=activation_cost,
        non_rankable_reasons=(),
    )


def _evaluate_phase1_candidate(
    instance: ProblemInstance,
    model_id: str,
    tier_id: str,
    uncovered_query_ids: tuple[str, ...],
    m1_selections: tuple[M1SelectionResult, ...],
) -> Phase1CandidateEvaluation:
    """Combine coverage and rental score without revalidating the instance."""

    coverage = _compute_phase1_pair_coverage(
        instance,
        model_id,
        tier_id,
        uncovered_query_ids,
        m1_selections,
    )
    return Phase1CandidateEvaluation(
        coverage=coverage,
        activation=_compute_phase1_activation_score(instance, coverage),
    )
