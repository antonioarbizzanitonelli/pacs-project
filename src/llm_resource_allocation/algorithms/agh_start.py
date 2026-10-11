"""Run GH, relocate, and consolidation for one supplied query ordering."""

from dataclasses import dataclass
from enum import Enum

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.math.objective import ObjectiveBreakdown, PaperVersion
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    ValidationTolerances,
)

from .agh_consolidation import (
    AGHConsolidationScanResult,
    _run_agh_consolidation_scan,
)
from .agh_ordering import AGHQueryOrdering
from .agh_relocate_pass import (
    AGH_RELOCATE_MAX_PASSES,
    AGHRelocateRunResult,
    _run_agh_relocate,
)
from .gh import GHResult, _run_gh_with_phase1
from .gh_phase1 import GHPhase1Config, GHPhase1Result, run_gh_phase1
from ._agh_placement_ops import (
    _AGHSearchContext,
    _AGHPlacementEvaluation,
    _build_agh_search_context,
)


class AGHStartCompletionStatus(str, Enum):
    """Record completion or the stage at which one AGH start stopped."""

    COMPLETED = "completed"
    STOPPED_AFTER_GH = "stopped_after_gh"
    STOPPED_AFTER_RELOCATE = "stopped_after_relocate"
    FINAL_EVALUATION_FAILED = "final_evaluation_failed"


class AGHStartFailureReason(str, Enum):
    """Describe why an AGH start could not produce a feasible final placement."""

    GH_ALGORITHM_NOT_COMPLETED = "gh_algorithm_not_completed"
    GH_CANONICALIZATION_NOT_COMPLETED = "gh_canonicalization_not_completed"
    GH_INDEPENDENTLY_INFEASIBLE = "gh_independently_infeasible"
    GH_OBJECTIVE_UNAVAILABLE = "gh_objective_unavailable"
    RELOCATE_INDEPENDENTLY_INFEASIBLE = "relocate_independently_infeasible"
    RELOCATE_OBJECTIVE_UNAVAILABLE = "relocate_objective_unavailable"
    FINAL_INDEPENDENTLY_INFEASIBLE = "final_independently_infeasible"
    FINAL_OBJECTIVE_UNAVAILABLE = "final_objective_unavailable"


@dataclass(frozen=True, slots=True)
class AGHStartResult:
    """Record one AGH start's stage outputs, validation, and completion status."""

    ordering: AGHQueryOrdering
    gh_result: GHResult
    relocate_result: AGHRelocateRunResult | None
    consolidation_result: AGHConsolidationScanResult | None
    completion_status: AGHStartCompletionStatus
    failure_reasons: tuple[AGHStartFailureReason, ...]

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))

    @property
    def completed(self) -> bool:
        """Return whether all stages completed with a feasible final placement."""

        return self.completion_status is AGHStartCompletionStatus.COMPLETED

    @property
    def gh_placement(self) -> PlacementDecisionState:
        """Return the placement produced by GH."""

        return self.gh_result.placement_decisions

    @property
    def relocate_placement(self) -> PlacementDecisionState | None:
        """Return the relocate output when that stage was reached."""

        if self.relocate_result is not None:
            return self.relocate_result.placement_after
        return None

    @property
    def consolidation_placement(self) -> PlacementDecisionState | None:
        """Return the consolidation output when that stage was reached."""

        if self.consolidation_result is not None:
            return self.consolidation_result.placement_after
        return None

    @property
    def final_placement(self) -> PlacementDecisionState | None:
        """Return the final placement when one is available."""

        return self.consolidation_placement

    @property
    def final_validation(self) -> PlacementValidationReport | None:
        """Return the final consolidation validation when that stage was reached."""

        if self.consolidation_result is not None:
            return self.consolidation_result.validation_after
        return None

    @property
    def final_objective_breakdown(self) -> ObjectiveBreakdown | None:
        """Return the final objective evidence from its validation report."""

        if self.final_validation is not None:
            return self.final_validation.objective_breakdown
        return None


def run_agh_start(
    instance: ProblemInstance,
    config: GHPhase1Config,
    ordering: AGHQueryOrdering,
    *,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHStartResult:
    """Run GH, up to three relocate passes, and one consolidation scan.

    Stop if GH is incomplete or a stage output is infeasible or lacks an
    objective. Final feasibility and cost come from independent validation.
    """

    if not isinstance(ordering, AGHQueryOrdering):
        raise ValueError("ordering must be an AGHQueryOrdering")
    context = _build_agh_search_context(instance, paper_version, validation_tolerances)
    phase1_result = run_gh_phase1(instance, config)
    return _run_agh_start(context, config, ordering, phase1_result)


def _gh_failure_reasons(result: GHResult) -> tuple[AGHStartFailureReason, ...]:
    reasons: list[AGHStartFailureReason] = []
    if not result.algorithm_completed:
        reasons.append(AGHStartFailureReason.GH_ALGORITHM_NOT_COMPLETED)
    if not result.canonicalization_completed:
        reasons.append(AGHStartFailureReason.GH_CANONICALIZATION_NOT_COMPLETED)
    if not result.independently_feasible:
        reasons.append(AGHStartFailureReason.GH_INDEPENDENTLY_INFEASIBLE)
    if result.objective_breakdown is None:
        reasons.append(AGHStartFailureReason.GH_OBJECTIVE_UNAVAILABLE)
    return tuple(reasons)


def _relocate_failure_reasons(
    result: AGHRelocateRunResult,
) -> tuple[AGHStartFailureReason, ...]:
    validation = result.pass_results[-1].validation_after
    reasons: list[AGHStartFailureReason] = []
    if not validation.is_feasible:
        reasons.append(AGHStartFailureReason.RELOCATE_INDEPENDENTLY_INFEASIBLE)
    if result.objective_breakdown_after is None:
        reasons.append(AGHStartFailureReason.RELOCATE_OBJECTIVE_UNAVAILABLE)
    return tuple(reasons)


def _final_failure_reasons(
    validation: PlacementValidationReport,
) -> tuple[AGHStartFailureReason, ...]:
    reasons: list[AGHStartFailureReason] = []
    if not validation.is_feasible:
        reasons.append(AGHStartFailureReason.FINAL_INDEPENDENTLY_INFEASIBLE)
    if validation.objective_breakdown is None:
        reasons.append(AGHStartFailureReason.FINAL_OBJECTIVE_UNAVAILABLE)
    return tuple(reasons)


def _run_agh_start(
    context: _AGHSearchContext,
    config: GHPhase1Config,
    ordering: AGHQueryOrdering,
    phase1_result: GHPhase1Result,
) -> AGHStartResult:
    """Run one AGH start using a shared context and precomputed GH Phase 1."""

    gh_result = _run_gh_with_phase1(
        context.problem_data.instance,
        config,
        phase1_result,
        paper_version=context.validator.paper_version,
        validation_tolerances=context.validator.tolerances,
        query_order_override=ordering.query_ids,
        problem_data=context.problem_data,
    )
    gh_placement = gh_result.placement_decisions
    gh_failures = _gh_failure_reasons(gh_result)
    if gh_failures:
        return AGHStartResult(
            ordering=ordering,
            gh_result=gh_result,
            relocate_result=None,
            consolidation_result=None,
            completion_status=AGHStartCompletionStatus.STOPPED_AFTER_GH,
            failure_reasons=gh_failures,
        )

    current_evaluation = _AGHPlacementEvaluation(
        context, gh_placement, gh_result.validation_report
    )
    relocate_result = _run_agh_relocate(
        current_evaluation,
        max_passes=AGH_RELOCATE_MAX_PASSES,
    )
    relocate_placement = relocate_result.placement_after
    relocate_failures = _relocate_failure_reasons(relocate_result)
    if relocate_failures:
        return AGHStartResult(
            ordering=ordering,
            gh_result=gh_result,
            relocate_result=relocate_result,
            consolidation_result=None,
            completion_status=(
                AGHStartCompletionStatus.STOPPED_AFTER_RELOCATE
            ),
            failure_reasons=relocate_failures,
        )

    current_evaluation = _AGHPlacementEvaluation(
        context, relocate_placement, relocate_result.pass_results[-1].validation_after
    )
    consolidation_result = _run_agh_consolidation_scan(current_evaluation)
    final_validation = consolidation_result.validation_after
    final_failures = _final_failure_reasons(final_validation)
    return AGHStartResult(
        ordering=ordering,
        gh_result=gh_result,
        relocate_result=relocate_result,
        consolidation_result=consolidation_result,
        completion_status=(
            AGHStartCompletionStatus.FINAL_EVALUATION_FAILED
            if final_failures
            else AGHStartCompletionStatus.COMPLETED
        ),
        failure_reasons=final_failures,
    )
