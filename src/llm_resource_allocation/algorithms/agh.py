"""Run AGH and independently revalidate the retained incumbent."""

from dataclasses import dataclass
from enum import Enum

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.math.objective import ObjectiveBreakdown, PaperVersion
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    ValidationTolerances,
    validate_placement,
)

from .agh_multistart import (
    AGHMultiStartResult,
    AGHMultiStartTerminationReason,
    run_agh_multistart,
)
from .agh_start import AGHStartResult
from .gh_phase1 import GHPhase1Config


class AGHCompletionStatus(str, Enum):
    """Record whether AGH retained a feasible and consistent final result."""

    COMPLETED = "completed"
    NO_FEASIBLE_INCUMBENT = "no_feasible_incumbent"
    FINAL_CONSISTENCY_FAILED = "final_consistency_failed"


class AGHFinalCheckFailureReason(str, Enum):
    """Describe a missing incumbent or a failed final validation or cost check."""

    NO_FEASIBLE_INCUMBENT = "no_feasible_incumbent"
    FINAL_VALIDATION_INFEASIBLE = "final_validation_infeasible"
    FINAL_OBJECTIVE_UNAVAILABLE = "final_objective_unavailable"
    FINAL_OBJECTIVE_MISMATCH = "final_objective_mismatch"


@dataclass(frozen=True, slots=True)
class AGHResult:
    """Record the AGH search and independent checks of its final incumbent."""

    multistart_result: AGHMultiStartResult
    final_validation: PlacementValidationReport | None
    completion_status: AGHCompletionStatus
    failure_reasons: tuple[AGHFinalCheckFailureReason, ...]

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))

    @property
    def completed(self) -> bool:
        """Return whether the final placement is feasible and its cost is consistent."""

        return self.completion_status is AGHCompletionStatus.COMPLETED

    @property
    def termination_reason(self) -> AGHMultiStartTerminationReason:
        """Return the multi-start search termination reason."""

        return self.multistart_result.termination_reason

    @property
    def executed_start_count(self) -> int:
        """Return the number of executed schedule entries."""

        return self.multistart_result.executed_start_count

    @property
    def unexecuted_start_count(self) -> int:
        """Return the number of schedule entries left after an early stop."""

        return self.multistart_result.unexecuted_start_count

    @property
    def scheduled_random_start_count(self) -> int:
        """Return the scheduled random-start count ``R``, before any early stop."""

        return self.multistart_result.scale.random_start_count

    @property
    def random_seed(self) -> int:
        """Return the seed used to generate random query orderings."""

        return self.multistart_result.random_seed

    @property
    def final_incumbent(self) -> AGHStartResult | None:
        """Return the incumbent retained by the multi-start search."""

        return self.multistart_result.final_incumbent

    @property
    def final_placement(self) -> PlacementDecisionState | None:
        """Return the final placement when one is available."""

        if self.final_incumbent is not None:
            return self.final_incumbent.final_placement
        return None

    @property
    def final_objective_breakdown(self) -> ObjectiveBreakdown | None:
        """Return the final objective evidence from its validation report."""

        if self.final_validation is not None:
            return self.final_validation.objective_breakdown
        return None


def _final_failure_reasons(
    validation: PlacementValidationReport,
    objective: ObjectiveBreakdown | None,
    incumbent: AGHStartResult,
) -> tuple[AGHFinalCheckFailureReason, ...]:
    """Return final feasibility and cost failures in diagnostic order."""

    reasons: list[AGHFinalCheckFailureReason] = []
    if not validation.is_feasible:
        reasons.append(AGHFinalCheckFailureReason.FINAL_VALIDATION_INFEASIBLE)
    if objective is None:
        reasons.append(AGHFinalCheckFailureReason.FINAL_OBJECTIVE_UNAVAILABLE)
    elif (
        incumbent.final_objective_breakdown is None
        or objective != incumbent.final_objective_breakdown
    ):
        reasons.append(AGHFinalCheckFailureReason.FINAL_OBJECTIVE_MISMATCH)
    return tuple(reasons)


def run_agh(
    instance: ProblemInstance,
    config: GHPhase1Config,
    *,
    random_seed: int,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHResult:
    """Run AGH and independently validate the retained incumbent.

    Validate its unchanged placement and compare recomputed cost components
    with the retained result. Record a missing or inconsistent incumbent in
    the completion status and failure reasons.
    """

    multistart_result = run_agh_multistart(
        instance,
        config,
        random_seed=random_seed,
        paper_version=paper_version,
        validation_tolerances=validation_tolerances,
    )
    incumbent = multistart_result.final_incumbent
    if incumbent is None:
        return AGHResult(
            multistart_result=multistart_result,
            final_validation=None,
            completion_status=AGHCompletionStatus.NO_FEASIBLE_INCUMBENT,
            failure_reasons=(AGHFinalCheckFailureReason.NO_FEASIBLE_INCUMBENT,),
        )

    final_placement = incumbent.final_placement
    final_validation = validate_placement(
        instance,
        final_placement,
        paper_version,
        validation_tolerances,
    )
    final_objective_breakdown = final_validation.objective_breakdown
    failure_reasons = _final_failure_reasons(
        final_validation, final_objective_breakdown, incumbent
    )

    return AGHResult(
        multistart_result=multistart_result,
        final_validation=final_validation,
        completion_status=(
            AGHCompletionStatus.COMPLETED
            if not failure_reasons
            else AGHCompletionStatus.FINAL_CONSISTENCY_FAILED
        ),
        failure_reasons=failure_reasons,
    )
