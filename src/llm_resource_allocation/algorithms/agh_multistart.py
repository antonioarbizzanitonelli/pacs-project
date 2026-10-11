"""Run deterministic and seeded random AGH starts, retaining the lowest cost."""

from dataclasses import dataclass
from enum import Enum

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.validation.placement import ValidationTolerances

from .agh_ordering import (
    DETERMINISTIC_AGH_ORDERING_SPECS,
    AGHQueryOrdering,
    _build_all_deterministic_agh_orderings,
    build_seeded_random_agh_ordering,
    compute_adaptive_agh_random_start_count,
)
from .agh_start import AGHStartResult, _run_agh_start
from .gh_phase1 import GHPhase1Config, run_gh_phase1
from ._agh_placement_ops import _build_agh_search_context


AGH_MAX_CONSECUTIVE_NONIMPROVING_STARTS = 5
"""Stop after this many consecutive starts leave the incumbent unchanged."""


class AGHStartComparisonOutcome(str, Enum):
    """Describe how one executed start affected the incumbent."""

    INITIAL_INCUMBENT_ESTABLISHED = "initial_incumbent_established"
    STRICT_IMPROVEMENT = "strict_improvement"
    NON_IMPROVING = "non_improving"


class AGHMultiStartTerminationReason(str, Enum):
    """Describe why the sequential schedule ended."""

    SCHEDULE_EXHAUSTED = "schedule_exhausted"
    FIVE_CONSECUTIVE_NONIMPROVING = "five_consecutive_nonimproving"


class AGHMultiStartIncumbentStatus(str, Enum):
    """Record whether a completed feasible start is available."""

    AVAILABLE = "available"
    NO_FEASIBLE_INCUMBENT = "no_feasible_incumbent"


@dataclass(frozen=True, slots=True)
class AGHMultiStartScaleEvidence:
    """Record the problem dimensions and scheduled random-start count."""

    query_type_count: int
    model_count: int
    gpu_tier_count: int

    @property
    def problem_scale(self) -> int:
        """Return the product of query, model, and GPU-tier counts."""

        return self.query_type_count * self.model_count * self.gpu_tier_count

    @property
    def random_start_count(self) -> int:
        """Return the scheduled random-start count for these dimensions."""

        return compute_adaptive_agh_random_start_count(
            self.query_type_count, self.model_count, self.gpu_tier_count
        )


@dataclass(frozen=True, slots=True)
class AGHMultiStartTrace:
    """Record one start's result, incumbent comparison, and stopping counter."""

    schedule_index: int
    start_result: AGHStartResult
    incumbent_before: AGHStartResult | None
    incumbent_after: AGHStartResult | None
    comparison_outcome: AGHStartComparisonOutcome
    consecutive_nonimprovements_before: int

    @property
    def ordering(self) -> AGHQueryOrdering:
        """Return the ordering used by this start."""

        return self.start_result.ordering

    @property
    def candidate_is_feasible_complete(self) -> bool:
        """Return whether this start completed with feasible objective evidence."""

        return _is_feasible_complete_start(self.start_result)

    @property
    def candidate_objective(self) -> float | None:
        """Return the candidate cost when the start is complete and feasible."""

        if self.candidate_is_feasible_complete:
            return self.start_result.final_objective_breakdown.total
        return None

    @property
    def incumbent_objective_before(self) -> float | None:
        """Return the retained incumbent's cost before this start, if available."""

        if self.incumbent_before is not None:
            return self.incumbent_before.final_objective_breakdown.total
        return None

    @property
    def incumbent_objective_after(self) -> float | None:
        """Return the retained incumbent's cost after this start, if available."""

        if self.incumbent_after is not None:
            return self.incumbent_after.final_objective_breakdown.total
        return None

    @property
    def strict_improvement(self) -> bool:
        """Return whether this start strictly improved an existing incumbent."""

        return self.comparison_outcome is AGHStartComparisonOutcome.STRICT_IMPROVEMENT

    @property
    def incumbent_changed(self) -> bool:
        """Return whether this start established or replaced the incumbent."""

        return self.incumbent_after is not self.incumbent_before

    @property
    def consecutive_nonimprovements_after(self) -> int:
        """Return the stopping counter after this start's outcome."""

        return 0 if self.incumbent_changed else self.consecutive_nonimprovements_before + 1


@dataclass(frozen=True, slots=True)
class AGHMultiStartResult:
    """Record executed starts, the retained incumbent, and termination evidence."""

    scale: AGHMultiStartScaleEvidence
    random_seed: int
    start_traces: tuple[AGHMultiStartTrace, ...]
    termination_reason: AGHMultiStartTerminationReason

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(self, "start_traces", tuple(self.start_traces))

    @property
    def has_feasible_incumbent(self) -> bool:
        """Return whether a completed feasible start was retained."""

        return self.incumbent_status is AGHMultiStartIncumbentStatus.AVAILABLE

    @property
    def start_results(self) -> tuple[AGHStartResult, ...]:
        """Return executed start results in schedule order."""

        return tuple(trace.start_result for trace in self.start_traces)

    @property
    def final_incumbent(self) -> AGHStartResult | None:
        """Return the incumbent retained by the last executed start."""

        return self.start_traces[-1].incumbent_after if self.start_traces else None

    @property
    def scheduled_start_count(self) -> int:
        """Return the deterministic and random starts in the full schedule."""

        return len(DETERMINISTIC_AGH_ORDERING_SPECS) + self.scale.random_start_count

    @property
    def executed_start_count(self) -> int:
        """Return the number of recorded starts."""

        return len(self.start_traces)

    @property
    def unexecuted_start_count(self) -> int:
        """Return the number of scheduled starts left unexecuted."""

        return self.scheduled_start_count - self.executed_start_count

    @property
    def incumbent_status(self) -> AGHMultiStartIncumbentStatus:
        """Return whether a final incumbent was retained."""

        if self.final_incumbent is not None:
            return AGHMultiStartIncumbentStatus.AVAILABLE
        return AGHMultiStartIncumbentStatus.NO_FEASIBLE_INCUMBENT

    @property
    def consecutive_nonimprovements(self) -> int:
        """Return the stopping counter at the end of the recorded starts."""

        if self.start_traces:
            return self.start_traces[-1].consecutive_nonimprovements_after
        return 0


def run_agh_multistart(
    instance: ProblemInstance,
    config: GHPhase1Config,
    *,
    random_seed: int,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHMultiStartResult:
    """Try eight deterministic starts followed by up to ``R`` seeded random starts.

    Retain the first completed feasible start, then replace it only for a
    strictly lower total cost. Starts that do not change the incumbent increment
    a shared counter; establishing or replacing it resets the counter. Stop at
    five consecutive nonimprovements or schedule exhaustion.
    """

    context = _build_agh_search_context(instance, paper_version, validation_tolerances)
    _validate_random_seed(random_seed)
    query_count, model_count, tier_count = instance.dimensions
    scale_evidence = AGHMultiStartScaleEvidence(
        query_type_count=query_count,
        model_count=model_count,
        gpu_tier_count=tier_count,
    )
    deterministic_orderings = _build_all_deterministic_agh_orderings(instance)
    phase1_result = run_gh_phase1(instance, config)
    deterministic_count = len(deterministic_orderings)
    scheduled_count = deterministic_count + scale_evidence.random_start_count

    traces: list[AGHMultiStartTrace] = []
    incumbent: AGHStartResult | None = None
    consecutive_nonimprovements = 0
    termination_reason = AGHMultiStartTerminationReason.SCHEDULE_EXHAUSTED

    for schedule_index in range(scheduled_count):
        if schedule_index < deterministic_count:
            ordering = deterministic_orderings[schedule_index]
        else:
            ordering = build_seeded_random_agh_ordering(
                instance.query_ids,
                seed=random_seed,
                random_start_index=schedule_index - deterministic_count,
            )

        start_result = _run_agh_start(context, config, ordering, phase1_result)
        candidate_is_feasible_complete = _is_feasible_complete_start(start_result)
        candidate_objective = (
            start_result.final_objective_breakdown.total
            if candidate_is_feasible_complete
            else None
        )
        incumbent_before = incumbent
        consecutive_nonimprovements_before = consecutive_nonimprovements

        if incumbent is None and candidate_is_feasible_complete:
            incumbent = start_result
            consecutive_nonimprovements = 0
            comparison_outcome = AGHStartComparisonOutcome.INITIAL_INCUMBENT_ESTABLISHED
        elif (
            incumbent is not None
            and candidate_is_feasible_complete
            and candidate_objective < incumbent.final_objective_breakdown.total
        ):
            incumbent = start_result
            consecutive_nonimprovements = 0
            comparison_outcome = AGHStartComparisonOutcome.STRICT_IMPROVEMENT
        else:
            consecutive_nonimprovements += 1
            comparison_outcome = AGHStartComparisonOutcome.NON_IMPROVING

        traces.append(
            AGHMultiStartTrace(
                schedule_index=schedule_index,
                start_result=start_result,
                incumbent_before=incumbent_before,
                incumbent_after=incumbent,
                comparison_outcome=comparison_outcome,
                consecutive_nonimprovements_before=consecutive_nonimprovements_before,
            )
        )
        if (
            consecutive_nonimprovements
            == AGH_MAX_CONSECUTIVE_NONIMPROVING_STARTS
        ):
            termination_reason = (
                AGHMultiStartTerminationReason.FIVE_CONSECUTIVE_NONIMPROVING
            )
            break

    return AGHMultiStartResult(
        scale=scale_evidence,
        random_seed=random_seed,
        start_traces=tuple(traces),
        termination_reason=termination_reason,
    )


def _is_feasible_complete_start(result: AGHStartResult) -> bool:
    """Return whether a start completed with feasible validation and an objective."""

    return (
        result.completed
        and result.final_validation is not None
        and result.final_validation.is_feasible
        and result.final_objective_breakdown is not None
    )


def _validate_random_seed(random_seed: int) -> None:
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
