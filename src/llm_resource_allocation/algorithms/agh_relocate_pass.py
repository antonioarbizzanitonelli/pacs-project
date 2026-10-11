"""Run deterministic first-improvement relocate passes for AGH.

Snapshot positive routes at each pass entry and refresh destinations after
accepted moves. Stop at the first unchanged pass or after three passes.
"""

from dataclasses import dataclass
from enum import Enum

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.math.objective import ObjectiveBreakdown, PaperVersion
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    ValidationTolerances,
)

from ._agh_placement_ops import (
    _AGHPlacementEvaluation,
    _build_agh_search_context,
    _deployment_pair_evidence,
)

from .agh_relocate import (
    AGHRelocateDestinationPair,
    AGHRelocateMoveResult,
    AGHRelocateSourceAssignment,
    _evaluate_agh_relocate_move,
)


AGH_RELOCATE_MAX_PASSES = 3


class AGHRelocateSourceStatus(str, Enum):
    """Describe the scan outcome for one source assignment."""

    NO_IMPROVING_DESTINATION = "no_improving_destination"
    MOVE_ACCEPTED = "move_accepted"
    SOURCE_NO_LONGER_ACTIVE = "source_no_longer_active"


class AGHRelocateRunTerminationReason(str, Enum):
    """Explain why the relocate passes stopped."""

    NO_CHANGE_PASS = "no_change_pass"
    MAX_PASSES_REACHED = "max_passes_reached"


@dataclass(frozen=True, slots=True)
class AGHRelocateAttemptTrace:
    """Record one destination attempt and its relocate result."""

    pass_index: int
    source_position: int
    destination_position: int
    move_result: AGHRelocateMoveResult

    @property
    def source(self) -> AGHRelocateSourceAssignment:
        """Return the source assignment evaluated by the move."""

        return self.move_result.source

    @property
    def destination(self) -> AGHRelocateDestinationPair:
        """Return the destination pair evaluated by the move."""

        return self.move_result.destination

    @property
    def placement_before(self) -> PlacementDecisionState:
        """Return the exact placement evaluated by the move."""

        return self.move_result.placement_before

    @property
    def placement_after(self) -> PlacementDecisionState:
        """Return the placement retained at the end of this operation."""

        return self.move_result.accepted_placement

    @property
    def accepted(self) -> bool:
        """Return the delegated move's acceptance outcome."""

        return self.move_result.accepted


@dataclass(frozen=True, slots=True)
class AGHRelocateSourceTrace:
    """Record the first-improvement destination scan for one source assignment."""

    pass_index: int
    source_position: int
    source: AGHRelocateSourceAssignment
    placement_before: PlacementDecisionState
    eligible_destinations: tuple[AGHRelocateDestinationPair, ...]
    attempts: tuple[AGHRelocateAttemptTrace, ...]
    status: AGHRelocateSourceStatus

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(
            self, "eligible_destinations", tuple(self.eligible_destinations)
        )
        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def accepted_destination(self) -> AGHRelocateDestinationPair | None:
        """Return the first improving destination, if one was accepted."""

        if self.attempts and self.attempts[-1].accepted:
            return self.attempts[-1].destination
        return None

    @property
    def placement_after(self) -> PlacementDecisionState:
        """Return the placement retained at the end of this operation."""

        return self.attempts[-1].placement_after if self.attempts else self.placement_before


@dataclass(frozen=True, slots=True)
class AGHRelocatePassResult:
    """Record one relocate pass, its accepted moves, and validation reports."""

    pass_index: int
    placement_before: PlacementDecisionState
    source_snapshot: tuple[AGHRelocateSourceAssignment, ...]
    source_traces: tuple[AGHRelocateSourceTrace, ...]
    validation_before: PlacementValidationReport
    validation_after: PlacementValidationReport

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(self, "source_snapshot", tuple(self.source_snapshot))
        object.__setattr__(self, "source_traces", tuple(self.source_traces))

    @property
    def attempt_traces(self) -> tuple[AGHRelocateAttemptTrace, ...]:
        """Return all destination attempts in source-scan order."""

        return tuple(attempt for trace in self.source_traces for attempt in trace.attempts)

    @property
    def accepted_move_count(self) -> int:
        """Return the number of accepted relocate moves."""

        return sum(attempt.accepted for attempt in self.attempt_traces)

    @property
    def placement_after(self) -> PlacementDecisionState:
        """Return the placement retained at the end of this operation."""

        if self.source_traces:
            return self.source_traces[-1].placement_after
        return self.placement_before

    @property
    def objective_breakdown_before(self) -> ObjectiveBreakdown | None:
        """Return the initial objective evidence from its validation report."""

        return self.validation_before.objective_breakdown

    @property
    def objective_breakdown_after(self) -> ObjectiveBreakdown | None:
        """Return the final objective evidence when a report is available."""

        return self.validation_after.objective_breakdown

    @property
    def total_objective_improvement(self) -> float | None:
        """Return the initial total minus the final total when both are available."""

        if self.objective_breakdown_before is None or self.objective_breakdown_after is None:
            return None
        return self.objective_breakdown_before.total - self.objective_breakdown_after.total

    @property
    def changed(self) -> bool:
        """Return whether at least one proposal was accepted."""

        return self.accepted_move_count > 0


@dataclass(frozen=True, slots=True)
class AGHRelocateRunResult:
    """Record up to three relocate passes and their termination reason."""

    placement_before: PlacementDecisionState
    max_passes: int
    pass_results: tuple[AGHRelocatePassResult, ...]
    termination_reason: AGHRelocateRunTerminationReason

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(self, "pass_results", tuple(self.pass_results))

    @property
    def passes_executed(self) -> int:
        """Return the number of executed passes."""

        return len(self.pass_results)

    @property
    def placement_after(self) -> PlacementDecisionState:
        """Return the placement retained at the end of this operation."""

        if self.pass_results:
            return self.pass_results[-1].placement_after
        return self.placement_before

    @property
    def accepted_move_count(self) -> int:
        """Return the number of accepted relocate moves."""

        return sum(result.accepted_move_count for result in self.pass_results)

    @property
    def objective_breakdown_before(self) -> ObjectiveBreakdown | None:
        """Return the initial objective evidence from its validation report."""

        return self.pass_results[0].objective_breakdown_before if self.pass_results else None

    @property
    def objective_breakdown_after(self) -> ObjectiveBreakdown | None:
        """Return the final objective evidence when a report is available."""

        return self.pass_results[-1].objective_breakdown_after if self.pass_results else None

    @property
    def total_objective_improvement(self) -> float | None:
        """Return the initial total minus the final total when both are available."""

        if self.objective_breakdown_before is None or self.objective_breakdown_after is None:
            return None
        return self.objective_breakdown_before.total - self.objective_breakdown_after.total


def run_agh_relocate_pass(
    instance: ProblemInstance,
    placement_before: PlacementDecisionState,
    *,
    pass_index: int = 0,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHRelocatePassResult:
    """Run one first-improvement pass over a snapshot of positive routes.

    Visit each source assignment once in route order. Adopt accepted moves
    immediately and select later destination candidates from the current placement.
    """

    _validate_pass_index(pass_index)
    context = _build_agh_search_context(instance, paper_version, validation_tolerances)
    return _run_agh_relocate_pass(context.validate(placement_before), pass_index=pass_index)


def run_agh_relocate(
    instance: ProblemInstance,
    placement_before: PlacementDecisionState,
    *,
    max_passes: int = AGH_RELOCATE_MAX_PASSES,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHRelocateRunResult:
    """Run up to ``max_passes`` relocate passes, stopping at the first unchanged pass.

    ``max_passes`` must be an integer from one through three.
    """

    _validate_max_passes(max_passes)
    context = _build_agh_search_context(instance, paper_version, validation_tolerances)
    return _run_agh_relocate(context.validate(placement_before), max_passes=max_passes)


def _active_destination_candidates(
    problem_data: PdmProblemData,
    placement: PlacementDecisionState,
    source: AGHRelocateSourceAssignment,
) -> tuple[AGHRelocateDestinationPair, ...]:
    """Return other coherent active pairs in model/tier order.

    Each candidate still requires move feasibility and objective evaluation.
    """

    source_pair = source[1:]
    return tuple(
        pair
        for pair in problem_data.variable_lattices.model_tier_keys
        if pair != source_pair
        and _deployment_pair_evidence(
            problem_data, placement, pair
        ).is_coherent_active_deployment
    )


def _scan_relocate_source(
    current_evaluation: _AGHPlacementEvaluation,
    source: AGHRelocateSourceAssignment,
    *,
    pass_index: int,
    source_position: int,
) -> AGHRelocateSourceTrace:
    """Try destination candidates until one improves the source assignment."""

    problem_data = current_evaluation.context.problem_data
    placement_before = current_evaluation.placement
    if placement_before.routing_fractions.get(source, 0.0) <= 0.0:
        return AGHRelocateSourceTrace(
            pass_index=pass_index,
            source_position=source_position,
            source=source,
            placement_before=placement_before,
            eligible_destinations=(),
            attempts=(),
            status=AGHRelocateSourceStatus.SOURCE_NO_LONGER_ACTIVE,
        )

    destinations = _active_destination_candidates(problem_data, placement_before, source)
    attempts: list[AGHRelocateAttemptTrace] = []
    for destination_position, destination in enumerate(destinations):
        move_result = _evaluate_agh_relocate_move(current_evaluation, source, destination)
        attempts.append(
            AGHRelocateAttemptTrace(
                pass_index=pass_index,
                source_position=source_position,
                destination_position=destination_position,
                move_result=move_result,
            )
        )
        if move_result.accepted:
            return AGHRelocateSourceTrace(
                pass_index=pass_index,
                source_position=source_position,
                source=source,
                placement_before=placement_before,
                eligible_destinations=destinations,
                attempts=tuple(attempts),
                status=AGHRelocateSourceStatus.MOVE_ACCEPTED,
            )

    return AGHRelocateSourceTrace(
        pass_index=pass_index,
        source_position=source_position,
        source=source,
        placement_before=placement_before,
        eligible_destinations=destinations,
        attempts=tuple(attempts),
        status=AGHRelocateSourceStatus.NO_IMPROVING_DESTINATION,
    )


def _run_agh_relocate_pass(
    current_evaluation: _AGHPlacementEvaluation,
    *,
    pass_index: int,
) -> AGHRelocatePassResult:
    """Run one pass while reusing the current placement's validation report."""

    problem_data = current_evaluation.context.problem_data
    placement_before = current_evaluation.placement
    validation_before = current_evaluation.validation
    source_snapshot = tuple(
        source
        for source in problem_data.variable_lattices.route_keys
        if placement_before.routing_fractions.get(source, 0.0) > 0.0
    )
    source_traces: list[AGHRelocateSourceTrace] = []
    for source_position, source in enumerate(source_snapshot):
        source_trace = _scan_relocate_source(
            current_evaluation,
            source,
            pass_index=pass_index,
            source_position=source_position,
        )
        source_traces.append(source_trace)
        if source_trace.status is AGHRelocateSourceStatus.MOVE_ACCEPTED:
            accepted_move = source_trace.attempts[-1].move_result
            current_evaluation = _AGHPlacementEvaluation(
                current_evaluation.context,
                accepted_move.accepted_placement,
                accepted_move.validation_report,
            )

    validation_after = current_evaluation.validation
    return AGHRelocatePassResult(
        pass_index=pass_index,
        placement_before=placement_before,
        source_snapshot=source_snapshot,
        source_traces=tuple(source_traces),
        validation_before=validation_before,
        validation_after=validation_after,
    )


def _run_agh_relocate(
    current_evaluation: _AGHPlacementEvaluation,
    *,
    max_passes: int = AGH_RELOCATE_MAX_PASSES,
) -> AGHRelocateRunResult:
    """Run relocate passes until no move is accepted or the pass limit is reached."""

    placement_before = current_evaluation.placement
    pass_results: list[AGHRelocatePassResult] = []
    termination_reason = AGHRelocateRunTerminationReason.MAX_PASSES_REACHED
    for pass_index in range(max_passes):
        pass_result = _run_agh_relocate_pass(current_evaluation, pass_index=pass_index)
        pass_results.append(pass_result)
        current_evaluation = _AGHPlacementEvaluation(
            current_evaluation.context, pass_result.placement_after, pass_result.validation_after
        )
        if not pass_result.changed:
            termination_reason = AGHRelocateRunTerminationReason.NO_CHANGE_PASS
            break

    return AGHRelocateRunResult(
        placement_before=placement_before,
        max_passes=max_passes,
        pass_results=tuple(pass_results),
        termination_reason=termination_reason,
    )


def _validate_pass_index(pass_index: int) -> None:
    """Require a nonnegative integer pass index."""

    if isinstance(pass_index, bool) or not isinstance(pass_index, int) or pass_index < 0:
        raise ValueError("pass_index must be a nonnegative integer")


def _validate_max_passes(max_passes: int) -> None:
    """Require an integer pass limit from one through three."""

    if (
        isinstance(max_passes, bool)
        or not isinstance(max_passes, int)
        or not 1 <= max_passes <= AGH_RELOCATE_MAX_PASSES
    ):
        raise ValueError("max_passes must be an integer in [1, 3]")
