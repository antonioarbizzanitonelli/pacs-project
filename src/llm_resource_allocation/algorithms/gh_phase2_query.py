"""Traverse one frozen GH Phase-2 ranking (v2 Algorithm 1, lines 15-21)."""

from dataclasses import dataclass
from enum import Enum

from llm_resource_allocation.algorithms.gh_phase2 import (
    GHPhase2CandidateEvaluation,
    GHPhase2QueryEvaluation,
    GHPhase2State,
    _evaluate_gh_phase2_query_candidates,
    _validate_phase2_state,
)
from llm_resource_allocation.algorithms.gh_phase2_commit import (
    GHPhase2CommitResult,
    _commit_gh_phase2_candidate,
)
from llm_resource_allocation.domain import ProblemInstance, validate_problem_instance


class GHPhase2QueryTerminationReason(str, Enum):
    """Explain why the one-query sorted-candidate traversal stopped."""

    QUERY_FULLY_SERVED = "query_fully_served"
    CANDIDATES_EXHAUSTED_WITH_RESIDUAL_DEMAND = (
        "candidates_exhausted_with_residual_demand"
    )


@dataclass(frozen=True, slots=True)
class GHPhase2CandidateAttempt:
    """Record one ranked candidate, commit result, and state transition."""

    attempt_index: int
    ranked_candidate_index: int
    candidate: GHPhase2CandidateEvaluation
    commit_result: GHPhase2CommitResult

    @property
    def rank_key(self) -> tuple[int, float] | None:
        """Return the ranking key retained by the candidate."""

        return self.candidate.rank_key

    @property
    def state_before(self) -> GHPhase2State:
        """Return the exact state evaluated by the commit authority."""

        return self.commit_result.state_before

    @property
    def state_after(self) -> GHPhase2State:
        """Return the state produced by the commit authority."""

        return self.commit_result.state_after

    @property
    def committed(self) -> bool:
        """Return whether the candidate commit succeeded."""

        return self.commit_result.committed


@dataclass(frozen=True, slots=True)
class GHPhase2QueryProcessResult:
    """Return the complete immutable result for exactly one query."""

    initial_state: GHPhase2State
    query_evaluation: GHPhase2QueryEvaluation
    attempts: tuple[GHPhase2CandidateAttempt, ...]
    termination_reason: GHPhase2QueryTerminationReason

    @property
    def query_id(self) -> str:
        """Return the evaluated query's identifier."""

        return self.query_evaluation.query_id

    @property
    def final_state(self) -> GHPhase2State:
        """Return the last attempt's state, or the unchanged entry state."""

        return self.attempts[-1].state_after if self.attempts else self.initial_state

    def __post_init__(self) -> None:
        """Freeze directly supplied attempt containers."""

        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def committed_attempt_count(self) -> int:
        """Return the number of successful candidate commits."""

        return sum(attempt.committed for attempt in self.attempts)

    @property
    def rejected_attempt_count(self) -> int:
        """Return the number of rejected candidate attempts."""

        return len(self.attempts) - self.committed_attempt_count


def process_gh_phase2_query_candidates(
    instance: ProblemInstance,
    state: GHPhase2State,
    query_id: str,
    query_evaluation: GHPhase2QueryEvaluation | None = None,
    *,
    expected_state_order: tuple[str, ...] | None = None,
) -> GHPhase2QueryProcessResult:
    """Attempt ranked candidates until the query is served or candidates run out.

    Evaluate once when no batch is supplied; otherwise verify the supplied
    batch against the entry state. Keep M2 values fixed throughout traversal.
    """

    validate_problem_instance(instance)
    _validate_phase2_state(
        instance, state, expected_state_order=expected_state_order
    )
    if query_evaluation is None:
        return _process_gh_phase2_query(instance, state, query_id)
    _validate_query_evaluation(instance, state, query_id, query_evaluation)
    return _process_gh_phase2_query_candidates(
        instance, state, query_id, query_evaluation
    )


def _process_gh_phase2_query(
    instance: ProblemInstance,
    state: GHPhase2State,
    query_id: str,
) -> GHPhase2QueryProcessResult:
    """Evaluate one query once at its validated entry state, then traverse it."""

    evaluation = _evaluate_gh_phase2_query_candidates(instance, state, query_id)
    return _process_gh_phase2_query_candidates(instance, state, query_id, evaluation)


def _process_gh_phase2_query_candidates(
    instance: ProblemInstance,
    state: GHPhase2State,
    query_id: str,
    query_evaluation: GHPhase2QueryEvaluation,
) -> GHPhase2QueryProcessResult:
    """Traverse an internally produced or verified batch without reevaluation."""

    current_state = state
    attempts: list[GHPhase2CandidateAttempt] = []
    context = None
    for ranked_index, candidate in enumerate(query_evaluation.ranked_candidates):
        if current_state.unmet_fraction_by_query[query_id] <= 0.0:
            break
        commit_result, context = _commit_gh_phase2_candidate(
            instance,
            current_state,
            candidate,
            context=context,
        )
        next_state = commit_result.state_after
        attempts.append(
            GHPhase2CandidateAttempt(
                attempt_index=len(attempts),
                ranked_candidate_index=ranked_index,
                candidate=candidate,
                commit_result=commit_result,
            )
        )
        current_state = next_state

    termination_reason = (
        GHPhase2QueryTerminationReason.QUERY_FULLY_SERVED
        if current_state.unmet_fraction_by_query[query_id] <= 0.0
        else GHPhase2QueryTerminationReason.CANDIDATES_EXHAUSTED_WITH_RESIDUAL_DEMAND
    )
    return GHPhase2QueryProcessResult(
        initial_state=state,
        query_evaluation=query_evaluation,
        attempts=tuple(attempts),
        termination_reason=termination_reason,
    )


def _validate_query_evaluation(
    instance: ProblemInstance,
    state: GHPhase2State,
    query_id: str,
    query_evaluation: GHPhase2QueryEvaluation,
) -> None:
    """Require exact candidate evidence for the query, instance, and entry state."""

    if not isinstance(query_evaluation, GHPhase2QueryEvaluation):
        raise ValueError("query_evaluation must be a GHPhase2QueryEvaluation")
    if query_id not in instance.query_ids:
        raise ValueError(f"unknown query ID: {query_id!r}")
    if query_evaluation.query_id != query_id:
        raise ValueError("query evaluation belongs to a different query")
    expected = _evaluate_gh_phase2_query_candidates(instance, state, query_id)
    if query_evaluation != expected:
        raise ValueError(
            "query evaluation is stale or incompatible with the instance and entry state"
        )
