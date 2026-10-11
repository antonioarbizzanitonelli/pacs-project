"""Run GH Phase-1 greedy activation (v2 Algorithm 1, lines 1-6)."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from llm_resource_allocation.algorithms.m1 import (
    _select_minimum_feasible_parallelism,
)
from llm_resource_allocation.algorithms.phase1 import (
    Phase1CandidateEvaluation,
    _evaluate_phase1_candidate,
    rank_phase1_candidates,
)
from llm_resource_allocation.domain import (
    ParallelismConfig,
    ProblemInstance,
    validate_problem_instance,
)


@dataclass(frozen=True, slots=True)
class GHPhase1Config:
    """Configure the printed Phase-1 share ``beta`` of global budget ``delta``."""

    budget_fraction: float

    def __post_init__(self) -> None:
        """Require a finite fraction without silently substituting paper beta=0.8."""

        value = self.budget_fraction
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("budget_fraction must be a real number")
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("budget_fraction must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class GHPhase1Deployment:
    """Record one Phase-1 activation and its common TP/PP configuration."""

    model_id: str
    tier_id: str
    configuration: ParallelismConfig
    covered_query_ids: tuple[str, ...]
    activation_cost_usd: float

    def __post_init__(self) -> None:
        """Protect the transition record from malformed or mutable inputs."""

        object.__setattr__(self, "covered_query_ids", tuple(self.covered_query_ids))
        if not self.covered_query_ids:
            raise ValueError("a Phase-1 deployment must cover at least one query")
        if len(set(self.covered_query_ids)) != len(self.covered_query_ids):
            raise ValueError("covered_query_ids must be unique")
        if not isfinite(self.activation_cost_usd) or self.activation_cost_usd < 0.0:
            raise ValueError("activation_cost_usd must be finite and nonnegative")

    @property
    def pair_id(self) -> tuple[str, str]:
        """Return the semantic active pair ``(j,k)``."""

        return (self.model_id, self.tier_id)

    @property
    def gpu_count(self) -> int:
        """Return the activated ``TP * PP`` count represented in ``y[j,k]``."""

        return self.configuration.gpu_count


@dataclass(frozen=True, slots=True)
class GHPhase1State:
    """Store uncovered queries, deployments, rental cost, cap, and iteration count."""

    uncovered_query_ids: tuple[str, ...]
    active_deployments: tuple[GHPhase1Deployment, ...]
    accumulated_rental_cost_usd: float
    rental_cap_usd: float

    def __post_init__(self) -> None:
        """Freeze sequences and enforce representation-level state invariants."""

        object.__setattr__(
            self, "uncovered_query_ids", tuple(self.uncovered_query_ids)
        )
        object.__setattr__(
            self, "active_deployments", tuple(self.active_deployments)
        )
        if len(set(self.uncovered_query_ids)) != len(self.uncovered_query_ids):
            raise ValueError("uncovered_query_ids must be unique")
        pair_ids = tuple(deployment.pair_id for deployment in self.active_deployments)
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("an active model-tier pair may appear only once")
        for name, value in (
            ("accumulated_rental_cost_usd", self.accumulated_rental_cost_usd),
            ("rental_cap_usd", self.rental_cap_usd),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def iteration_count(self) -> int:
        """Return the number of activations retained in the state."""

        return len(self.active_deployments)

    @property
    def active_pair_ids(self) -> tuple[tuple[str, str], ...]:
        """Return active pairs in activation order."""

        return tuple(deployment.pair_id for deployment in self.active_deployments)


class GHPhase1TerminationReason(str, Enum):
    """Explain why the Phase-1 loop stopped."""

    ALL_QUERIES_COVERED = "all_queries_covered"
    RENTAL_CAP_REACHED = "rental_cap_reached"
    NO_RANKABLE_CANDIDATE = "no_rankable_candidate"


@dataclass(frozen=True, slots=True)
class GHPhase1IterationTrace:
    """Retain complete immutable evidence for one greedy activation."""

    state_before: GHPhase1State
    candidate_evaluations: tuple[Phase1CandidateEvaluation, ...]
    ranked_candidates: tuple[Phase1CandidateEvaluation, ...]
    state_after: GHPhase1State

    @property
    def iteration_index(self) -> int:
        """Return the activation index carried by the entry state."""

        return self.state_before.iteration_count

    @property
    def selected_candidate(self) -> Phase1CandidateEvaluation:
        """Return the first candidate in the producer's greedy ranking."""

        return self.ranked_candidates[0]

    def __post_init__(self) -> None:
        """Freeze candidate sequences even for directly constructed traces."""

        object.__setattr__(
            self, "candidate_evaluations", tuple(self.candidate_evaluations)
        )
        object.__setattr__(self, "ranked_candidates", tuple(self.ranked_candidates))


@dataclass(frozen=True, slots=True)
class GHPhase1Result:
    """Return the Phase-1 deployment, trace, and explicit stopping reason."""

    initial_state: GHPhase1State
    iterations: tuple[GHPhase1IterationTrace, ...]
    termination_reason: GHPhase1TerminationReason
    terminal_candidate_evaluations: tuple[Phase1CandidateEvaluation, ...]

    @property
    def final_state(self) -> GHPhase1State:
        """Return the last activation state, or the unchanged initial state."""

        return (
            self.iterations[-1].state_after if self.iterations else self.initial_state
        )

    def __post_init__(self) -> None:
        """Freeze the complete trace sequence at the public result boundary."""

        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(
            self,
            "terminal_candidate_evaluations",
            tuple(self.terminal_candidate_evaluations),
        )

    @property
    def selected_pair_sequence(self) -> tuple[tuple[str, str], ...]:
        """Return the activation sequence for concise deterministic comparisons."""

        return self.final_state.active_pair_ids


def initialize_gh_phase1_state(
    instance: ProblemInstance,
    config: GHPhase1Config,
) -> GHPhase1State:
    """Initialize Algorithm 1 Phase 1 with ``I_unc=I``, ``y=0``, and cost zero."""

    validate_problem_instance(instance)
    return _initialize_gh_phase1_state(instance, config)


def evaluate_gh_phase1_candidates(
    instance: ProblemInstance,
    state: GHPhase1State,
) -> tuple[Phase1CandidateEvaluation, ...]:
    """Recompute ``F[j,k]`` and ``Cost(j,k)`` at the current ``I_unc``.

    Candidates follow model-then-tier semantic order, including active pairs,
    as required by Algorithm 1's all-pair loop (D-0036).
    """

    validate_problem_instance(instance)
    _validate_state_against_instance(instance, state)
    return _evaluate_gh_phase1_candidates(instance, state)


def apply_gh_phase1_activation(
    instance: ProblemInstance,
    state: GHPhase1State,
    selected_candidate: Phase1CandidateEvaluation,
) -> GHPhase1State:
    """Return the state after one selected Phase-1 activation (Algorithm 1).

    Remove exactly the Eq. (13) coverage set from ``I_unc`` in surviving order,
    store its common configuration as ``y``, and add Eq. (14) rental cost
    without mutating the input (D-0036).
    """

    validate_problem_instance(instance)
    _validate_state_against_instance(instance, state)
    _validate_activation_candidate(state, selected_candidate)
    return _apply_gh_phase1_activation(state, selected_candidate)


def run_gh_phase1(
    instance: ProblemInstance,
    config: GHPhase1Config,
) -> GHPhase1Result:
    """Run the GH Phase-1 activation loop (v2 Algorithm 1, lines 1-6).

    The guard is ``accumulated_rental < beta*delta`` before selection. An
    activation may reach or overshoot the cap; the next guard stops (D-0036).
    """

    validate_problem_instance(instance)
    return _run_gh_phase1(instance, config)


def _validate_state_against_instance(
    instance: ProblemInstance,
    state: GHPhase1State,
) -> None:
    """Reject state/instance identity mismatches at public transition boundaries."""

    if not isinstance(state, GHPhase1State):
        raise ValueError("state must be a GHPhase1State")
    if state.accumulated_rental_cost_usd != sum(
        deployment.activation_cost_usd for deployment in state.active_deployments
    ):
        raise ValueError("accumulated rental cost must equal activation costs")
    known_queries = set(instance.query_ids)
    if not set(state.uncovered_query_ids).issubset(known_queries):
        raise ValueError("state contains an unknown uncovered query ID")
    known_pairs = {
        (model.id, tier.id)
        for model in instance.models
        for tier in instance.gpu_tiers
    }
    if not set(state.active_pair_ids).issubset(known_pairs):
        raise ValueError("state contains an unknown active model-tier pair")
    tier_by_id = {tier.id: tier for tier in instance.gpu_tiers}
    for deployment in state.active_deployments:
        tier = tier_by_id[deployment.tier_id]
        if deployment.configuration not in tier.parallelism_configs:
            raise ValueError("deployment uses a configuration unsupported by its tier")
        expected_cost = (
            instance.horizon_hours
            * tier.hourly_rental_usd
            * deployment.configuration.gpu_count
        )
        if deployment.activation_cost_usd != expected_cost:
            raise ValueError("deployment cost does not equal Eq. (14) rental")
    covered_ids = tuple(
        query_id
        for deployment in state.active_deployments
        for query_id in deployment.covered_query_ids
    )
    if len(set(covered_ids)) != len(covered_ids):
        raise ValueError("a query may be covered by at most one activation")
    if not set(covered_ids).issubset(known_queries):
        raise ValueError("state contains an unknown covered query ID")
    if set(covered_ids).intersection(state.uncovered_query_ids):
        raise ValueError("covered and uncovered query IDs must be disjoint")
    if set(covered_ids).union(state.uncovered_query_ids) != known_queries:
        raise ValueError("state must partition the instance query IDs")


def _initialize_gh_phase1_state(
    instance: ProblemInstance,
    config: GHPhase1Config,
) -> GHPhase1State:
    """Initialize empty Phase 1 on a validated instance; check the config type."""

    if not isinstance(config, GHPhase1Config):
        raise ValueError("config must be a GHPhase1Config")
    return GHPhase1State(
        uncovered_query_ids=instance.query_ids,
        active_deployments=(),
        accumulated_rental_cost_usd=0.0,
        rental_cap_usd=config.budget_fraction * instance.cost_budget_usd,
    )


def _evaluate_gh_phase1_candidates(
    instance: ProblemInstance,
    state: GHPhase1State,
) -> tuple[Phase1CandidateEvaluation, ...]:
    """Reevaluate all pairs on an internally produced or checked Phase-1 state."""

    evaluations: list[Phase1CandidateEvaluation] = []
    for model in instance.models:
        for tier in instance.gpu_tiers:
            m1_selections = tuple(
                _select_minimum_feasible_parallelism(
                    instance,
                    query_id=query_id,
                    model_id=model.id,
                    tier_id=tier.id,
                )
                for query_id in state.uncovered_query_ids
            )
            evaluations.append(
                _evaluate_phase1_candidate(
                    instance,
                    model.id,
                    tier.id,
                    state.uncovered_query_ids,
                    m1_selections,
                )
            )
    return tuple(evaluations)


def _validate_activation_candidate(
    state: GHPhase1State,
    selected_candidate: Phase1CandidateEvaluation,
) -> None:
    """Check externally supplied activation evidence at the public boundary."""

    if selected_candidate.coverage.uncovered_query_ids != state.uncovered_query_ids:
        raise ValueError("selected candidate was not evaluated at the current state")
    if not selected_candidate.is_rankable:
        raise ValueError("selected candidate must be rankable")
    pair_id = (selected_candidate.model_id, selected_candidate.tier_id)
    if pair_id in state.active_pair_ids:
        raise ValueError("selected candidate pair is already active")
    covered = selected_candidate.coverage.covered_query_ids
    covered_set = set(covered)
    if not covered_set or not covered_set.issubset(state.uncovered_query_ids):
        raise ValueError("selected candidate must cover current uncovered queries")
    configuration = selected_candidate.coverage.common_configuration
    activation_cost = selected_candidate.activation.activation_cost_usd
    if configuration is None or activation_cost is None:
        raise ValueError("rankable activation lacks configuration or cost")


def _apply_gh_phase1_activation(
    state: GHPhase1State,
    selected_candidate: Phase1CandidateEvaluation,
) -> GHPhase1State:
    """Activate a producer-ranked or publicly checked candidate."""

    covered = selected_candidate.coverage.covered_query_ids
    covered_set = set(covered)
    configuration = selected_candidate.coverage.common_configuration
    activation_cost = selected_candidate.activation.activation_cost_usd
    remaining = tuple(
        query_id
        for query_id in state.uncovered_query_ids
        if query_id not in covered_set
    )
    deployment = GHPhase1Deployment(
        model_id=selected_candidate.model_id,
        tier_id=selected_candidate.tier_id,
        configuration=configuration,
        covered_query_ids=covered,
        activation_cost_usd=activation_cost,
    )
    return GHPhase1State(
        uncovered_query_ids=remaining,
        active_deployments=state.active_deployments + (deployment,),
        accumulated_rental_cost_usd=(
            state.accumulated_rental_cost_usd + activation_cost
        ),
        rental_cap_usd=state.rental_cap_usd,
    )


def _run_gh_phase1(
    instance: ProblemInstance,
    config: GHPhase1Config,
) -> GHPhase1Result:
    """Run Phase 1 on a validated instance, carrying immutable internal states."""

    initial_state = _initialize_gh_phase1_state(instance, config)
    state = initial_state
    traces: list[GHPhase1IterationTrace] = []

    while (
        state.uncovered_query_ids
        and state.accumulated_rental_cost_usd < state.rental_cap_usd
    ):
        evaluations = _evaluate_gh_phase1_candidates(instance, state)
        ranked = rank_phase1_candidates(evaluations)
        if not ranked:
            return GHPhase1Result(
                initial_state=initial_state,
                iterations=tuple(traces),
                termination_reason=GHPhase1TerminationReason.NO_RANKABLE_CANDIDATE,
                terminal_candidate_evaluations=evaluations,
            )
        selected = ranked[0]
        next_state = _apply_gh_phase1_activation(state, selected)
        traces.append(
            GHPhase1IterationTrace(
                state_before=state,
                candidate_evaluations=evaluations,
                ranked_candidates=ranked,
                state_after=next_state,
            )
        )
        state = next_state

    reason = (
        GHPhase1TerminationReason.ALL_QUERIES_COVERED
        if not state.uncovered_query_ids
        else GHPhase1TerminationReason.RENTAL_CAP_REACHED
    )
    return GHPhase1Result(
        initial_state=initial_state,
        iterations=tuple(traces),
        termination_reason=reason,
        terminal_candidate_evaluations=(),
    )
