"""Initialize GH Phase 2 and rank one query's candidates (Algorithm 1, lines 7-14)."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.algorithms.gh_phase1 import (
    GHPhase1Result,
    _validate_state_against_instance,
)
from llm_resource_allocation.algorithms.m1 import (
    M1SelectionResult,
    _select_minimum_feasible_parallelism,
)
from llm_resource_allocation.algorithms.m2 import (
    M2CandidateEvaluation,
    M2EvaluationInputs,
    _evaluate_m2_candidate,
    rank_m2_candidates,
)
from llm_resource_allocation.algorithms.m3 import (
    M3EvaluationState,
    M3UpgradeResult,
    _select_m3_parallelism_upgrade,
)
from llm_resource_allocation.domain import (
    ParallelismConfig,
    ProblemInstance,
    validate_problem_instance,
)
from llm_resource_allocation.math.latency import (
    compute_configuration_delay_seconds,
)


class GHPhase2ConfigurationSource(str, Enum):
    """Identify the Algorithm-1 branch supplying the configuration."""

    M1_INACTIVE_PAIR = "m1_inactive_pair"
    CURRENT_ACTIVE_CONFIGURATION = "current_active_configuration"
    M3_ACTIVE_PAIR_UPGRADE = "m3_active_pair_upgrade"


class GHPhase2CandidateRejectionReason(str, Enum):
    """Explain why one Phase-2 candidate cannot enter the M2 ranking."""

    M1_NO_FEASIBLE_CONFIGURATION = "m1_no_feasible_configuration"
    M3_NO_FEASIBLE_UPGRADE = "m3_no_feasible_upgrade"
    M2_ZERO_EFFECTIVE_COVERAGE = "m2_zero_effective_coverage"


@dataclass(frozen=True, slots=True)
class GHPhase2Deployment:
    """Represent current ``y[j,k]`` and TP/PP state for one active pair."""

    model_id: str
    tier_id: str
    configuration: ParallelismConfig

    @property
    def pair_id(self) -> tuple[str, str]:
        """Return the semantic model-tier pair."""

        return (self.model_id, self.tier_id)

    @property
    def gpu_count(self) -> int:
        """Return linked current ``y[j,k]=n*m``."""

        return self.configuration.gpu_count


@dataclass(frozen=True, slots=True)
class GHPhase2State:
    """Store the current GH Phase-2 state.

    Include active deployments, query processing order, routing fractions,
    per-query remaining demand, unmet fraction, cumulative error and delay,
    and used and remaining budget.
    """

    deployments: tuple[GHPhase2Deployment, ...]
    query_order: tuple[str, ...]
    routing_fraction_by_candidate: Mapping[tuple[str, str, str], float]
    remaining_demand_by_query: Mapping[str, float]
    unmet_fraction_by_query: Mapping[str, float]
    cumulative_error_by_query: Mapping[str, float]
    cumulative_delay_seconds_by_query: Mapping[str, float]
    accumulated_budget_cost_usd: float
    remaining_budget_usd: float

    def __post_init__(self) -> None:
        """Copy caller containers into immutable deterministic structures."""

        object.__setattr__(self, "deployments", tuple(self.deployments))
        object.__setattr__(self, "query_order", tuple(self.query_order))
        for field_name in (
            "routing_fraction_by_candidate",
            "remaining_demand_by_query",
            "unmet_fraction_by_query",
            "cumulative_error_by_query",
            "cumulative_delay_seconds_by_query",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )
        pair_ids = tuple(deployment.pair_id for deployment in self.deployments)
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("Phase-2 deployments must have unique pairs")
        if len(set(self.query_order)) != len(self.query_order):
            raise ValueError("query_order must contain unique query IDs")
        for name, values in (
            ("remaining demand", self.remaining_demand_by_query),
            ("unmet fraction", self.unmet_fraction_by_query),
            ("cumulative error", self.cumulative_error_by_query),
            ("cumulative delay", self.cumulative_delay_seconds_by_query),
        ):
            for value in values.values():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    or value < 0.0
                ):
                    raise ValueError(f"{name} values must be finite and nonnegative")
        if any(value > 1.0 for value in self.remaining_demand_by_query.values()):
            raise ValueError("remaining demand values must be <= 1")
        if any(value > 1.0 for value in self.unmet_fraction_by_query.values()):
            raise ValueError("unmet fraction values must be <= 1")
        for route, value in self.routing_fraction_by_candidate.items():
            if len(route) != 3 or not all(isinstance(item, str) for item in route):
                raise ValueError("routing keys must be (query, model, tier) strings")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
                or value > 1.0
            ):
                raise ValueError("sparse routing values must be finite and in (0, 1]")
        if (
            not isfinite(self.accumulated_budget_cost_usd)
            or self.accumulated_budget_cost_usd < 0.0
        ):
            raise ValueError(
                "accumulated_budget_cost_usd must be finite and nonnegative"
            )
        if not isfinite(self.remaining_budget_usd):
            raise ValueError("remaining_budget_usd must be finite")

    @property
    def active_pair_ids(self) -> tuple[tuple[str, str], ...]:
        """Return active pairs in inherited activation order."""

        return tuple(deployment.pair_id for deployment in self.deployments)

    def deployment_for_pair(
        self, model_id: str, tier_id: str
    ) -> GHPhase2Deployment | None:
        """Return current deployment evidence, or ``None`` for an inactive pair."""

        return next(
            (
                deployment
                for deployment in self.deployments
                if deployment.pair_id == (model_id, tier_id)
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class GHPhase2CandidateConfiguration:
    """Record a candidate TP/PP configuration, its source, and its delay."""

    query_id: str
    model_id: str
    tier_id: str
    source: GHPhase2ConfigurationSource
    selected_configuration: ParallelismConfig
    selected_fixed_configuration_delay_seconds: float

    def __post_init__(self) -> None:
        """Reject invalid delay evidence rather than repairing it."""

        delay = self.selected_fixed_configuration_delay_seconds
        if not isfinite(delay) or delay < 0.0:
            raise ValueError("selected configuration delay must be nonnegative")

    @property
    def is_feasible(self) -> bool:
        """Return true because this type represents only surviving selections."""

        return True


@dataclass(frozen=True, slots=True)
class GHPhase2CandidateEvaluation:
    """Retain one query/pair branch and all M1/M3/M2 evidence."""

    query_id: str
    model_id: str
    tier_id: str
    current_configuration: ParallelismConfig | None
    m1_selection: M1SelectionResult | None
    m3_upgrade: M3UpgradeResult | None
    candidate_configuration: GHPhase2CandidateConfiguration | None
    m2_inputs: M2EvaluationInputs
    m2_evaluation: M2CandidateEvaluation | None
    rejection_reasons: tuple[GHPhase2CandidateRejectionReason, ...]

    def __post_init__(self) -> None:
        """Own the immutable rejection evidence supplied by the producer."""

        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))

    @property
    def is_active_pair(self) -> bool:
        return self.current_configuration is not None

    @property
    def current_gpu_count(self) -> int:
        return self.m2_inputs.current_gpu_count

    @property
    def pair_id(self) -> tuple[str, str]:
        """Return the semantic candidate pair."""

        return (self.model_id, self.tier_id)

    @property
    def effective_configuration(self) -> ParallelismConfig | None:
        """Return the M1/current/M3 configuration passed to M2."""

        if self.candidate_configuration is None:
            return None
        return self.candidate_configuration.selected_configuration

    @property
    def is_rankable(self) -> bool:
        """Return whether M2 admits the candidate to ranking."""

        return self.m2_evaluation is not None and self.m2_evaluation.is_rankable

    @property
    def rank_key(self) -> tuple[int, float] | None:
        """Return the M2 ``(pi,kappa)`` key or ``None`` for a rejected candidate."""

        if self.m2_evaluation is None:
            return None
        return self.m2_evaluation.rank_key


@dataclass(frozen=True, slots=True)
class GHPhase2QueryEvaluation:
    """Return complete one-query candidates plus the M2-ranked subset."""

    query_id: str
    candidate_evaluations: tuple[GHPhase2CandidateEvaluation, ...]
    ranked_candidates: tuple[GHPhase2CandidateEvaluation, ...]

    def __post_init__(self) -> None:
        """Freeze both diagnostic sequences."""

        object.__setattr__(
            self, "candidate_evaluations", tuple(self.candidate_evaluations)
        )
        object.__setattr__(self, "ranked_candidates", tuple(self.ranked_candidates))

    @property
    def best_candidate(self) -> GHPhase2CandidateEvaluation | None:
        """Return the first M2-ranked candidate without committing it."""

        return self.ranked_candidates[0] if self.ranked_candidates else None


def order_gh_phase2_queries(instance: ProblemInstance) -> tuple[str, ...]:
    """Return descending ``lambda_i`` order (v2 Algorithm 1).

    ``lambda_i`` is ``query.arrival_rate_per_hour``. Exact ties retain instance
    semantic order.
    """

    validate_problem_instance(instance)
    return _order_gh_phase2_queries(instance)


def initialize_gh_phase2_state(
    instance: ProblemInstance,
    phase1_result: GHPhase1Result,
    *,
    query_order_override: tuple[str, ...] | None = None,
) -> GHPhase2State:
    """Transfer Phase-1 deployments and initialize Phase-2 query dynamics.

    Algorithm 1 starts with ``r_tilde=u=1`` and ``E_used=D_used=0``. Remaining
    budget is ``delta`` minus Phase-1 rental, retaining any cap overshoot.
    The default query order is descending arrival rate.
    """

    validate_problem_instance(instance)
    if not isinstance(phase1_result, GHPhase1Result):
        raise ValueError("phase1_result must be a GHPhase1Result")
    _validate_phase1_result_for_instance(instance, phase1_result)
    return _initialize_gh_phase2_state(
        instance, phase1_result, query_order_override=query_order_override
    )


def evaluate_gh_phase2_query_candidates(
    instance: ProblemInstance,
    state: GHPhase2State,
    query_id: str,
    *,
    expected_state_order: tuple[str, ...] | None = None,
) -> GHPhase2QueryEvaluation:
    """Evaluate and rank every ``(j,k)`` for one query (Algorithm 1, lines 9-14).

    Inactive pairs use M1; active pairs reuse the current shape when its delay
    meets the SLO, otherwise use M3 with remaining budget. Surviving shapes use
    M2 and stable ascending ``(pi,kappa)`` ranking, in model-then-tier semantic
    order for exact ties. Evaluation leaves state unchanged.
    """

    validate_problem_instance(instance)
    _validate_phase2_state(
        instance, state, expected_state_order=expected_state_order
    )
    return _evaluate_gh_phase2_query_candidates(instance, state, query_id)


def _order_gh_phase2_queries(instance: ProblemInstance) -> tuple[str, ...]:
    """Sort query IDs in an already validated instance."""

    return tuple(
        query.id
        for query in sorted(
            instance.queries,
            key=lambda query: query.arrival_rate_per_hour,
            reverse=True,
        )
    )


def _resolve_gh_phase2_query_order(
    instance: ProblemInstance,
    query_order_override: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Return the default order or validate one explicit exact permutation."""

    if query_order_override is None:
        return _order_gh_phase2_queries(instance)
    order = tuple(query_order_override)
    if len(order) != len(instance.query_ids):
        raise ValueError("query_order_override must contain exactly all query IDs")
    if any(not isinstance(query_id, str) for query_id in order):
        raise ValueError("query_order_override must contain string query IDs")
    if len(set(order)) != len(order):
        raise ValueError("query_order_override must not contain duplicate query IDs")
    if set(order) != set(instance.query_ids):
        raise ValueError("query_order_override must be an exact query-ID permutation")
    return order


def _evaluate_phase2_candidate(
    instance: ProblemInstance,
    query_id: str,
    model_id: str,
    tier_id: str,
    deployment: GHPhase2Deployment | None,
    remaining_budget_usd: float,
    m2_inputs: M2EvaluationInputs,
) -> GHPhase2CandidateEvaluation:
    """Select the inactive/current/M3 branch and evaluate M2."""

    m1_selection: M1SelectionResult | None = None
    m3_upgrade: M3UpgradeResult | None = None
    candidate_configuration: GHPhase2CandidateConfiguration | None = None
    rejection_reasons: tuple[GHPhase2CandidateRejectionReason, ...] = ()

    if deployment is None:
        m1_selection = _select_minimum_feasible_parallelism(
            instance, query_id, model_id, tier_id
        )
        if not m1_selection.is_feasible:
            rejection_reasons = (
                GHPhase2CandidateRejectionReason.M1_NO_FEASIBLE_CONFIGURATION,
            )
        else:
            selected_configuration = m1_selection.selected_configuration
            selected_delay = m1_selection.selected_fixed_configuration_delay_seconds
            candidate_configuration = GHPhase2CandidateConfiguration(
                query_id=query_id,
                model_id=model_id,
                tier_id=tier_id,
                source=GHPhase2ConfigurationSource.M1_INACTIVE_PAIR,
                selected_configuration=selected_configuration,
                selected_fixed_configuration_delay_seconds=selected_delay,
            )
            m2_evaluation = _evaluate_m2_candidate(
                instance, m1_selection, m2_inputs
            )
            return _complete_phase2_candidate(
                query_id,
                model_id,
                tier_id,
                deployment,
                m1_selection,
                None,
                candidate_configuration,
                m2_inputs,
                m2_evaluation,
            )
    else:
        coefficient_key = (query_id, model_id, tier_id)
        query = next(query for query in instance.queries if query.id == query_id)
        current_delay = compute_configuration_delay_seconds(
            instance.compute_delay_seconds_per_token[coefficient_key],
            instance.communication_delay_seconds_per_token[coefficient_key],
            query,
            deployment.configuration,
        )
        if current_delay <= query.delay_slo_seconds:
            candidate_configuration = GHPhase2CandidateConfiguration(
                query_id=query_id,
                model_id=model_id,
                tier_id=tier_id,
                source=GHPhase2ConfigurationSource.CURRENT_ACTIVE_CONFIGURATION,
                selected_configuration=deployment.configuration,
                selected_fixed_configuration_delay_seconds=current_delay,
            )
        else:
            m3_upgrade = _select_m3_parallelism_upgrade(
                instance,
                query_id,
                model_id,
                tier_id,
                M3EvaluationState(
                    current_configuration=deployment.configuration,
                    remaining_budget_usd=remaining_budget_usd,
                ),
            )
            if not m3_upgrade.has_upgrade:
                rejection_reasons = (
                    GHPhase2CandidateRejectionReason.M3_NO_FEASIBLE_UPGRADE,
                )
            else:
                selected = m3_upgrade.selected_diagnostic
                candidate_configuration = GHPhase2CandidateConfiguration(
                    query_id=query_id,
                    model_id=model_id,
                    tier_id=tier_id,
                    source=GHPhase2ConfigurationSource.M3_ACTIVE_PAIR_UPGRADE,
                    selected_configuration=selected.candidate_configuration,
                    selected_fixed_configuration_delay_seconds=(
                        selected.candidate_delay_seconds
                    ),
                )

    if rejection_reasons:
        return GHPhase2CandidateEvaluation(
            query_id=query_id,
            model_id=model_id,
            tier_id=tier_id,
            current_configuration=(
                None if deployment is None else deployment.configuration
            ),
            m1_selection=m1_selection,
            m3_upgrade=m3_upgrade,
            candidate_configuration=None,
            m2_inputs=m2_inputs,
            m2_evaluation=None,
            rejection_reasons=rejection_reasons,
        )

    m2_evaluation = _evaluate_m2_candidate(instance, candidate_configuration, m2_inputs)
    return _complete_phase2_candidate(
        query_id,
        model_id,
        tier_id,
        deployment,
        m1_selection,
        m3_upgrade,
        candidate_configuration,
        m2_inputs,
        m2_evaluation,
    )


def _complete_phase2_candidate(
    query_id: str,
    model_id: str,
    tier_id: str,
    deployment: GHPhase2Deployment | None,
    m1_selection: M1SelectionResult | None,
    m3_upgrade: M3UpgradeResult | None,
    candidate_configuration: GHPhase2CandidateConfiguration,
    m2_inputs: M2EvaluationInputs,
    m2_evaluation: M2CandidateEvaluation,
) -> GHPhase2CandidateEvaluation:
    """Attach the configuration-source provenance to M2 output."""

    reasons = (
        ()
        if m2_evaluation.is_rankable
        else (GHPhase2CandidateRejectionReason.M2_ZERO_EFFECTIVE_COVERAGE,)
    )
    return GHPhase2CandidateEvaluation(
        query_id=query_id,
        model_id=model_id,
        tier_id=tier_id,
        current_configuration=(
            None if deployment is None else deployment.configuration
        ),
        m1_selection=m1_selection,
        m3_upgrade=m3_upgrade,
        candidate_configuration=candidate_configuration,
        m2_inputs=m2_inputs,
        m2_evaluation=m2_evaluation,
        rejection_reasons=reasons,
    )


def _validate_phase2_state(
    instance: ProblemInstance,
    state: GHPhase2State,
    *,
    expected_state_order: tuple[str, ...] | None = None,
) -> None:
    """Validate state identities and exact Phase-2 mapping coverage."""

    if not isinstance(state, GHPhase2State):
        raise ValueError("state must be a GHPhase2State")
    query_ids = set(instance.query_ids)
    for field_name in (
        "remaining_demand_by_query",
        "unmet_fraction_by_query",
        "cumulative_error_by_query",
        "cumulative_delay_seconds_by_query",
    ):
        if set(getattr(state, field_name)) != query_ids:
            raise ValueError(f"{field_name} must contain exactly all query IDs")
    expected_order = _resolve_gh_phase2_query_order(instance, expected_state_order)
    if state.query_order != expected_order:
        if expected_state_order is None:
            raise ValueError("query_order must follow descending arrival rate")
        raise ValueError("query_order must match the explicit expected query order")
    model_ids = set(instance.model_ids)
    tier_by_id = {tier.id: tier for tier in instance.gpu_tiers}
    for deployment in state.deployments:
        if deployment.model_id not in model_ids or deployment.tier_id not in tier_by_id:
            raise ValueError("state contains an unknown active model-tier pair")
        if deployment.configuration not in tier_by_id[deployment.tier_id].parallelism_configs:
            raise ValueError("active configuration is unsupported by its GPU tier")
    if any(
        not isinstance(key, tuple)
        or key[0] not in query_ids
        or key[1] not in model_ids
        or key[2] not in tier_by_id
        for key in state.routing_fraction_by_candidate
    ):
        raise ValueError("state contains an unknown routing key")
    active_pairs = set(state.active_pair_ids)
    if any(
        (model_id, tier_id) not in active_pairs
        for _, model_id, tier_id in state.routing_fraction_by_candidate
    ):
        raise ValueError("every sparse route must target an active pair")
    _validate_phase2_budget(instance, state)


def _validate_phase2_budget(instance: ProblemInstance, state: GHPhase2State) -> None:
    """Preserve exact carried-budget consistency, including roundoff failures."""

    if (
        state.accumulated_budget_cost_usd + state.remaining_budget_usd
        != instance.cost_budget_usd
    ):
        raise ValueError("accumulated and remaining budget must sum to delta")


def _validate_phase1_result_for_instance(
    instance: ProblemInstance,
    phase1_result: GHPhase1Result,
) -> None:
    """Require consistent Phase-1 evidence for the supplied instance."""

    _validate_state_against_instance(instance, phase1_result.final_state)


def _initialize_gh_phase2_state(
    instance: ProblemInstance,
    phase1_result: GHPhase1Result,
    *,
    query_order_override: tuple[str, ...] | None = None,
) -> GHPhase2State:
    """Transfer producer-created or publicly checked Phase-1 evidence."""

    phase1_state = phase1_result.final_state
    deployments = tuple(
        GHPhase2Deployment(
            model_id=deployment.model_id,
            tier_id=deployment.tier_id,
            configuration=deployment.configuration,
        )
        for deployment in phase1_state.active_deployments
    )
    state = GHPhase2State(
        deployments=deployments,
        query_order=_resolve_gh_phase2_query_order(instance, query_order_override),
        routing_fraction_by_candidate={},
        remaining_demand_by_query={query.id: 1.0 for query in instance.queries},
        unmet_fraction_by_query={query.id: 1.0 for query in instance.queries},
        cumulative_error_by_query={query.id: 0.0 for query in instance.queries},
        cumulative_delay_seconds_by_query={
            query.id: 0.0 for query in instance.queries
        },
        accumulated_budget_cost_usd=phase1_state.accumulated_rental_cost_usd,
        remaining_budget_usd=(
            instance.cost_budget_usd
            - phase1_state.accumulated_rental_cost_usd
        ),
    )
    _validate_phase2_budget(instance, state)
    return state


def _evaluate_gh_phase2_query_candidates(
    instance: ProblemInstance,
    state: GHPhase2State,
    query_id: str,
) -> GHPhase2QueryEvaluation:
    """Evaluate one query at a checked entry state on a validated instance."""

    query = next((item for item in instance.queries if item.id == query_id), None)
    if query is None:
        raise ValueError(f"unknown query ID: {query_id!r}")

    evaluations: list[GHPhase2CandidateEvaluation] = []
    for model in instance.models:
        for tier in instance.gpu_tiers:
            deployment = state.deployment_for_pair(model.id, tier.id)
            m2_inputs = M2EvaluationInputs(
                remaining_demand=state.remaining_demand_by_query[query.id],
                cumulative_error=state.cumulative_error_by_query[query.id],
                cumulative_delay_seconds=(
                    state.cumulative_delay_seconds_by_query[query.id]
                ),
                current_gpu_count=(0 if deployment is None else deployment.gpu_count),
            )
            evaluations.append(
                _evaluate_phase2_candidate(
                    instance,
                    query.id,
                    model.id,
                    tier.id,
                    deployment,
                    state.remaining_budget_usd,
                    m2_inputs,
                )
            )

    candidate_evaluations = tuple(evaluations)
    m2_ranked = rank_m2_candidates(
        tuple(
            candidate.m2_evaluation
            for candidate in candidate_evaluations
            if candidate.m2_evaluation is not None
        )
    )
    candidate_by_pair = {
        candidate.pair_id: candidate for candidate in candidate_evaluations
    }
    ranked = tuple(
        candidate_by_pair[(candidate.model_id, candidate.tier_id)]
        for candidate in m2_ranked
    )
    return GHPhase2QueryEvaluation(
        query_id=query.id,
        candidate_evaluations=candidate_evaluations,
        ranked_candidates=ranked,
    )
