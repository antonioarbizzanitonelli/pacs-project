"""Check and commit one GH Phase-2 candidate (v2 Algorithm 1, lines 15-20)."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from llm_resource_allocation.algorithms.gh_phase2 import (
    GHPhase2CandidateEvaluation,
    GHPhase2ConfigurationSource,
    GHPhase2Deployment,
    GHPhase2State,
    _validate_phase2_budget,
    _validate_phase2_state,
)
from llm_resource_allocation.domain import (
    ParallelismConfig,
    ProblemInstance,
    validate_problem_instance,
)
from llm_resource_allocation.math.budget import BudgetDiagnostic, _compute_budget_residual
from llm_resource_allocation.math.compute_capacity import (
    ComputeCapacityDiagnostic,
    _compute_compute_capacity_diagnostics,
)
from llm_resource_allocation.math.error import compute_effective_error
from llm_resource_allocation.math.gpu_memory import (
    GpuMemoryDiagnostic,
    _compute_gpu_memory_residuals,
)
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import (
    PlacementDecisionState,
    _validate_placement_decision_domains,
)
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.math.storage_capacity import (
    QueryStorageCapacityDiagnostic,
    _compute_storage_capacity_residuals,
)


class GHPhase2CommitRejectionReason(str, Enum):
    """Identify a failed allocation or resource/budget check (Algorithm 1)."""

    ZERO_ALLOCATION = "zero_allocation"
    GPU_MEMORY = "gpu_memory"
    COMPUTE_CAPACITY = "compute_capacity"
    STORAGE = "storage"
    GLOBAL_BUDGET = "global_budget"


@dataclass(frozen=True, slots=True)
class GHPhase2AllocationProposal:
    """Record the unserved fraction and the candidate's M2 coverage."""

    unserved_fraction_before: float
    candidate_coverage_fraction: float

    @property
    def proposed_routing_fraction(self) -> float:
        return min(
            self.unserved_fraction_before,
            self.candidate_coverage_fraction,
        )


@dataclass(frozen=True, slots=True)
class GHPhase2ProspectiveDeployment:
    """Describe the candidate pair's prospective ``q,w,y`` transition."""

    model_id: str
    tier_id: str
    source: GHPhase2ConfigurationSource
    current_configuration: ParallelismConfig | None
    prospective_configuration: ParallelismConfig

    @property
    def was_active(self) -> bool:
        return self.current_configuration is not None

    @property
    def current_gpu_count(self) -> int:
        return 0 if self.current_configuration is None else self.current_configuration.gpu_count

    @property
    def prospective_gpu_count(self) -> int:
        return self.prospective_configuration.gpu_count

    @property
    def incremental_gpu_count(self) -> int:
        return self.prospective_gpu_count - self.current_gpu_count


@dataclass(frozen=True, slots=True)
class GHPhase2BudgetComparison:
    """Compare C1-C3 budget costs before and after the proposed assignment."""

    prior_accumulated_budget_cost_usd: float
    prior_budget_diagnostic: BudgetDiagnostic
    prospective_budget_diagnostic: BudgetDiagnostic

    @property
    def incremental_gpu_rental_cost_usd(self) -> float:
        return self.prospective_budget_diagnostic.gpu_rental_cost - self.prior_budget_diagnostic.gpu_rental_cost

    @property
    def incremental_model_storage_cost_usd(self) -> float:
        return self.prospective_budget_diagnostic.model_storage_cost - self.prior_budget_diagnostic.model_storage_cost

    @property
    def incremental_data_storage_cost_usd(self) -> float:
        return self.prospective_budget_diagnostic.data_storage_cost - self.prior_budget_diagnostic.data_storage_cost


@dataclass(frozen=True, slots=True)
class GHPhase2CommitDiagnostic:
    """Return complete evidence for one accepted or rejected candidate."""

    query_id: str
    model_id: str
    tier_id: str
    allocation: GHPhase2AllocationProposal
    prospective_deployment: GHPhase2ProspectiveDeployment | None
    gpu_memory_diagnostics: tuple[GpuMemoryDiagnostic, ...]
    compute_capacity_diagnostics: tuple[ComputeCapacityDiagnostic, ...]
    storage_capacity_diagnostics: tuple[QueryStorageCapacityDiagnostic, ...]
    budget_comparison: GHPhase2BudgetComparison | None
    rejection_reasons: tuple[GHPhase2CommitRejectionReason, ...]

    @property
    def is_feasible(self) -> bool:
        """Return whether this diagnostic permits the line-18 commit."""

        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class GHPhase2CommitResult:
    """Return an immutable one-candidate state transition and its evidence."""

    state_before: GHPhase2State
    state_after: GHPhase2State
    diagnostic: GHPhase2CommitDiagnostic

    @property
    def committed(self) -> bool:
        """Return whether the transition produced a new accepted state."""

        return self.diagnostic.is_feasible


@dataclass(frozen=True, slots=True)
class _GHPhase2Proposal:
    """Keep the prospective deployment, routing, and checked decisions together."""

    deployment_proposal: GHPhase2ProspectiveDeployment
    deployments: tuple[GHPhase2Deployment, ...]
    routes: Mapping[tuple[str, str, str], float]
    unmet: Mapping[str, float]
    decisions: PlacementDecisionState


@dataclass(frozen=True, slots=True)
class _GHPhase2CommitContext:
    """Bind the exact current state to its checked C1-C3 breakdown."""

    state: GHPhase2State
    budget: BudgetDiagnostic


def compute_gh_phase2_candidate_allocation(
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
) -> GHPhase2AllocationProposal:
    """Return the proposal ``min(u_i,x_bar[i,j,k])`` (Algorithm 1, line 18).

    ``x_bar`` is retained M2 evidence. Resource and budget checks must pass
    before this fraction becomes committed routing.
    """

    if not isinstance(state, GHPhase2State):
        raise ValueError("state must be a GHPhase2State")
    if not isinstance(candidate, GHPhase2CandidateEvaluation):
        raise ValueError("candidate must be a GHPhase2CandidateEvaluation")
    if not candidate.is_rankable or candidate.m2_evaluation is None:
        raise ValueError("candidate must be a positive-coverage ranked candidate")
    expected_gpu_count = (
        0 if candidate.current_configuration is None
        else candidate.current_configuration.gpu_count
    )
    if candidate.m2_inputs.current_gpu_count != expected_gpu_count:
        raise ValueError("M2 GPU count must match the current configuration")
    coverage_diagnostic = candidate.m2_evaluation.coverage_diagnostic
    if coverage_diagnostic is None:
        raise ValueError("ranked candidate lacks effective-coverage evidence")
    if candidate.query_id not in state.unmet_fraction_by_query:
        raise ValueError("candidate query is absent from Phase-2 state")
    return _compute_gh_phase2_candidate_allocation(state, candidate)


def _compute_gh_phase2_candidate_allocation(
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
) -> GHPhase2AllocationProposal:
    """Use retained coverage from a producer-ranked or publicly checked candidate."""

    coverage_diagnostic = candidate.m2_evaluation.coverage_diagnostic
    unserved_fraction_before = state.unmet_fraction_by_query[candidate.query_id]
    candidate_coverage_fraction = coverage_diagnostic.effective_coverage
    return GHPhase2AllocationProposal(
        unserved_fraction_before=unserved_fraction_before,
        candidate_coverage_fraction=candidate_coverage_fraction,
    )


def commit_gh_phase2_candidate(
    instance: ProblemInstance,
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
    *,
    candidate_evaluation_state: GHPhase2State | None = None,
    expected_state_order: tuple[str, ...] | None = None,
) -> GHPhase2CommitResult:
    """Attempt one candidate assignment after resource and budget checks.

    Return a new state on acceptance or the original state on rejection.
    Use candidate_evaluation_state when candidates were evaluated earlier.
    """

    validate_problem_instance(instance)
    _validate_phase2_state(
        instance, state, expected_state_order=expected_state_order
    )
    evaluation_state = (
        state
        if candidate_evaluation_state is None
        else candidate_evaluation_state
    )
    if evaluation_state is not state:
        _validate_phase2_state(
            instance,
            evaluation_state,
            expected_state_order=expected_state_order,
        )
    _validate_candidate_for_state(instance, state, candidate, evaluation_state)
    result, _ = _commit_gh_phase2_candidate(instance, state, candidate)
    return result


def _commit_gh_phase2_candidate(
    instance: ProblemInstance,
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
    *,
    context: _GHPhase2CommitContext | None = None,
) -> tuple[GHPhase2CommitResult, _GHPhase2CommitContext | None]:
    """Check a proposal on the same validated instance and exact current state."""

    allocation = _compute_gh_phase2_candidate_allocation(state, candidate)

    if allocation.proposed_routing_fraction <= 0.0:
        diagnostic = GHPhase2CommitDiagnostic(
            query_id=candidate.query_id,
            model_id=candidate.model_id,
            tier_id=candidate.tier_id,
            allocation=allocation,
            prospective_deployment=None,
            gpu_memory_diagnostics=(),
            compute_capacity_diagnostics=(),
            storage_capacity_diagnostics=(),
            budget_comparison=None,
            rejection_reasons=(GHPhase2CommitRejectionReason.ZERO_ALLOCATION,),
        )
        return GHPhase2CommitResult(state, state, diagnostic), context

    if context is None:
        decisions = _build_placement_decisions(
            state.deployments, state.routing_fraction_by_candidate
        )
        _validate_placement_decision_domains(instance, decisions)
        context = _GHPhase2CommitContext(
            state, _compute_budget_residual(instance, decisions)
        )

    proposal = _build_commit_proposal(state, candidate, allocation)
    diagnostic = _evaluate_prospective_commit(
        instance, state, candidate, allocation, proposal, context.budget
    )
    if not diagnostic.is_feasible:
        return GHPhase2CommitResult(state, state, diagnostic), context
    state_after = _apply_feasible_commit(
        instance,
        state,
        candidate,
        allocation,
        proposal,
        diagnostic.budget_comparison.prospective_budget_diagnostic,
    )
    return (
        GHPhase2CommitResult(state, state_after, diagnostic),
        _GHPhase2CommitContext(
            state_after,
            diagnostic.budget_comparison.prospective_budget_diagnostic,
        ),
    )


def _build_commit_proposal(
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
    allocation: GHPhase2AllocationProposal,
) -> _GHPhase2Proposal:
    """Build prospective routing/deployments and plain constraint decisions."""

    deployment_proposal, prospective_deployments = _prospective_deployments(
        state, candidate
    )
    route_key = (candidate.query_id, candidate.model_id, candidate.tier_id)
    prospective_routes = dict(state.routing_fraction_by_candidate)
    prospective_routes[route_key] = (
        prospective_routes.get(route_key, 0.0) + allocation.proposed_routing_fraction
    )

    prospective_unmet = dict(state.unmet_fraction_by_query)
    prospective_unmet[candidate.query_id] -= allocation.proposed_routing_fraction
    prospective_decisions = _build_placement_decisions(
        prospective_deployments,
        prospective_routes,
    )

    return _GHPhase2Proposal(
        deployment_proposal=deployment_proposal,
        deployments=prospective_deployments,
        routes=prospective_routes,
        unmet=prospective_unmet,
        decisions=prospective_decisions,
    )


def _evaluate_prospective_commit(
    instance: ProblemInstance,
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
    allocation: GHPhase2AllocationProposal,
    proposal: _GHPhase2Proposal,
    prior_budget_diagnostic: BudgetDiagnostic,
) -> GHPhase2CommitDiagnostic:
    """Record memory, compute, storage, and C1-C3 budget failures in paper order."""

    _validate_placement_decision_domains(instance, proposal.decisions)
    memory = _compute_gpu_memory_residuals(instance, proposal.decisions)
    compute = _compute_compute_capacity_diagnostics(
        instance, proposal.decisions.parallelism_decisions
    )
    storage = _compute_storage_capacity_residuals(instance, proposal.decisions)
    prospective_budget_diagnostic = _compute_budget_residual(instance, proposal.decisions)
    budget_comparison = GHPhase2BudgetComparison(
        prior_accumulated_budget_cost_usd=state.accumulated_budget_cost_usd,
        prior_budget_diagnostic=prior_budget_diagnostic,
        prospective_budget_diagnostic=prospective_budget_diagnostic,
    )

    rejection_reasons: list[GHPhase2CommitRejectionReason] = []
    if any(not residual.is_feasible() for residual in memory):
        rejection_reasons.append(GHPhase2CommitRejectionReason.GPU_MEMORY)
    if any(not residual.is_feasible() for residual in compute):
        rejection_reasons.append(GHPhase2CommitRejectionReason.COMPUTE_CAPACITY)
    if any(not residual.is_feasible() for residual in storage):
        rejection_reasons.append(GHPhase2CommitRejectionReason.STORAGE)
    if not prospective_budget_diagnostic.is_feasible():
        rejection_reasons.append(GHPhase2CommitRejectionReason.GLOBAL_BUDGET)

    return GHPhase2CommitDiagnostic(
        query_id=candidate.query_id,
        model_id=candidate.model_id,
        tier_id=candidate.tier_id,
        allocation=allocation,
        prospective_deployment=proposal.deployment_proposal,
        gpu_memory_diagnostics=memory,
        compute_capacity_diagnostics=compute,
        storage_capacity_diagnostics=storage,
        budget_comparison=budget_comparison,
        rejection_reasons=tuple(rejection_reasons),
    )


def _apply_feasible_commit(
    instance: ProblemInstance,
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
    allocation: GHPhase2AllocationProposal,
    proposal: _GHPhase2Proposal,
    prospective_budget_diagnostic: BudgetDiagnostic,
) -> GHPhase2State:
    """Return updated routing, query dynamics, deployments, and budget after success."""

    query = next(item for item in instance.queries if item.id == candidate.query_id)
    tier = next(item for item in instance.gpu_tiers if item.id == candidate.tier_id)
    effective_error = compute_effective_error(
        instance.base_error_rate[(query.id, candidate.model_id)],
        tier.quantization_error_multiplier,
    )
    remaining_demand = dict(state.remaining_demand_by_query)
    remaining_demand[query.id] -= allocation.proposed_routing_fraction
    cumulative_error = dict(state.cumulative_error_by_query)
    cumulative_error[query.id] += effective_error * allocation.proposed_routing_fraction
    cumulative_delay = dict(state.cumulative_delay_seconds_by_query)
    candidate_configuration = candidate.candidate_configuration
    cumulative_delay[query.id] += (
        candidate_configuration.selected_fixed_configuration_delay_seconds
        * allocation.proposed_routing_fraction
    )
    state_after = GHPhase2State(
        deployments=proposal.deployments,
        query_order=state.query_order,
        routing_fraction_by_candidate=proposal.routes,
        remaining_demand_by_query=remaining_demand,
        unmet_fraction_by_query=proposal.unmet,
        cumulative_error_by_query=cumulative_error,
        cumulative_delay_seconds_by_query=cumulative_delay,
        accumulated_budget_cost_usd=prospective_budget_diagnostic.total_budget_use,
        remaining_budget_usd=prospective_budget_diagnostic.slack,
    )
    _validate_phase2_budget(instance, state_after)
    return state_after


def _validate_candidate_for_state(
    instance: ProblemInstance,
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
    candidate_evaluation_state: GHPhase2State,
) -> None:
    """Reject identity, deployment, or declared evaluation-state mismatches."""

    if not isinstance(candidate, GHPhase2CandidateEvaluation):
        raise ValueError("candidate must be a GHPhase2CandidateEvaluation")
    if not candidate.is_rankable or candidate.candidate_configuration is None:
        raise ValueError("candidate must be a positive-coverage ranked candidate")
    if candidate.query_id not in instance.query_ids:
        raise ValueError("candidate has an unknown query ID")
    if (
        candidate.model_id not in instance.model_ids
        or candidate.tier_id not in instance.gpu_tier_ids
    ):
        raise ValueError("candidate has an unknown model-tier pair")
    if candidate.m2_evaluation.selected_configuration != candidate.effective_configuration:
        raise ValueError("candidate M2 configuration lacks matching selection evidence")
    current = state.deployment_for_pair(candidate.model_id, candidate.tier_id)
    evaluated = candidate_evaluation_state.deployment_for_pair(
        candidate.model_id, candidate.tier_id
    )
    source = candidate.candidate_configuration.source
    if source is GHPhase2ConfigurationSource.M1_INACTIVE_PAIR:
        if (
            current is not None
            or evaluated is not None
            or candidate.current_configuration is not None
        ):
            raise ValueError("M1 candidate no longer targets an inactive pair")
        if (
            candidate.m1_selection is None
            or candidate.m1_selection.selected_diagnostic is None
            or candidate.m1_selection.selected_configuration
            != candidate.effective_configuration
        ):
            raise ValueError("M1 candidate lacks matching selection evidence")
    else:
        if (
            evaluated is None
            or evaluated.configuration != candidate.current_configuration
        ):
            raise ValueError(
                "active candidate does not match its declared evaluation state"
            )
        if (
            current is None
            or current.configuration != candidate.current_configuration
        ):
            raise ValueError("active candidate does not match current deployment")
        if (
            source is GHPhase2ConfigurationSource.CURRENT_ACTIVE_CONFIGURATION
            and candidate.effective_configuration != current.configuration
        ):
            raise ValueError("current-active candidate changed configuration")
        if source is GHPhase2ConfigurationSource.M3_ACTIVE_PAIR_UPGRADE:
            if (
                candidate.m3_upgrade is None
                or not candidate.m3_upgrade.has_upgrade
                or candidate.m3_upgrade.selected_diagnostic is None
                or candidate.m3_upgrade.selected_configuration
                != candidate.effective_configuration
            ):
                raise ValueError("M3 candidate lacks matching upgrade evidence")
    expected_dynamic = (
        candidate_evaluation_state.remaining_demand_by_query[candidate.query_id],
        candidate_evaluation_state.cumulative_error_by_query[candidate.query_id],
        candidate_evaluation_state.cumulative_delay_seconds_by_query[
            candidate.query_id
        ],
        0 if evaluated is None else evaluated.gpu_count,
    )
    candidate_dynamic = (
        candidate.m2_inputs.remaining_demand,
        candidate.m2_inputs.cumulative_error,
        candidate.m2_inputs.cumulative_delay_seconds,
        candidate.m2_inputs.current_gpu_count,
    )
    if candidate_dynamic != expected_dynamic:
        raise ValueError(
            "candidate M2 evidence was not evaluated at its declared state"
        )


def _prospective_deployments(
    state: GHPhase2State,
    candidate: GHPhase2CandidateEvaluation,
) -> tuple[GHPhase2ProspectiveDeployment, tuple[GHPhase2Deployment, ...]]:
    """Derive the inactive/current/M3 ``q,w,y`` state without mutation."""

    candidate_configuration = candidate.candidate_configuration
    configuration = candidate.effective_configuration
    current = state.deployment_for_pair(candidate.model_id, candidate.tier_id)
    prospective = GHPhase2Deployment(
        candidate.model_id, candidate.tier_id, configuration
    )
    if current is None:
        deployments = state.deployments + (prospective,)
    else:
        deployments = tuple(
            prospective if item.pair_id == candidate.pair_id else item
            for item in state.deployments
        )
    deployment_proposal = GHPhase2ProspectiveDeployment(
        model_id=candidate.model_id,
        tier_id=candidate.tier_id,
        source=candidate_configuration.source,
        current_configuration=None if current is None else current.configuration,
        prospective_configuration=configuration,
    )
    return deployment_proposal, deployments


def _build_placement_decisions(
    deployments: tuple[GHPhase2Deployment, ...],
    routes: Mapping[tuple[str, str, str], float],
) -> PlacementDecisionState:
    """Build plain sparse decisions for Eq. (8f)-(8h) and Eq. (8c) checks."""

    route_mapping = dict(routes)
    deployment_by_pair = {item.pair_id: item for item in deployments}
    routing = RoutingDecisionState(
        routing_fractions=route_mapping,
        admission_indicators={key: 1 for key in route_mapping},
        deployment_indicators={item.pair_id: 1 for item in deployments},
        # Algorithm 1 initializes u=1 before it can satisfy the P_DM zeta
        # bound. Eq. (8f)-(8h) and Eq. (8c) do not consume u, so this narrow
        # adapter omits it instead of importing an unrelated domain check.
        unmet_fractions={},
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors={
            (
                item.model_id,
                item.tier_id,
                item.configuration.tensor_parallel_degree,
                item.configuration.pipeline_parallel_depth,
            ): 1
            for item in deployments
        },
        gpu_counts={item.pair_id: item.gpu_count for item in deployments},
    )
    # With one selected configuration per active pair, exact McCormick
    # products equal x on that selector and zero on every sparse omission.
    auxiliaries: dict[tuple[str, str, str, int, int], float] = {}
    for (query_id, model_id, tier_id), fraction in route_mapping.items():
        configuration = deployment_by_pair[(model_id, tier_id)].configuration
        auxiliaries[
            (
                query_id,
                model_id,
                tier_id,
                configuration.tensor_parallel_degree,
                configuration.pipeline_parallel_depth,
            )
        ] = fraction
    return PlacementDecisionState(
        parallelism_decisions=parallelism,
        auxiliary_products=auxiliaries,
    )
