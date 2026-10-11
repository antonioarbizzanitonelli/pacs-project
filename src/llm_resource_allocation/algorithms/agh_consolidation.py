"""Run one consolidation scan for AGH.

Order sources by compute utilization and move whole routes to the first feasible
active destination. Accept deactivation only after independent validation and
strict improvement of the full objective.
"""

from dataclasses import dataclass
from enum import Enum
from math import fsum

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.math.compute_capacity import (
    compute_available_compute_tflops_per_hour,
    compute_required_compute_tflops_per_hour,
)
from llm_resource_allocation.math.objective import ObjectiveBreakdown, PaperVersion
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    PlacementValidator,
    ValidationTolerances,
)

from ._agh_placement_ops import (
    AGHDeploymentPairEvidence,
    _AGHPlacementEvaluation,
    _build_agh_search_context,
    _transfer_route,
    _deployment_pair_evidence,
)


AGHConsolidationPair = tuple[str, str]
AGHConsolidationRoute = tuple[str, str, str]

# Audit transfer sums only; preserve route values, feasibility policy, and strict comparison.
AGH_CONSOLIDATION_CONSTRUCTION_ABSOLUTE_TOLERANCE = 1.0e-15


class AGHConsolidationRejectionReason(str, Enum):
    """Explain rejection of one source-pair consolidation proposal."""

    INVALID_PLACEMENT_DOMAINS = "invalid_placement_domains"
    INVALID_SOURCE = "invalid_source"
    SOURCE_NOT_ACTIVE = "source_not_active"
    INCOHERENT_SOURCE_DEPLOYMENT = "incoherent_source_deployment"
    REDISTRIBUTION_FAILED = "redistribution_failed"
    CONSTRUCTION_INCONSISTENCY = "construction_inconsistency"
    INFEASIBLE = "infeasible"
    NOT_STRICTLY_IMPROVING = "not_strictly_improving"


class AGHConsolidationRouteStatus(str, Enum):
    """Describe whether one source route found a feasible destination."""

    REDISTRIBUTED = "redistributed"
    NO_FEASIBLE_DESTINATION = "no_feasible_destination"


class AGHConsolidationSourceScanStatus(str, Enum):
    """Describe the consolidation outcome for one snapshotted source pair."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SOURCE_NO_LONGER_ACTIVE = "source_no_longer_active"


@dataclass(frozen=True, slots=True)
class AGHConsolidationUtilizationEvidence:
    """Record source compute utilization as required divided by available compute.

    Both compute values are in TFLOP/hour. Utilization is dimensionless and is
    zero when no compute is required.
    """

    pair: AGHConsolidationPair
    semantic_position: int
    required_compute_tflops_per_hour: float
    available_compute_tflops_per_hour: float
    pair_evidence: AGHDeploymentPairEvidence

    @property
    def utilization(self) -> float:
        """Return required divided by available compute, or zero for no demand."""

        if self.required_compute_tflops_per_hour == 0.0:
            return 0.0
        return self.required_compute_tflops_per_hour / self.available_compute_tflops_per_hour


@dataclass(frozen=True, slots=True)
class AGHConsolidationDestinationAttempt:
    """Record a provisional whole-route transfer and its feasibility report."""

    route_position: int
    destination_position: int
    source_route: AGHConsolidationRoute
    destination: AGHConsolidationPair
    moved_fraction: float
    placement_before: PlacementDecisionState
    prospective_placement: PlacementDecisionState
    validation_report: PlacementValidationReport

    @property
    def placement_after(self) -> PlacementDecisionState:
        """Return the placement retained at the end of this operation."""

        return self.prospective_placement if self.feasible else self.placement_before

    @property
    def feasible(self) -> bool:
        """Return whether the independent proposal validation is feasible."""

        return self.validation_report.is_feasible


@dataclass(frozen=True, slots=True)
class AGHConsolidationRouteTrace:
    """Record attempts to provisionally move one whole source route.

    The first feasible destination is selected before the source-pair proposal
    receives its final feasibility and objective checks.
    """

    route_position: int
    source_route: AGHConsolidationRoute
    moved_fraction: float
    placement_before: PlacementDecisionState
    eligible_destinations: tuple[AGHConsolidationPair, ...]
    attempts: tuple[AGHConsolidationDestinationAttempt, ...]
    status: AGHConsolidationRouteStatus

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(
            self, "eligible_destinations", tuple(self.eligible_destinations)
        )
        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def selected_destination(self) -> AGHConsolidationPair | None:
        """Return the first feasible destination, if one was found."""

        if self.attempts and self.attempts[-1].feasible:
            return self.attempts[-1].destination
        return None

    @property
    def placement_after(self) -> PlacementDecisionState:
        """Return the placement retained at the end of this operation."""

        return self.attempts[-1].placement_after if self.attempts else self.placement_before


@dataclass(frozen=True, slots=True)
class AGHConsolidationDemandEvidence:
    """Record routed and unmet fractions before and after consolidation."""

    query_id: str
    routed_before: float
    routed_after: float
    unmet_before: float
    unmet_after: float

    @property
    def routed_absolute_difference(self) -> float:
        """Return the absolute change in the query's total routed fraction."""

        return abs(self.routed_after - self.routed_before)

    @property
    def routed_demand_preserved(self) -> bool:
        """Check routed-fraction preservation within the construction tolerance."""

        return (
            self.routed_absolute_difference
            <= AGH_CONSOLIDATION_CONSTRUCTION_ABSOLUTE_TOLERANCE
        )

    @property
    def unmet_demand_preserved(self) -> bool:
        """Check that the copied unmet fraction is exactly unchanged."""

        return self.unmet_before == self.unmet_after

    @property
    def is_consistent(self) -> bool:
        """Return whether routed and unmet fractions are preserved."""

        return self.routed_demand_preserved and self.unmet_demand_preserved


@dataclass(frozen=True, slots=True)
class AGHConsolidationSourceResult:
    """Record a source-pair proposal, provisional transfers, and final acceptance.

    Rejection preserves the exact input placement and retains provisional evidence.
    """

    source: AGHConsolidationPair
    placement_before: PlacementDecisionState
    source_evidence: AGHDeploymentPairEvidence | None
    source_routes: tuple[AGHConsolidationRoute, ...]
    eligible_destinations: tuple[AGHConsolidationPair, ...]
    route_traces: tuple[AGHConsolidationRouteTrace, ...]
    redistributed_placement: PlacementDecisionState | None
    prospective_placement: PlacementDecisionState | None
    validation_before: PlacementValidationReport
    validation_report: PlacementValidationReport | None
    demand_evidence: tuple[AGHConsolidationDemandEvidence, ...]
    rejection_reasons: tuple[AGHConsolidationRejectionReason, ...]

    @property
    def objective_delta(self) -> float | None:
        """Return the proposed total objective minus the initial total, if available."""

        if self.objective_breakdown_before is None or self.objective_breakdown_after is None:
            return None
        return self.objective_breakdown_after.total - self.objective_breakdown_before.total

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(self, "source_routes", tuple(self.source_routes))
        object.__setattr__(
            self, "eligible_destinations", tuple(self.eligible_destinations)
        )
        object.__setattr__(self, "route_traces", tuple(self.route_traces))
        object.__setattr__(self, "demand_evidence", tuple(self.demand_evidence))
        object.__setattr__(
            self, "rejection_reasons", tuple(self.rejection_reasons)
        )

    @property
    def accepted_placement(self) -> PlacementDecisionState:
        """Return the accepted proposal or the exact input on rejection."""

        if self.accepted and self.prospective_placement is not None:
            return self.prospective_placement
        return self.placement_before

    @property
    def objective_breakdown_before(self) -> ObjectiveBreakdown | None:
        """Return the initial objective evidence from its validation report."""

        return self.validation_before.objective_breakdown

    @property
    def objective_breakdown_after(self) -> ObjectiveBreakdown | None:
        """Return the final objective evidence when a report is available."""

        if self.validation_report is not None:
            return self.validation_report.objective_breakdown
        return None

    @property
    def accepted(self) -> bool:
        """Return whether the proposal has no rejection reasons."""

        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class AGHConsolidationSourceTrace:
    """Record one source-pair scan entry in the initial utilization order."""

    source_position: int
    utilization_evidence: AGHConsolidationUtilizationEvidence
    placement_before: PlacementDecisionState
    source_result: AGHConsolidationSourceResult | None
    status: AGHConsolidationSourceScanStatus

    @property
    def placement_after(self) -> PlacementDecisionState:
        """Return the placement retained at the end of this operation."""

        if self.source_result is not None:
            return self.source_result.accepted_placement
        return self.placement_before


@dataclass(frozen=True, slots=True)
class AGHConsolidationScanResult:
    """Record one consolidation scan and its accepted source-pair proposals."""

    placement_before: PlacementDecisionState
    source_snapshot: tuple[AGHConsolidationUtilizationEvidence, ...]
    source_traces: tuple[AGHConsolidationSourceTrace, ...]
    validation_before: PlacementValidationReport
    validation_after: PlacementValidationReport

    def __post_init__(self) -> None:
        """Own supplied sequences without replaying the producing algorithm."""

        object.__setattr__(self, "source_snapshot", tuple(self.source_snapshot))
        object.__setattr__(self, "source_traces", tuple(self.source_traces))

    @property
    def accepted_source_count(self) -> int:
        """Return the number of accepted source-pair proposals."""

        return sum(trace.status is AGHConsolidationSourceScanStatus.ACCEPTED for trace in self.source_traces)

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

        return self.accepted_source_count > 0


def evaluate_agh_consolidation_source(
    instance: ProblemInstance,
    placement_before: PlacementDecisionState,
    source: AGHConsolidationPair,
    *,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHConsolidationSourceResult:
    """Try redistributing a source pair's traffic and deactivating its deployment.

    Move whole routes in semantic order to the first feasible active destination.
    After complete redistribution, deactivate the source and require independent
    feasibility and strict improvement of the full objective. Rejection returns
    the exact input placement.
    """

    context = _build_agh_search_context(instance, paper_version, validation_tolerances)
    return _evaluate_agh_consolidation_source(context.validate(placement_before), source)


def run_agh_consolidation_scan(
    instance: ProblemInstance,
    placement_before: PlacementDecisionState,
    *,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHConsolidationScanResult:
    """Scan source pairs once in ascending initial compute utilization.

    Skip sources no longer active; adopt accepted transactions immediately.
    Later transactions use current destinations without repeating the source scan.
    """

    context = _build_agh_search_context(instance, paper_version, validation_tolerances)
    return _run_agh_consolidation_scan(context.validate(placement_before))


def _reject_before_redistribution(
    *,
    source: AGHConsolidationPair,
    placement_before: PlacementDecisionState,
    validation_before: PlacementValidationReport,
    reason: AGHConsolidationRejectionReason,
    source_evidence: AGHDeploymentPairEvidence | None = None,
) -> AGHConsolidationSourceResult:
    """Record a precondition rejection before attempting any route transfers."""

    return AGHConsolidationSourceResult(
        source=source,
        placement_before=placement_before,
        source_evidence=source_evidence,
        source_routes=(),
        eligible_destinations=(),
        route_traces=(),
        redistributed_placement=None,
        prospective_placement=None,
        validation_before=validation_before,
        validation_report=None,
        demand_evidence=(),
        rejection_reasons=(reason,),
    )


def _active_destination_candidates(
    problem_data: PdmProblemData,
    placement: PlacementDecisionState,
    source: AGHConsolidationPair,
) -> tuple[AGHConsolidationPair, ...]:
    """Return other coherent active pairs in model/tier order.

    Route-transfer feasibility is checked separately for each candidate.
    """

    return tuple(
        pair
        for pair in problem_data.variable_lattices.model_tier_keys
        if pair != source
        and _deployment_pair_evidence(
            problem_data, placement, pair
        ).is_coherent_active_deployment
    )


def _deactivate_source_pair(
    problem_data: PdmProblemData,
    redistributed: PlacementDecisionState,
    source: AGHConsolidationPair,
) -> PlacementDecisionState:
    """Zero source q/y/w/x/z/v while preserving u and all other pairs."""

    lattices = problem_data.variable_lattices
    source_routes = {
        key for key in lattices.route_keys if key[1:] == source
    }
    source_configurations = {
        key for key in lattices.configuration_keys if key[:2] == source
    }
    x = {
        key: (
            0.0
            if key in source_routes
            else redistributed.routing_fractions.get(key, 0.0)
        )
        for key in lattices.route_keys
    }
    z = {
        key: (
            0
            if key in source_routes
            else redistributed.admission_indicators.get(key, 0.0)
        )
        for key in lattices.route_keys
    }
    q = {
        key: (0 if key == source else redistributed.deployment_indicators.get(key, 0.0))
        for key in lattices.model_tier_keys
    }
    u = {
        key: redistributed.unmet_fractions.get(key, 0.0)
        for key in lattices.unmet_keys
    }
    w = {
        key: (
            0
            if key in source_configurations
            else redistributed.configuration_selectors.get(key, 0.0)
        )
        for key in lattices.configuration_keys
    }
    y = {
        key: (0 if key == source else redistributed.gpu_counts.get(key, 0.0))
        for key in lattices.model_tier_keys
    }
    v = {
        key: (
            0.0
            if key[1:3] == source
            else redistributed.auxiliary_products.get(key, 0.0)
        )
        for key in lattices.auxiliary_product_keys
    }
    routing = RoutingDecisionState(x, z, q, u)
    parallelism = ParallelismDecisionState(routing, w, y)
    return PlacementDecisionState(parallelism, v)


def _audit_demand_consistency(
    problem_data: PdmProblemData,
    placement_before: PlacementDecisionState,
    placement_after: PlacementDecisionState,
) -> tuple[AGHConsolidationDemandEvidence, ...]:
    """Compare routed and unmet fractions for every query."""

    evidence: list[AGHConsolidationDemandEvidence] = []
    for query_id in problem_data.variable_lattices.unmet_keys:
        query_routes = tuple(
            key
            for key in problem_data.variable_lattices.route_keys
            if key[0] == query_id
        )
        routed_before = fsum(
            placement_before.routing_fractions.get(key, 0.0)
            for key in query_routes
        )
        routed_after = fsum(
            placement_after.routing_fractions.get(key, 0.0)
            for key in query_routes
        )
        unmet_before = placement_before.unmet_fractions.get(query_id, 0.0)
        unmet_after = placement_after.unmet_fractions.get(query_id, 0.0)
        evidence.append(
            AGHConsolidationDemandEvidence(
                query_id=query_id,
                routed_before=routed_before,
                routed_after=routed_after,
                unmet_before=unmet_before,
                unmet_after=unmet_after,
            )
        )
    return tuple(evidence)


def _redistribute_source_route(
    problem_data: PdmProblemData,
    validator: PlacementValidator,
    placement_before: PlacementDecisionState,
    source_route: AGHConsolidationRoute,
    destinations: tuple[AGHConsolidationPair, ...],
    *,
    route_position: int,
) -> AGHConsolidationRouteTrace:
    """Try a whole-route transfer and select the first feasible destination."""

    moved_fraction = placement_before.routing_fractions.get(source_route, 0.0)
    attempts: list[AGHConsolidationDestinationAttempt] = []
    selected_destination: AGHConsolidationPair | None = None
    for destination_position, destination in enumerate(destinations):
        prospective_placement = _transfer_route(
            problem_data,
            placement_before,
            source_route,
            destination,
            moved_fraction,
        )
        validation = validator.validate(prospective_placement)
        feasible = validation.is_feasible
        attempts.append(
            AGHConsolidationDestinationAttempt(
                route_position=route_position,
                destination_position=destination_position,
                source_route=source_route,
                destination=destination,
                moved_fraction=moved_fraction,
                placement_before=placement_before,
                prospective_placement=prospective_placement,
                validation_report=validation,
            )
        )
        if feasible:
            selected_destination = destination
            break

    status = (
        AGHConsolidationRouteStatus.REDISTRIBUTED
        if selected_destination is not None
        else AGHConsolidationRouteStatus.NO_FEASIBLE_DESTINATION
    )
    return AGHConsolidationRouteTrace(
        route_position=route_position,
        source_route=source_route,
        moved_fraction=moved_fraction,
        placement_before=placement_before,
        eligible_destinations=destinations,
        attempts=tuple(attempts),
        status=status,
    )


def _evaluate_agh_consolidation_source(
    current_evaluation: _AGHPlacementEvaluation,
    source: AGHConsolidationPair,
) -> AGHConsolidationSourceResult:
    """Evaluate source-pair deactivation using the current placement and report."""

    problem_data = current_evaluation.context.problem_data
    validator = current_evaluation.context.validator
    placement_before = current_evaluation.placement
    validation_before = current_evaluation.validation
    if not validation_before.domain_diagnostic.is_valid:
        return _reject_before_redistribution(
            source=source,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHConsolidationRejectionReason.INVALID_PLACEMENT_DOMAINS,
        )
    if (
        not isinstance(source, tuple)
        or len(source) != 2
        or source not in problem_data.variable_lattices.model_tier_keys
    ):
        return _reject_before_redistribution(
            source=source,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHConsolidationRejectionReason.INVALID_SOURCE,
        )

    source_evidence = _deployment_pair_evidence(problem_data, placement_before, source)
    if not source_evidence.is_active:
        return _reject_before_redistribution(
            source=source,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHConsolidationRejectionReason.SOURCE_NOT_ACTIVE,
            source_evidence=source_evidence,
        )
    if not source_evidence.is_coherent_active_deployment:
        return _reject_before_redistribution(
            source=source,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=(
                AGHConsolidationRejectionReason.INCOHERENT_SOURCE_DEPLOYMENT
            ),
            source_evidence=source_evidence,
        )

    source_routes = tuple(
        route
        for route in problem_data.variable_lattices.route_keys
        if route[1:] == source
        and placement_before.routing_fractions.get(route, 0.0) > 0.0
    )
    destinations = _active_destination_candidates(
        problem_data, placement_before, source
    )
    current_placement = placement_before
    route_traces: list[AGHConsolidationRouteTrace] = []
    for route_position, source_route in enumerate(source_routes):
        route_trace = _redistribute_source_route(
            problem_data,
            validator,
            current_placement,
            source_route,
            destinations,
            route_position=route_position,
        )
        route_traces.append(route_trace)
        if route_trace.status is AGHConsolidationRouteStatus.NO_FEASIBLE_DESTINATION:
            return AGHConsolidationSourceResult(
                source=source,
                placement_before=placement_before,
                source_evidence=source_evidence,
                source_routes=source_routes,
                eligible_destinations=destinations,
                route_traces=tuple(route_traces),
                redistributed_placement=current_placement,
                prospective_placement=None,
                validation_before=validation_before,
                validation_report=None,
                demand_evidence=(),
                rejection_reasons=(
                    AGHConsolidationRejectionReason.REDISTRIBUTION_FAILED,
                ),
            )
        current_placement = route_trace.placement_after

    prospective = _deactivate_source_pair(problem_data, current_placement, source)
    demand_evidence = _audit_demand_consistency(
        problem_data, placement_before, prospective
    )
    validation_after = validator.validate(prospective)
    objective_breakdown_before = validation_before.objective_breakdown
    objective_breakdown_after = validation_after.objective_breakdown
    reasons: list[AGHConsolidationRejectionReason] = []
    if not all(item.is_consistent for item in demand_evidence):
        reasons.append(
            AGHConsolidationRejectionReason.CONSTRUCTION_INCONSISTENCY
        )
    if not validation_after.is_feasible:
        reasons.append(AGHConsolidationRejectionReason.INFEASIBLE)
    if (
        objective_breakdown_before is not None
        and objective_breakdown_after is not None
        and not objective_breakdown_after.total < objective_breakdown_before.total
    ):
        reasons.append(
            AGHConsolidationRejectionReason.NOT_STRICTLY_IMPROVING
        )
    return AGHConsolidationSourceResult(
        source=source,
        placement_before=placement_before,
        source_evidence=source_evidence,
        source_routes=source_routes,
        eligible_destinations=destinations,
        route_traces=tuple(route_traces),
        redistributed_placement=current_placement,
        prospective_placement=prospective,
        validation_before=validation_before,
        validation_report=validation_after,
        demand_evidence=demand_evidence,
        rejection_reasons=tuple(reasons),
    )


def _build_source_utilization_snapshot(
    problem_data: PdmProblemData,
    placement: PlacementDecisionState,
    validation: PlacementValidationReport,
) -> tuple[AGHConsolidationUtilizationEvidence, ...]:
    """Snapshot coherent active sources by utilization, then model/tier order."""

    if validation.domain_diagnostic.is_valid:
        required_by_pair = {
            (item.model_id, item.gpu_tier_id): item.required_compute
            for item in validation.compute_capacity_diagnostics
        }
        available_by_pair = {
            (item.model_id, item.gpu_tier_id): item.available_compute
            for item in validation.compute_capacity_diagnostics
        }
    else:
        # Preserve the standalone scan's domain-error boundary on invalid inputs.
        required_by_pair = compute_required_compute_tflops_per_hour(
            problem_data.instance, placement.routing_decisions
        )
        available_by_pair = compute_available_compute_tflops_per_hour(
            problem_data.instance, placement.parallelism_decisions
        )
    evidence: list[AGHConsolidationUtilizationEvidence] = []
    for semantic_position, pair in enumerate(
        problem_data.variable_lattices.model_tier_keys
    ):
        pair_evidence = _deployment_pair_evidence(problem_data, placement, pair)
        if not pair_evidence.is_coherent_active_deployment:
            continue
        required = required_by_pair[pair]
        available = available_by_pair[pair]
        evidence.append(
            AGHConsolidationUtilizationEvidence(
                pair=pair,
                semantic_position=semantic_position,
                required_compute_tflops_per_hour=required,
                available_compute_tflops_per_hour=available,
                pair_evidence=pair_evidence,
            )
        )
    return tuple(sorted(evidence, key=lambda item: (item.utilization, item.semantic_position)))


def _run_agh_consolidation_scan(
    current_evaluation: _AGHPlacementEvaluation,
) -> AGHConsolidationScanResult:
    """Scan snapshotted source pairs while retaining the current validation report."""

    problem_data = current_evaluation.context.problem_data
    placement_before = current_evaluation.placement
    validation_before = current_evaluation.validation
    source_snapshot = _build_source_utilization_snapshot(
        problem_data, placement_before, validation_before
    )
    current_placement = placement_before
    source_traces: list[AGHConsolidationSourceTrace] = []
    for source_position, utilization_evidence in enumerate(source_snapshot):
        current_evidence = _deployment_pair_evidence(
            problem_data, current_placement, utilization_evidence.pair
        )
        if not current_evidence.is_active:
            source_trace = AGHConsolidationSourceTrace(
                source_position=source_position,
                utilization_evidence=utilization_evidence,
                placement_before=current_placement,
                source_result=None,
                status=(
                    AGHConsolidationSourceScanStatus.SOURCE_NO_LONGER_ACTIVE
                ),
            )
        else:
            source_result = _evaluate_agh_consolidation_source(
                current_evaluation, utilization_evidence.pair
            )
            source_trace = AGHConsolidationSourceTrace(
                source_position=source_position,
                utilization_evidence=utilization_evidence,
                placement_before=current_placement,
                source_result=source_result,
                status=(
                    AGHConsolidationSourceScanStatus.ACCEPTED
                    if source_result.accepted
                    else AGHConsolidationSourceScanStatus.REJECTED
                ),
            )
            if source_result.accepted:
                current_evaluation = _AGHPlacementEvaluation(
                    current_evaluation.context,
                    source_result.accepted_placement,
                    source_result.validation_report,
                )
        source_traces.append(source_trace)
        current_placement = source_trace.placement_after

    return AGHConsolidationScanResult(
        placement_before=placement_before,
        source_snapshot=source_snapshot,
        source_traces=tuple(source_traces),
        validation_before=validation_before,
        validation_after=current_evaluation.validation,
    )
