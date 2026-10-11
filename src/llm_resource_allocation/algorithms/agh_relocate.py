"""Evaluate a full-route AGH relocation using feasibility and total objective."""

from dataclasses import dataclass
from enum import Enum
from math import fsum

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.math.objective import ObjectiveBreakdown, PaperVersion
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    ValidationTolerances,
)


from ._agh_placement_ops import (
    AGHDeploymentPairEvidence,
    _AGHPlacementEvaluation,
    _build_agh_search_context,
    _transfer_route,
    _deployment_pair_evidence,
)


AGHRelocateSourceAssignment = tuple[str, str, str]
AGHRelocateDestinationPair = tuple[str, str]

# Audit transfer sums only; feasibility and strict objective comparison are separate.
AGH_RELOCATE_CONSTRUCTION_ABSOLUTE_TOLERANCE = 1.0e-15


class AGHRelocateRejectionReason(str, Enum):
    """Identify why a relocate proposal was rejected."""

    INVALID_PLACEMENT_DOMAINS = "invalid_placement_domains"
    INVALID_SOURCE = "invalid_source"
    NO_SOURCE_ROUTING = "no_source_routing"
    INCOHERENT_SOURCE_DEPLOYMENT = "incoherent_source_deployment"
    INVALID_DESTINATION = "invalid_destination"
    SAME_SOURCE_AND_DESTINATION = "same_source_and_destination"
    DESTINATION_NOT_ACTIVE = "destination_not_active"
    INCOHERENT_DESTINATION_DEPLOYMENT = "incoherent_destination_deployment"
    CONSTRUCTION_INCONSISTENCY = "construction_inconsistency"
    INFEASIBLE = "infeasible"
    NOT_STRICTLY_IMPROVING = "not_strictly_improving"


@dataclass(frozen=True, slots=True)
class AGHRelocateDemandConsistencyEvidence:
    """Record routed and unmet fractions before and after a transfer."""

    query_id: str
    routed_before: float
    routed_after: float
    construction_absolute_tolerance: float
    unmet_before: float
    unmet_after: float

    @property
    def routed_absolute_difference(self) -> float:
        """Return the absolute change in the query's total routed fraction."""

        return abs(self.routed_after - self.routed_before)

    @property
    def routed_demand_preserved(self) -> bool:
        """Return whether routed fractions agree within construction tolerance."""

        return self.routed_absolute_difference <= self.construction_absolute_tolerance

    @property
    def unmet_demand_preserved(self) -> bool:
        """Return whether the copied unmet fraction is exactly unchanged."""

        return self.unmet_before == self.unmet_after

    @property
    def is_consistent(self) -> bool:
        """Return whether routed and unmet fractions are preserved."""

        return self.routed_demand_preserved and self.unmet_demand_preserved


@dataclass(frozen=True, slots=True)
class AGHRelocateMoveResult:
    """Store a relocate proposal, its validation reports, and its outcome.

    Rejected proposals retain the exact input placement and any constructed
    candidate. Placement selection and objective breakdowns are derived properties.
    """

    source: AGHRelocateSourceAssignment
    destination: AGHRelocateDestinationPair
    moved_fraction: float | None
    placement_before: PlacementDecisionState
    prospective_placement: PlacementDecisionState | None
    source_evidence: AGHDeploymentPairEvidence | None
    destination_evidence: AGHDeploymentPairEvidence | None
    validation_before: PlacementValidationReport
    validation_report: PlacementValidationReport | None
    demand_consistency: AGHRelocateDemandConsistencyEvidence | None
    rejection_reasons: tuple[AGHRelocateRejectionReason, ...]

    @property
    def accepted_placement(self) -> PlacementDecisionState:
        """Return the accepted proposal or the exact input on rejection."""

        if self.accepted and self.prospective_placement is not None:
            return self.prospective_placement
        return self.placement_before

    @property
    def objective_breakdown_before(self) -> ObjectiveBreakdown | None:
        """Return the input placement's objective evidence from its report."""

        return self.validation_before.objective_breakdown

    @property
    def objective_breakdown_after(self) -> ObjectiveBreakdown | None:
        """Return candidate objective evidence when available in its report."""

        if self.validation_report is None:
            return None
        return self.validation_report.objective_breakdown

    @property
    def objective_delta(self) -> float | None:
        """Return candidate total minus input total when both are available."""

        if self.objective_breakdown_before is None or self.objective_breakdown_after is None:
            return None
        return self.objective_breakdown_after.total - self.objective_breakdown_before.total

    def __post_init__(self) -> None:
        """Freeze rejection reasons and check proposal/outcome consistency."""

        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))
        if self.accepted and self.prospective_placement is None:
            raise ValueError("accepted move requires a prospective placement")

    @property
    def accepted(self) -> bool:
        """Return whether the proposal has no rejection reasons."""

        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class _RelocatePreconditions:
    """Store a positive routed fraction and coherent source/destination evidence."""

    moved_fraction: float
    source_evidence: AGHDeploymentPairEvidence
    destination_evidence: AGHDeploymentPairEvidence


def evaluate_agh_relocate_move(
    instance: ProblemInstance,
    placement_before: PlacementDecisionState,
    source: AGHRelocateSourceAssignment,
    destination: AGHRelocateDestinationPair,
    *,
    paper_version: PaperVersion = PaperVersion.V2,
    validation_tolerances: ValidationTolerances | None = None,
) -> AGHRelocateMoveResult:
    """Evaluate a full-route transfer to a distinct active model/tier pair.

    Keep deployments, configurations, GPU counts, and unmet fractions fixed.
    Accept only a demand-preserving, independently feasible proposal with a
    strictly lower total objective. Rejection retains the exact input placement.
    """

    context = _build_agh_search_context(instance, paper_version, validation_tolerances)
    return _evaluate_agh_relocate_move(context.validate(placement_before), source, destination)


def _reject_before_construction(
    *,
    source: AGHRelocateSourceAssignment,
    destination: AGHRelocateDestinationPair,
    placement_before: PlacementDecisionState,
    validation_before: PlacementValidationReport,
    reason: AGHRelocateRejectionReason,
    moved_fraction: float | None = None,
    source_evidence: AGHDeploymentPairEvidence | None = None,
    destination_evidence: AGHDeploymentPairEvidence | None = None,
) -> AGHRelocateMoveResult:
    """Return an early rejection with the input placement retained."""

    return AGHRelocateMoveResult(
        source=source,
        destination=destination,
        moved_fraction=moved_fraction,
        placement_before=placement_before,
        prospective_placement=None,
        source_evidence=source_evidence,
        destination_evidence=destination_evidence,
        validation_before=validation_before,
        validation_report=None,
        demand_consistency=None,
        rejection_reasons=(reason,),
    )


def _audit_demand_consistency(
    problem_data: PdmProblemData,
    placement_before: PlacementDecisionState,
    placement_after: PlacementDecisionState,
    query_id: str,
) -> AGHRelocateDemandConsistencyEvidence:
    """Sum routed fractions and record unmet fractions around a transfer."""

    query_routes = tuple(
        key
        for key in problem_data.variable_lattices.route_keys
        if key[0] == query_id
    )
    routed_before = fsum(
        placement_before.routing_fractions.get(key, 0.0) for key in query_routes
    )
    routed_after = fsum(
        placement_after.routing_fractions.get(key, 0.0) for key in query_routes
    )
    unmet_before = placement_before.unmet_fractions.get(query_id, 0.0)
    unmet_after = placement_after.unmet_fractions.get(query_id, 0.0)
    return AGHRelocateDemandConsistencyEvidence(
        query_id=query_id,
        routed_before=routed_before,
        routed_after=routed_after,
        construction_absolute_tolerance=(
            AGH_RELOCATE_CONSTRUCTION_ABSOLUTE_TOLERANCE
        ),
        unmet_before=unmet_before,
        unmet_after=unmet_after,
    )


def _check_relocate_preconditions(
    problem_data: PdmProblemData,
    placement_before: PlacementDecisionState,
    source: AGHRelocateSourceAssignment,
    destination: AGHRelocateDestinationPair,
    validation_before: PlacementValidationReport,
) -> _RelocatePreconditions | AGHRelocateMoveResult:
    """Check move eligibility and return evidence or the first rejection."""

    if not validation_before.domain_diagnostic.is_valid:
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHRelocateRejectionReason.INVALID_PLACEMENT_DOMAINS,
        )

    if (
        not isinstance(source, tuple)
        or len(source) != 3
        or source not in problem_data.variable_lattices.route_keys
    ):
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHRelocateRejectionReason.INVALID_SOURCE,
        )
    moved_fraction = placement_before.routing_fractions.get(source, 0.0)
    if moved_fraction <= 0.0:
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHRelocateRejectionReason.NO_SOURCE_ROUTING,
            moved_fraction=moved_fraction,
        )

    source_pair = source[1:]
    source_evidence = _deployment_pair_evidence(
        problem_data, placement_before, source_pair
    )
    if (
        not source_evidence.is_coherent_active_deployment
        or placement_before.admission_indicators.get(source, 0.0) != 1
    ):
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHRelocateRejectionReason.INCOHERENT_SOURCE_DEPLOYMENT,
            moved_fraction=moved_fraction,
            source_evidence=source_evidence,
        )

    if (
        not isinstance(destination, tuple)
        or len(destination) != 2
        or destination not in problem_data.variable_lattices.model_tier_keys
    ):
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHRelocateRejectionReason.INVALID_DESTINATION,
            moved_fraction=moved_fraction,
            source_evidence=source_evidence,
        )
    if destination == source_pair:
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHRelocateRejectionReason.SAME_SOURCE_AND_DESTINATION,
            moved_fraction=moved_fraction,
            source_evidence=source_evidence,
        )

    destination_evidence = _deployment_pair_evidence(
        problem_data, placement_before, destination
    )
    if not destination_evidence.is_active:
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=AGHRelocateRejectionReason.DESTINATION_NOT_ACTIVE,
            moved_fraction=moved_fraction,
            source_evidence=source_evidence,
            destination_evidence=destination_evidence,
        )
    if not destination_evidence.is_coherent_active_deployment:
        return _reject_before_construction(
            source=source,
            destination=destination,
            placement_before=placement_before,
            validation_before=validation_before,
            reason=(
                AGHRelocateRejectionReason.INCOHERENT_DESTINATION_DEPLOYMENT
            ),
            moved_fraction=moved_fraction,
            source_evidence=source_evidence,
            destination_evidence=destination_evidence,
        )

    return _RelocatePreconditions(
        moved_fraction, source_evidence, destination_evidence
    )


def _evaluate_agh_relocate_move(
    current: _AGHPlacementEvaluation,
    source: AGHRelocateSourceAssignment,
    destination: AGHRelocateDestinationPair,
) -> AGHRelocateMoveResult:
    """Construct and evaluate a proposal using the current placement's report."""

    problem_data = current.context.problem_data
    validator = current.context.validator
    placement_before = current.placement
    validation_before = current.validation
    preconditions = _check_relocate_preconditions(
        problem_data, placement_before, source, destination, validation_before
    )
    if isinstance(preconditions, AGHRelocateMoveResult):
        return preconditions
    moved_fraction = preconditions.moved_fraction
    source_evidence = preconditions.source_evidence
    destination_evidence = preconditions.destination_evidence

    prospective = _transfer_route(
        problem_data,
        placement_before,
        source,
        destination,
        moved_fraction,
    )
    demand_evidence = _audit_demand_consistency(
        problem_data, placement_before, prospective, source[0]
    )
    validation_after = validator.validate(prospective)
    objective_breakdown_before = validation_before.objective_breakdown
    objective_breakdown_after = validation_after.objective_breakdown
    reasons: list[AGHRelocateRejectionReason] = []
    if not demand_evidence.is_consistent:
        reasons.append(AGHRelocateRejectionReason.CONSTRUCTION_INCONSISTENCY)
    if not validation_after.is_feasible:
        reasons.append(AGHRelocateRejectionReason.INFEASIBLE)
    if (
        objective_breakdown_before is not None
        and objective_breakdown_after is not None
        and not objective_breakdown_after.total < objective_breakdown_before.total
    ):
        # Algorithm 2 requires exact strict improvement, without an objective epsilon.
        reasons.append(AGHRelocateRejectionReason.NOT_STRICTLY_IMPROVING)

    return AGHRelocateMoveResult(
        source=source,
        destination=destination,
        moved_fraction=moved_fraction,
        placement_before=placement_before,
        prospective_placement=prospective,
        source_evidence=source_evidence,
        destination_evidence=destination_evidence,
        validation_before=validation_before,
        validation_report=validation_after,
        demand_consistency=demand_evidence,
        rejection_reasons=tuple(reasons),
    )
