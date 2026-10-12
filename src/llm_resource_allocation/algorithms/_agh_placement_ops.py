"""Prepare AGH search data, inspect deployments, and construct route transfers."""

from dataclasses import dataclass

from llm_resource_allocation.domain import ParallelismConfig, ProblemInstance
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    PlacementValidator,
    ValidationTolerances,
)


@dataclass(frozen=True, slots=True)
class _AGHSearchContext:
    """Store fixed problem data and one validation policy for an AGH search."""

    problem_data: PdmProblemData
    validator: PlacementValidator

    def validate(self, placement: PlacementDecisionState) -> "_AGHPlacementEvaluation":
        """Evaluate a placement and retain its independent validation report."""

        return _AGHPlacementEvaluation(self, placement, self.validator.validate(placement))


@dataclass(frozen=True, slots=True)
class _AGHPlacementEvaluation:
    """Store a placement and its validation report within one search context."""

    context: _AGHSearchContext
    placement: PlacementDecisionState
    validation: PlacementValidationReport

@dataclass(frozen=True, slots=True)
class AGHDeploymentPairEvidence:
    """Record activation, configurations, and GPU count for a model/tier pair."""

    model_id: str
    gpu_tier_id: str
    deployment_indicator: float
    selected_configurations: tuple[ParallelismConfig, ...]
    gpu_count: float

    def __post_init__(self) -> None:
        """Own the configuration sequence independently of the caller."""

        object.__setattr__(
            self, "selected_configurations", tuple(self.selected_configurations)
        )

    @property
    def is_active(self) -> bool:
        """Return whether ``q=1`` and ``y>0`` for this pair."""

        return self.deployment_indicator == 1 and self.gpu_count > 0

    @property
    def is_coherent_active_deployment(self) -> bool:
        """Return whether the active pair has one configuration with ``y=n*m``."""

        return (
            self.is_active
            and len(self.selected_configurations) == 1
            and self.gpu_count == self.selected_configurations[0].gpu_count
        )


def _build_agh_search_context(
    instance: ProblemInstance,
    paper_version: PaperVersion,
    validation_tolerances: ValidationTolerances | None,
) -> _AGHSearchContext:
    """Prepare problem data and the configured independent placement validator."""

    selected_tolerances = (
        ValidationTolerances() if validation_tolerances is None else validation_tolerances
    )
    return _AGHSearchContext(
        PdmProblemData(instance),
        PlacementValidator(instance, paper_version, selected_tolerances),
    )


def _deployment_pair_evidence(
    problem_data: PdmProblemData,
    placement: PlacementDecisionState,
    pair: tuple[str, str],
) -> AGHDeploymentPairEvidence:
    """Read deployment and configuration evidence for one model/tier pair."""

    configuration_keys = tuple(
        key
        for key in problem_data.variable_lattices.configuration_keys
        if key[:2] == pair
    )
    selected = tuple(
        ParallelismConfig(key[2], key[3])
        for key in configuration_keys
        if placement.configuration_selectors.get(key, 0.0) == 1
    )
    deployment = placement.deployment_indicators.get(pair, 0.0)
    gpu_count = placement.gpu_counts.get(pair, 0.0)
    return AGHDeploymentPairEvidence(
        model_id=pair[0],
        gpu_tier_id=pair[1],
        deployment_indicator=deployment,
        selected_configurations=selected,
        gpu_count=gpu_count,
    )


def _transfer_route(
    problem_data: PdmProblemData,
    placement_before: PlacementDecisionState,
    source: tuple[str, str, str],
    destination: tuple[str, str],
    moved_fraction: float,
) -> PlacementDecisionState:
    """Transfer a source route's fraction and update its admission and products.

    Preserve deployments, configurations, GPU counts, unmet fractions, and
    unrelated auxiliary products. Derive affected ``z`` from positive ``x`` and
    affected ``v`` from ``x*w``.
    """

    lattices = problem_data.variable_lattices
    destination_route = (source[0], *destination)
    x = {
        key: placement_before.routing_fractions.get(key, 0.0)
        for key in lattices.route_keys
    }
    x[source] = 0.0
    x[destination_route] = x[destination_route] + moved_fraction

    # Positive-route admission: z=1 exactly when x>0.
    z = {
        key: placement_before.admission_indicators.get(key, 0.0)
        for key in lattices.route_keys
    }
    z[source] = int(x[source] > 0.0)
    z[destination_route] = int(x[destination_route] > 0.0)

    q = {
        key: placement_before.deployment_indicators.get(key, 0.0)
        for key in lattices.model_tier_keys
    }
    u = {
        key: placement_before.unmet_fractions.get(key, 0.0)
        for key in lattices.unmet_keys
    }
    w = {
        key: placement_before.configuration_selectors.get(key, 0.0)
        for key in lattices.configuration_keys
    }
    y = {
        key: placement_before.gpu_counts.get(key, 0.0)
        for key in lattices.model_tier_keys
    }

    affected_routes = {source, destination_route}
    v = {
        key: (
            x[key[:3]] * w[(key[1], key[2], key[3], key[4])]
            if key[:3] in affected_routes
            else placement_before.auxiliary_products.get(key, 0.0)
        )
        for key in lattices.auxiliary_product_keys
    }
    routing = RoutingDecisionState(
        routing_fractions=x,
        admission_indicators=z,
        deployment_indicators=q,
        unmet_fractions=u,
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors=w,
        gpu_counts=y,
    )
    return PlacementDecisionState(
        parallelism_decisions=parallelism,
        auxiliary_products=v,
    )
