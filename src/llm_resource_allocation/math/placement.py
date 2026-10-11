"""Placement decisions and full-index McCormick diagnostics (v2 Eqs. 7a-7b).

``PlacementDecisionState`` shares routing and parallelism mappings and adds
supplied auxiliaries ``v[i,j,k,n,m]`` for the product ``x[i,j,k]*w[j,k,n,m]``.
"""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.domain import ProblemInstance, validate_problem_instance

from .mccormick import (
    McCormickResiduals,
    compute_mccormick_residuals,
    satisfies_mccormick_envelope,
)
from .parallelism import (
    ConfigurationSelectorKey,
    ParallelismDecisionState,
    _validate_parallelism_configuration_domains,
)
from .routing import (
    DecisionDomainError,
    ModelTierKey,
    QueryModelTierKey,
    RoutingDecisionState,
    _validate_routing_decision_domains,
)


AuxiliaryProductKey = tuple[str, str, str, int, int]


@dataclass(frozen=True, slots=True)
class PlacementDecisionState:
    """Store x, z, q, u, w, y, and v for the v2 formulation (Table 1).

    ``parallelism_decisions`` supplies w, y, and nested routing decisions;
    ``auxiliary_products[(i,j,k,n,m)]`` stores v. Missing known entries mean zero.
    Processing delay D_proc is derived from v and D_cfg after Eqs. (7a)-(7b).
    """

    parallelism_decisions: ParallelismDecisionState
    auxiliary_products: Mapping[AuxiliaryProductKey, float]

    def __post_init__(self) -> None:
        """Copy ``auxiliary_products`` (v) into a read-only mapping."""

        object.__setattr__(
            self,
            "auxiliary_products",
            MappingProxyType(dict(self.auxiliary_products)),
        )

    @property
    def routing_decisions(self) -> RoutingDecisionState:
        """Return the shared routing state containing x, z, q, and u."""

        return self.parallelism_decisions.routing_decisions

    @property
    def routing_fractions(self) -> Mapping[QueryModelTierKey, float]:
        """Return ``(i,j,k) -> x[i,j,k]`` from ``routing_decisions``."""

        return self.routing_decisions.routing_fractions

    @property
    def admission_indicators(self) -> Mapping[QueryModelTierKey, float]:
        """Return ``(i,j,k) -> z[i,j,k]`` from ``routing_decisions``."""

        return self.routing_decisions.admission_indicators

    @property
    def deployment_indicators(self) -> Mapping[ModelTierKey, float]:
        """Return ``(j,k) -> q[j,k]`` from ``parallelism_decisions``."""

        return self.parallelism_decisions.deployment_indicators

    @property
    def unmet_fractions(self) -> Mapping[str, float]:
        """Return ``i -> u[i]`` from ``routing_decisions``."""

        return self.routing_decisions.unmet_fractions

    @property
    def configuration_selectors(
        self,
    ) -> Mapping[ConfigurationSelectorKey, float]:
        """Return ``(j,k,n,m) -> w[j,k,n,m]`` from ``parallelism_decisions``."""

        return self.parallelism_decisions.configuration_selectors

    @property
    def gpu_counts(self) -> Mapping[ModelTierKey, float]:
        """Return ``(j,k) -> y[j,k]`` from ``parallelism_decisions``."""

        return self.parallelism_decisions.gpu_counts


@dataclass(frozen=True, slots=True)
class McCormickConsistencyDiagnostic:
    """Record product consistency for one ``(i,j,k,n,m)`` (v2 Eqs. 7a-7b).

    ``routing_fraction`` is x; ``configuration_selector`` is w;
    ``auxiliary_product`` is supplied v. ``residuals`` derives the four envelope
    slacks from these values through the scalar mathematical authority.
    """

    query_id: str
    model_id: str
    gpu_tier_id: str
    tensor_parallel_degree: int
    pipeline_parallel_depth: int
    routing_fraction: float
    configuration_selector: float
    auxiliary_product: float

    @property
    def residuals(self) -> McCormickResiduals:
        """Derive the envelope slacks from the supplied x, w, and v."""

        return compute_mccormick_residuals(
            self.routing_fraction,
            self.configuration_selector,
            self.auxiliary_product,
        )

    @property
    def minimum_residual(self) -> float:
        """Return ``min(x-v, w-v, v-x-w+1, v)`` from ``residuals``."""

        return self.residuals.minimum_residual

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Recompute the four slacks from x, w, v and check ``>= -tolerance``.

        This tests the McCormick envelope in v2 Eqs. (7a)-(7b).
        """

        return satisfies_mccormick_envelope(
            self.routing_fraction,
            self.configuration_selector,
            self.auxiliary_product,
            tolerance,
        )


def validate_placement_decision_domains(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> None:
    """Validate x, z, q, u, w, y, and v domains and indices (v2 Table 1).

    ``auxiliary_products`` must contain finite ``v[i,j,k,n,m] in [0,1]``;
    these bounds follow from Eqs. (7a)-(7b). Keys require known IDs and valid
    tier configurations. Missing known entries mean zero; domains are exact.
    """

    validate_problem_instance(instance)
    _validate_placement_decision_domains(instance, state)


def _validate_placement_decision_domains(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> None:
    """Check all decision domains after instance validation; inspect shared q once."""

    # Domains alone do not establish any of Eqs. (7)-(8k).
    _validate_routing_decision_domains(instance, state.routing_decisions)
    _validate_parallelism_configuration_domains(instance, state.parallelism_decisions)

    auxiliary_product_keys = {
        (
            query_id,
            model_id,
            tier.id,
            config.tensor_parallel_degree,
            config.pipeline_parallel_depth,
        )
        for query_id in instance.query_ids
        for model_id in instance.model_ids
        for tier in instance.gpu_tiers
        for config in tier.parallelism_configs
    }
    errors: list[str] = []
    unknown_keys = {
        key
        for key in state.auxiliary_products
        if (
            not isinstance(key, tuple)
            or len(key) != 5
            or isinstance(key[3], bool)
            or not isinstance(key[3], int)
            or isinstance(key[4], bool)
            or not isinstance(key[4], int)
            or key not in auxiliary_product_keys
        )
    }
    if unknown_keys:
        errors.append(
            "auxiliary_products has unknown or infeasible keys: "
            f"{sorted(unknown_keys, key=repr)!r}"
        )

    for key, value in state.auxiliary_products.items():
        label = f"auxiliary_products[{key!r}]"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{label} must be a real number in [0, 1]; got {value!r}")
        elif not isfinite(value):
            errors.append(f"{label} must be finite; got {value!r}")
        elif value < 0.0 or value > 1.0:
            errors.append(f"{label} must be in [0, 1]; got {value!r}")

    if errors:
        raise DecisionDomainError(
            "Invalid placement decision state:\n- " + "\n- ".join(errors)
        )


def compute_mccormick_consistency_diagnostics(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> tuple[McCormickConsistencyDiagnostic, ...]:
    """Compute McCormick slacks for every ``(i,j,k,n,m)`` (v2 Eqs. 7a-7b).

    Read supplied x, w, and v, treating missing known entries as zero.
    Return ``x-v``, ``w-v``, ``v-x-w+1``, and ``v`` in instance/configuration order.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_mccormick_consistency_diagnostics(instance, state)


def _compute_mccormick_consistency_diagnostics(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> tuple[McCormickConsistencyDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""
    diagnostics: list[McCormickConsistencyDiagnostic] = []
    for query_id in instance.query_ids:
        for model_id in instance.model_ids:
            for tier in instance.gpu_tiers:
                routing_fraction = state.routing_fractions.get(
                    (query_id, model_id, tier.id), 0.0
                )
                for config in tier.parallelism_configs:
                    selector_key = (
                        model_id,
                        tier.id,
                        config.tensor_parallel_degree,
                        config.pipeline_parallel_depth,
                    )
                    auxiliary_product_key = (query_id, *selector_key)
                    selector = state.configuration_selectors.get(
                        selector_key, 0.0
                    )
                    auxiliary = state.auxiliary_products.get(auxiliary_product_key, 0.0)
                    diagnostics.append(
                        McCormickConsistencyDiagnostic(
                            query_id=query_id,
                            model_id=model_id,
                            gpu_tier_id=tier.id,
                            tensor_parallel_degree=(
                                config.tensor_parallel_degree
                            ),
                            pipeline_parallel_depth=(
                                config.pipeline_parallel_depth
                            ),
                            routing_fraction=routing_fraction,
                            configuration_selector=selector,
                            auxiliary_product=auxiliary,
                        )
                    )
    return tuple(diagnostics)
