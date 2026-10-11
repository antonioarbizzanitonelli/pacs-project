"""Check TP/PP selection and GPU counts for each model/tier pair.

Paper v2 Eq. (8d): ``sum_(n,m) w[j,k,n,m] = q[j,k]``.
Paper v2 Eq. (8e): ``y[j,k] = sum_(n,m) n*m*w[j,k,n,m]``.
Here ``n`` is the TP degree and ``m`` is the PP depth.
"""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.domain import ProblemInstance, validate_problem_instance

from .routing import (
    DecisionDomainError,
    ModelTierKey,
    RoutingDecisionState,
    _validate_deployment_indicator_domains,
)


ConfigurationSelectorKey = tuple[str, str, int, int]


@dataclass(frozen=True, slots=True)
class ParallelismDecisionState:
    """Store parallelism decisions for Paper v2 Eqs. (8d)-(8e).

    ``configuration_selectors[(j,k,n,m)]`` stores ``w[j,k,n,m]``;
    ``gpu_counts[(j,k)]`` stores ``y[j,k]``. ``routing_decisions`` supplies
    ``q[j,k]``. Missing known entries are treated as zero.
    """

    routing_decisions: RoutingDecisionState
    configuration_selectors: Mapping[ConfigurationSelectorKey, float]
    gpu_counts: Mapping[ModelTierKey, float]

    def __post_init__(self) -> None:
        """Copy ``configuration_selectors`` (w) and ``gpu_counts`` (y) read-only."""

        for field_name in ("configuration_selectors", "gpu_counts"):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )

    @property
    def deployment_indicators(self) -> Mapping[ModelTierKey, float]:
        """Return ``(j,k) -> q[j,k]`` from ``routing_decisions``.

        ``q[j,k]`` is 1 when model ``j`` is deployed on GPU tier ``k``,
        otherwise 0. This same mapping is used in Eqs. (8d) and (8k).
        """

        return self.routing_decisions.deployment_indicators


@dataclass(frozen=True, slots=True)
class ConfigurationSelectionDiagnostic:
    """Record configuration selection for one model/tier pair (v2 Eq. 8d).

    ``selected_configuration_count = sum_(n,m) w[j,k,n,m]``;
    ``deployment_indicator = q[j,k]``. Their difference is ``signed_residual``.
    """

    model_id: str
    gpu_tier_id: str
    selected_configuration_count: float
    deployment_indicator: float

    @property
    def signed_residual(self) -> float:
        """Return ``sum_(n,m) w[j,k,n,m] - q[j,k]`` (v2 Eq. 8d)."""

        return self.selected_configuration_count - self.deployment_indicator

    @property
    def absolute_violation(self) -> float:
        """Return ``abs(signed_residual)`` for the Eq. (8d) equality."""

        return abs(self.signed_residual)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``abs(sum_(n,m) w[j,k,n,m] - q[j,k]) <= tolerance``.

        The tolerance applies to the Eq. (8d) residual, not the binary domains.
        """

        return self.absolute_violation <= _validate_tolerance(tolerance)


@dataclass(frozen=True, slots=True)
class GpuCountLinkingDiagnostic:
    """Record GPU-count linking for one model/tier pair (v2 Eq. 8e).

    ``allocated_gpu_count = y[j,k]``;
    ``implied_gpu_count = sum_(n,m) n*m*w[j,k,n,m]``.
    Their difference is ``signed_residual``.
    """

    model_id: str
    gpu_tier_id: str
    allocated_gpu_count: float
    implied_gpu_count: float

    @property
    def signed_residual(self) -> float:
        """Return ``y[j,k] - sum_(n,m) n*m*w[j,k,n,m]`` (v2 Eq. 8e)."""

        return self.allocated_gpu_count - self.implied_gpu_count

    @property
    def absolute_violation(self) -> float:
        """Return ``abs(signed_residual)`` for the Eq. (8e) equality."""

        return abs(self.signed_residual)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``abs(y[j,k] - sum_(n,m) n*m*w[j,k,n,m]) <= tolerance``.

        The tolerance applies to the Eq. (8e) residual, not the decision domains.
        """

        return self.absolute_violation <= _validate_tolerance(tolerance)


def _validate_tolerance(tolerance: float) -> float:
    """Require a finite non-negative equality tolerance without coercion."""

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("tolerance must be a finite non-negative real number")
    return tolerance


def _append_binary_selector_error(
    value: object,
    label: str,
    errors: list[str],
) -> None:
    """Append an error if a selector ``w[j,k,n,m]`` is not exactly 0 or 1."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value not in (0, 1)
    ):
        errors.append(f"{label} must be exactly binary (0 or 1); got {value!r}")


def _append_nonnegative_integer_error(
    value: object,
    label: str,
    errors: list[str],
) -> None:
    """Append an error if ``y[j,k]`` is not a finite non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a real integer value; got {value!r}")
        return
    if not isfinite(value):
        errors.append(f"{label} must be finite; got {value!r}")
        return
    # Exact solver-style values such as 8.0 represent integers; fractional or
    # near-integer values are never rounded by a feasibility tolerance.
    if value < 0 or value != int(value):
        errors.append(
            f"{label} must be a non-negative integer value; got {value!r}"
        )


def validate_parallelism_decision_domains(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> None:
    """Validate parallelism decision domains and indices (v2 Table 1).

    Require binary ``w[j,k,n,m]`` and ``q[j,k]``, non-negative integer
    ``y[j,k]``, known model/tier IDs, and ``(n,m) in N_k x M_k``.
    Missing known entries mean zero; domains are checked without tolerance.
    """

    validate_problem_instance(instance)
    _validate_deployment_indicator_domains(instance, state.deployment_indicators)
    _validate_parallelism_configuration_domains(instance, state)


def _validate_parallelism_configuration_domains(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> None:
    """Check w and y after the instance and shared q domains are validated."""

    pair_keys = {
        (model_id, tier_id)
        for model_id in instance.model_ids
        for tier_id in instance.gpu_tier_ids
    }
    selector_keys = {
        (
            model_id,
            tier.id,
            config.tensor_parallel_degree,
            config.pipeline_parallel_depth,
        )
        for model_id in instance.model_ids
        for tier in instance.gpu_tiers
        for config in tier.parallelism_configs
    }
    errors: list[str] = []

    unknown_selectors = {
        key
        for key in state.configuration_selectors
        if (
            not isinstance(key, tuple)
            or len(key) != 4
            or isinstance(key[2], bool)
            or not isinstance(key[2], int)
            or isinstance(key[3], bool)
            or not isinstance(key[3], int)
            or key not in selector_keys
        )
    }
    if unknown_selectors:
        errors.append(
            "configuration_selectors has unknown or infeasible keys: "
            f"{sorted(unknown_selectors, key=repr)!r}"
        )
    unknown_counts = set(state.gpu_counts) - pair_keys
    if unknown_counts:
        errors.append(
            "gpu_counts has unknown semantic IDs: "
            f"{sorted(unknown_counts, key=repr)!r}"
        )

    for key, value in state.configuration_selectors.items():
        _append_binary_selector_error(
            value, f"configuration_selectors[{key!r}]", errors
        )
    for key, value in state.gpu_counts.items():
        _append_nonnegative_integer_error(
            value, f"gpu_counts[{key!r}]", errors
        )

    if errors:
        raise DecisionDomainError(
            "Invalid parallelism decision state:\n- " + "\n- ".join(errors)
        )


def compute_configuration_selection_residuals(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> tuple[ConfigurationSelectionDiagnostic, ...]:
    """Compute the selection residual for each model/tier pair (v2 Eq. 8d).

    ``sum_(n,m) w[j,k,n,m] - q[j,k]`` compares the number of selected
    configurations with the deployment indicator. Results follow instance order.
    """

    validate_parallelism_decision_domains(instance, state)
    return _compute_configuration_selection_residuals(instance, state)


def compute_gpu_count_linking_residuals(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> tuple[GpuCountLinkingDiagnostic, ...]:
    """Compute the GPU-count residual for each model/tier pair (v2 Eq. 8e).

    ``y[j,k] - sum_(n,m) n*m*w[j,k,n,m]`` compares allocated GPUs with the
    count required by selected configurations. Results follow instance order.
    """

    validate_parallelism_decision_domains(instance, state)
    return _compute_gpu_count_linking_residuals(instance, state)


def _compute_configuration_selection_residuals(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> tuple[ConfigurationSelectionDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""
    residuals: list[ConfigurationSelectionDiagnostic] = []
    for model_id in instance.model_ids:
        for tier in instance.gpu_tiers:
            # Exact binary terms need no numerical stabilization; ordinary
            # summation keeps the exact-one count transparent.
            selected_count = sum(
                state.configuration_selectors.get(
                    (
                        model_id,
                        tier.id,
                        config.tensor_parallel_degree,
                        config.pipeline_parallel_depth,
                    ),
                    0.0,
                )
                for config in tier.parallelism_configs
            )
            deployment_indicator = state.deployment_indicators.get(
                (model_id, tier.id), 0.0
            )
            residuals.append(
                ConfigurationSelectionDiagnostic(
                    model_id=model_id,
                    gpu_tier_id=tier.id,
                    selected_configuration_count=selected_count,
                    deployment_indicator=deployment_indicator,
                )
            )
    return tuple(residuals)


def _compute_gpu_count_linking_residuals(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> tuple[GpuCountLinkingDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""
    residuals: list[GpuCountLinkingDiagnostic] = []
    for model_id in instance.model_ids:
        for tier in instance.gpu_tiers:
            # Paper v2 Eq. (8e) is a direct finite sum of integer products.
            implied_gpu_count = sum(
                config.gpu_count
                * state.configuration_selectors.get(
                    (
                        model_id,
                        tier.id,
                        config.tensor_parallel_degree,
                        config.pipeline_parallel_depth,
                    ),
                    0.0,
                )
                for config in tier.parallelism_configs
            )
            allocated_gpu_count = state.gpu_counts.get(
                (model_id, tier.id), 0.0
            )
            residuals.append(
                GpuCountLinkingDiagnostic(
                    model_id=model_id,
                    gpu_tier_id=tier.id,
                    allocated_gpu_count=allocated_gpu_count,
                    implied_gpu_count=implied_gpu_count,
                )
            )
    return tuple(residuals)
