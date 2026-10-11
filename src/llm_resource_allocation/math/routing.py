"""Demand balance and routing consistency (Paper v2 Eqs. 8b, 8k).

For each query ``i``, require ``sum_(j,k) x[i,j,k] + u[i] = 1``.
Each route must satisfy ``0 <= x[i,j,k] <= z[i,j,k] <= q[j,k]``.
"""

from dataclasses import dataclass
from math import fsum, isfinite
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.domain import ProblemInstance, validate_problem_instance


ModelTierKey = tuple[str, str]
QueryModelTierKey = tuple[str, str, str]


class DecisionDomainError(ValueError):
    """Report invalid decision keys or values."""


@dataclass(frozen=True, slots=True)
class RoutingDecisionState:
    """Store routing and deployment decisions (v2 Table 1).

    ``routing_fractions[(i,j,k)]`` stores x; ``admission_indicators[(i,j,k)]``
    stores z; ``deployment_indicators[(j,k)]`` stores q; ``unmet_fractions[i]``
    stores u. Missing known entries are treated as zero.
    """

    routing_fractions: Mapping[QueryModelTierKey, float]
    admission_indicators: Mapping[QueryModelTierKey, float]
    deployment_indicators: Mapping[ModelTierKey, float]
    unmet_fractions: Mapping[str, float]

    def __post_init__(self) -> None:
        """Copy the x, z, q, and u mappings into read-only views."""

        for field_name in (
            "routing_fractions",
            "admission_indicators",
            "deployment_indicators",
            "unmet_fractions",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )


@dataclass(frozen=True, slots=True)
class DemandBalanceDiagnostic:
    """Record one query's demand balance (v2 Eq. 8b).

    ``routed_fraction = sum_(j,k) x[i,j,k]``; ``unmet_fraction = u[i]``;
    ``lhs = sum_(j,k) x[i,j,k] + u[i]``. ``signed_residual = lhs - 1``.
    """

    query_id: str
    routed_fraction: float
    unmet_fraction: float
    lhs: float

    @property
    def signed_residual(self) -> float:
        """Return ``sum_(j,k) x[i,j,k] + u[i] - 1`` (v2 Eq. 8b)."""

        return self.lhs - 1.0

    @property
    def absolute_violation(self) -> float:
        """Return ``abs(signed_residual)`` for the Eq. (8b) equality."""

        return abs(self.signed_residual)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``abs(signed_residual) <= tolerance`` (v2 Eq. 8b)."""

        return self.absolute_violation <= _validate_tolerance(tolerance)


@dataclass(frozen=True, slots=True)
class RoutingConsistencyDiagnostic:
    """Record slacks for one route's ``0 <= x <= z <= q`` (v2 Eq. 8k).

    ``routing_fraction = x[i,j,k]``, ``admission_indicator = z[i,j,k]``,
    ``deployment_indicator = q[j,k]``. Slacks x, z-x, and q-z are non-negative
    when their respective inequalities hold.
    """

    query_id: str
    model_id: str
    gpu_tier_id: str
    routing_fraction: float
    admission_indicator: float
    deployment_indicator: float

    @property
    def routing_nonnegative_slack(self) -> float:
        """Return ``x[i,j,k]``, the slack for ``0 <= x[i,j,k]`` (v2 Eq. 8k)."""

        return self.routing_fraction

    @property
    def admission_slack(self) -> float:
        """Return ``z[i,j,k] - x[i,j,k]`` for ``x <= z`` (v2 Eq. 8k)."""

        return self.admission_indicator - self.routing_fraction

    @property
    def deployment_slack(self) -> float:
        """Return ``q[j,k] - z[i,j,k]`` for ``z <= q`` (v2 Eq. 8k)."""

        return self.deployment_indicator - self.admission_indicator

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether all three slacks are ``>= -tolerance`` (v2 Eq. 8k)."""

        validated_tolerance = _validate_tolerance(tolerance)
        return min(
            self.routing_nonnegative_slack,
            self.admission_slack,
            self.deployment_slack,
        ) >= -validated_tolerance


def _validate_tolerance(tolerance: float) -> float:
    """Require a finite, non-negative tolerance."""

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("tolerance must be a finite non-negative real number")
    return tolerance


def _append_real_domain_error(
    value: object,
    label: str,
    errors: list[str],
    *,
    maximum: float,
) -> None:
    """Report values outside ``[0, maximum]`` or not finite real numbers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a real number; got {value!r}")
        return
    if not isfinite(value):
        errors.append(f"{label} must be finite; got {value!r}")
        return
    if value < 0.0 or value > maximum:
        errors.append(f"{label} must be in [0, {maximum}]; got {value!r}")


def _append_binary_domain_error(
    value: object, label: str, errors: list[str]
) -> None:
    """Report values other than numeric zero or one."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value not in (0, 1)
    ):
        errors.append(f"{label} must be exactly binary (0 or 1); got {value!r}")


def _append_deployment_indicator_errors(
    instance: ProblemInstance,
    deployment_indicators: Mapping[ModelTierKey, float],
    errors: list[str],
) -> None:
    """Append errors for unknown ``(j,k)`` keys or non-binary ``q[j,k]``."""

    pair_keys = {
        (model_id, tier_id)
        for model_id in instance.model_ids
        for tier_id in instance.gpu_tier_ids
    }
    unknown_keys = set(deployment_indicators) - pair_keys
    if unknown_keys:
        errors.append(
            "deployment_indicators has unknown semantic IDs: "
            f"{sorted(unknown_keys, key=repr)!r}"
        )
    for key, value in deployment_indicators.items():
        _append_binary_domain_error(
            value, f"deployment_indicators[{key!r}]", errors
        )


def validate_deployment_indicator_domains(
    instance: ProblemInstance,
    deployment_indicators: Mapping[ModelTierKey, float],
) -> None:
    """Validate known ``(j,k)`` keys and binary ``q[j,k]`` (v2 Table 1).

    Missing known pairs mean zero. Invalid instance data raise
    ``InstanceValidationError``; invalid q entries raise ``DecisionDomainError``.
    """

    validate_problem_instance(instance)
    _validate_deployment_indicator_domains(instance, deployment_indicators)


def _validate_deployment_indicator_domains(
    instance: ProblemInstance,
    deployment_indicators: Mapping[ModelTierKey, float],
) -> None:
    """Check q domains after the caller has validated the instance."""

    errors: list[str] = []
    _append_deployment_indicator_errors(instance, deployment_indicators, errors)
    if errors:
        raise DecisionDomainError(
            "Invalid deployment indicator state:\n- " + "\n- ".join(errors)
        )


def validate_routing_decision_domains(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> None:
    """Validate routing decision indices and exact domains (v2 Table 1).

    Require ``x[i,j,k] in [0,1]``, binary ``z[i,j,k]`` and ``q[j,k]``, and
    ``u[i] in [0,zeta_i]``, where ``zeta_i = query.max_unmet_fraction``.
    Known IDs are required; missing known entries mean zero.
    """

    validate_problem_instance(instance)
    _validate_routing_decision_domains(instance, state)


def _validate_routing_decision_domains(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> None:
    """Check x, z, q and u domains after instance validation."""

    query_ids = set(instance.query_ids)
    triple_keys = {
        (query_id, model_id, tier_id)
        for query_id in instance.query_ids
        for model_id in instance.model_ids
        for tier_id in instance.gpu_tier_ids
    }
    errors: list[str] = []

    for label, mapping, expected_keys in (
        ("routing_fractions", state.routing_fractions, triple_keys),
        ("admission_indicators", state.admission_indicators, triple_keys),
        ("unmet_fractions", state.unmet_fractions, query_ids),
    ):
        unknown_keys = set(mapping) - expected_keys
        if unknown_keys:
            errors.append(
                f"{label} has unknown semantic IDs: "
                f"{sorted(unknown_keys, key=repr)!r}"
            )

    for key, value in state.routing_fractions.items():
        _append_real_domain_error(
            value, f"routing_fractions[{key!r}]", errors, maximum=1.0
        )
    for key, value in state.admission_indicators.items():
        _append_binary_domain_error(
            value, f"admission_indicators[{key!r}]", errors
        )
    _append_deployment_indicator_errors(
        instance, state.deployment_indicators, errors
    )

    max_unmet_by_query = {
        query.id: query.max_unmet_fraction for query in instance.queries
    }
    for query_id, value in state.unmet_fractions.items():
        # Unknown IDs were reported above; avoiding lookup here keeps the error
        # aggregated rather than replacing it with an incidental KeyError.
        if query_id in max_unmet_by_query:
            _append_real_domain_error(
                value,
                f"unmet_fractions[{query_id!r}]",
                errors,
                maximum=max_unmet_by_query[query_id],
            )

    if errors:
        raise DecisionDomainError(
            "Invalid routing decision state:\n- " + "\n- ".join(errors)
        )


def compute_demand_balance_residuals(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> tuple[DemandBalanceDiagnostic, ...]:
    """Compute ``sum_(j,k) x[i,j,k] + u[i] - 1`` per query (v2 Eq. 8b).

    Compare routed plus unmet demand with one; return diagnostics in query order.
    """

    validate_routing_decision_domains(instance, state)
    return _compute_demand_balance_residuals(instance, state)


def compute_routing_consistency_residuals(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> tuple[RoutingConsistencyDiagnostic, ...]:
    """Compute slacks for ``0 <= x[i,j,k] <= z[i,j,k] <= q[j,k]`` (v2 Eq. 8k).

    Return x, z-x, and q-z for every route in query/model/tier instance order.
    Non-negative slacks satisfy the respective inequalities.
    """

    validate_routing_decision_domains(instance, state)
    return _compute_routing_consistency_residuals(instance, state)


def _compute_demand_balance_residuals(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> tuple[DemandBalanceDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""
    residuals: list[DemandBalanceDiagnostic] = []
    for query_id in instance.query_ids:
        routing_values = tuple(
            state.routing_fractions.get((query_id, model_id, tier_id), 0.0)
            for model_id in instance.model_ids
            for tier_id in instance.gpu_tier_ids
        )
        routed_fraction = fsum(routing_values)
        unmet_fraction = state.unmet_fractions.get(query_id, 0.0)
        # A single stable sum avoids manufacturing a residual for hand cases
        # such as 0.6 + 0.3 + 0.1 solely through addition order.
        lhs = fsum((*routing_values, unmet_fraction))
        residuals.append(
            DemandBalanceDiagnostic(
                query_id=query_id,
                routed_fraction=routed_fraction,
                unmet_fraction=unmet_fraction,
                lhs=lhs,
            )
        )
    return tuple(residuals)


def _compute_routing_consistency_residuals(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> tuple[RoutingConsistencyDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""
    residuals: list[RoutingConsistencyDiagnostic] = []
    for query_id in instance.query_ids:
        for model_id in instance.model_ids:
            for tier_id in instance.gpu_tier_ids:
                triple_key = (query_id, model_id, tier_id)
                pair_key = (model_id, tier_id)
                routing_fraction = state.routing_fractions.get(triple_key, 0.0)
                admission_indicator = state.admission_indicators.get(
                    triple_key, 0.0
                )
                deployment_indicator = state.deployment_indicators.get(
                    pair_key, 0.0
                )
                residuals.append(
                    RoutingConsistencyDiagnostic(
                        query_id=query_id,
                        model_id=model_id,
                        gpu_tier_id=tier_id,
                        routing_fraction=routing_fraction,
                        admission_indicator=admission_indicator,
                        deployment_indicator=deployment_indicator,
                    )
                )
    return tuple(residuals)
