"""Weighted-error diagnostics for ``sum_(j,k) e_bar[i,j,k]*x[i,j,k] <= epsilon_i``."""

from dataclasses import dataclass
from math import fsum, isfinite

from llm_resource_allocation.domain import ProblemInstance

from .error import compute_effective_error
from .routing import RoutingDecisionState, validate_routing_decision_domains


@dataclass(frozen=True, slots=True)
class ErrorSloDiagnostic:
    """Record weighted error and its SLO for one query (v2 Eq. 8j).

    ``weighted_error = sum_(j,k) e_bar[i,j,k]*x[i,j,k]``;
    ``error_limit = epsilon_i``. ``slack = error_limit - weighted_error``.
    All values are dimensionless.
    """

    query_id: str
    weighted_error: float
    error_limit: float

    @property
    def slack(self) -> float:
        """Return ``epsilon_i - sum_(j,k) e_bar[i,j,k]*x[i,j,k]`` (v2 Eq. 8j)."""

        return self.error_limit - self.weighted_error

    @property
    def violation(self) -> float:
        """Return ``max(weighted_error - error_limit, 0)``."""

        return max(-self.slack, 0.0)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``slack >= -tolerance`` for Eq. (8j).

        The tolerance is dimensionless.
        """

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("tolerance must be a finite non-negative real number")
        return self.slack >= -tolerance


def compute_error_slo_residuals(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> tuple[ErrorSloDiagnostic, ...]:
    """Compute weighted-error slacks per query (v2 Eq. 8j).

    ``slack = epsilon_i - sum_(j,k) e_bar[i,j,k]*x[i,j,k]``, with
    ``e_bar = mu[k]*e_base[i,j]`` from Eq. (1). Return diagnostics in query order;
    the weighted sum uses routing fractions directly, without normalization.
    """

    validate_routing_decision_domains(instance, state)
    return _compute_error_slo_residuals(instance, state)


def _compute_error_slo_residuals(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> tuple[ErrorSloDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""
    tier_by_id = {tier.id: tier for tier in instance.gpu_tiers}
    residuals: list[ErrorSloDiagnostic] = []

    for query in instance.queries:
        weighted_terms = (
            compute_effective_error(
                instance.base_error_rate[(query.id, model_id)],
                tier_by_id[tier_id].quantization_error_multiplier,
            )
            * state.routing_fractions.get((query.id, model_id, tier_id), 0.0)
            for model_id in instance.model_ids
            for tier_id in instance.gpu_tier_ids
        )
        # Stable summation preserves the printed full-lattice sum while
        # minimizing addition-order artifacts in diagnostic residuals.
        weighted_error = fsum(weighted_terms)
        residuals.append(
            ErrorSloDiagnostic(
                query_id=query.id,
                weighted_error=weighted_error,
                error_limit=query.error_slo,
            )
        )

    return tuple(residuals)
