"""Storage-capacity diagnostics for Paper v2 Eq. (8h).

Resolve the paper's free query index with one storage constraint per query.
Coherent units multiply the data rate by explicit steady-state residency;
literal units retain the historical GB plus GB/hour arithmetic.
"""

from dataclasses import dataclass
from math import fsum, isfinite

from llm_resource_allocation.domain import ProblemInstance, UnitConvention

from .placement import PlacementDecisionState, validate_placement_decision_domains


@dataclass(frozen=True, slots=True)
class QueryStorageCapacityDiagnostic:
    """Record the per-query storage comparison (v2 Eq. 8h).

    ``model_storage_usage_gb = sum_(j,k) B_j*z[i,j,k]``;
    ``data_storage_rate_gb_per_hour = sum_(j,k) theta_i*r_i*lambda_i*x[i,j,k]``;
    ``storage_limit_gb = C_s``. ``data_residency_hours`` and ``unit_convention``
    retain the selected policy. The coherent data footprint is rate times
    residency; literal mode preserves the printed rate's numerical value.
    """

    query_id: str
    model_storage_usage_gb: float
    data_storage_rate_gb_per_hour: float
    storage_limit_gb: float
    data_residency_hours: float = 1.0
    unit_convention: UnitConvention = UnitConvention.LITERAL_PAPER

    @property
    def data_storage_usage_gb(self) -> float:
        """Return coherent resident data in GB; reject the literal rate."""

        if self.unit_convention is UnitConvention.LITERAL_PAPER:
            raise ValueError("literal data storage is a GB/hour rate, not a GB stock")
        return self.data_storage_rate_gb_per_hour * self.data_residency_hours

    @property
    def storage_usage_gb(self) -> float:
        """Return model plus data storage under the recorded convention."""

        return self.model_storage_usage_gb + self.data_storage_usage_gb

    @property
    def printed_storage_lhs(self) -> float:
        """Return the selected storage LHS; historical attribute retained."""

        if self.unit_convention is UnitConvention.LITERAL_PAPER:
            return self.model_storage_usage_gb + self.data_storage_rate_gb_per_hour
        return self.storage_usage_gb

    @property
    def slack(self) -> float:
        """Return ``C_s - printed_storage_lhs`` (v2 Eq. 8h)."""

        return self.storage_limit_gb - self.printed_storage_lhs

    @property
    def violation(self) -> float:
        """Return ``max(printed_storage_lhs - C_s, 0)``."""

        return max(-self.slack, 0.0)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``slack >= -tolerance`` for the per-query Eq. (8h).

        The tolerance uses GB in coherent mode, or the legacy comparison scale.
        """

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("tolerance must be a finite non-negative real number")
        return self.slack >= -tolerance


def compute_model_storage_usage_by_query(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[str, float]:
    """Return ``i -> sum_(j,k) B_j*z[i,j,k]`` in GB (v2 Eq. 8h).

    ``model.weight_gb`` is B_j; ``state.admission_indicators`` supplies z.
    Each admitted model/tier entry contributes its weight for that query.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_model_storage_usage_by_query(instance, state)


def compute_data_storage_rate_by_query(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[str, float]:
    """Return ``i -> sum_(j,k) theta_i*r_i*lambda_i*x[i,j,k]`` (v2 Eq. 8h).

    ``query.storage_gb_per_token`` is theta_i (converted to GB/token at input);
    ``query.total_tokens`` is r_i; ``query.arrival_rate_per_hour`` is lambda_i;
    ``state.routing_fractions`` supplies x. The result is GB/hour, without a horizon.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_data_storage_rate_by_query(instance, state)


def compute_storage_capacity_residuals(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> tuple[QueryStorageCapacityDiagnostic, ...]:
    """Return storage-capacity diagnostics per query (v2 Eq. 8h).

    Compute ``model_storage_usage_gb = sum_(j,k) B_j*z[i,j,k]`` and
    ``data_storage_rate_gb_per_hour = sum_(j,k) theta_i*r_i*lambda_i*x[i,j,k]``.
    Each diagnostic stores both terms and ``storage_limit_gb = C_s``, deriving
    ``slack = C_s - storage_usage_gb``. Coherent mode includes the configured
    steady-state data residency; literal mode preserves the printed comparison.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_storage_capacity_residuals(instance, state)


def _compute_model_storage_usage_by_query(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[str, float]:
    """Calculate from instance and decision domains validated by the caller."""
    weight_by_model = {model.id: model.weight_gb for model in instance.models}
    return {
        query_id: fsum(
            weight_by_model[model_id]
            * state.admission_indicators.get(
                (query_id, model_id, tier_id), 0.0
            )
            for model_id in instance.model_ids
            for tier_id in instance.gpu_tier_ids
        )
        for query_id in instance.query_ids
    }


def _compute_data_storage_rate_by_query(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[str, float]:
    """Calculate from instance and decision domains validated by the caller."""
    return {
        query.id: fsum(
            query.storage_gb_per_token
            * query.total_tokens
            * query.arrival_rate_per_hour
            * state.routing_fractions.get(
                (query.id, model_id, tier_id), 0.0
            )
            for model_id in instance.model_ids
            for tier_id in instance.gpu_tier_ids
        )
        for query in instance.queries
    }


def _compute_storage_capacity_residuals(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> tuple[QueryStorageCapacityDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""

    model_usage = _compute_model_storage_usage_by_query(instance, state)
    data_rate = _compute_data_storage_rate_by_query(instance, state)
    return tuple(
        QueryStorageCapacityDiagnostic(
            query_id=query_id,
            model_storage_usage_gb=model_usage[query_id],
            data_storage_rate_gb_per_hour=data_rate[query_id],
            storage_limit_gb=instance.storage_capacity_gb,
            data_residency_hours=instance.data_storage_duration_hours,
            unit_convention=instance.unit_convention,
        )
        for query_id in instance.query_ids
    )
