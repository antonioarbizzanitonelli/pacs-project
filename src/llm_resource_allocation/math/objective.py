"""Objective terms C1-C5 from Paper v2 Eq. (8a) and v1 Eq. (5a).

Primitives consume already validated values; omitted terms contribute zero.
``DeltaT = instance.horizon_hours``, ``p_s = instance.storage_price_usd_per_gb_hour``,
``B_j = model.weight_gb``, ``theta_i = query.storage_gb_per_token``,
``r_i = query.total_tokens``, and ``lambda_i = query.arrival_rate_per_hour``.
The instance selects coherent physical units or historical printed arithmetic.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.domain.units import MILLISECONDS_PER_SECOND


ModelTierKey = tuple[str, str]
QueryModelTierKey = tuple[str, str, str]


class PaperVersion(str, Enum):
    """Select v1 or v2 objective semantics, including the C5 horizon factor."""

    V1 = "v1"
    V2 = "v2"


@dataclass(frozen=True, slots=True)
class ObjectiveState:
    """Store inputs to C1-C5 (v2 Eq. 8a; v1 Eq. 5a).

    ``gpu_counts`` stores y; ``admission_indicators`` stores z;
    ``routing_fractions`` stores x; ``unmet_fractions`` stores u;
    ``processing_delays_seconds`` stores D_proc. Mappings are copied read-only;
    missing terms contribute zero. This record does not validate decisions.
    """

    gpu_counts: Mapping[ModelTierKey, float]
    admission_indicators: Mapping[QueryModelTierKey, float]
    routing_fractions: Mapping[QueryModelTierKey, float]
    unmet_fractions: Mapping[str, float]
    processing_delays_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        """Copy the objective input mappings into read-only views."""

        for field_name in (
            "gpu_counts",
            "admission_indicators",
            "routing_fractions",
            "unmet_fractions",
            "processing_delays_seconds",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    """Record the five objective terms (v2 Eq. 8a; v1 Eq. 5a).

    ``gpu_rental_cost = C1``, ``model_storage_cost = C2``,
    ``data_storage_cost = C3``, ``delay_penalty = C4``,
    ``unmet_demand_penalty = C5``; ``total = C1 + C2 + C3 + C4 + C5``.
    """

    gpu_rental_cost: float
    model_storage_cost: float
    data_storage_cost: float
    delay_penalty: float
    unmet_demand_penalty: float

    @property
    def total(self) -> float:
        """Return ``C1 + C2 + C3 + C4 + C5`` (v2 Eq. 8a; v1 Eq. 5a).

        Coherent inputs produce a USD total. Literal-paper inputs retain the
        historical C3/C4 dimensional mismatches for comparison.
        """

        return (
            self.gpu_rental_cost
            + self.model_storage_cost
            + self.data_storage_cost
            + self.delay_penalty
            + self.unmet_demand_penalty
        )


def compute_gpu_rental_cost(
    instance: ProblemInstance,
    gpu_counts: Mapping[ModelTierKey, float],
) -> float:
    """Return GPU rental ``C1 = DeltaT*sum_(j,k) p_c[k]*y[j,k]`` (v2 Eq. 8a).

    ``tier.hourly_rental_usd`` is ``p_c[k]`` in USD/GPU-hour;
    ``gpu_counts`` supplies ``y[j,k]``. The result is in USD.
    """

    hourly_price_by_tier = {
        tier.id: tier.hourly_rental_usd for tier in instance.gpu_tiers
    }
    hourly_rental = sum(
        hourly_price_by_tier[tier_id] * gpu_count
        for (_, tier_id), gpu_count in gpu_counts.items()
    )
    return instance.horizon_hours * hourly_rental


def compute_model_storage_cost(
    instance: ProblemInstance,
    admission_indicators: Mapping[QueryModelTierKey, float],
) -> float:
    """Return model-storage ``C2 = DeltaT*sum_(i,j,k) p_s*B_j*z[i,j,k]`` (v2 Eq. 8a).

    ``admission_indicators`` supplies z. Each admitted query contributes its
    model-weight term, including queries sharing a model/tier pair; output is USD.
    """

    weight_by_model = {model.id: model.weight_gb for model in instance.models}
    # Retaining the query index is a paper-specific replication choice: two
    # admitted query types on the same pair contribute two printed C2 terms.
    indexed_weight = sum(
        weight_by_model[model_id] * admitted
        for (_, model_id, _), admitted in admission_indicators.items()
    )
    return (
        instance.horizon_hours
        * instance.storage_price_usd_per_gb_hour
        * indexed_weight
    )


def compute_data_storage_cost(
    instance: ProblemInstance,
    routing_fractions: Mapping[QueryModelTierKey, float],
) -> float:
    """Return the horizon cost of resident query data.

    In coherent mode, ``C3 = DeltaT*p_s*data_residency_hours*sum(theta*r*lambda*x)``
    is in USD. The configured residency converts data rate to steady-state stock.
    Literal mode retains the earlier printed expression without that factor.
    """

    query_by_id = {query.id: query for query in instance.queries}
    indexed_storage = sum(
        query_by_id[query_id].storage_gb_per_token
        * query_by_id[query_id].total_tokens
        * query_by_id[query_id].arrival_rate_per_hour
        * routing_fraction
        for (query_id, _, _), routing_fraction in routing_fractions.items()
    )
    cost = (
        instance.horizon_hours
        * instance.storage_price_usd_per_gb_hour
        * indexed_storage
    )
    return cost * instance.data_residency_hours if instance.uses_coherent_units else cost


def compute_delay_penalty(
    instance: ProblemInstance,
    processing_delays_seconds: Mapping[str, float],
) -> float:
    """Return delay penalties under the selected unit convention.

    ``query.delay_penalty_usd_per_millisecond_query`` supplies rho;
    ``processing_delays_seconds`` supplies D_proc in seconds. Coherent mode
    converts seconds to milliseconds and charges all ``lambda_i*DeltaT``
    arriving requests, producing USD. Literal mode retains ``sum(rho*D_proc)``.
    """

    query_by_id = {query.id: query for query in instance.queries}
    if instance.uses_coherent_units:
        return sum(
            query_by_id[query_id].delay_penalty_usd_per_millisecond_query
            * processing_delay
            * MILLISECONDS_PER_SECOND
            * query_by_id[query_id].arrival_rate_per_hour
            * instance.horizon_hours
            for query_id, processing_delay in processing_delays_seconds.items()
        )
    return sum(
        query_by_id[query_id].delay_penalty_usd_per_millisecond_query * processing_delay
        for query_id, processing_delay in processing_delays_seconds.items()
    )


def compute_unmet_demand_penalty(
    instance: ProblemInstance,
    unmet_fractions: Mapping[str, float],
    paper_version: PaperVersion,
) -> float:
    """Return unmet-demand C5 from ``unmet_fractions`` (u) for the selected version.

    With ``phi_i = query.unmet_penalty_usd_per_hour``, v1 Eq. (5a) uses
    ``sum_i phi_i*u[i]``; v2 Eq. (8a) uses ``DeltaT*sum_i phi_i*u[i]``.
    Coherent units require v2; v1 remains available with literal-paper units.
    """

    if instance.uses_coherent_units and paper_version is PaperVersion.V1:
        raise ValueError("coherent units require PaperVersion.V2; v1 uses LITERAL_PAPER")

    penalty_by_query = {
        query.id: query.unmet_penalty_usd_per_hour for query in instance.queries
    }
    base_penalty = sum(
        penalty_by_query[query_id] * unmet_fraction
        for query_id, unmet_fraction in unmet_fractions.items()
    )
    if paper_version is PaperVersion.V1:
        return base_penalty
    if paper_version is PaperVersion.V2:
        return instance.horizon_hours * base_penalty
    raise ValueError("paper_version must be PaperVersion.V1 or PaperVersion.V2")


def compute_objective_breakdown(
    instance: ProblemInstance,
    state: ObjectiveState,
    paper_version: PaperVersion,
) -> ObjectiveBreakdown:
    """Return C1-C5 and their total (v2 Eq. 8a or v1 Eq. 5a).

    ``state`` supplies y, z, x, u, and D_proc; ``paper_version`` selects C5
    scaling. Evaluation consumes already validated decisions.
    Coherent units give costs in USD; literal units retain historical C3/C4
    arithmetic for versioned comparison.
    """

    return ObjectiveBreakdown(
        gpu_rental_cost=compute_gpu_rental_cost(instance, state.gpu_counts),
        model_storage_cost=compute_model_storage_cost(
            instance, state.admission_indicators
        ),
        data_storage_cost=compute_data_storage_cost(
            instance, state.routing_fractions
        ),
        delay_penalty=compute_delay_penalty(
            instance, state.processing_delays_seconds
        ),
        unmet_demand_penalty=compute_unmet_demand_penalty(
            instance, state.unmet_fractions, paper_version
        ),
    )
