"""Deterministic synthetic benchmark inputs, separate from paper experiments.

The scaling family repeats the two existing heterogeneous query types. It
changes only their identifiers and the explicitly assumed budget/storage
scaling rule; no coefficient is tuned to favor a method or force feasibility.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from llm_resource_allocation.domain.types import ProblemInstance
from llm_resource_allocation.domain.units import UnitConvention
from llm_resource_allocation.domain.validation import validate_problem_instance

from .tiny_instances import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
    make_parallelism_tradeoff_tiny_instance,
)


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkCase:
    """Pair an immutable validated instance with its assumed input provenance."""

    case_id: str
    family: str
    instance: ProblemInstance
    provenance: Mapping[str, str | int | float]

    def __post_init__(self) -> None:
        for name, value in (("case_id", self.case_id), ("family", self.family)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.instance, ProblemInstance):
            raise TypeError("instance must be a ProblemInstance")
        provenance = dict(self.provenance)
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (str, int, float))
            for key, value in provenance.items()
        ):
            raise TypeError("provenance must contain string keys and scalar values")
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        validate_problem_instance(self.instance)
        if not self.instance.uses_coherent_units:
            raise ValueError("public benchmark cases require coherent units")


def _provenance(
    instance: ProblemInstance,
    *,
    source_fixture: str,
    description: str,
    capacity_multiplier: float = 1.0,
) -> dict[str, str | int | float]:
    return {
        "classification": "ASSUMED",
        "description": description,
        "source_fixture": source_fixture,
        "query_count": len(instance.queries),
        "model_count": len(instance.models),
        "gpu_tier_count": len(instance.gpu_tiers),
        "capacity_multiplier": capacity_multiplier,
        "cost_budget_usd": instance.cost_budget_usd,
        "storage_capacity_gb": instance.storage_capacity_gb,
        "storage_price_usd_per_gb_hour": instance.storage_price_usd_per_gb_hour,
        "horizon_hours": instance.horizon_hours,
        "compute_utilization": instance.compute_utilization,
        "seconds_per_hour": instance.seconds_per_hour,
        "unit_convention": instance.unit_convention.value,
        "data_residency_hours": instance.data_residency_hours,
    }


def _make_query_scaling_instance(
    base: ProblemInstance, query_count: int
) -> ProblemInstance:
    source_queries = tuple(
        base.queries[index % 2] for index in range(query_count)
    )
    queries = tuple(
        replace(query, id=f"{query.id}_{index + 1:04d}")
        for index, query in enumerate(source_queries)
    )
    triple_coefficients = {
        field_name: {
            (query.id, model.id, tier.id): getattr(base, field_name)[
                (source.id, model.id, tier.id)
            ]
            for query, source in zip(queries, source_queries, strict=True)
            for model in base.models
            for tier in base.gpu_tiers
        }
        for field_name in (
            "compute_delay_seconds_per_token",
            "communication_delay_seconds_per_token",
            "compute_gflop_per_token",
            "kv_cache_residency_seconds",
        )
    }
    return replace(
        base,
        id=f"synthetic_query_scaling_{query_count}",
        queries=queries,
        base_error_rate={
            (query.id, model.id): base.base_error_rate[(source.id, model.id)]
            for query, source in zip(queries, source_queries, strict=True)
            for model in base.models
        },
        **triple_coefficients,
        cost_budget_usd=base.cost_budget_usd * (query_count / 2),
        storage_capacity_gb=base.storage_capacity_gb * (query_count / 2),
    )


def make_synthetic_benchmark_cases(
    query_counts: tuple[int, ...] = (2, 4, 8, 16),
    *,
    data_residency_hours: float = 1.0,
) -> tuple[SyntheticBenchmarkCase, ...]:
    """Return three existing fixtures followed by controlled query-count cases.

    Counts must be distinct positive integers and retain caller order. An empty
    tuple selects only the three existing fixtures. The scaling instances keep
    models, tiers, every query attribute, and all inherited coefficients fixed;
    budget and storage capacity equal the 2-query source values times ``I/2``.
    Every case uses coherent units and the explicitly configured data residency.
    """

    counts = tuple(query_counts)
    seen: set[int] = set()
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("query_counts must contain positive integers")
        if count in seen:
            raise ValueError("query_counts must contain distinct values")
        seen.add(count)

    def coherent(instance: ProblemInstance) -> ProblemInstance:
        return replace(instance, unit_convention=UnitConvention.COHERENT,
                       data_residency_hours=data_residency_hours)

    base = coherent(make_heterogeneous_tiny_instance())
    specifications = (
        (
            "minimal",
            coherent(make_minimal_tiny_instance()),
            "make_minimal_tiny_instance",
            "Synthetic 1-query, 1-model, 1-tier hand-checkable fixture.",
        ),
        (
            "parallelism_tradeoff",
            coherent(make_parallelism_tradeoff_tiny_instance()),
            "make_parallelism_tradeoff_tiny_instance",
            "Synthetic minimal fixture with the existing 0.2-second delay SLO.",
        ),
        (
            "heterogeneous",
            base,
            "make_heterogeneous_tiny_instance",
            "Original synthetic 2-query, 2-model, 2-tier heterogeneous fixture.",
        ),
    )
    cases = [
        SyntheticBenchmarkCase(
            case_id,
            "verification",
            instance,
            _provenance(instance, source_fixture=source, description=description),
        )
        for case_id, instance, source, description in specifications
    ]
    for count in counts:
        instance = _make_query_scaling_instance(base, count)
        provenance = _provenance(
            instance,
            source_fixture="make_heterogeneous_tiny_instance",
            description=(
                "Synthetic alternating clones of q_short and q_long; only IDs, "
                "budget, and storage capacity differ from the source."
            ),
            capacity_multiplier=count / 2,
        )
        provenance["query_cloning_rule"] = "alternating q_short, q_long in input order"
        provenance["capacity_scaling_rule"] = "source budget and storage capacity * I/2"
        provenance["source_query_count"] = 2
        cases.append(
            SyntheticBenchmarkCase(
                f"query_scaling_{count}", "query_scaling", instance, provenance
            )
        )
    return tuple(cases)
