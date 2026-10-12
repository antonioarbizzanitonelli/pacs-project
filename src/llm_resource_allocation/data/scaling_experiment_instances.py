"""Controlled synthetic I/J/K scaling, with explicit experiment assumptions.

These inputs extend the heterogeneous hand-checkable fixture; they are neither
Azure-calibrated coefficients nor recovered paper experiment inputs. The nested
interpolation menu preserves every previously available model/tier alternative.
"""

from dataclasses import replace
from fractions import Fraction
from math import isfinite

from llm_resource_allocation.domain.types import FoundationModel, GpuTier
from llm_resource_allocation.domain.units import UnitConvention

from .synthetic_benchmark_instances import SyntheticBenchmarkCase
from .tiny_instances import make_heterogeneous_tiny_instance


def _nested_coordinates(count: int) -> tuple[Fraction, ...]:
    """Return endpoints followed by dyadic interior points in increasing depth."""

    coordinates = [Fraction(0), Fraction(1)]
    denominator = 2
    while len(coordinates) < count:
        for numerator in range(1, denominator, 2):
            coordinates.append(Fraction(numerator, denominator))
            if len(coordinates) == count:
                break
        denominator *= 2
    return tuple(coordinates)


def _interpolate(left: float, right: float, coordinate: Fraction) -> float:
    """Preserve source endpoint arithmetic, interpolate only added alternatives."""

    if coordinate == 0:
        return left
    if coordinate == 1:
        return right
    return left + float(coordinate) * (right - left)


def _coordinate_id(prefix: str, coordinate: Fraction) -> str:
    return f"{prefix}_{coordinate.numerator}_{coordinate.denominator}"


def make_scaling_experiment_case(
    query_count: int,
    model_count: int,
    gpu_tier_count: int,
    *,
    load_multiplier: float = 1.0,
    delay_slo_multiplier: float = 1.0,
    budget_multiplier: float = 1.0,
    data_residency_hours: float = 1.0,
) -> SyntheticBenchmarkCase:
    """Build one deterministic, independently domain-validated scaling case.

    All cardinalities must be integers at least two. Query types alternate the
    original short/long fixture, preserving its first two IDs. Budget and storage
    capacity scale by ``I/2``; the explicit budget multiplier acts afterward.
    Models and tiers retain both source endpoints, then add nested coordinates
    ``1/2, 1/4, 3/4, 1/8, ...``. Numeric attributes interpolate linearly; every
    triple coefficient interpolates first over model, then over tier. Original
    TP/PP sets remain intact. New tiers use their union, ``TP=PP=(1,2)``.

    Load, delay SLO and budget multipliers are separate assumed interventions.
    Coefficients, caps, storage prices, utilization and the two-hour horizon do
    not depend on any algorithm result. Setting two cardinalities constant gives
    query-only, model-only, tier-only or combined options-only controls. Units
    are coherent, with explicitly configured data residency (one hour by default).
    """

    for name, value in (
        ("query_count", query_count),
        ("model_count", model_count),
        ("gpu_tier_count", gpu_tier_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer at least two")
    for name, value in (
        ("load_multiplier", load_multiplier),
        ("delay_slo_multiplier", delay_slo_multiplier),
        ("budget_multiplier", budget_multiplier),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")

    base = replace(make_heterogeneous_tiny_instance(),
                   unit_convention=UnitConvention.COHERENT,
                   data_residency_hours=data_residency_hours)
    model_coordinates = _nested_coordinates(model_count)
    tier_coordinates = _nested_coordinates(gpu_tier_count)
    models = tuple(
        base.models[index]
        if index < 2
        else FoundationModel(
            id=_coordinate_id("m_interpolated", coordinate),
            weight_gb=_interpolate(
                base.models[0].weight_gb, base.models[1].weight_gb, coordinate
            ),
            kv_cache_gb_per_token=_interpolate(
                base.models[0].kv_cache_gb_per_token,
                base.models[1].kv_cache_gb_per_token,
                coordinate,
            ),
        )
        for index, coordinate in enumerate(model_coordinates)
    )
    tier_fields = (
        "memory_capacity_gb",
        "compute_throughput_tflops",
        "hourly_rental_usd",
        "memory_bandwidth_gb_per_second",
        "quantization_latency_scale",
        "quantization_error_multiplier",
    )
    tiers = tuple(
        base.gpu_tiers[index]
        if index < 2
        else GpuTier(
            id=_coordinate_id("gpu_interpolated", coordinate),
            **{
                name: _interpolate(
                    getattr(base.gpu_tiers[0], name),
                    getattr(base.gpu_tiers[1], name),
                    coordinate,
                )
                for name in tier_fields
            },
            tensor_parallel_degrees=(1, 2),
            pipeline_parallel_depths=(1, 2),
        )
        for index, coordinate in enumerate(tier_coordinates)
    )
    source_queries = tuple(base.queries[index % 2] for index in range(query_count))
    queries = tuple(
        replace(
            source,
            id=source.id if index < 2 else f"{source.id}_{index + 1:04d}",
            arrival_rate_per_hour=source.arrival_rate_per_hour * load_multiplier,
            delay_slo_seconds=source.delay_slo_seconds * delay_slo_multiplier,
        )
        for index, source in enumerate(source_queries)
    )
    triple_coefficients = {}
    for field_name in (
        "compute_delay_seconds_per_token",
        "communication_delay_seconds_per_token",
        "compute_gflop_per_token",
        "kv_cache_residency_seconds",
    ):
        source_table = getattr(base, field_name)
        triple_coefficients[field_name] = {
            (query.id, model.id, tier.id): _interpolate(
                _interpolate(
                    source_table[(source.id, base.models[0].id, base.gpu_tiers[0].id)],
                    source_table[(source.id, base.models[1].id, base.gpu_tiers[0].id)],
                    model_coordinate,
                ),
                _interpolate(
                    source_table[(source.id, base.models[0].id, base.gpu_tiers[1].id)],
                    source_table[(source.id, base.models[1].id, base.gpu_tiers[1].id)],
                    model_coordinate,
                ),
                tier_coordinate,
            )
            for query, source in zip(queries, source_queries, strict=True)
            for model, model_coordinate in zip(models, model_coordinates, strict=True)
            for tier, tier_coordinate in zip(tiers, tier_coordinates, strict=True)
        }
    dimensions_id = f"{query_count}x{model_count}x{gpu_tier_count}"
    instance = replace(
        base,
        id=f"synthetic_multidimensional_scaling_{dimensions_id}",
        queries=queries,
        models=models,
        gpu_tiers=tiers,
        base_error_rate={
            (query.id, model.id): _interpolate(
                base.base_error_rate[(source.id, base.models[0].id)],
                base.base_error_rate[(source.id, base.models[1].id)],
                coordinate,
            )
            for query, source in zip(queries, source_queries, strict=True)
            for model, coordinate in zip(models, model_coordinates, strict=True)
        },
        **triple_coefficients,
        cost_budget_usd=base.cost_budget_usd * (query_count / 2) * budget_multiplier,
        storage_capacity_gb=base.storage_capacity_gb * (query_count / 2),
    )
    return SyntheticBenchmarkCase(
        case_id=f"scaling_{dimensions_id}",
        family="multidimensional_scaling",
        instance=instance,
        provenance={
            "classification": "ASSUMED",
            "description": (
                "Synthetic multidimensional expansion of heterogeneous tiny inputs; "
                "not Azure-calibrated or recovered paper coefficients."
            ),
            "source_fixture": "make_heterogeneous_tiny_instance",
            "query_count": query_count,
            "model_count": model_count,
            "gpu_tier_count": gpu_tier_count,
            "query_cloning_rule": "alternating q_short, q_long in input order",
            "coordinate_rule": "0,1; dyadic odd numerators by increasing denominator",
            "model_coordinates": ",".join(str(value) for value in model_coordinates),
            "gpu_tier_coordinates": ",".join(str(value) for value in tier_coordinates),
            "attribute_interpolation_rule": "linear from original two endpoints",
            "coefficient_interpolation_rule": "bilinear: model first, then GPU tier",
            "new_tier_parallelism_rule": "union of source sets: TP=(1,2), PP=(1,2)",
            "capacity_scaling_rule": "source budget and storage capacity * I/2",
            "capacity_multiplier": query_count / 2,
            "load_multiplier": load_multiplier,
            "delay_slo_multiplier": delay_slo_multiplier,
            "budget_multiplier": budget_multiplier,
            "cost_budget_usd": instance.cost_budget_usd,
            "storage_capacity_gb": instance.storage_capacity_gb,
            "storage_price_usd_per_gb_hour": instance.storage_price_usd_per_gb_hour,
            "horizon_hours": instance.horizon_hours,
            "compute_utilization": instance.compute_utilization,
            "seconds_per_hour": instance.seconds_per_hour,
            "unit_convention": instance.unit_convention.value,
            "data_residency_hours": instance.data_residency_hours,
        },
    )
