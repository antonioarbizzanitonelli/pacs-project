"""Semantic validation for solver-independent problem instances."""

from collections.abc import Iterable, Mapping
from math import isfinite
import re

from .types import PairKey, ProblemInstance, TripleKey
from .units import SECONDS_PER_HOUR, UnitConvention


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class InstanceValidationError(ValueError):
    """Report one or more structural or numerical instance-data violations."""


def _validate_identifier(identifier: object, label: str, errors: list[str]) -> None:
    """Append an error when an ID is not a stable ASCII interchange key."""

    if not isinstance(identifier, str) or not _IDENTIFIER_PATTERN.fullmatch(identifier):
        errors.append(
            f"{label} must match {_IDENTIFIER_PATTERN.pattern!r}; got {identifier!r}"
        )


def _validate_number(
    value: object,
    label: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    """Validate a finite real value and optional inclusive/exclusive bounds."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a real number; got {value!r}")
        return
    if not isfinite(value):
        errors.append(f"{label} must be finite; got {value!r}")
        return
    if minimum is not None:
        violates_minimum = value < minimum if minimum_inclusive else value <= minimum
        if violates_minimum:
            comparator = ">=" if minimum_inclusive else ">"
            errors.append(f"{label} must be {comparator} {minimum}; got {value!r}")
    if maximum is not None and value > maximum:
        errors.append(f"{label} must be <= {maximum}; got {value!r}")


def _validate_unique_ids(
    identifiers: Iterable[str], label: str, errors: list[str]
) -> None:
    """Validate every ID and report duplicates without relying on set order."""

    seen: set[str] = set()
    for identifier in identifiers:
        _validate_identifier(identifier, label, errors)
        if identifier in seen:
            errors.append(f"duplicate {label}: {identifier!r}")
        seen.add(identifier)


def _validate_parallelism_degrees(
    degrees: tuple[int, ...], label: str, errors: list[str]
) -> None:
    """Require a non-empty set-like sequence of positive integer degrees."""

    if not degrees:
        errors.append(f"{label} must not be empty")
        return
    seen: set[int] = set()
    for degree in degrees:
        if isinstance(degree, bool) or not isinstance(degree, int) or degree <= 0:
            errors.append(f"{label} entries must be positive integers; got {degree!r}")
        if degree in seen:
            errors.append(f"{label} contains duplicate degree {degree!r}")
        seen.add(degree)


def _validate_coefficient_mapping(
    coefficients: Mapping[PairKey | TripleKey, float],
    expected_keys: set[PairKey] | set[TripleKey],
    label: str,
    errors: list[str],
    *,
    maximum: float | None = None,
) -> None:
    """Check exact index coverage and finite non-negative coefficients."""

    actual_keys = set(coefficients)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        errors.append(f"{label} is missing keys: {sorted(missing, key=repr)!r}")
    if extra:
        errors.append(f"{label} has unexpected keys: {sorted(extra, key=repr)!r}")
    for key, value in coefficients.items():
        _validate_number(
            value,
            f"{label}[{key!r}]",
            errors,
            minimum=0.0,
            maximum=maximum,
        )


def validate_problem_instance(instance: ProblemInstance) -> None:
    """Validate a complete solver-independent paper-v2 problem instance.

    Validation covers identifiers, set cardinality, physical/numerical ranges,
    TP/PP sets, and exact `(i,j)`/`(i,j,k)` coefficient coverage. It deliberately
    does not test feasibility of Paper v2, Eq. (8), because no decision variables
    or optimizer belong to this domain-foundation milestone.

    Args:
        instance: The immutable problem instance to validate.

    Raises:
        InstanceValidationError: If one or more semantic checks fail. The
            exception contains every detected issue to support data auditing.
    """

    errors: list[str] = []
    _validate_identifier(instance.id, "instance ID", errors)

    if not instance.queries:
        errors.append("queries must not be empty")
    if not instance.models:
        errors.append("models must not be empty")
    if not instance.gpu_tiers:
        errors.append("gpu_tiers must not be empty")

    _validate_unique_ids(instance.query_ids, "query ID", errors)
    _validate_unique_ids(instance.model_ids, "model ID", errors)
    _validate_unique_ids(instance.gpu_tier_ids, "GPU-tier ID", errors)

    for query in instance.queries:
        prefix = f"query {query.id!r}"
        for field_name, value in (
            ("arrival_rate_per_hour", query.arrival_rate_per_hour),
            ("input_tokens", query.input_tokens),
            ("output_tokens", query.output_tokens),
            ("storage_gb_per_token", query.storage_gb_per_token),
            (
                "delay_penalty_usd_per_millisecond_query",
                query.delay_penalty_usd_per_millisecond_query,
            ),
            ("unmet_penalty_usd_per_hour", query.unmet_penalty_usd_per_hour),
        ):
            _validate_number(value, f"{prefix}.{field_name}", errors, minimum=0.0)
        _validate_number(
            query.delay_slo_seconds,
            f"{prefix}.delay_slo_seconds",
            errors,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _validate_number(
            query.error_slo,
            f"{prefix}.error_slo",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
        _validate_number(
            query.max_unmet_fraction,
            f"{prefix}.max_unmet_fraction",
            errors,
            minimum=0.0,
            maximum=1.0,
        )

    for model in instance.models:
        prefix = f"model {model.id!r}"
        _validate_number(
            model.weight_gb,
            f"{prefix}.weight_gb",
            errors,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _validate_number(
            model.kv_cache_gb_per_token,
            f"{prefix}.kv_cache_gb_per_token",
            errors,
            minimum=0.0,
        )

    for tier in instance.gpu_tiers:
        prefix = f"GPU tier {tier.id!r}"
        for field_name, value in (
            ("memory_capacity_gb", tier.memory_capacity_gb),
            ("compute_throughput_tflops", tier.compute_throughput_tflops),
            ("memory_bandwidth_gb_per_second", tier.memory_bandwidth_gb_per_second),
        ):
            _validate_number(
                value,
                f"{prefix}.{field_name}",
                errors,
                minimum=0.0,
                minimum_inclusive=False,
            )
        _validate_number(
            tier.hourly_rental_usd,
            f"{prefix}.hourly_rental_usd",
            errors,
            minimum=0.0,
        )
        _validate_number(
            tier.quantization_latency_scale,
            f"{prefix}.quantization_latency_scale",
            errors,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        _validate_number(
            tier.quantization_error_multiplier,
            f"{prefix}.quantization_error_multiplier",
            errors,
            minimum=1.0,
        )
        _validate_parallelism_degrees(
            tier.tensor_parallel_degrees,
            f"{prefix}.tensor_parallel_degrees",
            errors,
        )
        _validate_parallelism_degrees(
            tier.pipeline_parallel_depths,
            f"{prefix}.pipeline_parallel_depths",
            errors,
        )

    for field_name, value, minimum, inclusive, maximum in (
        ("storage_price_usd_per_gb_hour", instance.storage_price_usd_per_gb_hour, 0.0, True, None),
        ("cost_budget_usd", instance.cost_budget_usd, 0.0, True, None),
        ("storage_capacity_gb", instance.storage_capacity_gb, 0.0, False, None),
        ("horizon_hours", instance.horizon_hours, 0.0, False, None),
        ("compute_utilization", instance.compute_utilization, 0.0, False, 1.0),
        ("seconds_per_hour", instance.seconds_per_hour, 0.0, False, None),
        ("data_residency_hours", instance.data_residency_hours, 0.0, True, None),
    ):
        _validate_number(
            value,
            f"instance.{field_name}",
            errors,
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=inclusive,
        )

    if not isinstance(instance.unit_convention, UnitConvention):
        errors.append("instance.unit_convention must be a UnitConvention")
    if instance.uses_coherent_units and instance.seconds_per_hour != SECONDS_PER_HOUR:
        errors.append(
            f"coherent units require instance.seconds_per_hour = {SECONDS_PER_HOUR}"
        )

    pair_keys: set[PairKey] = {
        (query_id, model_id)
        for query_id in instance.query_ids
        for model_id in instance.model_ids
    }
    triple_keys: set[TripleKey] = {
        (query_id, model_id, tier_id)
        for query_id in instance.query_ids
        for model_id in instance.model_ids
        for tier_id in instance.gpu_tier_ids
    }
    _validate_coefficient_mapping(
        instance.base_error_rate,
        pair_keys,
        "base_error_rate",
        errors,
        maximum=1.0,
    )
    for label, coefficients in (
        ("compute_delay_seconds_per_token", instance.compute_delay_seconds_per_token),
        (
            "communication_delay_seconds_per_token",
            instance.communication_delay_seconds_per_token,
        ),
        ("compute_gflop_per_token", instance.compute_gflop_per_token),
        ("kv_cache_residency_seconds", instance.kv_cache_residency_seconds),
    ):
        _validate_coefficient_mapping(coefficients, triple_keys, label, errors)

    if errors:
        message = "Invalid problem instance:\n- " + "\n- ".join(errors)
        raise InstanceValidationError(message)
