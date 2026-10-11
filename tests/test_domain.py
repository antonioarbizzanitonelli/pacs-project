"""Unit tests for typed domain data and semantic instance validation."""

from dataclasses import replace
from math import inf, nan
from types import MappingProxyType
import unittest

from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import (
    InstanceValidationError,
    ParallelismConfig,
    validate_problem_instance,
)
from llm_resource_allocation.domain.units import kilobytes_to_gigabytes


class DomainValidationTests(unittest.TestCase):
    """Exercise structural, dimensional, range, and finiteness validation."""

    def test_valid_instance_and_derived_quantities(self) -> None:
        """Validate the minimal fixture and its non-redundant derived fields."""

        instance = make_minimal_tiny_instance()
        validate_problem_instance(instance)
        self.assertEqual((1, 1, 1), instance.dimensions)
        self.assertEqual(3.0, instance.queries[0].total_tokens)
        self.assertEqual(
            (ParallelismConfig(1, 1), ParallelismConfig(2, 1)),
            instance.gpu_tiers[0].parallelism_configs,
        )

    def test_invalid_and_duplicate_ids_are_rejected(self) -> None:
        """Reject non-interchangeable IDs and duplicates within an index set."""

        instance = make_minimal_tiny_instance()
        invalid_query = replace(instance.queries[0], id="bad id")
        duplicate_queries = (invalid_query, invalid_query)
        invalid = replace(instance, queries=duplicate_queries)
        with self.assertRaises(InstanceValidationError) as context:
            validate_problem_instance(invalid)
        self.assertIn("query ID", str(context.exception))
        self.assertIn("duplicate", str(context.exception))

    def test_empty_dimension_is_rejected(self) -> None:
        """Require non-empty query, model, and tier dimensions."""

        invalid = replace(make_minimal_tiny_instance(), models=())
        with self.assertRaisesRegex(InstanceValidationError, "models must not be empty"):
            validate_problem_instance(invalid)

    def test_missing_tensor_coefficient_is_rejected(self) -> None:
        """Require every `(i,j,k)` entry in each triple coefficient mapping."""

        instance = make_minimal_tiny_instance()
        incomplete = dict(instance.compute_delay_seconds_per_token)
        incomplete.pop(next(iter(incomplete)))
        invalid = replace(instance, compute_delay_seconds_per_token=incomplete)
        with self.assertRaisesRegex(InstanceValidationError, "is missing keys"):
            validate_problem_instance(invalid)

    def test_invalid_parallelism_sets_are_rejected(self) -> None:
        """Reject empty, duplicate, non-integer, and non-positive TP/PP sets."""

        instance = make_minimal_tiny_instance()
        invalid_tier = replace(
            instance.gpu_tiers[0],
            tensor_parallel_degrees=(),
            pipeline_parallel_depths=(1, 1, 0),
        )
        invalid = replace(instance, gpu_tiers=(invalid_tier,))
        with self.assertRaises(InstanceValidationError) as context:
            validate_problem_instance(invalid)
        message = str(context.exception)
        self.assertIn("tensor_parallel_degrees must not be empty", message)
        self.assertIn("duplicate degree", message)
        self.assertIn("positive integers", message)

    def test_invalid_slo_cap_and_horizon_are_rejected(self) -> None:
        """Enforce probability-like SLO/cap bounds and a positive horizon."""

        instance = make_minimal_tiny_instance()
        invalid_query = replace(
            instance.queries[0], error_slo=1.01, max_unmet_fraction=-0.01
        )
        invalid = replace(instance, queries=(invalid_query,), horizon_hours=0.0)
        with self.assertRaises(InstanceValidationError) as context:
            validate_problem_instance(invalid)
        message = str(context.exception)
        self.assertIn("error_slo must be <= 1.0", message)
        self.assertIn("max_unmet_fraction must be >= 0.0", message)
        self.assertIn("horizon_hours must be > 0.0", message)

    def test_non_finite_values_are_rejected(self) -> None:
        """Reject NaN and infinity in entity fields and coefficient tensors."""

        instance = make_minimal_tiny_instance()
        invalid_query = replace(instance.queries[0], arrival_rate_per_hour=nan)
        invalid_coefficients = dict(instance.compute_gflop_per_token)
        invalid_coefficients[next(iter(invalid_coefficients))] = inf
        invalid = replace(
            instance,
            queries=(invalid_query,),
            compute_gflop_per_token=invalid_coefficients,
        )
        with self.assertRaises(InstanceValidationError) as context:
            validate_problem_instance(invalid)
        self.assertGreaterEqual(str(context.exception).count("must be finite"), 2)

    def test_negative_scientific_quantities_are_rejected(self) -> None:
        """Reject negative rates, footprints, prices, and coefficients."""

        instance = make_minimal_tiny_instance()
        invalid_query = replace(instance.queries[0], arrival_rate_per_hour=-1.0)
        invalid_coefficients = dict(instance.communication_delay_seconds_per_token)
        invalid_coefficients[next(iter(invalid_coefficients))] = -0.01
        invalid = replace(
            instance,
            queries=(invalid_query,),
            communication_delay_seconds_per_token=invalid_coefficients,
            storage_price_usd_per_gb_hour=-0.5,
        )
        with self.assertRaises(InstanceValidationError) as context:
            validate_problem_instance(invalid)
        message = str(context.exception)
        self.assertIn("arrival_rate_per_hour must be >= 0.0", message)
        self.assertIn("storage_price_usd_per_gb_hour must be >= 0.0", message)
        self.assertIn("communication_delay_seconds_per_token", message)

    def test_coefficient_mappings_are_copied_and_read_only(self) -> None:
        """Prevent caller mutation from changing an immutable instance."""

        instance = make_minimal_tiny_instance()
        original = instance.base_error_rate
        mutable_source = dict(original)
        copied = replace(instance, base_error_rate=mutable_source)
        key = next(iter(mutable_source))
        mutable_source[key] = 0.99

        self.assertIsInstance(copied.base_error_rate, MappingProxyType)
        self.assertEqual(original[key], copied.base_error_rate[key])
        with self.assertRaises(TypeError):
            copied.base_error_rate[key] = 0.50

    def test_canonical_decimal_kilobyte_conversion(self) -> None:
        """Make the selected KB-to-GB input conversion explicit and testable."""

        self.assertEqual(1.0, kilobytes_to_gigabytes(1_000_000.0))
        self.assertEqual(1e-6, kilobytes_to_gigabytes(1.0))
        for invalid in (-1.0, inf, nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    kilobytes_to_gigabytes(invalid)


if __name__ == "__main__":
    unittest.main()
