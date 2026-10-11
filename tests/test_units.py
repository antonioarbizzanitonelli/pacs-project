"""Physical conversion ratios and the canonical decimal storage boundary."""

import unittest
from dataclasses import fields, replace

from llm_resource_allocation.data.tiny_instances import (
    make_coherent_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.domain import (
    InstanceValidationError,
    ProblemInstance,
    UnitConvention,
    validate_problem_instance,
)

from llm_resource_allocation.domain.units import (
    GFLOPS_PER_TFLOP,
    KILOBYTES_PER_GIGABYTE,
    MILLISECONDS_PER_SECOND,
    SECONDS_PER_HOUR,
    kilobytes_to_gigabytes,
)


class PhysicalUnitTests(unittest.TestCase):
    def test_named_conversion_ratios(self) -> None:
        self.assertEqual(1_000_000.0, KILOBYTES_PER_GIGABYTE)
        self.assertEqual(1_000.0, GFLOPS_PER_TFLOP)
        self.assertEqual(1_000.0, MILLISECONDS_PER_SECOND)
        self.assertEqual(3_600.0, SECONDS_PER_HOUR)

    def test_decimal_storage_conversion(self) -> None:
        self.assertEqual(0.0, kilobytes_to_gigabytes(0))
        self.assertEqual(1.0, kilobytes_to_gigabytes(1_000_000))
        self.assertEqual(0.000128, kilobytes_to_gigabytes(128.0))

    def test_storage_boundary_rejects_invalid_inputs(self) -> None:
        invalid_values = (
            -1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            True,
            False,
            "1",
            None,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    kilobytes_to_gigabytes(value)


class UnitConventionTests(unittest.TestCase):
    def test_new_instances_default_to_coherent_units(self) -> None:
        historical = make_minimal_tiny_instance()
        inputs = {
            field.name: getattr(historical, field.name)
            for field in fields(ProblemInstance)
            if field.name not in ("unit_convention", "data_residency_hours", "seconds_per_hour")
        }
        instance = ProblemInstance(**inputs)
        validate_problem_instance(instance)
        self.assertIs(instance.unit_convention, UnitConvention.COHERENT)
        self.assertTrue(instance.uses_coherent_units)
        self.assertEqual(1.0, instance.data_residency_hours)
        self.assertEqual(1.0, instance.data_storage_duration_hours)
        self.assertEqual(3600.0, instance.seconds_per_hour)

    def test_historical_fixture_retains_literal_convention(self) -> None:
        instance = replace(make_minimal_tiny_instance(), data_residency_hours=2.0)
        self.assertIs(instance.unit_convention, UnitConvention.LITERAL_PAPER)
        self.assertFalse(instance.uses_coherent_units)
        self.assertEqual(1.0, instance.data_storage_duration_hours)

    def test_coherent_fixture_uses_explicit_data_residency(self) -> None:
        instance = replace(make_coherent_tiny_instance(), data_residency_hours=2.0)
        validate_problem_instance(instance)
        self.assertTrue(instance.uses_coherent_units)
        self.assertEqual(2.0, instance.data_storage_duration_hours)
        validate_problem_instance(replace(instance, data_residency_hours=0.0))

    def test_unit_convention_requires_enum(self) -> None:
        for value in ("coherent", "literal_paper", True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(InstanceValidationError, "unit_convention"):
                    validate_problem_instance(
                        replace(make_minimal_tiny_instance(), unit_convention=value)
                    )

    def test_data_residency_rejects_invalid_inputs(self) -> None:
        for value in (-1.0, float("inf"), float("nan"), True, "1", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(InstanceValidationError, "data_residency_hours"):
                    validate_problem_instance(
                        replace(make_coherent_tiny_instance(), data_residency_hours=value)
                    )

    def test_coherent_seconds_per_hour_is_physical_constant(self) -> None:
        instance = replace(make_coherent_tiny_instance(), seconds_per_hour=1.0)
        with self.assertRaisesRegex(InstanceValidationError, "seconds_per_hour"):
            validate_problem_instance(instance)
        validate_problem_instance(
            replace(instance, unit_convention=UnitConvention.LITERAL_PAPER)
        )
