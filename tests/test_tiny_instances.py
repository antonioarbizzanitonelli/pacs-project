"""Tests for deterministic, explicitly synthetic tiny problem instances."""

import unittest

from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.domain import validate_problem_instance


class TinyInstanceTests(unittest.TestCase):
    """Verify fixture determinism, dimensions, and qualitative heterogeneity."""

    def test_minimal_instance_is_deterministic_and_valid(self) -> None:
        """Recreate the 1 x 1 x 1 fixture without hidden random state."""

        first = make_minimal_tiny_instance()
        second = make_minimal_tiny_instance()
        self.assertEqual(first, second)
        self.assertEqual((1, 1, 1), first.dimensions)
        validate_problem_instance(first)

    def test_heterogeneous_instance_is_deterministic_and_valid(self) -> None:
        """Recreate the 2 x 2 x 2 fixture and validate its full coefficient lattice."""

        first = make_heterogeneous_tiny_instance()
        second = make_heterogeneous_tiny_instance()
        self.assertEqual(first, second)
        self.assertEqual((2, 2, 2), first.dimensions)
        self.assertEqual(4, len(first.base_error_rate))
        self.assertEqual(8, len(first.compute_delay_seconds_per_token))
        validate_problem_instance(first)

    def test_heterogeneous_fixture_contains_real_choice_contrasts(self) -> None:
        """Preserve simple model/tier contrasts needed by later hand checks."""

        instance = make_heterogeneous_tiny_instance()
        compact, accurate = instance.models
        budget, fast = instance.gpu_tiers
        self.assertLess(compact.weight_gb, accurate.weight_gb)
        self.assertLess(budget.hourly_rental_usd, fast.hourly_rental_usd)
        self.assertLess(budget.memory_capacity_gb, fast.memory_capacity_gb)
        self.assertNotEqual(
            budget.quantization_error_multiplier,
            fast.quantization_error_multiplier,
        )


if __name__ == "__main__":
    unittest.main()
