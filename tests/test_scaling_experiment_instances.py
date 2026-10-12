"""Hand-derived coefficients for coherent nested synthetic scaling."""

from dataclasses import replace
import unittest

from llm_resource_allocation.data.scaling_experiment_instances import make_scaling_experiment_case
from llm_resource_allocation.data.synthetic_benchmark_instances import make_synthetic_benchmark_cases
from llm_resource_allocation.domain import UnitConvention


class ScalingInstanceTests(unittest.TestCase):
    def test_coherent_query_clones_and_capacity_rule(self):
        cases = make_synthetic_benchmark_cases((4,))
        for case in cases:
            self.assertIs(case.instance.unit_convention, UnitConvention.COHERENT)
            self.assertEqual(case.instance.data_residency_hours, 1.0)
        instance = cases[-1].instance
        self.assertEqual(instance.dimensions, (4, 2, 2))
        self.assertEqual(tuple(query.arrival_rate_per_hour for query in instance.queries), (12.0, 4.0, 12.0, 4.0))
        self.assertEqual(tuple(query.total_tokens for query in instance.queries), (4.0, 10.0, 4.0, 10.0))
        self.assertEqual(instance.cost_budget_usd, 100.0)
        self.assertEqual(instance.storage_capacity_gb, 400.0)

    def test_nested_menu_retains_old_alternatives_and_uses_bilinear_coefficients(self):
        smaller = make_scaling_experiment_case(4, 3, 3).instance
        larger = make_scaling_experiment_case(4, 5, 5).instance
        self.assertEqual(larger.models[:3], smaller.models)
        self.assertEqual(larger.gpu_tiers[:3], smaller.gpu_tiers)
        midpoint_model, midpoint_tier = smaller.models[2], smaller.gpu_tiers[2]
        self.assertEqual(midpoint_model.weight_gb, 4.0)
        self.assertEqual(midpoint_tier.memory_capacity_gb, 16.0)
        self.assertEqual(midpoint_tier.hourly_rental_usd, 1.25)
        self.assertEqual(midpoint_tier.tensor_parallel_degrees, (1, 2))
        self.assertEqual(midpoint_tier.pipeline_parallel_depths, (1, 2))
        key = ("q_short", midpoint_model.id, midpoint_tier.id)
        # Across models: budget .1->.3 gives .2; fast .05->.15 gives .1;
        # across tiers: (.2+.1)/2 gives .15 seconds/token.
        self.assertAlmostEqual(smaller.compute_delay_seconds_per_token[key], .15)
        self.assertEqual(smaller.compute_gflop_per_token[key], 3.0)
        self.assertAlmostEqual(smaller.base_error_rate[("q_short", midpoint_model.id)], .045)

    def test_declared_load_changes_only_arrivals_with_independent_budget_scale(self):
        nominal = make_scaling_experiment_case(4, 3, 3).instance
        loaded = make_scaling_experiment_case(4, 3, 3, load_multiplier=100000.0, budget_multiplier=2.0).instance
        self.assertEqual(tuple(query.arrival_rate_per_hour for query in loaded.queries), (1200000.0, 400000.0, 1200000.0, 400000.0))
        self.assertEqual(loaded.cost_budget_usd, 200.0)
        self.assertEqual(loaded.storage_capacity_gb, nominal.storage_capacity_gb)
        self.assertEqual(loaded.compute_gflop_per_token, nominal.compute_gflop_per_token)
        self.assertEqual(loaded.compute_delay_seconds_per_token, nominal.compute_delay_seconds_per_token)
        self.assertEqual(loaded.models, nominal.models)
        self.assertEqual(loaded.gpu_tiers, nominal.gpu_tiers)

    def test_residency_override_is_explicit_and_literal_cases_are_rejected(self):
        case = make_scaling_experiment_case(2, 2, 2, data_residency_hours=2.0)
        self.assertEqual(case.instance.data_residency_hours, 2.0)
        self.assertEqual(case.provenance["data_residency_hours"], 2.0)
        with self.assertRaisesRegex(ValueError, "coherent"):
            replace(case, instance=replace(case.instance, unit_convention=UnitConvention.LITERAL_PAPER))

    def test_invalid_scaling_boundaries_reject_inputs(self):
        for dimensions in ((1, 2, 2), (2, True, 2), (2, 2, 1)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                make_scaling_experiment_case(*dimensions)
        for changes in ({"load_multiplier": 0}, {"budget_multiplier": -1}, {"delay_slo_multiplier": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_scaling_experiment_case(2, 2, 2, **changes)


if __name__ == "__main__":
    unittest.main()
