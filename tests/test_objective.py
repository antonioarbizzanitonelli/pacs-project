"""Hand-calculated tests for published objective-component arithmetic."""

from dataclasses import replace
from types import MappingProxyType
import unittest

from llm_resource_allocation.data import make_heterogeneous_tiny_instance
from llm_resource_allocation.domain import ProblemInstance, validate_problem_instance
from llm_resource_allocation.math.objective import (
    ObjectiveState,
    PaperVersion,
    compute_data_storage_cost,
    compute_delay_penalty,
    compute_gpu_rental_cost,
    compute_model_storage_cost,
    compute_objective_breakdown,
    compute_unmet_demand_penalty,
)


def make_objective_test_instance() -> ProblemInstance:
    """Build one valid synthetic instance with transparent objective inputs."""

    base = make_heterogeneous_tiny_instance()
    q_short = replace(
        base.queries[0],
        arrival_rate_per_hour=10.0,
        input_tokens=60.0,
        output_tokens=40.0,
        storage_gb_per_token=0.01,
        delay_penalty_usd_per_millisecond_query=2.0,
        unmet_penalty_usd_per_hour=10.0,
    )
    q_long = replace(
        base.queries[1],
        delay_penalty_usd_per_millisecond_query=3.0,
        unmet_penalty_usd_per_hour=20.0,
    )
    m_compact = replace(base.models[0], weight_gb=20.0)
    m_accurate = replace(base.models[1], weight_gb=30.0)
    budget_tier = replace(base.gpu_tiers[0], hourly_rental_usd=1.5)
    fast_tier = replace(base.gpu_tiers[1], hourly_rental_usd=0.5)
    instance = replace(
        base,
        queries=(q_short, q_long),
        models=(m_compact, m_accurate),
        gpu_tiers=(budget_tier, fast_tier),
        horizon_hours=24.0,
        storage_price_usd_per_gb_hour=0.01,
    )
    validate_problem_instance(instance)
    return instance


def make_nonzero_objective_state() -> ObjectiveState:
    """Return sparse plain values that make every objective component nonzero."""

    return ObjectiveState(
        gpu_counts={('m_compact', 'gpu_budget_int8'): 2.0},
        admission_indicators={
            ('q_short', 'm_compact', 'gpu_budget_int8'): 1.0
        },
        routing_fractions={
            ('q_short', 'm_compact', 'gpu_budget_int8'): 0.25
        },
        unmet_fractions={'q_short': 0.25},
        processing_delays_seconds={'q_short': 0.5},
    )


class ObjectiveComponentTests(unittest.TestCase):
    """Verify every term in Paper v2 Eq. (8a) independently by hand."""

    def setUp(self) -> None:
        """Create a valid coefficient source shared by arithmetic-only tests."""

        self.instance = make_objective_test_instance()

    def test_c1_gpu_rental_hand_calculation(self) -> None:
        """Check ``24 * (1.5*2 + 0.5*1) = 84`` without rounding counts."""

        gpu_counts = {
            ('m_compact', 'gpu_budget_int8'): 2.0,
            ('m_accurate', 'gpu_fast_fp16'): 1.0,
        }
        self.assertAlmostEqual(
            84.0,
            compute_gpu_rental_cost(self.instance, gpu_counts),
        )

    def test_c2_retains_repeated_query_indexing(self) -> None:
        """Check that the printed C2 charges one shared model weight twice."""

        ten_hour_instance = replace(self.instance, horizon_hours=10.0)
        admissions = {
            ('q_short', 'm_compact', 'gpu_budget_int8'): 1.0,
            ('q_long', 'm_compact', 'gpu_budget_int8'): 1.0,
        }
        # Paper v2 Eq. (8a): 10 * 0.01 * 20 * 2 = 4, not a deduplicated 2.
        self.assertAlmostEqual(
            4.0,
            compute_model_storage_cost(ten_hour_instance, admissions),
        )

    def test_c3_data_storage_hand_calculation(self) -> None:
        """Check printed C3 arithmetic, not its unresolved dimensional meaning."""

        two_hour_instance = replace(
            self.instance,
            horizon_hours=2.0,
            storage_price_usd_per_gb_hour=0.5,
        )
        routing = {('q_short', 'm_compact', 'gpu_budget_int8'): 0.25}
        # Synthetic arithmetic: 2 * 0.5 * 0.01 * 100 * 10 * 0.25 = 2.5.
        self.assertAlmostEqual(
            2.5,
            compute_data_storage_cost(two_hour_instance, routing),
        )

    def test_c4_delay_penalty_hand_calculation(self) -> None:
        """Check ``2*0.5 + 3*0.25 = 1.75`` with no extra paper-absent factors."""

        processing_delays = {'q_short': 0.5, 'q_long': 0.25}
        self.assertAlmostEqual(
            1.75,
            compute_delay_penalty(self.instance, processing_delays),
        )

    def test_c5_explicit_v1_v2_hand_calculation(self) -> None:
        """Make the material horizon-factor difference visible numerically."""

        unmet = {'q_short': 0.25}
        v1_penalty = compute_unmet_demand_penalty(
            self.instance, unmet, PaperVersion.V1
        )
        v2_penalty = compute_unmet_demand_penalty(
            self.instance, unmet, PaperVersion.V2
        )
        self.assertAlmostEqual(2.5, v1_penalty)
        self.assertAlmostEqual(60.0, v2_penalty)
        self.assertAlmostEqual(24.0, v2_penalty / v1_penalty)

    def test_invalid_version_representation_is_not_silently_accepted(self) -> None:
        """Require the explicit enum instead of interpreting an ad-hoc flag."""

        with self.assertRaises(ValueError):
            compute_unmet_demand_penalty(
                self.instance,
                {'q_short': 0.25},
                'v2',  # type: ignore[arg-type]
            )


class ObjectiveCompositionTests(unittest.TestCase):
    """Check zero, scaling, immutability, and additive composition semantics."""

    def setUp(self) -> None:
        """Create shared coefficient and nonzero decision fixtures."""

        self.instance = make_objective_test_instance()
        self.state = make_nonzero_objective_state()

    def test_zero_state_has_zero_components_and_total(self) -> None:
        """Make every printed objective sum zero for empty sparse mappings."""

        zero_state = ObjectiveState({}, {}, {}, {}, {})
        breakdown = compute_objective_breakdown(
            self.instance, zero_state, PaperVersion.V2
        )
        self.assertEqual(0.0, breakdown.gpu_rental_cost)
        self.assertEqual(0.0, breakdown.model_storage_cost)
        self.assertEqual(0.0, breakdown.data_storage_cost)
        self.assertEqual(0.0, breakdown.delay_penalty)
        self.assertEqual(0.0, breakdown.unmet_demand_penalty)
        self.assertEqual(0.0, breakdown.total)

    def test_horizon_scaling_preserves_version_semantics(self) -> None:
        """Scale C1-C3/v2-C5 while keeping C4 and v1-C5 horizon-independent."""

        two_hours = replace(self.instance, horizon_hours=2.0)
        six_hours = replace(self.instance, horizon_hours=6.0)
        v2_short = compute_objective_breakdown(
            two_hours, self.state, PaperVersion.V2
        )
        v2_long = compute_objective_breakdown(
            six_hours, self.state, PaperVersion.V2
        )
        self.assertAlmostEqual(3.0 * v2_short.gpu_rental_cost, v2_long.gpu_rental_cost)
        self.assertAlmostEqual(
            3.0 * v2_short.model_storage_cost, v2_long.model_storage_cost
        )
        self.assertAlmostEqual(
            3.0 * v2_short.data_storage_cost, v2_long.data_storage_cost
        )
        self.assertAlmostEqual(v2_short.delay_penalty, v2_long.delay_penalty)
        self.assertAlmostEqual(
            3.0 * v2_short.unmet_demand_penalty,
            v2_long.unmet_demand_penalty,
        )

        v1_short = compute_objective_breakdown(
            two_hours, self.state, PaperVersion.V1
        )
        v1_long = compute_objective_breakdown(
            six_hours, self.state, PaperVersion.V1
        )
        self.assertAlmostEqual(
            v1_short.unmet_demand_penalty,
            v1_long.unmet_demand_penalty,
        )

    def test_breakdown_total_is_sum_of_five_named_components(self) -> None:
        """Keep the total an auditable additive composition, not a second formula."""

        breakdown = compute_objective_breakdown(
            self.instance, self.state, PaperVersion.V2
        )
        independent_sum = (
            breakdown.gpu_rental_cost
            + breakdown.model_storage_cost
            + breakdown.data_storage_cost
            + breakdown.delay_penalty
            + breakdown.unmet_demand_penalty
        )
        self.assertAlmostEqual(independent_sum, breakdown.total)

    def test_state_copies_and_freezes_caller_mappings(self) -> None:
        """Prevent later caller mutations from changing recomputed objectives."""

        caller_counts = {('m_compact', 'gpu_budget_int8'): 2.0}
        state = ObjectiveState(caller_counts, {}, {}, {}, {})
        caller_counts[('m_compact', 'gpu_budget_int8')] = 99.0
        self.assertIsInstance(state.gpu_counts, MappingProxyType)
        self.assertEqual(2.0, state.gpu_counts[('m_compact', 'gpu_budget_int8')])
        with self.assertRaises(TypeError):
            state.gpu_counts[('m_compact', 'gpu_budget_int8')] = 3.0

    def test_arithmetic_does_not_clip_logically_invalid_values(self) -> None:
        """Leave decision-domain rejection to independent validation callers."""

        negative_rental = compute_gpu_rental_cost(
            self.instance,
            {('m_compact', 'gpu_budget_int8'): -1.0},
        )
        oversized_routing = compute_data_storage_cost(
            self.instance,
            {('q_short', 'm_compact', 'gpu_budget_int8'): 1.5},
        )
        self.assertAlmostEqual(-36.0, negative_rental)
        self.assertAlmostEqual(
            6.0 * compute_data_storage_cost(
                self.instance,
                {('q_short', 'm_compact', 'gpu_budget_int8'): 0.25},
            ),
            oversized_routing,
        )


if __name__ == '__main__':
    unittest.main()
