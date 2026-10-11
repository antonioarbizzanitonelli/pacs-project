"""Hand-calculated tests for independent Paper v2 Eq. (8g) diagnostics."""

from dataclasses import replace
import unittest

from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.domain import InstanceValidationError
from llm_resource_allocation.math.routing import (
    DecisionDomainError,
    RoutingDecisionState,
    compute_demand_balance_residuals,
    compute_routing_consistency_residuals,
)
from llm_resource_allocation.math.parallelism import (
    ParallelismDecisionState,
    compute_configuration_selection_residuals,
    compute_gpu_count_linking_residuals,
)
from llm_resource_allocation.math.compute_capacity import (
    compute_available_compute_tflops_per_hour,
    compute_compute_capacity_diagnostics,
    compute_required_compute_tflops_per_hour,
)


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ROUTE[1:]
SELECTOR_1_1 = (*PAIR, 1, 1)


def make_compute_instance(*, throughput=0.1, utilization=0.5):
    """Return a one-pair instance with transparent Eq. (8g) coefficients."""

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        arrival_rate_per_hour=10.0,
        input_tokens=60.0,
        output_tokens=40.0,
    )
    tier = replace(
        base.gpu_tiers[0],
        compute_throughput_tflops=throughput,
    )
    return replace(
        base,
        queries=(query,),
        gpu_tiers=(tier,),
        compute_gflop_per_token={ROUTE: 100.0},
        compute_utilization=utilization,
        seconds_per_hour=3600.0,
    )


def make_state(*, x=None, z=None, q=None, u=None, w=None, y=None):
    """Build a sparse domain-valid parallelism state for Eq. (8g) tests."""

    routing = RoutingDecisionState(
        routing_fractions={} if x is None else x,
        admission_indicators={} if z is None else z,
        deployment_indicators={} if q is None else q,
        unmet_fractions={} if u is None else u,
    )
    return ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors={} if w is None else w,
        gpu_counts={} if y is None else y,
    )


def result_for_pair(results, pair=PAIR):
    """Select one model/tier diagnostic from deterministic output."""

    return next(
        result
        for result in results
        if (result.model_id, result.gpu_tier_id) == pair
    )


class ComputeCapacityHandCalculationTests(unittest.TestCase):
    """Verify feasible, violated, and exact-boundary arithmetic explicitly."""

    def test_required_and_available_feasible_hand_case(self):
        """Check required `50`, available `180`, and slack `130` TFLOP/hour."""

        instance = make_compute_instance(throughput=0.1)
        state = make_state(x={ROUTE: 0.5}, y={PAIR: 1})
        result = result_for_pair(compute_compute_capacity_diagnostics(instance, state))

        # Paper v2 Eq. (8g): 100*100*10*.5/1000 = 50 TFLOP/hour.
        expected_required = 100.0 * 100.0 * 10.0 * 0.5 / 1000.0
        # Paper v2 Eq. (8g): .5*3600*.1*1 = 180 TFLOP/hour.
        expected_available = 0.5 * 3600.0 * 0.1 * 1
        self.assertEqual(expected_required, 50.0)
        self.assertEqual(expected_available, 180.0)
        self.assertEqual(result.required_compute, 50.0)
        self.assertEqual(result.available_compute, 180.0)
        self.assertEqual(result.slack, 130.0)
        self.assertEqual(result.violation, 0.0)
        self.assertTrue(result.is_feasible())

    def test_violation_has_negative_slack_and_positive_excess(self):
        """Check required `50`, available `40`, slack `-10`, violation `10`."""

        instance = make_compute_instance(throughput=1.0 / 45.0)
        result = result_for_pair(
            compute_compute_capacity_diagnostics(
                instance, make_state(x={ROUTE: 0.5}, y={PAIR: 1})
            )
        )

        self.assertAlmostEqual(result.required_compute, 50.0)
        self.assertAlmostEqual(result.available_compute, 40.0)
        self.assertAlmostEqual(result.slack, -10.0)
        self.assertAlmostEqual(result.violation, 10.0)
        self.assertFalse(result.is_feasible())

    def test_exact_boundary_is_feasible(self):
        """Check required and available compute both equal `50` TFLOP/hour."""

        instance = make_compute_instance(throughput=1.0 / 36.0)
        result = result_for_pair(
            compute_compute_capacity_diagnostics(
                instance, make_state(x={ROUTE: 0.5}, y={PAIR: 1})
            )
        )

        self.assertAlmostEqual(result.required_compute, 50.0)
        self.assertAlmostEqual(result.available_compute, 50.0)
        self.assertAlmostEqual(result.slack, 0.0)
        self.assertTrue(result.is_feasible())

    def test_required_and_available_helpers_are_independently_exposed(self):
        """Keep both sides of Eq. (8g) auditable before residual composition."""

        instance = make_compute_instance()
        state = make_state(x={ROUTE: 0.5}, y={PAIR: 1})

        required = compute_required_compute_tflops_per_hour(
            instance, state.routing_decisions
        )
        available = compute_available_compute_tflops_per_hour(instance, state)
        self.assertEqual(required, {PAIR: 50.0})
        self.assertEqual(available, {PAIR: 180.0})


class ComputeAggregationAndScalingTests(unittest.TestCase):
    """Exercise the printed sum over queries and linear `x`/`y` behavior."""

    def test_two_query_contributions_sum_to_fifty(self):
        """Check manual contributions `20 + 30 = 50` on one pair."""

        base = make_heterogeneous_tiny_instance()
        target_pair = ("m_compact", "gpu_fast_fp16")
        route_a = ("q_short", *target_pair)
        route_b = ("q_long", *target_pair)
        queries = tuple(
            replace(
                query,
                arrival_rate_per_hour=(4.0 if query.id == "q_short" else 6.0),
                input_tokens=60.0,
                output_tokens=40.0,
            )
            for query in base.queries
        )
        compute_costs = dict(base.compute_gflop_per_token)
        compute_costs[route_a] = 100.0
        compute_costs[route_b] = 100.0
        instance = replace(
            base,
            queries=queries,
            compute_gflop_per_token=compute_costs,
        )
        state = make_state(x={route_a: 0.5, route_b: 0.5})

        required = compute_required_compute_tflops_per_hour(
            instance, state.routing_decisions
        )[target_pair]
        # Query A: 100*100*4*.5/1000=20; query B similarly gives 30.
        self.assertEqual(required, 20.0 + 30.0)

    def test_required_load_scales_linearly_with_routing_fraction(self):
        """Check `x=1,.5,.25` gives required load `100,50,25`."""

        instance = make_compute_instance()
        results = {}
        for routing_fraction in (1.0, 0.5, 0.25):
            results[routing_fraction] = compute_required_compute_tflops_per_hour(
                instance,
                make_state(x={ROUTE: routing_fraction}).routing_decisions,
            )[PAIR]

        self.assertEqual(results, {1.0: 100.0, 0.5: 50.0, 0.25: 25.0})

    def test_available_capacity_scales_linearly_with_gpu_count(self):
        """Check `y=1,2,4` gives capacity `180,360,720`."""

        instance = make_compute_instance()
        results = {
            gpu_count: compute_available_compute_tflops_per_hour(
                instance, make_state(y={PAIR: gpu_count})
            )[PAIR]
            for gpu_count in (1, 2, 4)
        }

        self.assertEqual(results, {1: 180.0, 2: 360.0, 4: 720.0})


class ComputeCapacityBoundaryTests(unittest.TestCase):
    """Verify zero capacity, sparse data, domain reuse, and tolerance isolation."""

    def test_zero_load_and_zero_capacity_is_feasible(self):
        """Check the inactive sparse boundary `required=available=0`."""

        result = result_for_pair(
            compute_compute_capacity_diagnostics(
                make_compute_instance(), make_state()
            )
        )

        self.assertEqual(result.required_compute, 0.0)
        self.assertEqual(result.available_compute, 0.0)
        self.assertEqual(result.slack, 0.0)
        self.assertTrue(result.is_feasible())

    def test_positive_load_with_zero_capacity_is_infeasible(self):
        """Diagnose routed work at `y=0` without relying on Eq. (8k)."""

        result = result_for_pair(
            compute_compute_capacity_diagnostics(
                make_compute_instance(), make_state(x={ROUTE: 0.5})
            )
        )

        self.assertEqual(result.required_compute, 50.0)
        self.assertEqual(result.available_compute, 0.0)
        self.assertEqual(result.violation, 50.0)
        self.assertFalse(result.is_feasible())

    def test_sparse_state_reports_every_pair_in_deterministic_order(self):
        """Treat missing known `x/y` as zero and retain inactive pairs."""

        instance = make_heterogeneous_tiny_instance()
        results = compute_compute_capacity_diagnostics(instance, make_state())
        expected_pairs = tuple(
            (model_id, tier_id)
            for model_id in instance.model_ids
            for tier_id in instance.gpu_tier_ids
        )

        self.assertEqual(
            tuple((result.model_id, result.gpu_tier_id) for result in results),
            expected_pairs,
        )
        self.assertTrue(
            all(
                result.required_compute == result.available_compute == 0.0
                and result.is_feasible()
                for result in results
            )
        )

    def test_unknown_decisions_and_missing_alpha_are_rejected(self):
        """Keep sparse decisions distinct from complete scientific coefficients."""

        instance = make_compute_instance()
        unknown_x = make_state(
            x={("q_unknown", "m_small", "gpu_basic_fp16"): 0.0}
        )
        unknown_y = make_state(y={("m_unknown", "gpu_basic_fp16"): 0})
        for state in (unknown_x, unknown_y):
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    compute_compute_capacity_diagnostics(instance, state)

        with self.assertRaises(InstanceValidationError):
            compute_compute_capacity_diagnostics(
                replace(instance, compute_gflop_per_token={}), make_state()
            )

    def test_existing_x_and_y_domains_are_not_relaxed(self):
        """Reject out-of-range routing and fractional GPU counts exactly."""

        instance = make_compute_instance()
        invalid_states = (
            make_state(x={ROUTE: 1.000000001}),
            make_state(y={PAIR: 0.999999999}),
        )
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    compute_compute_capacity_diagnostics(instance, state)

    def test_tolerance_changes_only_inequality_classification(self):
        """Preserve raw required/available/slack under a tiny tolerance."""

        required = 50.0
        available = required - 5e-10
        throughput = available / (0.5 * 3600.0)
        result = result_for_pair(
            compute_compute_capacity_diagnostics(
                make_compute_instance(throughput=throughput),
                make_state(x={ROUTE: 0.5}, y={PAIR: 1}),
            )
        )
        raw_values = (
            result.required_compute,
            result.available_compute,
            result.slack,
        )

        self.assertFalse(result.is_feasible())
        self.assertTrue(result.is_feasible(tolerance=1e-9))
        self.assertEqual(
            (result.required_compute, result.available_compute, result.slack),
            raw_values,
        )

    def test_invalid_tolerance_is_rejected(self):
        """Require finite non-negative TFLOP/hour tolerance values."""

        result = result_for_pair(
            compute_compute_capacity_diagnostics(
                make_compute_instance(), make_state()
            )
        )
        for invalid in (-1e-9, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    result.is_feasible(invalid)


class ComputeCapacityIntegrationTests(unittest.TestCase):
    """Prove independence from other constraints and valid Eq. (8e) wiring."""

    def test_eq8g_computes_despite_eq8b_eq8d_eq8e_eq8k_failures(self):
        """Report simultaneous violations while retaining supplied `x` and `y`."""

        instance = make_compute_instance()
        state = make_state(
            x={ROUTE: 0.5},
            q={PAIR: 1},
            # Omitted w makes both active-selection and y-link equalities fail.
            y={PAIR: 1},
        )

        demand = compute_demand_balance_residuals(
            instance, state.routing_decisions
        )[0]
        selection = result_for_pair(
            compute_configuration_selection_residuals(instance, state)
        )
        gpu_link = result_for_pair(
            compute_gpu_count_linking_residuals(instance, state)
        )
        routing = next(
            result
            for result in compute_routing_consistency_residuals(
                instance, state.routing_decisions
            )
            if (result.query_id, result.model_id, result.gpu_tier_id) == ROUTE
        )
        compute = result_for_pair(
            compute_compute_capacity_diagnostics(instance, state)
        )

        self.assertFalse(demand.is_feasible())
        self.assertFalse(selection.is_feasible())
        self.assertFalse(gpu_link.is_feasible())
        self.assertFalse(routing.is_feasible())
        self.assertEqual(compute.required_compute, 50.0)
        self.assertEqual(compute.available_compute, 180.0)
        self.assertTrue(compute.is_feasible())

    def test_valid_eq8e_gpu_count_feeds_eq8g_without_core_coupling(self):
        """Check `w -> y` in tests while production Eq. (8g) reads only `y`."""

        instance = make_compute_instance()
        state = make_state(
            x={ROUTE: 0.5},
            z={ROUTE: 1},
            q={PAIR: 1},
            w={SELECTOR_1_1: 1},
            y={PAIR: 1},
        )
        gpu_link = result_for_pair(
            compute_gpu_count_linking_residuals(instance, state)
        )
        compute = result_for_pair(
            compute_compute_capacity_diagnostics(instance, state)
        )

        self.assertTrue(gpu_link.is_feasible())
        self.assertEqual(gpu_link.implied_gpu_count, 1)
        self.assertEqual(compute.available_compute, 180.0)
        self.assertTrue(compute.is_feasible())


if __name__ == "__main__":
    unittest.main()
