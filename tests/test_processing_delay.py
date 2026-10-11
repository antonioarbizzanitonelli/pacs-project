"""Hand-calculated tests for `D_proc`, Paper v2 Eq. (8i), and C4 integration."""

from dataclasses import replace
from math import fsum, inf, nextafter, ulp
from unittest.mock import patch
import unittest

from llm_resource_allocation.data import make_heterogeneous_tiny_instance
from llm_resource_allocation.domain import InstanceValidationError, ParallelismConfig
from llm_resource_allocation.math.parallelism import (
    compute_configuration_selection_residuals,
)
from llm_resource_allocation.math.routing import (
    compute_demand_balance_residuals,
)
from llm_resource_allocation.math.mccormick import compute_binary_continuous_product
from llm_resource_allocation.math.latency import compute_configuration_delay_seconds
from llm_resource_allocation.math.objective import compute_delay_penalty
from llm_resource_allocation.math.processing_delay import (
    compute_delay_slo_residuals,
    compute_processing_delays_seconds,
)
from llm_resource_allocation.math.error_slo import compute_error_slo_residuals

if __package__:
    from .placement_support import make_sparse_placement_state as make_state
else:
    from placement_support import make_sparse_placement_state as make_state


ROUTE_A = ("q_short", "m_compact", "gpu_fast_fp16")
PAIR_A = ROUTE_A[1:]
SELECTOR_A = (*PAIR_A, 2, 2)
AUXILIARY_A = (*ROUTE_A, 2, 2)
ROUTE_B = ("q_short", "m_accurate", "gpu_budget_int8")
PAIR_B = ROUTE_B[1:]
SELECTOR_B = (*PAIR_B, 1, 1)
AUXILIARY_B = (*ROUTE_B, 1, 1)


def make_delay_instance(*, delay_limit=0.25, delay_penalty=0.01):
    """Create coefficients giving transparent `D_cfg_A=.16`, `D_cfg_B=.30`."""

    base = make_heterogeneous_tiny_instance()
    queries = tuple(
        replace(
            query,
            input_tokens=100.0,
            output_tokens=50.0,
            delay_slo_seconds=delay_limit,
            delay_penalty_usd_per_millisecond_query=delay_penalty,
        )
        if query.id == "q_short"
        else query
        for query in base.queries
    )
    compute_delays = dict(base.compute_delay_seconds_per_token)
    communication_delays = dict(base.communication_delay_seconds_per_token)
    compute_delays[ROUTE_A] = 0.002
    communication_delays[ROUTE_A] = 0.0001
    compute_delays[ROUTE_B] = 0.002
    communication_delays[ROUTE_B] = 0.0
    return replace(
        base,
        queries=queries,
        compute_delay_seconds_per_token=compute_delays,
        communication_delay_seconds_per_token=communication_delays,
    )


def make_hand_state():
    """Return the exact-product two-route state yielding `D_proc=.216 s`."""

    return make_state(
        x={ROUTE_A: 0.6, ROUTE_B: 0.4},
        z={ROUTE_A: 1, ROUTE_B: 1},
        q={PAIR_A: 1, PAIR_B: 1},
        w={SELECTOR_A: 1, SELECTOR_B: 1},
        y={PAIR_A: 4, PAIR_B: 1},
        v={AUXILIARY_A: 0.6, AUXILIARY_B: 0.4},
    )


def result_for_query(results, query_id="q_short"):
    """Select one query result from deterministic instance-order output."""

    return next(result for result in results if result.query_id == query_id)


class ProcessingDelayAggregationTests(unittest.TestCase):
    """Verify the linearized fixed-configuration aggregation by hand."""

    def setUp(self):
        """Use the transparent two-route latency fixture."""

        self.instance = make_delay_instance()
        self.state = make_hand_state()

    def test_required_two_route_hand_calculation(self):
        """Check `0.6*0.16 + 0.4*0.30 = 0.216 seconds`."""

        delays = compute_processing_delays_seconds(self.instance, self.state)

        self.assertAlmostEqual(delays["q_short"], 0.216)
        self.assertEqual(delays["q_long"], 0.0)

    def test_fixed_configuration_primitive_is_reused(self):
        """Delegate every `D_cfg` term to the already tested latency primitive."""

        query = next(q for q in self.instance.queries if q.id == "q_short")
        config = ParallelismConfig(2, 2)
        with patch(
            "llm_resource_allocation.math.processing_delay."
            "compute_configuration_delay_seconds",
            wraps=compute_configuration_delay_seconds,
        ) as primitive:
            compute_processing_delays_seconds(self.instance, self.state)

        primitive.assert_any_call(0.002, 0.0001, query, config)
        self.assertAlmostEqual(
            compute_configuration_delay_seconds(0.002, 0.0001, query, config),
            0.16,
        )

    def test_production_uses_extracted_v_not_recomputed_x_times_w(self):
        """Keep a McCormick mismatch visible instead of silently repairing `v`."""

        mismatched = make_state(
            x={ROUTE_A: 0.6},
            w={SELECTOR_A: 1},
            v={AUXILIARY_A: 0.5},
        )
        delay = compute_processing_delays_seconds(
            self.instance, mismatched
        )["q_short"]

        self.assertAlmostEqual(delay, 0.5 * 0.16)
        self.assertNotAlmostEqual(delay, 0.6 * 0.16)

    def test_oracle_comparison_agrees_when_v_is_exact_product(self):
        """Cross-check production `v` aggregation against independent `x*w`."""

        query = next(q for q in self.instance.queries if q.id == "q_short")
        oracle = fsum(
            (
                compute_binary_continuous_product(0.6, 1)
                * compute_configuration_delay_seconds(
                    0.002, 0.0001, query, ParallelismConfig(2, 2)
                ),
                compute_binary_continuous_product(0.4, 1)
                * compute_configuration_delay_seconds(
                    0.002, 0.0, query, ParallelismConfig(1, 1)
                ),
            )
        )

        production = compute_processing_delays_seconds(
            self.instance, self.state
        )["q_short"]
        self.assertAlmostEqual(production, oracle)

    def test_sparse_missing_auxiliaries_are_exact_zero(self):
        """Traverse the full lattice while omitted known `v` contributes zero."""

        delays = compute_processing_delays_seconds(self.instance, make_state())
        self.assertEqual(delays, {"q_short": 0.0, "q_long": 0.0})

    def test_missing_scientific_delay_coefficient_is_not_sparse_zero(self):
        """Keep incomplete scientific tensors distinct from sparse decisions."""

        incomplete = dict(self.instance.compute_delay_seconds_per_token)
        incomplete.pop(ROUTE_A)
        invalid_instance = replace(
            self.instance, compute_delay_seconds_per_token=incomplete
        )

        with self.assertRaises(InstanceValidationError):
            compute_processing_delays_seconds(invalid_instance, make_state())


class DelaySloTests(unittest.TestCase):
    """Verify Paper v2 Eq. (8i) feasible, violated, and boundary cases."""

    def test_feasible_delay_has_positive_slack(self):
        """Check `Delta=.25`, `D_proc=.216`, and slack `.034`."""

        result = result_for_query(
            compute_delay_slo_residuals(
                make_delay_instance(delay_limit=0.25), make_hand_state()
            )
        )

        self.assertAlmostEqual(result.processing_delay, 0.216)
        self.assertEqual(result.delay_limit, 0.25)
        self.assertAlmostEqual(result.slack, 0.034)
        self.assertEqual(result.violation, 0.0)
        self.assertTrue(result.is_feasible())

    def test_violated_delay_exposes_positive_excess(self):
        """Check `Delta=.20`, slack `-.016`, and violation `.016`."""

        result = result_for_query(
            compute_delay_slo_residuals(
                make_delay_instance(delay_limit=0.20), make_hand_state()
            )
        )

        self.assertAlmostEqual(result.slack, -0.016)
        self.assertAlmostEqual(result.violation, 0.016)
        self.assertFalse(result.is_feasible())

    def test_exact_delay_boundary_is_feasible(self):
        """Check the exactly representable boundary `D_proc = Delta = .125`."""

        instance = make_delay_instance(delay_limit=0.125)
        instance = replace(
            instance,
            queries=tuple(
                replace(query, input_tokens=1.0, output_tokens=1.0)
                if query.id == "q_short"
                else query
                for query in instance.queries
            ),
            compute_delay_seconds_per_token={
                **instance.compute_delay_seconds_per_token,
                ROUTE_A: 0.125,
            },
            communication_delay_seconds_per_token={
                **instance.communication_delay_seconds_per_token,
                ROUTE_A: 0.0,
            },
        )
        result = result_for_query(
            compute_delay_slo_residuals(
                instance, make_state(v={AUXILIARY_A: 1.0})
            )
        )

        # TP=2: .125*1/2 + (.125/2)*1 = .125 exactly in binary.
        self.assertEqual(result.processing_delay, 0.125)
        self.assertEqual(result.slack, 0.0)
        self.assertTrue(result.is_feasible())

    def test_decimal_boundary_preserves_composition_roundoff(self):
        """Keep the one-ULP excess from TTFT+generation visible at zero tolerance."""

        result = result_for_query(
            compute_delay_slo_residuals(
                make_delay_instance(delay_limit=0.216), make_hand_state()
            )
        )

        self.assertEqual(result.processing_delay, nextafter(0.216, inf))
        self.assertEqual(result.violation, ulp(0.216))
        self.assertFalse(result.is_feasible())
        self.assertTrue(result.is_feasible(tolerance=ulp(0.216)))
        self.assertEqual(result.delay_limit, 0.216)

    def test_tolerance_changes_only_classification(self):
        """Preserve raw delay/SLO/slack under a seconds-valued tolerance."""

        result = result_for_query(
            compute_delay_slo_residuals(
                make_delay_instance(delay_limit=0.216 - 5e-10),
                make_hand_state(),
            )
        )
        raw_values = (result.processing_delay, result.delay_limit, result.slack)

        self.assertFalse(result.is_feasible())
        self.assertTrue(result.is_feasible(tolerance=1e-9))
        self.assertEqual(
            (result.processing_delay, result.delay_limit, result.slack),
            raw_values,
        )

    def test_invalid_tolerance_is_rejected(self):
        """Require a finite non-negative real delay tolerance."""

        result = result_for_query(
            compute_delay_slo_residuals(make_delay_instance(), make_hand_state())
        )
        for invalid in (-1e-9, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    result.is_feasible(invalid)


class DelayIntegrationTests(unittest.TestCase):
    """Check TP/PP effects, independent failures, and objective C4 wiring."""

    def _single_config_delay(self, n, m):
        """Return a consistent one-route delay for a selected configuration."""

        selector = (*PAIR_A, n, m)
        auxiliary = (*ROUTE_A, n, m)
        state = make_state(
            x={ROUTE_A: 1.0},
            z={ROUTE_A: 1},
            q={PAIR_A: 1},
            w={selector: 1},
            y={PAIR_A: n * m},
            v={auxiliary: 1.0},
        )
        return compute_processing_delays_seconds(
            make_delay_instance(), state
        )["q_short"]

    def test_tp_reduces_compute_and_pp_adds_communication(self):
        """Verify only the local formula effects, not a global optimum claim."""

        low_tp = self._single_config_delay(1, 1)
        high_tp = self._single_config_delay(2, 1)
        deeper_pp = self._single_config_delay(2, 2)

        self.assertGreater(low_tp, high_tp)
        self.assertGreater(deeper_pp, high_tp)

    def test_delay_computes_despite_eq8b_eq8j_and_eq8d_failures(self):
        """Expose simultaneous independent diagnostics for domain-valid values."""

        instance = make_delay_instance()
        queries = tuple(
            replace(query, error_slo=0.001)
            if query.id == "q_short"
            else query
            for query in instance.queries
        )
        instance = replace(instance, queries=queries)
        selector_other = (*PAIR_A, 1, 1)
        auxiliary_other = (*ROUTE_A, 1, 1)
        state = make_state(
            x={ROUTE_A: 0.6},
            q={PAIR_A: 1},
            w={SELECTOR_A: 1, selector_other: 1},
            y={PAIR_A: 5},
            v={AUXILIARY_A: 0.6, auxiliary_other: 0.6},
        )

        balance = result_for_query(
            compute_demand_balance_residuals(instance, state.routing_decisions)
        )
        error = result_for_query(
            compute_error_slo_residuals(instance, state.routing_decisions)
        )
        selection = next(
            result
            for result in compute_configuration_selection_residuals(
                instance, state.parallelism_decisions
            )
            if (result.model_id, result.gpu_tier_id) == PAIR_A
        )
        delay = compute_processing_delays_seconds(instance, state)["q_short"]

        self.assertFalse(balance.is_feasible())
        self.assertFalse(error.is_feasible())
        self.assertFalse(selection.is_feasible())
        self.assertAlmostEqual(delay, 0.6 * (0.16 + 0.305))

    def test_processing_delay_feeds_existing_c4_unchanged(self):
        """Check `rho=2` and `D_proc=.216` give literal C4 `.432`."""

        instance = make_delay_instance(delay_penalty=2.0)
        delays = compute_processing_delays_seconds(instance, make_hand_state())

        self.assertAlmostEqual(compute_delay_penalty(instance, delays), 0.432)


if __name__ == "__main__":
    unittest.main()
