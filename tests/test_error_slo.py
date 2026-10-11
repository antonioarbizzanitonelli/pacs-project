"""Hand-calculated tests for independent Paper v2 Eq. (8j) diagnostics."""

from dataclasses import replace
from unittest.mock import patch
import unittest

from llm_resource_allocation.data import make_heterogeneous_tiny_instance
from llm_resource_allocation.domain import InstanceValidationError
from llm_resource_allocation.math.routing import (
    DecisionDomainError,
    RoutingDecisionState,
    compute_demand_balance_residuals,
)
from llm_resource_allocation.math.error import compute_effective_error
from llm_resource_allocation.math.error_slo import compute_error_slo_residuals


LOW_ERROR_ROUTE = ("q_short", "m_compact", "gpu_fast_fp16")
HIGH_ERROR_ROUTE = ("q_short", "m_accurate", "gpu_fast_fp16")
INT8_ROUTE = ("q_short", "m_compact", "gpu_budget_int8")


def make_state(*, routing, unmet=None) -> RoutingDecisionState:
    """Build a sparse domain-valid state without imposing other constraints."""

    return RoutingDecisionState(
        routing_fractions=routing,
        admission_indicators={},
        deployment_indicators={},
        unmet_fractions={} if unmet is None else unmet,
    )


def make_error_instance(
    *,
    error_limit: float,
    compact_base_error: float = 0.02,
    accurate_base_error: float = 0.08,
):
    """Return a valid 2x2x2 instance with transparent `q_short` errors."""

    base = make_heterogeneous_tiny_instance()
    queries = tuple(
        replace(query, error_slo=error_limit)
        if query.id == "q_short"
        else query
        for query in base.queries
    )
    base_errors = dict(base.base_error_rate)
    base_errors[("q_short", "m_compact")] = compact_base_error
    base_errors[("q_short", "m_accurate")] = accurate_base_error
    return replace(base, queries=queries, base_error_rate=base_errors)


def result_for_query(results, query_id):
    """Select one query diagnostic from deterministic result order."""

    return next(result for result in results if result.query_id == query_id)


class ErrorSloHandCalculationTests(unittest.TestCase):
    """Verify feasible, violated, and exact-boundary weighted mixtures."""

    def test_feasible_weighted_mixture(self):
        """Check `0.02*0.6 + 0.08*0.4 = 0.044 <= 0.05`."""

        instance = make_error_instance(error_limit=0.05)
        state = make_state(routing={LOW_ERROR_ROUTE: 0.6, HIGH_ERROR_ROUTE: 0.4})
        result = result_for_query(
            compute_error_slo_residuals(instance, state), "q_short"
        )

        self.assertAlmostEqual(result.weighted_error, 0.044)
        self.assertEqual(result.error_limit, 0.05)
        self.assertAlmostEqual(result.slack, 0.006)
        self.assertEqual(result.violation, 0.0)
        self.assertTrue(result.is_feasible())

    def test_violated_weighted_mixture(self):
        """Check the same `E=0.044` violates `epsilon=0.03` by `0.014`."""

        instance = make_error_instance(error_limit=0.03)
        state = make_state(routing={LOW_ERROR_ROUTE: 0.6, HIGH_ERROR_ROUTE: 0.4})
        result = result_for_query(
            compute_error_slo_residuals(instance, state), "q_short"
        )

        self.assertAlmostEqual(result.weighted_error, 0.044)
        self.assertAlmostEqual(result.slack, -0.014)
        self.assertAlmostEqual(result.violation, 0.014)
        self.assertFalse(result.is_feasible())

    def test_exact_boundary(self):
        """Check `0.02*0.5 + 0.08*0.5 = epsilon = 0.05`."""

        instance = make_error_instance(error_limit=0.05)
        state = make_state(routing={LOW_ERROR_ROUTE: 0.5, HIGH_ERROR_ROUTE: 0.5})
        result = result_for_query(
            compute_error_slo_residuals(instance, state), "q_short"
        )

        self.assertAlmostEqual(result.weighted_error, 0.05)
        self.assertAlmostEqual(result.slack, 0.0)
        self.assertEqual(result.violation, 0.0)
        self.assertTrue(result.is_feasible())


class EffectiveErrorIntegrationTests(unittest.TestCase):
    """Verify the complete Eq. (1) to Eq. (8j) calculation chain."""

    def test_eq_1_primitive_is_reused_for_int8_contribution(self):
        """Use `e_base=0.04`, `mu=1.15`, and `x=0.5` to obtain `0.023`."""

        instance = make_error_instance(
            error_limit=0.05,
            compact_base_error=0.04,
        )
        state = make_state(routing={INT8_ROUTE: 0.5})

        with patch(
            "llm_resource_allocation.math.error_slo.compute_effective_error",
            wraps=compute_effective_error,
        ) as effective_error_primitive:
            result = result_for_query(
                compute_error_slo_residuals(instance, state), "q_short"
            )

        effective_error_primitive.assert_any_call(0.04, 1.15)
        self.assertAlmostEqual(compute_effective_error(0.04, 1.15), 0.046)
        self.assertAlmostEqual(result.weighted_error, 0.046 * 0.5)

    def test_int8_contribution_is_proportionally_above_fp16(self):
        """Check only the published `mu=1.15` versus `mu=1.0` error scaling."""

        instance = make_error_instance(
            error_limit=0.05,
            compact_base_error=0.04,
        )
        fp16 = result_for_query(
            compute_error_slo_residuals(
                instance, make_state(routing={LOW_ERROR_ROUTE: 0.5})
            ),
            "q_short",
        )
        int8 = result_for_query(
            compute_error_slo_residuals(
                instance, make_state(routing={INT8_ROUTE: 0.5})
            ),
            "q_short",
        )

        self.assertAlmostEqual(fp16.weighted_error, 0.02)
        self.assertAlmostEqual(int8.weighted_error, 0.023)
        self.assertAlmostEqual(int8.weighted_error / fp16.weighted_error, 1.15)


class ErrorSloBoundaryTests(unittest.TestCase):
    """Verify sparse data, independent diagnostics, and tolerance isolation."""

    def test_omitted_routes_are_zero_across_the_full_lattice(self):
        """Count only supplied known routes and make an empty state exactly zero."""

        instance = make_error_instance(error_limit=0.05)
        one_route = result_for_query(
            compute_error_slo_residuals(
                instance, make_state(routing={LOW_ERROR_ROUTE: 0.6})
            ),
            "q_short",
        )
        empty = result_for_query(
            compute_error_slo_residuals(instance, make_state(routing={})),
            "q_short",
        )

        self.assertAlmostEqual(one_route.weighted_error, 0.012)
        self.assertEqual(empty.weighted_error, 0.0)

    def test_unknown_route_and_missing_scientific_error_are_rejected(self):
        """Keep sparse decision zeros separate from scientific coefficient errors."""

        instance = make_error_instance(error_limit=0.05)
        unknown_state = make_state(
            routing={("q_unknown", "m_compact", "gpu_fast_fp16"): 0.0}
        )
        with self.assertRaises(DecisionDomainError):
            compute_error_slo_residuals(instance, unknown_state)

        incomplete_errors = dict(instance.base_error_rate)
        incomplete_errors.pop(("q_short", "m_compact"))
        invalid_instance = replace(instance, base_error_rate=incomplete_errors)
        with self.assertRaises(InstanceValidationError):
            compute_error_slo_residuals(invalid_instance, make_state(routing={}))

    def test_invalid_x_is_rejected_by_existing_domain_boundary(self):
        """Reuse M3D validation rather than clipping or redefining routing domains."""

        instance = make_error_instance(error_limit=0.05)
        with self.assertRaises(DecisionDomainError):
            compute_error_slo_residuals(
                instance, make_state(routing={LOW_ERROR_ROUTE: 1.000001})
            )

    def test_error_is_independent_of_demand_balance_and_unmet_fraction(self):
        """Do not normalize `x`, add `u`, or require Eq. (8b) feasibility."""

        instance = make_error_instance(error_limit=0.05)
        routing = {LOW_ERROR_ROUTE: 0.4, HIGH_ERROR_ROUTE: 0.4}
        without_unmet = make_state(routing=routing)
        with_unmet = make_state(routing=routing, unmet={"q_short": 0.1})

        first = result_for_query(
            compute_error_slo_residuals(instance, without_unmet), "q_short"
        )
        second = result_for_query(
            compute_error_slo_residuals(instance, with_unmet), "q_short"
        )
        balance = result_for_query(
            compute_demand_balance_residuals(instance, without_unmet), "q_short"
        )

        # The literal Eq. (8j) gives 0.04. Dividing by routed demand 0.8
        # would instead produce 0.05, which the paper does not print.
        self.assertAlmostEqual(first.weighted_error, 0.04)
        self.assertAlmostEqual(second.weighted_error, 0.04)
        self.assertAlmostEqual(balance.signed_residual, -0.2)

    def test_tolerance_changes_only_feasibility_classification(self):
        """Preserve raw `E` and slack when a tiny violation is tolerated."""

        instance = make_error_instance(
            error_limit=0.05,
            compact_base_error=0.1,
        )
        state = make_state(routing={LOW_ERROR_ROUTE: 0.500000005})
        result = result_for_query(
            compute_error_slo_residuals(instance, state), "q_short"
        )
        original_values = (result.weighted_error, result.slack)

        self.assertFalse(result.is_feasible(tolerance=0.0))
        self.assertTrue(result.is_feasible(tolerance=1e-9))
        self.assertEqual((result.weighted_error, result.slack), original_values)
        self.assertAlmostEqual(result.violation, 5e-10)

    def test_invalid_tolerance_is_rejected(self):
        """Require a finite non-negative real feasibility tolerance."""

        result = result_for_query(
            compute_error_slo_residuals(
                make_error_instance(error_limit=0.05),
                make_state(routing={}),
            ),
            "q_short",
        )
        for invalid in (-1e-9, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    result.is_feasible(tolerance=invalid)


if __name__ == "__main__":
    unittest.main()
