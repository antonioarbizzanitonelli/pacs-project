"""Tests for independent demand-balance and routing-consistency validation."""

from dataclasses import replace
import unittest

from llm_resource_allocation.data import make_heterogeneous_tiny_instance
from llm_resource_allocation.domain import InstanceValidationError
from llm_resource_allocation.math.routing import (
    DecisionDomainError,
    RoutingDecisionState,
    compute_demand_balance_residuals,
    compute_routing_consistency_residuals,
    validate_routing_decision_domains,
)


ROUTE_A = ("q_short", "m_compact", "gpu_budget_int8")
ROUTE_B = ("q_short", "m_accurate", "gpu_fast_fp16")
PAIR_A = ("m_compact", "gpu_budget_int8")


def make_state(
    *,
    x=None,
    z=None,
    q=None,
    u=None,
) -> RoutingDecisionState:
    """Build a compact sparse state for hand-worked validation cases."""

    return RoutingDecisionState(
        routing_fractions={} if x is None else x,
        admission_indicators={} if z is None else z,
        deployment_indicators={} if q is None else q,
        unmet_fractions={} if u is None else u,
    )


def result_for_query(results, query_id):
    """Return one query diagnostic from deterministic residual output."""

    return next(result for result in results if result.query_id == query_id)


def result_for_route(results, route):
    """Return one `(i,j,k)` diagnostic from deterministic residual output."""

    return next(
        result
        for result in results
        if (result.query_id, result.model_id, result.gpu_tier_id) == route
    )


class DemandBalanceTests(unittest.TestCase):
    """Exercise Paper v2 Eq. (8b) / v1 Eq. (5b) by hand."""

    def setUp(self):
        """Use the deterministic 2x2x2 structural fixture."""

        self.instance = make_heterogeneous_tiny_instance()

    def test_exact_balance(self):
        """Check the required `0.6 + 0.3 + 0.1 = 1.0` case."""

        state = make_state(x={ROUTE_A: 0.6, ROUTE_B: 0.3}, u={"q_short": 0.1})
        result = result_for_query(
            compute_demand_balance_residuals(self.instance, state), "q_short"
        )

        self.assertAlmostEqual(result.routed_fraction, 0.9)
        self.assertEqual(result.unmet_fraction, 0.1)
        self.assertEqual(result.lhs, 1.0)
        self.assertEqual(result.signed_residual, 0.0)
        self.assertEqual(result.absolute_violation, 0.0)
        self.assertTrue(result.is_feasible())

    def test_under_allocation_preserves_negative_residual(self):
        """Check `0.4 + 0.3 + 0.1 = 0.8` and expose residual `-0.2`."""

        state = make_state(x={ROUTE_A: 0.4, ROUTE_B: 0.3}, u={"q_short": 0.1})
        result = result_for_query(
            compute_demand_balance_residuals(self.instance, state), "q_short"
        )

        self.assertAlmostEqual(result.lhs, 0.8)
        self.assertAlmostEqual(result.signed_residual, -0.2)
        self.assertAlmostEqual(result.absolute_violation, 0.2)
        self.assertFalse(result.is_feasible())

    def test_replacing_lhs_recomputes_signed_residual(self):
        result = result_for_query(
            compute_demand_balance_residuals(self.instance, make_state()),
            "q_short",
        )
        changed = replace(result, lhs=0.75)
        self.assertEqual(changed.signed_residual, -0.25)
        self.assertEqual(result.signed_residual, -1.0)

    def test_over_allocation_preserves_positive_residual(self):
        """Check `0.6 + 0.5 = 1.1` and expose residual `+0.1`."""

        state = make_state(x={ROUTE_A: 0.6, ROUTE_B: 0.5})
        result = result_for_query(
            compute_demand_balance_residuals(self.instance, state), "q_short"
        )

        self.assertAlmostEqual(result.lhs, 1.1)
        self.assertAlmostEqual(result.signed_residual, 0.1)
        self.assertAlmostEqual(result.absolute_violation, 0.1)
        self.assertFalse(result.is_feasible())

    def test_tolerance_applies_only_to_equality_feasibility(self):
        """Allow a tiny residual without changing or repairing decision values."""

        state = make_state(x={ROUTE_A: 0.9000000005}, u={"q_short": 0.1})
        result = result_for_query(
            compute_demand_balance_residuals(self.instance, state), "q_short"
        )

        self.assertFalse(result.is_feasible())
        self.assertTrue(result.is_feasible(tolerance=1e-9))
        self.assertEqual(result.routed_fraction, 0.9000000005)


class RoutingConsistencyTests(unittest.TestCase):
    """Exercise the three named slacks of Paper v2 Eq. (8k)."""

    def setUp(self):
        """Use the deterministic 2x2x2 structural fixture."""

        self.instance = make_heterogeneous_tiny_instance()

    def _evaluate(self, *, x, z, q):
        """Evaluate the canonical route while all omitted lattice entries are zero."""

        state = make_state(x={ROUTE_A: x}, z={ROUTE_A: z}, q={PAIR_A: q})
        return result_for_route(
            compute_routing_consistency_residuals(self.instance, state), ROUTE_A
        )

    def test_valid_admitted_route(self):
        """Check `x=0.4, z=1, q=1` and all named non-negative slacks."""

        result = self._evaluate(x=0.4, z=1, q=1)

        self.assertEqual(result.routing_nonnegative_slack, 0.4)
        self.assertEqual(result.admission_slack, 0.6)
        self.assertEqual(result.deployment_slack, 0)
        self.assertTrue(result.is_feasible())

    def test_valid_deployed_but_not_admitted_route(self):
        """Check the valid boundary `x=0, z=0, q=1`."""

        result = self._evaluate(x=0, z=0, q=1)

        self.assertEqual(result.routing_nonnegative_slack, 0)
        self.assertEqual(result.admission_slack, 0)
        self.assertEqual(result.deployment_slack, 1)
        self.assertTrue(result.is_feasible())

    def test_valid_all_zero_route(self):
        """Check the undeployed valid boundary `x=z=q=0`."""

        self.assertTrue(self._evaluate(x=0, z=0, q=0).is_feasible())

    def test_routing_without_admission_fails_named_link(self):
        """Check `x=0.2, z=0, q=1` fails only `x <= z` for this triple."""

        result = self._evaluate(x=0.2, z=0, q=1)

        self.assertEqual(result.admission_slack, -0.2)
        self.assertEqual(result.deployment_slack, 1)
        self.assertFalse(result.is_feasible())

    def test_admission_without_deployment_fails_named_link(self):
        """Check `x=0, z=1, q=0` fails `z <= q`."""

        result = self._evaluate(x=0, z=1, q=0)

        self.assertEqual(result.admission_slack, 1)
        self.assertEqual(result.deployment_slack, -1)
        self.assertFalse(result.is_feasible())

    def test_chain_implies_zero_downstream_values(self):
        """Show that `q=0 => z=0,x=0` and `z=0 => x=0` are enforced."""

        self.assertFalse(self._evaluate(x=0, z=1, q=0).is_feasible())
        self.assertFalse(self._evaluate(x=0.2, z=0, q=1).is_feasible())
        self.assertTrue(self._evaluate(x=0, z=0, q=0).is_feasible())


class RoutingDecisionDomainTests(unittest.TestCase):
    """Keep decision domains exact and sparse-index semantics explicit."""

    def setUp(self):
        """Use the deterministic 2x2x2 structural fixture."""

        self.instance = make_heterogeneous_tiny_instance()

    def test_routing_fraction_outside_domain_is_rejected(self):
        """Reject negative and above-one `x` values before Eq. (8k) tolerance."""

        for value in (-0.001, 1.001):
            with self.subTest(value=value):
                with self.assertRaises(DecisionDomainError):
                    compute_routing_consistency_residuals(
                        self.instance, make_state(x={ROUTE_A: value})
                    )

    def test_nonbinary_indicators_are_rejected_without_rounding(self):
        """Reject near-binary `z` and fractional `q` exactly as supplied."""

        states = (
            make_state(z={ROUTE_A: 0.999999}),
            make_state(q={PAIR_A: 0.5}),
        )
        for state in states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    validate_routing_decision_domains(self.instance, state)

    def test_nonfinite_and_boolean_values_are_rejected(self):
        """Reject pseudo-numeric values that cannot be scientific decisions."""

        states = (
            make_state(x={ROUTE_A: float("nan")}),
            make_state(x={ROUTE_A: float("inf")}),
            make_state(z={ROUTE_A: True}),
            make_state(q={PAIR_A: False}),
        )
        for state in states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    validate_routing_decision_domains(self.instance, state)

    def test_strict_two_percent_unmet_cap_boundaries(self):
        """Accept `u=0,0.01,0.02` and reject values outside `[0,0.02]`."""

        strict_queries = tuple(
            replace(query, max_unmet_fraction=0.02)
            if query.id == "q_short"
            else query
            for query in self.instance.queries
        )
        strict_instance = replace(self.instance, queries=strict_queries)

        for value in (0, 0.01, 0.02):
            with self.subTest(valid=value):
                validate_routing_decision_domains(
                    strict_instance, make_state(u={"q_short": value})
                )
        for value in (-0.001, 0.020001):
            with self.subTest(invalid=value):
                with self.assertRaises(DecisionDomainError):
                    validate_routing_decision_domains(
                        strict_instance, make_state(u={"q_short": value})
                    )

    def test_sparse_missing_entries_are_exact_zero(self):
        """Interpret omitted decision entries as zero across the full lattice."""

        state = make_state()
        demand = compute_demand_balance_residuals(self.instance, state)
        routing = compute_routing_consistency_residuals(self.instance, state)

        self.assertTrue(all(result.routed_fraction == 0 for result in demand))
        self.assertTrue(all(result.unmet_fraction == 0 for result in demand))
        self.assertTrue(all(result.signed_residual == -1 for result in demand))
        self.assertTrue(all(result.is_feasible() for result in routing))

    def test_unknown_semantic_ids_are_rejected_in_every_mapping(self):
        """Prevent sparse storage from silently discarding out-of-instance keys."""

        states = (
            make_state(x={("q_unknown", "m_compact", "gpu_budget_int8"): 0}),
            make_state(z={("q_short", "m_unknown", "gpu_budget_int8"): 0}),
            make_state(q={("m_compact", "gpu_unknown"): 0}),
            make_state(u={"q_unknown": 0}),
        )
        for state in states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    validate_routing_decision_domains(self.instance, state)

    def test_missing_scientific_coefficient_remains_instance_error(self):
        """Never apply sparse-decision zero semantics to scientific tensors."""

        incomplete = dict(self.instance.compute_delay_seconds_per_token)
        incomplete.pop(ROUTE_A)
        invalid_instance = replace(
            self.instance, compute_delay_seconds_per_token=incomplete
        )

        with self.assertRaises(InstanceValidationError):
            compute_demand_balance_residuals(invalid_instance, make_state())

    def test_state_copies_caller_mappings(self):
        """Keep diagnostic input stable when a caller later mutates its dictionary."""

        routing = {ROUTE_A: 0.4}
        state = make_state(x=routing)
        routing[ROUTE_A] = 0.8

        self.assertEqual(state.routing_fractions[ROUTE_A], 0.4)


if __name__ == "__main__":
    unittest.main()
