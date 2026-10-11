"""Tests for extracted McCormick auxiliaries and their full-lattice diagnostics."""

from types import MappingProxyType
from dataclasses import replace
from unittest.mock import patch
import unittest

from llm_resource_allocation.data import make_heterogeneous_tiny_instance
from llm_resource_allocation.math.routing import (
    DecisionDomainError,
    RoutingDecisionState,
)
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import (
    PlacementDecisionState,
    compute_mccormick_consistency_diagnostics,
    validate_placement_decision_domains,
)
from llm_resource_allocation.math.mccormick import compute_mccormick_residuals


ROUTE = ("q_short", "m_compact", "gpu_fast_fp16")
PAIR = ROUTE[1:]
SELECTOR = (*PAIR, 2, 2)
AUXILIARY = (*ROUTE, 2, 2)


def make_state(*, x=None, w=None, v=None) -> PlacementDecisionState:
    """Build one sparse aggregate without imposing constraint feasibility."""

    routing = RoutingDecisionState(
        routing_fractions={} if x is None else x,
        admission_indicators={},
        deployment_indicators={},
        unmet_fractions={},
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors={} if w is None else w,
        gpu_counts={},
    )
    return PlacementDecisionState(
        parallelism_decisions=parallelism,
        auxiliary_products={} if v is None else v,
    )


def diagnostic_for_key(diagnostics, key=AUXILIARY):
    """Select one semantic auxiliary diagnostic from deterministic output."""

    return next(
        diagnostic
        for diagnostic in diagnostics
        if (
            diagnostic.query_id,
            diagnostic.model_id,
            diagnostic.gpu_tier_id,
            diagnostic.tensor_parallel_degree,
            diagnostic.pipeline_parallel_depth,
        )
        == key
    )


class PlacementStateArchitectureTests(unittest.TestCase):
    """Check that the aggregate adds `v` without duplicating prior decisions."""

    def test_one_owner_for_routing_parallelism_and_deployment(self):
        """Forward the exact nested `x`, `w`, and `q` mappings by identity."""

        routing = RoutingDecisionState(
            routing_fractions={ROUTE: 0.6},
            admission_indicators={ROUTE: 1},
            deployment_indicators={PAIR: 1},
            unmet_fractions={"q_short": 0.4},
        )
        parallelism = ParallelismDecisionState(
            routing_decisions=routing,
            configuration_selectors={SELECTOR: 1},
            gpu_counts={PAIR: 4},
        )
        auxiliaries = {AUXILIARY: 0.6}
        state = PlacementDecisionState(parallelism, auxiliaries)
        auxiliaries[AUXILIARY] = 0.1

        self.assertIs(state.routing_decisions, routing)
        self.assertIs(state.routing_fractions, routing.routing_fractions)
        self.assertIs(
            state.configuration_selectors,
            parallelism.configuration_selectors,
        )
        self.assertIs(
            state.deployment_indicators,
            routing.deployment_indicators,
        )
        self.assertNotIn("routing_fractions", state.__dataclass_fields__)
        self.assertNotIn("configuration_selectors", state.__dataclass_fields__)
        self.assertNotIn("deployment_indicators", state.__dataclass_fields__)
        self.assertNotIn("processing_delays_seconds", state.__dataclass_fields__)
        self.assertIsInstance(state.auxiliary_products, MappingProxyType)
        self.assertEqual(state.auxiliary_products[AUXILIARY], 0.6)


class AuxiliaryDomainTests(unittest.TestCase):
    """Keep extracted `v` domains and semantic lattice membership exact."""

    def setUp(self):
        """Use the deterministic tier-specific 2x2x2 lattice."""

        self.instance = make_heterogeneous_tiny_instance()

    def test_closed_unit_interval_is_accepted_without_clipping(self):
        """Accept finite exact boundaries and an interior auxiliary value."""

        for value in (0, 0.6, 1, 0.0, 1.0):
            with self.subTest(value=value):
                state = make_state(v={AUXILIARY: value})
                validate_placement_decision_domains(self.instance, state)
                self.assertEqual(state.auxiliary_products[AUXILIARY], value)

    def test_out_of_range_nonfinite_boolean_and_nonreal_v_are_rejected(self):
        """Reject values tolerance must never clip into `[0,1]`."""

        for value in (
            -1e-12,
            1.0 + 1e-12,
            float("nan"),
            float("inf"),
            True,
            "0.6",
        ):
            with self.subTest(value=value):
                with self.assertRaises(DecisionDomainError):
                    validate_placement_decision_domains(
                        self.instance, make_state(v={AUXILIARY: value})
                    )

    def test_unknown_ids_and_tier_infeasible_degrees_are_rejected(self):
        """Require known IDs and the actual tier `N_k x M_k` lattice."""

        invalid_keys = (
            ("q_unknown", *AUXILIARY[1:]),
            (AUXILIARY[0], "m_unknown", *AUXILIARY[2:]),
            (*AUXILIARY[:2], "gpu_unknown", *AUXILIARY[3:]),
            (*ROUTE, 4, 2),
            (*ROUTE, 2, 4),
            (*ROUTE, 2.0, 2),
            (*ROUTE, True, 2),
        )
        for key in invalid_keys:
            with self.subTest(key=key):
                with self.assertRaises(DecisionDomainError):
                    validate_placement_decision_domains(
                        self.instance, make_state(v={key: 0.0})
                    )


class McCormickDecisionDiagnosticTests(unittest.TestCase):
    """Exercise the required binary cases on extracted auxiliary values."""

    def setUp(self):
        """Use one known route and tier-feasible `(TP,PP)=(2,2)` selector."""

        self.instance = make_heterogeneous_tiny_instance()

    def _evaluate(self, *, w, v):
        """Evaluate one key while keeping its routing fraction fixed at `0.6`."""

        state = make_state(
            x={ROUTE: 0.6},
            w={SELECTOR: w},
            v={AUXILIARY: v},
        )
        return diagnostic_for_key(
            compute_mccormick_consistency_diagnostics(self.instance, state)
        )

    def test_w_zero_requires_v_zero(self):
        """Check `w=0,x=0.6,v=0` feasible and `v=0.1` infeasible."""

        feasible = self._evaluate(w=0, v=0)
        violated = self._evaluate(w=0, v=0.1)

        self.assertTrue(feasible.is_feasible())
        self.assertFalse(violated.is_feasible())
        self.assertAlmostEqual(violated.residuals.w_minus_v, -0.1)

    def test_w_one_requires_v_equal_x(self):
        """Check `w=1,x=0.6,v=0.6` feasible and `v=0.5` infeasible."""

        feasible = self._evaluate(w=1, v=0.6)
        violated = self._evaluate(w=1, v=0.5)

        self.assertTrue(feasible.is_feasible())
        self.assertFalse(violated.is_feasible())
        self.assertAlmostEqual(
            violated.residuals.v_minus_x_minus_w_plus_one,
            -0.1,
        )

    def test_replacing_source_values_updates_every_derived_mccormick_view(self):
        original = self._evaluate(w=1, v=0.6)
        changed = replace(original, auxiliary_product=0.5)

        self.assertTrue(original.is_feasible())
        self.assertFalse(changed.is_feasible())
        self.assertAlmostEqual(changed.residuals.x_minus_v, 0.1)
        self.assertAlmostEqual(changed.residuals.w_minus_v, 0.5)
        self.assertAlmostEqual(changed.residuals.v_minus_x_minus_w_plus_one, -0.1)
        self.assertEqual(changed.residuals.v_nonnegative, 0.5)
        self.assertAlmostEqual(changed.minimum_residual, -0.1)

    def test_sparse_inactive_lattice_is_all_zero_and_feasible(self):
        """Interpret every omitted known `x`, `w`, and `v` as exact zero."""

        diagnostics = compute_mccormick_consistency_diagnostics(
            self.instance, make_state()
        )

        expected_count = len(self.instance.queries) * len(self.instance.models) * sum(
            len(tier.parallelism_configs) for tier in self.instance.gpu_tiers
        )
        self.assertEqual(len(diagnostics), expected_count)
        self.assertTrue(all(diagnostic.is_feasible() for diagnostic in diagnostics))
        self.assertTrue(
            all(
                diagnostic.routing_fraction
                == diagnostic.configuration_selector
                == diagnostic.auxiliary_product
                == 0
                for diagnostic in diagnostics
            )
        )

    def test_full_lattice_diagnostic_reuses_scalar_residual_primitive(self):
        """Retain semantic IDs while delegating every inequality calculation."""

        state = make_state(
            x={ROUTE: 0.6},
            w={SELECTOR: 1},
            v={AUXILIARY: 0.6},
        )
        with patch(
            "llm_resource_allocation.math.placement.compute_mccormick_residuals",
            wraps=compute_mccormick_residuals,
        ) as primitive:
            diagnostic = diagnostic_for_key(
                compute_mccormick_consistency_diagnostics(self.instance, state)
            )
            residuals = diagnostic.residuals

        primitive.assert_any_call(0.6, 1, 0.6)
        self.assertEqual(residuals.minimum_residual, 0.0)
        self.assertEqual(
            (diagnostic.query_id, diagnostic.model_id, diagnostic.gpu_tier_id),
            ROUTE,
        )

    def test_tolerance_classifies_residual_only_and_never_domain(self):
        """Preserve raw `v` while rejecting even tiny out-of-domain values."""

        raw_v = 0.6000000005
        diagnostic = self._evaluate(w=1, v=raw_v)

        self.assertFalse(diagnostic.is_feasible())
        self.assertTrue(diagnostic.is_feasible(tolerance=1e-9))
        self.assertEqual(diagnostic.auxiliary_product, raw_v)
        self.assertAlmostEqual(diagnostic.minimum_residual, -5e-10)

        with self.assertRaises(DecisionDomainError):
            compute_mccormick_consistency_diagnostics(
                self.instance,
                make_state(v={AUXILIARY: 1.0 + 5e-10}),
            )


if __name__ == "__main__":
    unittest.main()
