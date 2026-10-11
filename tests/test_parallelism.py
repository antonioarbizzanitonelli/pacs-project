"""Hand-calculated tests for Paper v2 Eqs. (8d)-(8e)."""

from dataclasses import replace
from types import MappingProxyType
import unittest

from llm_resource_allocation.data import make_heterogeneous_tiny_instance
from llm_resource_allocation.math.parallelism import (
    ConfigurationSelectionDiagnostic,
    GpuCountLinkingDiagnostic,
    ParallelismDecisionState,
    compute_configuration_selection_residuals,
    compute_gpu_count_linking_residuals,
    validate_parallelism_decision_domains,
)
from llm_resource_allocation.math.routing import (
    DecisionDomainError,
    RoutingDecisionState,
)


PAIR = ("m_compact", "gpu_fast_fp16")
W_2_2 = (*PAIR, 2, 2)
W_2_4 = (*PAIR, 2, 4)
W_4_1 = (*PAIR, 4, 1)


def make_parallelism_instance():
    """Extend one synthetic tier with the configurations needed for hand cases."""

    instance = make_heterogeneous_tiny_instance()
    tiers = tuple(
        replace(
            tier,
            tensor_parallel_degrees=(1, 2, 4),
            pipeline_parallel_depths=(1, 2, 4),
        )
        if tier.id == "gpu_fast_fp16"
        else tier
        for tier in instance.gpu_tiers
    )
    return replace(instance, gpu_tiers=tiers)


def make_state(*, q=None, w=None, y=None) -> ParallelismDecisionState:
    """Build a sparse state whose deployment mapping has one existing owner."""

    routing = RoutingDecisionState(
        routing_fractions={},
        admission_indicators={},
        deployment_indicators={} if q is None else q,
        unmet_fractions={},
    )
    return ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors={} if w is None else w,
        gpu_counts={} if y is None else y,
    )


def result_for_pair(results, pair=PAIR):
    """Select one model/tier diagnostic from deterministic full-lattice output."""

    return next(
        result
        for result in results
        if (result.model_id, result.gpu_tier_id) == pair
    )


class ConfigurationSelectionTests(unittest.TestCase):
    """Verify Eq. (8d) for inactive, active, absent, and multiple selectors."""

    def setUp(self):
        """Use a valid synthetic lattice containing `(2,2)` and `(4,1)`."""

        self.instance = make_parallelism_instance()

    def _evaluate(self, *, q, w):
        """Return the canonical pair's Eq. (8d) diagnostic."""

        return result_for_pair(
            compute_configuration_selection_residuals(
                self.instance, make_state(q={PAIR: q}, w=w)
            )
        )

    def test_inactive_pair_with_no_configuration_is_feasible(self):
        """Check `sum w=0`, `q=0`, and residual zero."""

        result = self._evaluate(q=0, w={})
        self.assertEqual(result.selected_configuration_count, 0)
        self.assertEqual(result.deployment_indicator, 0)
        self.assertEqual(result.signed_residual, 0)
        self.assertTrue(result.is_feasible())

    def test_active_pair_with_exactly_one_configuration_is_feasible(self):
        """Check active `(TP,PP)=(2,2)` gives `sum w-q=1-1=0`."""

        result = self._evaluate(q=1, w={W_2_2: 1})
        self.assertEqual(result.selected_configuration_count, 1)
        self.assertEqual(result.signed_residual, 0)
        self.assertTrue(result.is_feasible())

    def test_active_pair_without_configuration_has_negative_residual(self):
        """Check `sum w-q=0-1=-1` without hiding the missing choice."""

        result = self._evaluate(q=1, w={})
        self.assertEqual(result.signed_residual, -1)
        self.assertEqual(result.absolute_violation, 1)
        self.assertFalse(result.is_feasible())

    def test_active_pair_with_two_configurations_has_positive_residual(self):
        """Check selectors `(2,2)` and `(4,1)` give residual `2-1=+1`."""

        result = self._evaluate(q=1, w={W_2_2: 1, W_4_1: 1})
        self.assertEqual(result.selected_configuration_count, 2)
        self.assertEqual(result.signed_residual, 1)
        self.assertFalse(result.is_feasible())

    def test_replacing_base_value_recomputes_signed_residual(self):
        result = self._evaluate(q=1, w={W_2_2: 1})
        changed = replace(result, selected_configuration_count=2)
        self.assertEqual(changed.signed_residual, 1)
        self.assertEqual(result.signed_residual, 0)

    def test_inactive_pair_with_configuration_has_positive_residual(self):
        """Check an inactive pair cannot select `(2,2)`: residual `1-0=+1`."""

        result = self._evaluate(q=0, w={W_2_2: 1})
        self.assertEqual(result.signed_residual, 1)
        self.assertFalse(result.is_feasible())


class GpuCountLinkingTests(unittest.TestCase):
    """Verify Eq. (8e) and its independence from Eq. (8d)."""

    def setUp(self):
        """Use a tier whose feasible lattice contains all required products."""

        self.instance = make_parallelism_instance()

    def _evaluate(self, *, w, y):
        """Return the canonical pair's Eq. (8e) diagnostic."""

        return result_for_pair(
            compute_gpu_count_linking_residuals(
                self.instance, make_state(w=w, y={PAIR: y})
            )
        )

    def test_two_by_four_implies_eight_and_y_seven_is_short_by_one(self):
        """Check `(2,4)` gives 8 GPUs, with signed residual `7-8=-1`."""

        feasible = self._evaluate(w={W_2_4: 1}, y=8)
        short = self._evaluate(w={W_2_4: 1}, y=7)

        self.assertEqual(feasible.implied_gpu_count, 8)
        self.assertEqual(feasible.allocated_gpu_count, 8)
        self.assertEqual(feasible.signed_residual, 0)
        self.assertTrue(feasible.is_feasible())
        self.assertEqual(short.signed_residual, -1)
        self.assertEqual(short.absolute_violation, 1)
        self.assertFalse(short.is_feasible())

    def test_four_by_one_implies_four(self):
        """Check `(4,1)` contributes the literal product `4*1=4`."""

        result = self._evaluate(w={W_4_1: 1}, y=4)
        self.assertEqual(result.implied_gpu_count, 4)
        self.assertTrue(result.is_feasible())

    def test_inactive_sparse_pair_has_zero_count_link(self):
        """Check omitted `w` and `y` both mean exact zero."""

        result = result_for_pair(
            compute_gpu_count_linking_residuals(self.instance, make_state())
        )
        self.assertEqual(result.allocated_gpu_count, 0)
        self.assertEqual(result.implied_gpu_count, 0)
        self.assertTrue(result.is_feasible())

    def test_multiple_selectors_are_still_summed_independently(self):
        """Evaluate Eq. (8e) even though the same state violates Eq. (8d)."""

        state = make_state(
            q={PAIR: 1},
            w={W_2_2: 1, W_4_1: 1},
            y={PAIR: 8},
        )
        selection = result_for_pair(
            compute_configuration_selection_residuals(self.instance, state)
        )
        gpu_link = result_for_pair(
            compute_gpu_count_linking_residuals(self.instance, state)
        )

        self.assertEqual(selection.signed_residual, 1)
        self.assertFalse(selection.is_feasible())
        self.assertEqual(gpu_link.implied_gpu_count, 4 + 4)
        self.assertEqual(gpu_link.signed_residual, 0)
        self.assertTrue(gpu_link.is_feasible())


class ParallelismDecisionDomainTests(unittest.TestCase):
    """Keep `w`, `y`, and shared `q` domains exact and sparse keys auditable."""

    def setUp(self):
        """Use the extended deterministic parallelism lattice."""

        self.instance = make_parallelism_instance()

    def test_exact_binary_w_values_are_accepted(self):
        """Accept integer and exact solver-style float zero/one values."""

        for value in (0, 1, 0.0, 1.0):
            with self.subTest(value=value):
                validate_parallelism_decision_domains(
                    self.instance, make_state(w={W_2_2: value})
                )

    def test_fractional_nonfinite_and_boolean_w_values_are_rejected(self):
        """Reject values that tolerance must never round into a binary selector."""

        for value in (-1, 2, 0.5, 0.9999999999, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(DecisionDomainError):
                    validate_parallelism_decision_domains(
                        self.instance, make_state(w={W_2_2: value})
                    )

    def test_nonnegative_integer_y_boundaries_are_accepted(self):
        """Accept `y=0,1,8` and their exact float representations."""

        for value in (0, 1, 8, 0.0, 1.0, 8.0):
            with self.subTest(value=value):
                validate_parallelism_decision_domains(
                    self.instance, make_state(y={PAIR: value})
                )

    def test_negative_fractional_nonfinite_and_boolean_y_are_rejected(self):
        """Reject `-1`, `1.5`, infinities, NaN, and booleans as GPU counts."""

        for value in (-1, 1.5, float("inf"), float("nan"), True, False):
            with self.subTest(value=value):
                with self.assertRaises(DecisionDomainError):
                    validate_parallelism_decision_domains(
                        self.instance, make_state(y={PAIR: value})
                    )

    def test_fractional_q_uses_the_existing_exact_routing_authority(self):
        """Reject fractional deployment without introducing a second q policy."""

        with self.assertRaises(DecisionDomainError):
            validate_parallelism_decision_domains(
                self.instance, make_state(q={PAIR: 0.9999999999})
            )

    def test_tolerance_never_repairs_fractional_w_or_y(self):
        """Keep residual tolerance separate from exact integrality domains."""

        invalid_states = (
            make_state(w={W_2_2: 1 - 5e-10}),
            make_state(y={PAIR: 8 - 5e-10}),
        )
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    compute_gpu_count_linking_residuals(self.instance, state)

    def test_tolerance_classifies_only_tiny_equality_residuals(self):
        """Permit solver-like residual noise while preserving raw diagnostics."""

        selection = ConfigurationSelectionDiagnostic(
            "m", "k", 0.9999999995, 1.0
        )
        gpu_link = GpuCountLinkingDiagnostic("m", "k", 7.9999999995, 8.0)

        for result in (selection, gpu_link):
            with self.subTest(result=result):
                self.assertFalse(result.is_feasible())
                self.assertTrue(result.is_feasible(tolerance=1e-9))
                self.assertAlmostEqual(result.signed_residual, -5e-10, delta=1e-15)

    def test_unknown_ids_and_infeasible_tier_degrees_are_rejected(self):
        """Reject keys outside both semantic IDs and the tier-specific lattice."""

        invalid_states = (
            make_state(w={("m_unknown", PAIR[1], 2, 2): 0}),
            make_state(w={(PAIR[0], "gpu_unknown", 2, 2): 0}),
            make_state(w={(*PAIR, 8, 2): 0}),
            make_state(w={(*PAIR, 2, 8): 0}),
            make_state(w={(*PAIR, 1.0, 2): 0}),
            make_state(w={(*PAIR, True, 2): 0}),
            make_state(y={("m_unknown", PAIR[1]): 0}),
            make_state(y={(PAIR[0], "gpu_unknown"): 0}),
        )
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    validate_parallelism_decision_domains(self.instance, state)

    def test_sparse_missing_w_y_q_values_are_all_zero(self):
        """Evaluate every known pair as zero without accepting unknown keys."""

        state = make_state()
        selections = compute_configuration_selection_residuals(self.instance, state)
        gpu_links = compute_gpu_count_linking_residuals(self.instance, state)

        self.assertTrue(all(result.signed_residual == 0 for result in selections))
        self.assertTrue(all(result.signed_residual == 0 for result in gpu_links))
        self.assertTrue(all(result.is_feasible() for result in selections))
        self.assertTrue(all(result.is_feasible() for result in gpu_links))

    def test_q_has_one_owner_and_new_mappings_are_immutable_copies(self):
        """Prevent q duplication and caller mutation of w/y diagnostic inputs."""

        q = {PAIR: 1}
        w = {W_2_2: 1}
        y = {PAIR: 4}
        state = make_state(q=q, w=w, y=y)
        w[W_2_2] = 0
        y[PAIR] = 99
        q[PAIR] = 0

        self.assertIs(
            state.deployment_indicators,
            state.routing_decisions.deployment_indicators,
        )
        self.assertNotIn("deployment_indicators", state.__dataclass_fields__)
        self.assertIsInstance(state.configuration_selectors, MappingProxyType)
        self.assertIsInstance(state.gpu_counts, MappingProxyType)
        self.assertEqual(state.configuration_selectors[W_2_2], 1)
        self.assertEqual(state.gpu_counts[PAIR], 4)
        self.assertEqual(state.deployment_indicators[PAIR], 1)

    def test_invalid_tolerance_is_rejected(self):
        """Require finite non-negative real equality tolerances."""

        result = ConfigurationSelectionDiagnostic("m", "k", 0, 0)
        for invalid in (-1e-9, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    result.is_feasible(invalid)


if __name__ == "__main__":
    unittest.main()
