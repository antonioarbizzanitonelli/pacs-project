"""Hand-derived and isolation tests for Paper v2 Eq. (12), mechanism M3."""

import unittest

from placement_support import snapshot_value
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    M3EvaluationState,
    M3NoUpgradeReason,
    select_m3_parallelism_upgrade,
)
from llm_resource_allocation.algorithms import m3 as m3_module
from llm_resource_allocation.data.tiny_instances import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig
from llm_resource_allocation.math.latency import compute_configuration_delay_seconds


QUERY_ID = "q_text"
MODEL_ID = "m_small"
TIER_ID = "gpu_basic_fp16"


def make_m3_instance(
    *,
    delay_slo_seconds: float = 0.2,
    hourly_rental_usd: float = 3.0,
    horizon_hours: float = 2.0,
    compute_delay: float = 0.10,
    communication_delay: float = 0.01,
    tensor_degrees: tuple[int, ...] = (1, 2, 4),
    pipeline_depths: tuple[int, ...] = (1,),
):
    """Return a transparent valid 1x1x1 fixture for M3 arithmetic."""

    base = make_minimal_tiny_instance()
    query = replace(base.queries[0], delay_slo_seconds=delay_slo_seconds)
    tier = replace(
        base.gpu_tiers[0],
        hourly_rental_usd=hourly_rental_usd,
        tensor_parallel_degrees=tensor_degrees,
        pipeline_parallel_depths=pipeline_depths,
    )
    key = (QUERY_ID, MODEL_ID, TIER_ID)
    return replace(
        base,
        queries=(query,),
        gpu_tiers=(tier,),
        compute_delay_seconds_per_token={key: compute_delay},
        communication_delay_seconds_per_token={key: communication_delay},
        horizon_hours=horizon_hours,
    )


def select_upgrade(instance, state):
    """Evaluate the canonical query/model/tier M3 case."""

    return select_m3_parallelism_upgrade(
        instance,
        query_id=QUERY_ID,
        model_id=MODEL_ID,
        tier_id=TIER_ID,
        state=state,
    )


def diagnostic_for(result, configuration):
    """Return one expected tier-valid diagnostic for concise assertions."""

    return next(
        diagnostic
        for diagnostic in result.configuration_diagnostics
        if diagnostic.candidate_configuration == configuration
    )


class M3HandDerivedSelectionTests(unittest.TestCase):
    """Verify applicability, exact conditions, and minimum-GPU selection."""

    def test_hand_baseline_selects_exact_budget_boundary_upgrade(self):
        """Current D=.31>.2; TP2 D=.16 and extra rental is 2*3*(2-1)=6."""

        instance = make_m3_instance()
        state = M3EvaluationState(ParallelismConfig(1, 1), 6.0)

        result = select_upgrade(instance, state)
        selected = result.selected_diagnostic

        self.assertTrue(result.is_applicable)
        self.assertTrue(result.has_upgrade)
        self.assertAlmostEqual(result.current_delay_seconds, 0.31)
        self.assertEqual(result.selected_configuration, ParallelismConfig(2, 1))
        self.assertEqual(selected.current_gpu_count, 1)
        self.assertEqual(selected.candidate_gpu_count, 2)
        self.assertEqual(selected.incremental_gpu_count, 1)
        self.assertEqual(selected.incremental_gpu_cost_usd, 6.0)
        self.assertAlmostEqual(selected.candidate_delay_seconds, 0.16)
        self.assertTrue(selected.delay_condition_passes)
        self.assertTrue(selected.budget_condition_passes)
        self.assertEqual(result.no_upgrade_reasons, ())

    def test_producer_retains_the_selected_upgrade_diagnostic(self):
        for budget, expected in ((6.0, ParallelismConfig(2, 1)), (5.99, None)):
            with self.subTest(budget=budget):
                result = select_upgrade(
                    make_m3_instance(),
                    M3EvaluationState(ParallelismConfig(1, 1), budget),
                )
                self.assertEqual(result.selected_configuration, expected)
                matching = tuple(
                    diagnostic
                    for diagnostic in result.configuration_diagnostics
                    if diagnostic.candidate_configuration == expected
                )
                self.assertEqual(len(matching), int(expected is not None))
                if matching:
                    self.assertIs(result.selected_diagnostic, matching[0])
                    self.assertTrue(matching[0].is_feasible_upgrade)
                else:
                    self.assertIsNone(result.selected_diagnostic)

    def test_budget_blocked_delay_feasible_upgrade_is_not_selected(self):
        instance = make_m3_instance()
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 5.99),
        )
        tp2 = diagnostic_for(result, ParallelismConfig(2, 1))

        self.assertTrue(tp2.delay_condition_passes)
        self.assertEqual(tp2.incremental_gpu_cost_usd, 6.0)
        self.assertFalse(tp2.budget_condition_passes)
        self.assertFalse(result.has_upgrade)
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.INSUFFICIENT_REMAINING_BUDGET,),
        )

    def test_multiple_feasible_upgrades_choose_minimum_gpu_count(self):
        instance = make_m3_instance()
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 100.0),
        )

        tp2 = diagnostic_for(result, ParallelismConfig(2, 1))
        tp4 = diagnostic_for(result, ParallelismConfig(4, 1))
        self.assertTrue(tp2.is_feasible_upgrade)
        self.assertTrue(tp4.is_feasible_upgrade)
        self.assertEqual(tp2.candidate_gpu_count, 2)
        self.assertEqual(tp4.candidate_gpu_count, 4)
        self.assertEqual(result.selected_configuration, ParallelismConfig(2, 1))

    def test_no_delay_feasible_upgrade_has_specific_reason(self):
        instance = make_m3_instance(delay_slo_seconds=0.05)
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 100.0),
        )

        self.assertTrue(result.is_applicable)
        self.assertFalse(result.has_upgrade)
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.NO_DELAY_FEASIBLE_CONFIGURATION,),
        )


class M3ApplicabilityTests(unittest.TestCase):
    """Protect active-pair and current-delay preconditions."""

    def test_inactive_pair_is_explicitly_non_applicable(self):
        instance = make_m3_instance()
        result = select_upgrade(instance, M3EvaluationState(None, 100.0))

        self.assertFalse(result.is_applicable)
        self.assertFalse(result.has_upgrade)
        self.assertEqual(result.current_gpu_count, 0)
        self.assertIsNone(result.current_delay_seconds)
        self.assertEqual(result.configuration_diagnostics, ())
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.INACTIVE_PAIR,),
        )

    def test_current_configuration_already_meets_slo(self):
        instance = make_m3_instance(delay_slo_seconds=0.5)
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 100.0),
        )

        self.assertFalse(result.is_applicable)
        self.assertFalse(result.has_upgrade)
        self.assertEqual(result.configuration_diagnostics, ())
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.CURRENT_DELAY_WITHIN_SLO,),
        )

    def test_current_delay_exact_slo_boundary_is_not_applicable(self):
        # D_cfg = .0625*3 + .0625*1 = .25 exactly in binary.
        instance = make_m3_instance(
            delay_slo_seconds=0.25,
            compute_delay=0.0625,
            communication_delay=0.0625,
        )
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 100.0),
        )

        self.assertEqual(result.current_delay_seconds, 0.25)
        self.assertFalse(result.is_applicable)
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.CURRENT_DELAY_WITHIN_SLO,),
        )

    def test_already_maximal_configuration_has_no_upgrade(self):
        instance = make_m3_instance(delay_slo_seconds=0.05)
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(4, 1), 100.0),
        )

        self.assertTrue(result.is_applicable)
        self.assertFalse(result.has_upgrade)
        self.assertFalse(
            any(
                diagnostic.higher_gpu_count_passes
                for diagnostic in result.configuration_diagnostics
            )
        )
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.NO_HIGHER_GPU_CONFIGURATION,),
        )


class M3ConfigurationAuthorityAndTieTests(unittest.TestCase):
    """Protect tier authority, product eligibility, and stable ties."""

    def test_unsupported_intermediate_configuration_is_never_diagnosed(self):
        instance = make_m3_instance(tensor_degrees=(1, 4))
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 100.0),
        )

        considered = tuple(
            diagnostic.candidate_configuration
            for diagnostic in result.configuration_diagnostics
        )
        self.assertEqual(
            considered,
            (ParallelismConfig(1, 1), ParallelismConfig(4, 1)),
        )
        self.assertNotIn(ParallelismConfig(2, 1), considered)
        self.assertEqual(result.selected_configuration, ParallelismConfig(4, 1))

    def test_upgrade_uses_gpu_product_not_tp_only(self):
        instance = make_m3_instance(
            delay_slo_seconds=2.0,
            compute_delay=1.0,
            communication_delay=0.10,
            tensor_degrees=(1, 2, 3, 6),
            pipeline_depths=(4, 3, 2, 1),
            hourly_rental_usd=1.0,
            horizon_hours=1.0,
        )
        current = ParallelismConfig(1, 4)
        result = select_upgrade(instance, M3EvaluationState(current, 100.0))

        # (2,3) reduces TP/PP shape while increasing product 4 -> 6.
        self.assertEqual(result.selected_configuration, ParallelismConfig(2, 3))
        self.assertEqual(result.selected_diagnostic.incremental_gpu_count, 2)
        self.assertAlmostEqual(result.selected_diagnostic.candidate_delay_seconds, 1.8)

    def test_equal_minimum_gpu_tie_preserves_tier_semantic_order(self):
        common = dict(
            delay_slo_seconds=2.0,
            compute_delay=1.0,
            communication_delay=0.10,
            pipeline_depths=(4, 3, 2, 1),
            hourly_rental_usd=1.0,
            horizon_hours=1.0,
        )
        first_2x3 = make_m3_instance(
            tensor_degrees=(1, 2, 3, 6),
            **common,
        )
        first_6x1 = make_m3_instance(
            tensor_degrees=(6, 3, 2, 1),
            **common,
        )
        state = M3EvaluationState(ParallelismConfig(1, 4), 100.0)

        forward = select_upgrade(first_2x3, state)
        reversed_order = select_upgrade(first_6x1, state)

        self.assertEqual(forward.selected_configuration, ParallelismConfig(2, 3))
        self.assertEqual(
            reversed_order.selected_configuration,
            ParallelismConfig(6, 1),
        )
        self.assertEqual(forward.selected_diagnostic.candidate_gpu_count, 6)
        self.assertEqual(reversed_order.selected_diagnostic.candidate_gpu_count, 6)


class M3BudgetAndIncrementalCostTests(unittest.TestCase):
    """Verify positive-part arithmetic and zero/negative remaining budgets."""

    def test_same_configuration_has_less_equal_and_greater_current_counts(self):
        instance = make_m3_instance(
            delay_slo_seconds=0.1,
            tensor_degrees=(1, 2),
            pipeline_depths=(1, 2, 4),
        )
        target = ParallelismConfig(2, 1)
        results = tuple(
            select_upgrade(
                instance,
                M3EvaluationState(current, 100.0),
            )
            for current in (
                ParallelismConfig(1, 1),
                ParallelismConfig(1, 2),
                ParallelismConfig(1, 4),
            )
        )
        diagnostics = tuple(diagnostic_for(result, target) for result in results)

        self.assertEqual(
            tuple(item.current_gpu_count for item in diagnostics),
            (1, 2, 4),
        )
        self.assertEqual(
            tuple(item.incremental_gpu_count for item in diagnostics),
            (1, 0, 0),
        )
        self.assertEqual(
            tuple(item.incremental_gpu_cost_usd for item in diagnostics),
            (6.0, 0.0, 0.0),
        )
        self.assertEqual(
            tuple(item.higher_gpu_count_passes for item in diagnostics),
            (True, False, False),
        )

    def test_zero_budget_blocks_positive_cost_upgrade(self):
        instance = make_m3_instance()
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 0.0),
        )

        self.assertFalse(result.has_upgrade)
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.INSUFFICIENT_REMAINING_BUDGET,),
        )

    def test_zero_cost_upgrade_is_allowed_at_zero_budget(self):
        instance = make_m3_instance(hourly_rental_usd=0.0)
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), 0.0),
        )

        self.assertTrue(result.has_upgrade)
        self.assertEqual(result.selected_diagnostic.incremental_gpu_cost_usd, 0.0)
        self.assertTrue(result.selected_diagnostic.budget_condition_passes)

    def test_negative_remaining_budget_blocks_even_zero_cost_upgrade(self):
        instance = make_m3_instance(hourly_rental_usd=0.0)
        result = select_upgrade(
            instance,
            M3EvaluationState(ParallelismConfig(1, 1), -0.01),
        )

        self.assertFalse(result.has_upgrade)
        self.assertEqual(result.selected_diagnostic, None)
        self.assertEqual(
            result.no_upgrade_reasons,
            (M3NoUpgradeReason.INSUFFICIENT_REMAINING_BUDGET,),
        )


class M3QueryDependenceTests(unittest.TestCase):
    """Verify the same active model-tier pair responds to query-specific SLOs."""

    def test_relaxed_query_needs_no_upgrade_but_tight_query_selects_one(self):
        base = make_m3_instance(delay_slo_seconds=0.4)
        relaxed = base.queries[0]
        tight = replace(relaxed, id="q_tight", delay_slo_seconds=0.2)
        queries = (relaxed, tight)
        pairs = {(query.id, MODEL_ID): 0.05 for query in queries}
        triples = tuple((query.id, MODEL_ID, TIER_ID) for query in queries)
        instance = replace(
            base,
            queries=queries,
            base_error_rate=pairs,
            compute_delay_seconds_per_token={key: 0.10 for key in triples},
            communication_delay_seconds_per_token={key: 0.01 for key in triples},
            compute_gflop_per_token={key: 1.0 for key in triples},
            kv_cache_residency_seconds={key: 1.0 for key in triples},
        )
        state = M3EvaluationState(ParallelismConfig(1, 1), 100.0)

        relaxed_result = select_m3_parallelism_upgrade(
            instance, "q_text", MODEL_ID, TIER_ID, state
        )
        tight_result = select_m3_parallelism_upgrade(
            instance, "q_tight", MODEL_ID, TIER_ID, state
        )

        self.assertFalse(relaxed_result.is_applicable)
        self.assertEqual(
            relaxed_result.no_upgrade_reasons,
            (M3NoUpgradeReason.CURRENT_DELAY_WITHIN_SLO,),
        )
        self.assertTrue(tight_result.is_applicable)
        self.assertEqual(
            tight_result.selected_configuration,
            ParallelismConfig(2, 1),
        )


class M3ImmutabilityReuseAndIsolationTests(unittest.TestCase):
    """Keep M3 deterministic, primitive-based, and outside GH/P_DM layers."""

    def test_results_are_deterministic_immutable_and_inputs_unchanged(self):
        instance = make_m3_instance()
        state = M3EvaluationState(ParallelismConfig(1, 1), 100.0)
        before_instance = snapshot_value(instance)
        before_state = snapshot_value(state)

        first = select_upgrade(instance, state)
        second = select_upgrade(instance, state)

        self.assertEqual(first, second)
        self.assertEqual(snapshot_value(instance), before_instance)
        self.assertEqual(snapshot_value(state), before_state)
        with self.assertRaises(FrozenInstanceError):
            state.remaining_budget_usd = 0.0
        with self.assertRaises(FrozenInstanceError):
            first.selected_configuration = ParallelismConfig(4, 1)
        with self.assertRaises(TypeError):
            first.selected_diagnostic.incremental_gpu_count = 99

    def test_invalid_state_and_unsupported_current_configuration_fail_clearly(self):
        with self.assertRaises(ValueError):
            M3EvaluationState("not-a-configuration", 1.0)
        for invalid_budget in (float("inf"), float("nan"), True, "1"):
            with self.subTest(value=invalid_budget), self.assertRaises(ValueError):
                M3EvaluationState(ParallelismConfig(1, 1), invalid_budget)

        instance = make_m3_instance(tensor_degrees=(1, 4))
        with self.assertRaisesRegex(ValueError, "not valid for its GPU tier"):
            select_upgrade(
                instance,
                M3EvaluationState(ParallelismConfig(2, 1), 100.0),
            )

    def test_existing_configuration_delay_primitive_is_reused(self):
        instance = make_m3_instance()
        state = M3EvaluationState(ParallelismConfig(1, 1), 100.0)
        with patch.object(
            m3_module,
            "compute_configuration_delay_seconds",
            wraps=compute_configuration_delay_seconds,
        ) as delay_primitive:
            result = select_upgrade(instance, state)

        self.assertEqual(
            delay_primitive.call_count,
            1 + len(result.configuration_diagnostics),
        )



if __name__ == "__main__":
    unittest.main()
