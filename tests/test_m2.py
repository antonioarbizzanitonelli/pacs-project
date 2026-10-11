"""Hand-derived and isolation tests for Paper v2 Eqs. (10)-(11), M2."""

import unittest

from placement_support import snapshot_value
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    M2CoverageLimiter,
    M2EvaluationInputs,
    M2UnrankableReason,
    compute_m2_effective_coverage,
    compute_m2_marginal_cost,
    evaluate_m2_candidate,
    rank_m2_candidates,
    select_minimum_feasible_parallelism,
)
from llm_resource_allocation.algorithms import m1 as m1_module
from llm_resource_allocation.data.tiny_instances import make_minimal_tiny_instance


QUERY_ID = "q_text"
MODEL_ID = "m_small"
TIER_ID = "gpu_basic_fp16"


def make_m2_instance(
    *,
    arrival_rate_per_hour: float = 10.0,
    storage_gb_per_token: float = 0.01,
    delay_slo_seconds: float = 10.0,
    error_slo: float = 0.5,
    delay_penalty: float = 4.0,
    weight_gb: float = 2.0,
    memory_capacity_gb: float = 8.0,
    hourly_rental_usd: float = 3.0,
    base_error_rate: float = 0.2,
    compute_delay: float = 0.10,
    communication_delay: float = 0.01,
    tensor_degrees: tuple[int, ...] = (1, 2),
    horizon_hours: float = 2.0,
    storage_price: float = 0.5,
):
    """Return a transparent valid 1x1x1 fixture for M2 arithmetic."""

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        arrival_rate_per_hour=arrival_rate_per_hour,
        storage_gb_per_token=storage_gb_per_token,
        delay_slo_seconds=delay_slo_seconds,
        error_slo=error_slo,
        delay_penalty_usd_per_millisecond_query=delay_penalty,
    )
    model = replace(base.models[0], weight_gb=weight_gb)
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=memory_capacity_gb,
        hourly_rental_usd=hourly_rental_usd,
        tensor_parallel_degrees=tensor_degrees,
    )
    triple = (QUERY_ID, MODEL_ID, TIER_ID)
    return replace(
        base,
        queries=(query,),
        models=(model,),
        gpu_tiers=(tier,),
        base_error_rate={(QUERY_ID, MODEL_ID): base_error_rate},
        compute_delay_seconds_per_token={triple: compute_delay},
        communication_delay_seconds_per_token={triple: communication_delay},
        horizon_hours=horizon_hours,
        storage_price_usd_per_gb_hour=storage_price,
    )


def select(instance, query_id: str = QUERY_ID, tier_id: str = TIER_ID):
    """Run M1 for one semantic candidate."""

    return select_minimum_feasible_parallelism(
        instance,
        query_id=query_id,
        model_id=MODEL_ID,
        tier_id=tier_id,
    )


class M2HandDerivedArithmeticTests(unittest.TestCase):
    """Verify Eq. (10), Eq. (11), and exact rank values by hand."""

    def test_producer_retains_cost_and_coverage_with_each_selected_candidate(self):
        for memory_capacity, available in ((8.0, True), (0.01, False)):
            with self.subTest(memory_capacity=memory_capacity):
                instance = make_m2_instance(memory_capacity_gb=memory_capacity)
                result = evaluate_m2_candidate(
                    instance,
                    select(instance),
                    M2EvaluationInputs(0.8, 0.1, 1.0, 0),
                )
                self.assertEqual(result.selected_configuration is not None, available)
                self.assertEqual(result.cost_breakdown is not None, available)
                self.assertEqual(result.coverage_diagnostic is not None, available)

    def test_hand_baseline_cost_coverage_and_rank(self):
        """Derive cost 9.54, full coverage .8, pi=0, and kappa=11.925."""

        instance = make_m2_instance()
        m1 = select(instance)
        evaluation_inputs = M2EvaluationInputs(0.8, 0.1, 1.0, 0)

        result = evaluate_m2_candidate(instance, m1, evaluation_inputs)
        cost = result.cost_breakdown
        coverage = result.coverage_diagnostic

        self.assertEqual(cost.required_gpu_count, 1)
        self.assertEqual(cost.incremental_gpu_count, 1)
        self.assertEqual(cost.incremental_gpu_cost, 6.0)
        # 2 * .5 * (2 + .01 * 3 * 10) = 2.3
        self.assertAlmostEqual(cost.model_and_data_storage_cost, 2.3)
        # rho * D_cfg = 4 * .31 = 1.24, without a unit conversion.
        self.assertAlmostEqual(cost.delay_penalty_cost, 1.24)
        self.assertAlmostEqual(cost.marginal_cost, 9.54)
        self.assertAlmostEqual(coverage.error_coverage_limit, 2.0)
        self.assertAlmostEqual(coverage.delay_coverage_limit, 9.0 / 0.31)
        self.assertEqual(coverage.effective_coverage, 0.8)
        self.assertEqual(
            coverage.limiting_factors,
            (M2CoverageLimiter.REMAINING_DEMAND,),
        )
        self.assertEqual(result.pi, 0)
        self.assertAlmostEqual(result.kappa, 11.925)
        self.assertEqual(result.rank_key, (result.pi, result.kappa))

    def test_incremental_gpu_positive_part_and_already_available_case(self):
        """Required TP=2 charges 2, 1, then 0 GPUs as current y grows."""

        instance = make_m2_instance(delay_slo_seconds=0.2)
        m1 = select(instance)
        self.assertEqual(m1.selected_configuration.gpu_count, 2)

        costs = tuple(
            compute_m2_marginal_cost(
                instance,
                m1,
                M2EvaluationInputs(1.0, 0.0, 0.0, current),
            )
            for current in (0, 1, 2, 3)
        )

        self.assertEqual(
            tuple(item.incremental_gpu_count for item in costs),
            (2, 1, 0, 0),
        )
        self.assertEqual(
            tuple(item.incremental_gpu_cost for item in costs),
            (12.0, 6.0, 0.0, 0.0),
        )
        # Eq. (10) keeps storage and delay terms even when incremental GPU
        # cost is zero; the total marginal cost therefore does not vanish.
        self.assertGreater(costs[2].marginal_cost, 0.0)
        self.assertEqual(
            costs[2].marginal_cost,
            costs[2].model_and_data_storage_cost + costs[2].delay_penalty_cost,
        )
        already_available = evaluate_m2_candidate(
            instance,
            m1,
            M2EvaluationInputs(1.0, 0.0, 0.0, 2),
        )
        self.assertTrue(already_available.is_rankable)
        self.assertGreater(already_available.kappa, 0.0)

    def test_public_cost_and_coverage_primitives_match_composed_result(self):
        instance = make_m2_instance()
        m1 = select(instance)
        evaluation_inputs = M2EvaluationInputs(0.6, 0.1, 1.0, 1)

        result = evaluate_m2_candidate(instance, m1, evaluation_inputs)

        self.assertEqual(
            result.cost_breakdown,
            compute_m2_marginal_cost(instance, m1, evaluation_inputs),
        )
        self.assertEqual(
            result.coverage_diagnostic,
            compute_m2_effective_coverage(instance, m1, evaluation_inputs),
        )


class M2CoverageBoundaryTests(unittest.TestCase):
    """Exercise each printed cap, simultaneous limits, and unsafe boundaries."""

    def test_remaining_demand_is_the_only_active_limit(self):
        instance = make_m2_instance()
        result = evaluate_m2_candidate(
            instance,
            select(instance),
            M2EvaluationInputs(0.25, 0.0, 0.0, 0),
        )

        self.assertEqual(result.coverage_diagnostic.effective_coverage, 0.25)
        self.assertTrue(result.coverage_diagnostic.covers_remaining_demand)
        self.assertEqual(result.pi, 0)

    def test_error_budget_is_the_active_limit(self):
        instance = make_m2_instance(error_slo=0.1)
        result = evaluate_m2_candidate(
            instance,
            select(instance),
            M2EvaluationInputs(1.0, 0.06, 0.0, 0),
        )

        self.assertAlmostEqual(result.coverage_diagnostic.error_coverage_limit, 0.2)
        self.assertAlmostEqual(result.coverage_diagnostic.effective_coverage, 0.2)
        self.assertEqual(
            result.coverage_diagnostic.limiting_factors,
            (M2CoverageLimiter.ERROR_BUDGET,),
        )
        self.assertEqual(result.pi, 1)

    def test_delay_budget_is_the_active_limit(self):
        # D_cfg = .0625*3 + .0625*1 = .25 exactly in binary.
        instance = make_m2_instance(
            delay_slo_seconds=1.0,
            compute_delay=0.0625,
            communication_delay=0.0625,
        )
        result = evaluate_m2_candidate(
            instance,
            select(instance),
            M2EvaluationInputs(1.0, 0.0, 0.9375, 0),
        )

        self.assertAlmostEqual(result.coverage_diagnostic.delay_coverage_limit, 0.25)
        self.assertAlmostEqual(result.coverage_diagnostic.effective_coverage, 0.25)
        self.assertEqual(
            result.coverage_diagnostic.limiting_factors,
            (M2CoverageLimiter.DELAY_BUDGET,),
        )

    def test_simultaneous_exact_limits_retain_all_limiting_factors(self):
        # Every cap is exactly 0.5 in binary: .125/.25 for both budgets.
        instance = make_m2_instance(
            error_slo=0.25,
            base_error_rate=0.25,
            delay_slo_seconds=1.0,
            compute_delay=0.0625,
            communication_delay=0.0625,
        )
        result = evaluate_m2_candidate(
            instance,
            select(instance),
            M2EvaluationInputs(0.5, 0.125, 0.875, 0),
        )

        coverage = result.coverage_diagnostic
        self.assertEqual(coverage.error_coverage_limit, 0.5)
        self.assertEqual(coverage.delay_coverage_limit, 0.5)
        self.assertEqual(coverage.effective_coverage, 0.5)
        self.assertEqual(
            coverage.limiting_factors,
            (
                M2CoverageLimiter.REMAINING_DEMAND,
                M2CoverageLimiter.ERROR_BUDGET,
                M2CoverageLimiter.DELAY_BUDGET,
            ),
        )

    def test_exhausted_and_exceeded_error_budgets_are_explicitly_unrankable(self):
        instance = make_m2_instance(error_slo=0.1)
        m1 = select(instance)
        exhausted = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(1.0, 0.1, 0.0, 0)
        )
        exceeded = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(1.0, 0.11, 0.0, 0)
        )

        self.assertEqual(exhausted.coverage_diagnostic.error_coverage_limit, 0.0)
        self.assertEqual(exhausted.coverage_diagnostic.effective_coverage, 0.0)
        self.assertEqual(exhausted.pi, 1)
        self.assertIsNone(exhausted.kappa)
        self.assertIn(
            M2UnrankableReason.ERROR_BUDGET_EXHAUSTED,
            exhausted.unrankable_reasons,
        )
        self.assertAlmostEqual(exceeded.coverage_diagnostic.error_coverage_limit, -0.05)
        self.assertEqual(exceeded.coverage_diagnostic.effective_coverage, 0.0)
        self.assertIn(
            M2UnrankableReason.ERROR_BUDGET_EXCEEDED,
            exceeded.unrankable_reasons,
        )

    def test_exhausted_and_exceeded_delay_budgets_are_explicitly_unrankable(self):
        instance = make_m2_instance()
        m1 = select(instance)
        exhausted = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(1.0, 0.0, 10.0, 0)
        )
        exceeded = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(1.0, 0.0, 10.1, 0)
        )

        self.assertIsNone(exhausted.kappa)
        self.assertIn(
            M2UnrankableReason.DELAY_BUDGET_EXHAUSTED,
            exhausted.unrankable_reasons,
        )
        self.assertLess(exceeded.coverage_diagnostic.delay_coverage_limit, 0.0)
        self.assertEqual(exceeded.coverage_diagnostic.effective_coverage, 0.0)
        self.assertIn(
            M2UnrankableReason.DELAY_BUDGET_EXCEEDED,
            exceeded.unrankable_reasons,
        )

    def test_zero_denominators_are_nonbinding_unless_state_already_violates(self):
        instance = make_m2_instance(
            base_error_rate=0.0,
            compute_delay=0.0,
            communication_delay=0.0,
        )
        m1 = select(instance)
        within = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(0.7, 0.1, 1.0, 0)
        )
        violated = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(0.7, 0.6, 11.0, 0)
        )

        self.assertIsNone(within.coverage_diagnostic.error_coverage_limit)
        self.assertIsNone(within.coverage_diagnostic.delay_coverage_limit)
        self.assertEqual(within.coverage_diagnostic.effective_coverage, 0.7)
        self.assertEqual(violated.coverage_diagnostic.error_coverage_limit, 0.0)
        self.assertEqual(violated.coverage_diagnostic.delay_coverage_limit, 0.0)
        self.assertFalse(violated.is_rankable)
        self.assertEqual(
            violated.unrankable_reasons,
            (
                M2UnrankableReason.ERROR_BUDGET_EXCEEDED,
                M2UnrankableReason.DELAY_BUDGET_EXCEEDED,
            ),
        )

    def test_zero_remaining_demand_is_diagnosed_and_not_ranked(self):
        instance = make_m2_instance()
        result = evaluate_m2_candidate(
            instance,
            select(instance),
            M2EvaluationInputs(0.0, 0.0, 0.0, 0),
        )

        self.assertEqual(result.pi, 0)
        self.assertIsNone(result.kappa)
        self.assertFalse(result.is_rankable)
        self.assertEqual(
            result.unrankable_reasons,
            (M2UnrankableReason.NO_REMAINING_DEMAND,),
        )


class M2M1CompositionTests(unittest.TestCase):
    """Protect explicit no-feasible handling and M1 evidence reuse."""

    def test_no_feasible_m1_result_is_retained_without_cost_or_coverage(self):
        instance = make_m2_instance(weight_gb=100.0, memory_capacity_gb=1.0)
        m1 = select(instance)
        self.assertFalse(m1.is_feasible)

        result = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(1.0, 0.0, 0.0, 0)
        )

        self.assertIsNone(result.selected_configuration)
        self.assertIsNone(result.cost_breakdown)
        self.assertIsNone(result.coverage_diagnostic)
        self.assertEqual(
            result.unrankable_reasons,
            (M2UnrankableReason.NO_FEASIBLE_CONFIGURATION,),
        )
        self.assertFalse(result.is_rankable)

    def test_m2_reuses_m1_delay_evidence_without_recomputing_m1_or_latency(self):
        instance = make_m2_instance()
        m1 = select(instance)
        evaluation_inputs = M2EvaluationInputs(1.0, 0.0, 0.0, 0)

        with (
            patch.object(
                m1_module,
                "select_minimum_feasible_parallelism",
                side_effect=AssertionError("M1 was recomputed"),
            ),
            patch.object(
                m1_module,
                "compute_configuration_delay_seconds",
                side_effect=AssertionError("D_cfg was recomputed"),
            ),
        ):
            result = evaluate_m2_candidate(instance, m1, evaluation_inputs)

        self.assertEqual(
            result.coverage_diagnostic.fixed_configuration_delay_seconds,
            m1.selected_fixed_configuration_delay_seconds,
        )


class M2RankingAndQueryDependenceTests(unittest.TestCase):
    """Verify exact lexicographic ranking and query-specific state."""

    def make_three_tier_instance(self, *, equal_prices: bool = False):
        base = make_m2_instance(hourly_rental_usd=5.0)
        prices = (5.0, 1.0, 2.0) if not equal_prices else (1.0, 1.0, 1.0)
        tiers = tuple(
            replace(base.gpu_tiers[0], id=tier_id, hourly_rental_usd=price)
            for tier_id, price in zip(("tier_a", "tier_b", "tier_c"), prices)
        )
        triples = tuple((QUERY_ID, MODEL_ID, tier.id) for tier in tiers)
        return replace(
            base,
            gpu_tiers=tiers,
            compute_delay_seconds_per_token={key: 0.10 for key in triples},
            communication_delay_seconds_per_token={key: 0.01 for key in triples},
            compute_gflop_per_token={key: 1.0 for key in triples},
            kv_cache_residency_seconds={key: 1.0 for key in triples},
        )

    def test_three_candidates_rank_best_even_when_it_is_not_first(self):
        instance = self.make_three_tier_instance()
        evaluation_inputs = M2EvaluationInputs(1.0, 0.0, 0.0, 0)
        evaluations = tuple(
            evaluate_m2_candidate(instance, select(instance, tier_id=tier.id), evaluation_inputs)
            for tier in instance.gpu_tiers
        )

        ranked = rank_m2_candidates(evaluations)

        self.assertEqual(
            tuple(item.tier_id for item in ranked),
            ("tier_b", "tier_c", "tier_a"),
        )

    def test_exact_rank_ties_preserve_supplied_semantic_order(self):
        instance = self.make_three_tier_instance(equal_prices=True)
        evaluation_inputs = M2EvaluationInputs(1.0, 0.0, 0.0, 0)
        evaluations = tuple(
            evaluate_m2_candidate(instance, select(instance, tier_id=tier.id), evaluation_inputs)
            for tier in instance.gpu_tiers
        )
        supplied = (evaluations[2], evaluations[0], evaluations[1])

        ranked = rank_m2_candidates(supplied)

        self.assertEqual(ranked, supplied)

    def test_full_coverage_precedes_cheaper_partial_coverage(self):
        instance = make_m2_instance(error_slo=0.2)
        m1 = select(instance)
        full = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(0.5, 0.0, 0.0, 0)
        )
        partial = evaluate_m2_candidate(
            instance, m1, M2EvaluationInputs(1.0, 0.1, 0.0, 1)
        )
        self.assertEqual(full.pi, 0)
        self.assertEqual(partial.pi, 1)
        self.assertLess(
            partial.cost_breakdown.marginal_cost,
            full.cost_breakdown.marginal_cost,
        )

        self.assertEqual(rank_m2_candidates((partial, full)), (full, partial))

    def test_same_model_tier_is_query_dependent(self):
        base = make_m2_instance(error_slo=0.2)
        relaxed = base.queries[0]
        tight = replace(relaxed, id="q_tight", error_slo=0.05)
        queries = (relaxed, tight)
        pairs = {(query.id, MODEL_ID): 0.1 for query in queries}
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
        evaluation_inputs = M2EvaluationInputs(1.0, 0.0, 0.0, 0)

        relaxed_result = evaluate_m2_candidate(instance, select(instance), evaluation_inputs)
        tight_result = evaluate_m2_candidate(
            instance, select(instance, query_id="q_tight"), evaluation_inputs
        )

        self.assertEqual(relaxed_result.coverage_diagnostic.effective_coverage, 1.0)
        self.assertEqual(tight_result.coverage_diagnostic.effective_coverage, 0.5)
        self.assertEqual(relaxed_result.pi, 0)
        self.assertEqual(tight_result.pi, 1)


class M2ImmutabilityAndIsolationTests(unittest.TestCase):
    """Keep M2 deterministic, immutable, and outside solver/GH orchestration."""

    def test_inputs_and_diagnostics_are_frozen_and_inputs_are_unchanged(self):
        instance = make_m2_instance()
        m1 = select(instance)
        evaluation_inputs = M2EvaluationInputs(0.8, 0.1, 1.0, 0)
        before_instance = snapshot_value(instance)
        before_m1 = snapshot_value(m1)
        before_state = snapshot_value(evaluation_inputs)

        result = evaluate_m2_candidate(instance, m1, evaluation_inputs)

        self.assertEqual(snapshot_value(instance), before_instance)
        self.assertEqual(snapshot_value(m1), before_m1)
        self.assertEqual(snapshot_value(evaluation_inputs), before_state)
        with self.assertRaises(FrozenInstanceError):
            evaluation_inputs.current_gpu_count = 2
        with self.assertRaises(TypeError):
            result.pi = 1
        with self.assertRaises(TypeError):
            result.cost_breakdown.incremental_gpu_count = 99

    def test_invalid_dynamic_inputs_is_rejected_without_clipping(self):
        invalid = (
            (-0.1, 0.0, 0.0, 0),
            (1.1, 0.0, 0.0, 0),
            (1.0, -0.1, 0.0, 0),
            (1.0, 0.0, -0.1, 0),
            (1.0, 0.0, 0.0, -1),
            (1.0, 0.0, 0.0, 1.5),
            (float("inf"), 0.0, 0.0, 0),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                M2EvaluationInputs(*values)



if __name__ == "__main__":
    unittest.main()
