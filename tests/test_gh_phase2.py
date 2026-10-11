"""Hand-derived tests for GH Phase-2 initialization and one-query ranking."""

import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    GHPhase1Config,
    GHPhase2CandidateRejectionReason,
    GHPhase2ConfigurationSource,
    GHPhase2Deployment,
    M3NoUpgradeReason,
    evaluate_gh_phase2_query_candidates,
    initialize_gh_phase2_state,
    order_gh_phase2_queries,
    rank_m2_candidates,
    run_gh_phase1,
)
from llm_resource_allocation.algorithms import gh_phase2 as gh_phase2_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig


def make_branch_instance(
    *,
    arrivals: tuple[float, float, float] = (30.0, 20.0, 10.0),
):
    """Return a 3-query/3-model fixture with all Phase-2 branches.

    The active model wins Phase 1 with q_high on one GPU. For q_mid its current
    0.03-second delay requires a two-GPU M3 upgrade costing the exact remaining
    dollar. For q_low, a four-GPU upgrade costs three dollars and fails budget.
    The inactive model uses M1; the blocked model never fits GPU memory.
    """

    base = make_minimal_tiny_instance()
    query_template = base.queries[0]
    queries = tuple(
        replace(
            query_template,
            id=query_id,
            arrival_rate_per_hour=arrival,
            delay_slo_seconds=delay_slo,
            error_slo=0.10,
        )
        for query_id, arrival, delay_slo in zip(
            ("q_high", "q_mid", "q_low"),
            arrivals,
            (0.04, 0.02, 0.008),
        )
    )
    model_weights = (
        ("model_active", 1.0),
        ("model_inactive", 2.0),
        ("model_blocked", 5.0),
    )
    models = tuple(
        replace(base.models[0], id=model_id, weight_gb=weight)
        for model_id, weight in model_weights
    )
    tier = replace(
        base.gpu_tiers[0],
        id="tier",
        memory_capacity_gb=1.0,
        hourly_rental_usd=1.0,
        tensor_parallel_degrees=(1, 2, 4),
        pipeline_parallel_depths=(1,),
    )
    base_error_rate = {
        (query.id, model.id): (
            0.05
            if model.id != "model_active" or query.id == "q_high"
            else 0.20
        )
        for query in queries
        for model in models
    }
    triples = tuple(
        (query.id, model.id, tier.id)
        for query in queries
        for model in models
    )
    return replace(
        base,
        id="gh_phase2_branch_fixture",
        queries=queries,
        models=models,
        gpu_tiers=(tier,),
        base_error_rate=base_error_rate,
        compute_delay_seconds_per_token={key: 0.01 for key in triples},
        communication_delay_seconds_per_token={key: 0.0 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=2.0,
        horizon_hours=1.0,
    )


def initialized_branch_case():
    """Return fixture, its one-activation Phase-1 result, and Phase-2 state."""

    instance = make_branch_instance()
    phase1 = run_gh_phase1(instance, GHPhase1Config(0.5))
    state = initialize_gh_phase2_state(instance, phase1)
    return instance, phase1, state


def candidate_for(result, model_id: str):
    """Resolve the sole-tier candidate for concise branch assertions."""

    return next(
        candidate
        for candidate in result.candidate_evaluations
        if candidate.model_id == model_id
    )


def make_ranking_instance(
    model_specs: tuple[tuple[str, float, float], ...],
):
    """Return a one-query inactive-pair fixture for transparent M2 rankings."""

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        id="q",
        arrival_rate_per_hour=10.0,
        delay_slo_seconds=1.0,
        error_slo=0.10,
    )
    models = tuple(
        replace(base.models[0], id=model_id, weight_gb=weight)
        for model_id, weight, _ in model_specs
    )
    tier = replace(
        base.gpu_tiers[0],
        id="tier",
        memory_capacity_gb=1.0,
        hourly_rental_usd=1.0,
        tensor_parallel_degrees=(1, 2, 4),
        pipeline_parallel_depths=(1,),
    )
    triples = tuple((query.id, model.id, tier.id) for model in models)
    return replace(
        base,
        id="gh_phase2_ranking_fixture",
        queries=(query,),
        models=models,
        gpu_tiers=(tier,),
        base_error_rate={
            (query.id, model_id): error
            for model_id, _, error in model_specs
        },
        compute_delay_seconds_per_token={key: 0.01 for key in triples},
        communication_delay_seconds_per_token={key: 0.0 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=20.0,
        horizon_hours=1.0,
    )


def inactive_phase2_state(instance):
    """Enter Phase 2 from the literal zero-cap Phase-1 stop."""

    phase1 = run_gh_phase1(instance, GHPhase1Config(0.0))
    return initialize_gh_phase2_state(instance, phase1)


class GHPhase2InitializationTests(unittest.TestCase):
    """Verify exact Phase-1 handoff and descending-lambda metadata."""

    def test_phase1_deployment_configuration_gpu_and_cost_transfer_exactly(self):
        instance, phase1, state = initialized_branch_case()

        self.assertEqual(phase1.selected_pair_sequence, (("model_active", "tier"),))
        self.assertEqual(state.active_pair_ids, phase1.selected_pair_sequence)
        self.assertEqual(state.deployments[0].configuration, ParallelismConfig(1, 1))
        self.assertEqual(state.deployments[0].gpu_count, 1)
        self.assertEqual(state.accumulated_budget_cost_usd, 1.0)
        self.assertEqual(state.remaining_budget_usd, 1.0)
        self.assertEqual(state.remaining_demand_by_query, {query_id: 1.0 for query_id in instance.query_ids})
        self.assertEqual(state.cumulative_error_by_query, {query_id: 0.0 for query_id in instance.query_ids})
        self.assertEqual(state.cumulative_delay_seconds_by_query, {query_id: 0.0 for query_id in instance.query_ids})

    def test_query_order_is_descending_lambda(self):
        instance = make_branch_instance()

        self.assertEqual(order_gh_phase2_queries(instance), ("q_high", "q_mid", "q_low"))

    def test_equal_lambda_tie_preserves_instance_order_not_lexical_id(self):
        instance = make_branch_instance(arrivals=(20.0, 20.0, 10.0))

        self.assertEqual(order_gh_phase2_queries(instance), ("q_high", "q_mid", "q_low"))

    def test_phase2_state_is_deeply_immutable(self):
        _, _, state = initialized_branch_case()

        with self.assertRaises(FrozenInstanceError):
            state.remaining_budget_usd = 0.0
        with self.assertRaises(TypeError):
            state.remaining_demand_by_query["q_high"] = 0.5


class GHPhase2BudgetConstructionTests(unittest.TestCase):
    """Retain the original exact-budget failure at the construction boundary."""

    def test_phase1_overshoot_roundoff_is_rejected_during_initialization(self):
        instance = replace(make_minimal_tiny_instance(), cost_budget_usd=0.001)
        result = run_gh_phase1(instance, GHPhase1Config(0.8))
        self.assertEqual(result.final_state.accumulated_rental_cost_usd, 1.0)

        with self.assertRaisesRegex(ValueError, "budget must sum to delta"):
            initialize_gh_phase2_state(instance, result)


class GHPhase2BranchTests(unittest.TestCase):
    """Exercise inactive, current-active, M3 success, and M3 failure paths."""

    def test_inactive_candidate_uses_m1_and_maps_exact_m2_inputs(self):
        instance, _, state = initialized_branch_case()

        candidate = candidate_for(
            evaluate_gh_phase2_query_candidates(instance, state, "q_mid"),
            "model_inactive",
        )

        self.assertFalse(candidate.is_active_pair)
        self.assertIsNotNone(candidate.m1_selection)
        self.assertIsNone(candidate.m3_upgrade)
        self.assertEqual(candidate.effective_configuration, ParallelismConfig(2, 1))
        self.assertEqual(candidate.candidate_configuration.source, GHPhase2ConfigurationSource.M1_INACTIVE_PAIR)
        self.assertEqual(candidate.m2_inputs.remaining_demand, 1.0)
        self.assertEqual(candidate.m2_inputs.cumulative_error, 0.0)
        self.assertEqual(candidate.m2_inputs.cumulative_delay_seconds, 0.0)
        self.assertEqual(candidate.m2_inputs.current_gpu_count, 0)
        self.assertEqual(candidate.m2_evaluation.cost_breakdown.incremental_gpu_count, 2)
        self.assertEqual(candidate.m2_evaluation.coverage_diagnostic.effective_coverage, 1.0)
        self.assertEqual(candidate.rank_key[0], 0)

    def test_inactive_m1_failure_is_explicit_and_skips_m2(self):
        instance, _, state = initialized_branch_case()

        candidate = candidate_for(
            evaluate_gh_phase2_query_candidates(instance, state, "q_mid"),
            "model_blocked",
        )

        self.assertFalse(candidate.m1_selection.is_feasible)
        self.assertIsNone(candidate.candidate_configuration)
        self.assertIsNone(candidate.m2_evaluation)
        self.assertEqual(candidate.rejection_reasons, (GHPhase2CandidateRejectionReason.M1_NO_FEASIBLE_CONFIGURATION,))

    def test_active_suitable_pair_reuses_current_configuration_and_skips_m3(self):
        instance, _, state = initialized_branch_case()
        with patch.object(
            gh_phase2_module,
            "_select_m3_parallelism_upgrade",
            side_effect=AssertionError("M3 must not run"),
        ):
            candidate = candidate_for(
                evaluate_gh_phase2_query_candidates(instance, state, "q_high"),
                "model_active",
            )

        self.assertIsNone(candidate.m1_selection)
        self.assertIsNone(candidate.m3_upgrade)
        self.assertEqual(candidate.current_configuration, ParallelismConfig(1, 1))
        self.assertEqual(candidate.effective_configuration, ParallelismConfig(1, 1))
        self.assertEqual(candidate.candidate_configuration.source, GHPhase2ConfigurationSource.CURRENT_ACTIVE_CONFIGURATION)
        self.assertEqual(candidate.m2_evaluation.cost_breakdown.incremental_gpu_count, 0)

    def test_active_delay_violation_uses_m3_and_feeds_upgrade_to_m2(self):
        instance, _, state = initialized_branch_case()

        candidate = candidate_for(
            evaluate_gh_phase2_query_candidates(instance, state, "q_mid"),
            "model_active",
        )

        self.assertIsNone(candidate.m1_selection)
        self.assertTrue(candidate.m3_upgrade.has_upgrade)
        self.assertEqual(candidate.current_configuration, ParallelismConfig(1, 1))
        self.assertEqual(candidate.effective_configuration, ParallelismConfig(2, 1))
        self.assertEqual(candidate.candidate_configuration.source, GHPhase2ConfigurationSource.M3_ACTIVE_PAIR_UPGRADE)
        self.assertEqual(candidate.current_gpu_count, 1)
        self.assertEqual(candidate.m3_upgrade.selected_diagnostic.incremental_gpu_cost_usd, 1.0)
        self.assertEqual(candidate.m2_evaluation.cost_breakdown.incremental_gpu_count, 1)
        self.assertAlmostEqual(candidate.m2_evaluation.coverage_diagnostic.fixed_configuration_delay_seconds, 0.015)

    def test_active_m3_budget_failure_is_explicit_and_skips_m2(self):
        instance, _, state = initialized_branch_case()

        candidate = candidate_for(
            evaluate_gh_phase2_query_candidates(instance, state, "q_low"),
            "model_active",
        )

        self.assertFalse(candidate.m3_upgrade.has_upgrade)
        self.assertEqual(candidate.m3_upgrade.no_upgrade_reasons, (M3NoUpgradeReason.INSUFFICIENT_REMAINING_BUDGET,))
        self.assertIsNone(candidate.candidate_configuration)
        self.assertIsNone(candidate.m2_evaluation)
        self.assertEqual(candidate.rejection_reasons, (GHPhase2CandidateRejectionReason.M3_NO_FEASIBLE_UPGRADE,))


class GHPhase2RankingTests(unittest.TestCase):
    """Protect exact M2 lexicographic ranking and stable semantic ties."""

    def test_full_coverage_beats_cheaper_partial_candidate(self):
        instance = make_ranking_instance(
            (("expensive_full", 4.0, 0.05), ("cheap_partial", 1.0, 0.20))
        )
        state = inactive_phase2_state(instance)

        result = evaluate_gh_phase2_query_candidates(instance, state, "q")
        full, partial = result.candidate_evaluations

        self.assertEqual(full.rank_key[0], 0)
        self.assertEqual(partial.rank_key[0], 1)
        self.assertGreater(full.rank_key[1], partial.rank_key[1])
        self.assertEqual(result.best_candidate.model_id, "expensive_full")

    def test_best_of_three_is_not_first_semantic_candidate(self):
        instance = make_ranking_instance(
            (("first", 4.0, 0.05), ("best", 1.0, 0.05), ("third", 2.0, 0.05))
        )

        result = evaluate_gh_phase2_query_candidates(
            instance, inactive_phase2_state(instance), "q"
        )

        self.assertEqual(tuple(item.model_id for item in result.ranked_candidates), ("best", "third", "first"))

    def test_exact_tie_preserves_and_reverses_semantic_order_only(self):
        forward = make_ranking_instance(
            (("model_z", 1.0, 0.05), ("model_a", 1.0, 0.05))
        )
        reverse = make_ranking_instance(
            (("model_a", 1.0, 0.05), ("model_z", 1.0, 0.05))
        )

        forward_result = evaluate_gh_phase2_query_candidates(forward, inactive_phase2_state(forward), "q")
        reverse_result = evaluate_gh_phase2_query_candidates(reverse, inactive_phase2_state(reverse), "q")

        self.assertEqual(tuple(item.model_id for item in forward_result.ranked_candidates), ("model_z", "model_a"))
        self.assertEqual(tuple(item.model_id for item in reverse_result.ranked_candidates), ("model_a", "model_z"))

    def test_zero_effective_coverage_is_diagnosed_and_not_ranked(self):
        instance = make_ranking_instance((("model", 1.0, 0.05),))
        state = inactive_phase2_state(instance)
        exhausted = replace(state, cumulative_error_by_query={"q": 0.10})

        result = evaluate_gh_phase2_query_candidates(instance, exhausted, "q")
        candidate = result.candidate_evaluations[0]

        self.assertEqual(candidate.m2_evaluation.coverage_diagnostic.effective_coverage, 0.0)
        self.assertEqual(candidate.rejection_reasons, (GHPhase2CandidateRejectionReason.M2_ZERO_EFFECTIVE_COVERAGE,))
        self.assertEqual(result.ranked_candidates, ())

    def test_existing_m2_ranking_is_called_once(self):
        instance = make_ranking_instance(
            (("first", 2.0, 0.05), ("second", 1.0, 0.05))
        )
        with patch.object(
            gh_phase2_module,
            "rank_m2_candidates",
            wraps=rank_m2_candidates,
        ) as ranking:
            evaluate_gh_phase2_query_candidates(
                instance, inactive_phase2_state(instance), "q"
            )

        self.assertEqual(ranking.call_count, 1)


class GHPhase2DynamicStateAndIsolationTests(unittest.TestCase):
    """Verify state-sensitive M2 mapping without committing any allocation."""

    def test_query_dynamic_values_change_m2_without_mutating_state(self):
        instance, phase1, state = initialized_branch_case()
        changed = replace(
            state,
            remaining_demand_by_query={"q_high": 1.0, "q_mid": 0.5, "q_low": 1.0},
            cumulative_error_by_query={"q_high": 0.0, "q_mid": 0.025, "q_low": 0.0},
            cumulative_delay_seconds_by_query={"q_high": 0.0, "q_mid": 0.005, "q_low": 0.0},
        )

        before = candidate_for(evaluate_gh_phase2_query_candidates(instance, state, "q_mid"), "model_inactive")
        after = candidate_for(evaluate_gh_phase2_query_candidates(instance, changed, "q_mid"), "model_inactive")

        self.assertEqual(after.m2_inputs.remaining_demand, 0.5)
        self.assertEqual(after.m2_inputs.cumulative_error, 0.025)
        self.assertEqual(after.m2_inputs.cumulative_delay_seconds, 0.005)
        self.assertNotEqual(before.m2_evaluation.coverage_diagnostic, after.m2_evaluation.coverage_diagnostic)
        self.assertEqual(phase1.final_state.active_deployments[0].configuration, ParallelismConfig(1, 1))
        self.assertEqual(state.remaining_demand_by_query["q_mid"], 1.0)

    def test_current_gpu_count_changes_incremental_m2_cost(self):
        instance, _, state = initialized_branch_case()
        inactive = candidate_for(evaluate_gh_phase2_query_candidates(instance, state, "q_mid"), "model_inactive")
        with_active_pair = replace(
            state,
            deployments=state.deployments + (
                GHPhase2Deployment("model_inactive", "tier", ParallelismConfig(1, 1)),
            ),
        )

        active = candidate_for(evaluate_gh_phase2_query_candidates(instance, with_active_pair, "q_mid"), "model_inactive")

        self.assertEqual(inactive.m2_inputs.current_gpu_count, 0)
        self.assertEqual(active.m2_inputs.current_gpu_count, 1)
        self.assertEqual(inactive.m2_evaluation.cost_breakdown.incremental_gpu_count, 2)
        self.assertEqual(active.m2_evaluation.cost_breakdown.incremental_gpu_count, 1)

    def test_evaluation_does_not_commit_routing_or_change_any_state(self):
        instance, _, state = initialized_branch_case()

        result = evaluate_gh_phase2_query_candidates(instance, state, "q_mid")

        self.assertEqual(state.remaining_demand_by_query["q_mid"], 1.0)
        self.assertEqual(state.cumulative_error_by_query["q_mid"], 0.0)
        self.assertEqual(state.cumulative_delay_seconds_by_query["q_mid"], 0.0)
        self.assertTrue(result.best_candidate.is_rankable)



if __name__ == "__main__":
    unittest.main()
