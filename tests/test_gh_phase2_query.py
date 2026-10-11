"""Hand-derived tests for one-query GH Phase-2 candidate traversal."""

import sys
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    GHPhase1Config,
    GHPhase2CommitRejectionReason,
    GHPhase2ConfigurationSource,
    GHPhase2Deployment,
    GHPhase2QueryTerminationReason,
    GHPhase2State,
    commit_gh_phase2_candidate,
    evaluate_gh_phase2_query_candidates,
    initialize_gh_phase2_state,
    process_gh_phase2_query_candidates,
    run_gh_phase1,
)
from llm_resource_allocation.algorithms import gh_phase2_query as query_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig


QUERY = "q_text"
TIER = "gpu_basic_fp16"


def make_loop_instance(
    model_specs=(
        ("model_a", 1.0, 0.05, 0.001, 1.0),
        ("model_b", 1.0, 0.05, 0.001, 1.0),
    ),
    *,
    error_slo=0.10,
    delay_slo_seconds=1.0,
    cost_budget_usd=20.0,
    storage_capacity_gb=100.0,
):
    """Return a transparent one-query candidate-loop fixture.

    Each model spec is ``(id, weight GB, base error, KV GB/token, alpha)``.
    Identical specs intentionally create exact stable M2 ties.
    """

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        error_slo=error_slo,
        delay_slo_seconds=delay_slo_seconds,
    )
    models = tuple(
        replace(
            base.models[0],
            id=model_id,
            weight_gb=weight,
            kv_cache_gb_per_token=kv,
        )
        for model_id, weight, _, kv, _ in model_specs
    )
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=8.0,
        tensor_parallel_degrees=(1, 2),
        pipeline_parallel_depths=(1,),
    )
    triples = tuple((QUERY, model.id, TIER) for model in models)
    error_by_model = {model_id: error for model_id, _, error, _, _ in model_specs}
    alpha_by_model = {model_id: alpha for model_id, _, _, _, alpha in model_specs}
    return replace(
        base,
        id="gh_phase2_query_loop_fixture",
        queries=(query,),
        models=models,
        gpu_tiers=(tier,),
        base_error_rate={(QUERY, model.id): error_by_model[model.id] for model in models},
        compute_delay_seconds_per_token={key: 0.10 for key in triples},
        communication_delay_seconds_per_token={key: 0.01 for key in triples},
        compute_gflop_per_token={
            key: alpha_by_model[key[1]] for key in triples
        },
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=cost_budget_usd,
        storage_capacity_gb=storage_capacity_gb,
    )


def inactive_state(instance):
    """Enter Phase 2 through the literal zero-cap Phase-1 stop."""

    phase1 = run_gh_phase1(instance, GHPhase1Config(0.0))
    return initialize_gh_phase2_state(instance, phase1)


def evaluate_and_process(instance, state=None):
    """Evaluate once through 5F, then process that exact frozen ranking."""

    state = inactive_state(instance) if state is None else state
    evaluation = evaluate_gh_phase2_query_candidates(instance, state, QUERY)
    return evaluation, process_gh_phase2_query_candidates(
        instance, state, QUERY, evaluation
    )


def observe_query_evaluations(operation):
    """Observe actual evaluation code across aliases, restoring any profiler."""

    evaluated = []
    evaluator_code = query_module._evaluate_gh_phase2_query_candidates.__code__
    previous_profile = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is evaluator_code:
            evaluated.append((frame.f_locals["state"], frame.f_locals["query_id"]))

    sys.setprofile(profile)
    try:
        result = operation()
    finally:
        sys.setprofile(previous_profile)
    return result, evaluated


class GHPhase2QueryLoopFlowTests(unittest.TestCase):
    """Exercise success, rejection continuation, exhaustion, and zero work."""

    def test_first_candidate_fully_serves_query_and_stops(self):
        instance = make_loop_instance()

        evaluation, result = evaluate_and_process(instance)

        self.assertEqual(result.termination_reason, GHPhase2QueryTerminationReason.QUERY_FULLY_SERVED)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].candidate, evaluation.ranked_candidates[0])
        self.assertTrue(result.attempts[0].committed)
        self.assertEqual(result.final_state.unmet_fraction_by_query[QUERY], 0.0)
        self.assertEqual(result.committed_attempt_count, 1)
        self.assertEqual(result.rejected_attempt_count, 0)

    def test_first_rejected_second_succeeds_in_original_rank_order(self):
        instance = make_loop_instance(
            (
                ("memory_first", 1.0, 0.05, 100.0, 1.0),
                ("safe_second", 1.0, 0.05, 0.001, 1.0),
            )
        )

        evaluation, result = evaluate_and_process(instance)

        self.assertEqual(
            tuple(item.model_id for item in evaluation.ranked_candidates),
            ("memory_first", "safe_second"),
        )
        self.assertEqual(
            tuple(item.candidate.model_id for item in result.attempts),
            ("memory_first", "safe_second"),
        )
        self.assertIn(
            GHPhase2CommitRejectionReason.GPU_MEMORY,
            result.attempts[0].commit_result.diagnostic.rejection_reasons,
        )
        self.assertFalse(result.attempts[0].committed)
        self.assertTrue(result.attempts[1].committed)
        self.assertEqual(result.termination_reason, GHPhase2QueryTerminationReason.QUERY_FULLY_SERVED)

    def test_multiple_rejections_retain_different_5g_reasons(self):
        instance = make_loop_instance(
            (
                ("memory", 1.0, 0.05, 100.0, 1.0),
                ("compute", 2.0, 0.05, 0.001, 100_000_000.0),
                ("budget", 3.0, 0.05, 0.001, 1.0),
            ),
            cost_budget_usd=0.5,
        )

        _, result = evaluate_and_process(instance)

        self.assertEqual(len(result.attempts), 3)
        reasons = tuple(
            attempt.commit_result.diagnostic.rejection_reasons
            for attempt in result.attempts
        )
        self.assertIn(GHPhase2CommitRejectionReason.GPU_MEMORY, reasons[0])
        self.assertIn(GHPhase2CommitRejectionReason.COMPUTE_CAPACITY, reasons[1])
        self.assertEqual(reasons[2], (GHPhase2CommitRejectionReason.GLOBAL_BUDGET,))
        self.assertEqual(result.rejected_attempt_count, 3)
        self.assertIs(result.final_state, result.initial_state)
        self.assertEqual(
            result.termination_reason,
            GHPhase2QueryTerminationReason.CANDIDATES_EXHAUSTED_WITH_RESIDUAL_DEMAND,
        )

    def test_no_rankable_candidates_terminates_with_residual_and_no_attempts(self):
        instance = make_loop_instance((("blocked", 100.0, 0.05, 0.001, 1.0),))

        evaluation, result = evaluate_and_process(instance)

        self.assertEqual(evaluation.ranked_candidates, ())
        self.assertEqual(result.attempts, ())
        self.assertIs(result.final_state, result.initial_state)
        self.assertEqual(
            result.termination_reason,
            GHPhase2QueryTerminationReason.CANDIDATES_EXHAUSTED_WITH_RESIDUAL_DEMAND,
        )

    def test_already_satisfied_query_performs_no_zero_allocation_attempt(self):
        instance = make_loop_instance()
        state = inactive_state(instance)
        satisfied = replace(
            state,
            remaining_demand_by_query={QUERY: 0.0},
            unmet_fraction_by_query={QUERY: 0.0},
        )
        evaluation = evaluate_gh_phase2_query_candidates(instance, satisfied, QUERY)

        with patch.object(
            query_module,
            "_commit_gh_phase2_candidate",
            side_effect=AssertionError("u_i=0 must stop before 5G"),
        ):
            result = process_gh_phase2_query_candidates(
                instance, satisfied, QUERY, evaluation
            )

        self.assertEqual(result.attempts, ())
        self.assertIs(result.final_state, satisfied)
        self.assertEqual(result.termination_reason, GHPhase2QueryTerminationReason.QUERY_FULLY_SERVED)


class GHPhase2QueryFrozenEvidenceTests(unittest.TestCase):
    """Pin the paper-literal one-evaluation/one-sort behavior after partial work."""

    def test_owned_evaluation_runs_once_even_with_two_accepted_attempts(self):
        instance = make_loop_instance(error_slo=0.025)
        state = inactive_state(instance)
        result, evaluated = observe_query_evaluations(
            lambda: process_gh_phase2_query_candidates(instance, state, QUERY)
        )

        self.assertEqual(len(evaluated), 1)
        self.assertIs(evaluated[0][0], state)
        self.assertEqual(evaluated[0][1], QUERY)
        self.assertEqual(result.committed_attempt_count, 2)
        self.assertEqual(result.final_state.unmet_fraction_by_query[QUERY], 0.0)
        self.assertEqual(result.final_state.cumulative_error_by_query[QUERY], 0.05)
        for attempt in result.attempts:
            self.assertEqual(attempt.candidate.m2_inputs.cumulative_error, 0.0)

    def test_supplied_batch_is_verified_once_and_kept_by_identity(self):
        instance = make_loop_instance(error_slo=0.025)
        state = inactive_state(instance)
        batch = evaluate_gh_phase2_query_candidates(instance, state, QUERY)
        result, evaluated = observe_query_evaluations(
            lambda: process_gh_phase2_query_candidates(instance, state, QUERY, batch)
        )

        self.assertEqual(len(evaluated), 1)
        self.assertIs(evaluated[0][0], state)
        self.assertIs(result.query_evaluation, batch)
        self.assertEqual(result.committed_attempt_count, 2)

    def test_partial_then_second_candidate_has_complete_hand_calculated_state(self):
        instance = make_loop_instance(error_slo=0.025)
        state = inactive_state(instance)
        evaluation = evaluate_gh_phase2_query_candidates(instance, state, QUERY)

        result = process_gh_phase2_query_candidates(instance, state, QUERY, evaluation)

        self.assertEqual(len(result.attempts), 2)
        first, second = result.attempts
        self.assertEqual(first.commit_result.diagnostic.allocation.proposed_routing_fraction, 0.5)
        self.assertEqual(second.commit_result.diagnostic.allocation.proposed_routing_fraction, 0.5)
        self.assertEqual(first.state_after.unmet_fraction_by_query[QUERY], 0.5)
        self.assertEqual(first.state_after.remaining_demand_by_query[QUERY], 0.5)
        self.assertEqual(first.state_after.cumulative_error_by_query[QUERY], 0.025)
        self.assertAlmostEqual(first.state_after.cumulative_delay_seconds_by_query[QUERY], 0.155)
        self.assertEqual(result.final_state.unmet_fraction_by_query[QUERY], 0.0)
        self.assertEqual(result.final_state.remaining_demand_by_query[QUERY], 0.0)
        # The printed batch is frozen: the second original x_bar remains .5.
        # Consequently cumulative error reaches .05 although epsilon=.025.
        self.assertEqual(second.candidate.m2_inputs.cumulative_error, 0.0)
        self.assertEqual(result.final_state.cumulative_error_by_query[QUERY], 0.05)
        self.assertAlmostEqual(result.final_state.cumulative_delay_seconds_by_query[QUERY], 0.31)
        self.assertEqual(tuple(item.gpu_count for item in result.final_state.deployments), (1, 1))
        self.assertAlmostEqual(result.final_state.accumulated_budget_cost_usd, 2.00200003)
        self.assertEqual(
            result.final_state.routing_fraction_by_candidate,
            {
                (QUERY, "model_a", TIER): 0.5,
                (QUERY, "model_b", TIER): 0.5,
            },
        )

    def test_default_5g_rejects_stale_candidate_but_loop_declares_entry_context(self):
        instance = make_loop_instance(error_slo=0.025)
        state = inactive_state(instance)
        evaluation = evaluate_gh_phase2_query_candidates(instance, state, QUERY)
        first = commit_gh_phase2_candidate(instance, state, evaluation.ranked_candidates[0])

        with self.assertRaisesRegex(ValueError, "declared state"):
            commit_gh_phase2_candidate(
                instance, first.state_after, evaluation.ranked_candidates[1]
            )

        loop_result = process_gh_phase2_query_candidates(
            instance, state, QUERY, evaluation
        )
        self.assertEqual(len(loop_result.attempts), 2)
        self.assertTrue(loop_result.attempts[1].committed)

    def test_one_partial_candidate_exhausts_list_with_residual_demand(self):
        instance = make_loop_instance(
            (("only", 1.0, 0.05, 0.001, 1.0),),
            error_slo=0.025,
        )

        _, result = evaluate_and_process(instance)

        self.assertEqual(len(result.attempts), 1)
        self.assertTrue(result.attempts[0].committed)
        self.assertEqual(result.final_state.unmet_fraction_by_query[QUERY], 0.5)
        self.assertEqual(
            result.termination_reason,
            GHPhase2QueryTerminationReason.CANDIDATES_EXHAUSTED_WITH_RESIDUAL_DEMAND,
        )

    def test_m3_upgrade_then_another_frozen_candidate_attempt(self):
        instance = make_loop_instance(
            (
                ("active", 1.0, 0.05, 0.001, 1.0),
                ("fresh", 1.0, 0.05, 0.001, 1.0),
            ),
            error_slo=0.025,
            delay_slo_seconds=0.2,
        )
        state = GHPhase2State(
            deployments=(
                GHPhase2Deployment("active", TIER, ParallelismConfig(1, 1)),
            ),
            query_order=(QUERY,),
            routing_fraction_by_candidate={},
            remaining_demand_by_query={QUERY: 1.0},
            unmet_fraction_by_query={QUERY: 1.0},
            cumulative_error_by_query={QUERY: 0.0},
            cumulative_delay_seconds_by_query={QUERY: 0.0},
            accumulated_budget_cost_usd=1.0,
            remaining_budget_usd=19.0,
        )

        evaluation, result = evaluate_and_process(instance, state)

        self.assertEqual(
            evaluation.ranked_candidates[0].candidate_configuration.source,
            GHPhase2ConfigurationSource.M3_ACTIVE_PAIR_UPGRADE,
        )
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(
            result.final_state.deployment_for_pair("active", TIER).configuration,
            ParallelismConfig(2, 1),
        )
        self.assertEqual(
            result.final_state.deployment_for_pair("fresh", TIER).configuration,
            ParallelismConfig(2, 1),
        )
        self.assertEqual(result.final_state.unmet_fraction_by_query[QUERY], 0.0)

    def test_later_budget_check_observes_earlier_successful_state(self):
        instance = make_loop_instance(error_slo=0.025, cost_budget_usd=1.5)

        _, result = evaluate_and_process(instance)

        first, second = result.attempts
        self.assertTrue(first.committed)
        self.assertFalse(second.committed)
        self.assertEqual(
            second.commit_result.diagnostic.rejection_reasons,
            (GHPhase2CommitRejectionReason.GLOBAL_BUDGET,),
        )
        self.assertEqual(
            second.commit_result.diagnostic.budget_comparison.prior_accumulated_budget_cost_usd,
            first.state_after.accumulated_budget_cost_usd,
        )
        self.assertEqual(result.final_state.unmet_fraction_by_query[QUERY], 0.5)
        self.assertIs(second.state_after, second.state_before)


class GHPhase2QueryCompatibilityAndTraceTests(unittest.TestCase):
    """Protect old routes, exact input evidence, trace chaining, and determinism."""

    def test_previous_query_routing_is_preserved(self):
        base = make_loop_instance()
        template = base.queries[0]
        queries = (
            replace(template, id="q_prior", arrival_rate_per_hour=20.0),
            replace(template, id=QUERY, arrival_rate_per_hour=10.0),
        )
        triples = tuple(
            (query.id, model.id, TIER)
            for query in queries
            for model in base.models
        )
        instance = replace(
            base,
            queries=queries,
            base_error_rate={
                (query.id, model.id): 0.05
                for query in queries
                for model in base.models
            },
            compute_delay_seconds_per_token={key: 0.10 for key in triples},
            communication_delay_seconds_per_token={key: 0.01 for key in triples},
            compute_gflop_per_token={key: 1.0 for key in triples},
            kv_cache_residency_seconds={key: 1.0 for key in triples},
        )
        prior_route = ("q_prior", "model_a", TIER)
        prior_cost = 1.001000015
        state = GHPhase2State(
            deployments=(GHPhase2Deployment("model_a", TIER, ParallelismConfig(1, 1)),),
            query_order=("q_prior", QUERY),
            routing_fraction_by_candidate={prior_route: 0.25},
            remaining_demand_by_query={"q_prior": 0.75, QUERY: 1.0},
            unmet_fraction_by_query={"q_prior": 0.75, QUERY: 1.0},
            cumulative_error_by_query={"q_prior": 0.0125, QUERY: 0.0},
            cumulative_delay_seconds_by_query={"q_prior": 0.0775, QUERY: 0.0},
            accumulated_budget_cost_usd=prior_cost,
            remaining_budget_usd=instance.cost_budget_usd - prior_cost,
        )

        _, result = evaluate_and_process(instance, state)

        self.assertEqual(result.final_state.routing_fraction_by_candidate[prior_route], 0.25)
        self.assertEqual(state.routing_fraction_by_candidate, {prior_route: 0.25})

    def test_stale_evaluation_is_rejected_before_any_commit(self):
        instance = make_loop_instance()
        entry = inactive_state(instance)
        evaluation = evaluate_gh_phase2_query_candidates(instance, entry, QUERY)
        changed = replace(
            entry,
            remaining_demand_by_query={QUERY: 0.5},
            unmet_fraction_by_query={QUERY: 0.5},
        )

        with patch.object(query_module, "_commit_gh_phase2_candidate") as commit:
            with self.assertRaisesRegex(ValueError, "stale or incompatible"):
                process_gh_phase2_query_candidates(
                    instance, changed, QUERY, evaluation
                )

        commit.assert_not_called()

    def test_wrong_query_or_reordered_evidence_is_rejected(self):
        instance = make_loop_instance()
        state = inactive_state(instance)
        evaluation = evaluate_gh_phase2_query_candidates(instance, state, QUERY)
        wrong_query = replace(evaluation, query_id="other_query")
        reordered = replace(
            evaluation,
            ranked_candidates=tuple(reversed(evaluation.ranked_candidates)),
        )

        with self.assertRaisesRegex(ValueError, "different query"):
            process_gh_phase2_query_candidates(instance, state, QUERY, wrong_query)
        with self.assertRaisesRegex(ValueError, "stale or incompatible"):
            process_gh_phase2_query_candidates(instance, state, QUERY, reordered)

    def test_supplied_batch_from_same_ids_with_other_coefficients_is_rejected(self):
        instance = make_loop_instance()
        state = inactive_state(instance)
        evaluation = evaluate_gh_phase2_query_candidates(instance, state, QUERY)
        changed = replace(
            instance,
            base_error_rate={key: 0.08 for key in instance.base_error_rate},
        )

        with patch.object(query_module, "_commit_gh_phase2_candidate") as commit:
            with self.assertRaisesRegex(ValueError, "stale or incompatible"):
                process_gh_phase2_query_candidates(changed, state, QUERY, evaluation)
        commit.assert_not_called()

    def test_owned_evaluation_rejects_unknown_query_before_any_commit(self):
        instance = make_loop_instance()
        state = inactive_state(instance)
        with patch.object(query_module, "_commit_gh_phase2_candidate") as commit:
            with self.assertRaisesRegex(ValueError, "unknown query ID"):
                process_gh_phase2_query_candidates(instance, state, "missing")
        commit.assert_not_called()

    def test_supplied_batch_type_is_rejected_before_any_commit(self):
        instance = make_loop_instance()
        state = inactive_state(instance)
        with patch.object(query_module, "_commit_gh_phase2_candidate") as commit:
            with self.assertRaisesRegex(ValueError, "GHPhase2QueryEvaluation"):
                process_gh_phase2_query_candidates(instance, state, QUERY, ())
        commit.assert_not_called()

    def test_trace_reconstructs_every_5g_transition_and_rank(self):
        instance = make_loop_instance(error_slo=0.025)
        evaluation, result = evaluate_and_process(instance)

        current = result.initial_state
        for index, attempt in enumerate(result.attempts):
            self.assertEqual(attempt.attempt_index, index)
            self.assertEqual(attempt.ranked_candidate_index, index)
            self.assertEqual(attempt.candidate, evaluation.ranked_candidates[index])
            self.assertEqual(attempt.rank_key, attempt.candidate.rank_key)
            self.assertIs(attempt.state_before, current)
            self.assertIs(attempt.commit_result.state_before, current)
            self.assertIs(attempt.state_after, attempt.commit_result.state_after)
            current = attempt.state_after
        self.assertIs(current, result.final_state)
        self.assertNotIn("final_state", {item.name for item in fields(result)})
        self.assertNotIn("query_id", {item.name for item in fields(result)})
        self.assertTrue(
            {"rank_key", "state_before", "state_after"}.isdisjoint(
                item.name for item in fields(result.attempts[0])
            )
        )

    def test_exact_ties_are_stable_and_reversible_with_model_order(self):
        forward = make_loop_instance(
            (("model_z", 1.0, 0.05, 0.001, 1.0), ("model_a", 1.0, 0.05, 0.001, 1.0))
        )
        reverse = make_loop_instance(
            (("model_a", 1.0, 0.05, 0.001, 1.0), ("model_z", 1.0, 0.05, 0.001, 1.0))
        )

        forward_eval, forward_result = evaluate_and_process(forward)
        reverse_eval, reverse_result = evaluate_and_process(reverse)

        self.assertEqual(tuple(c.model_id for c in forward_eval.ranked_candidates), ("model_z", "model_a"))
        self.assertEqual(tuple(c.model_id for c in reverse_eval.ranked_candidates), ("model_a", "model_z"))
        self.assertEqual(forward_result.attempts[0].candidate.model_id, "model_z")
        self.assertEqual(reverse_result.attempts[0].candidate.model_id, "model_a")

    def test_results_are_deterministic_immutable_and_inputs_unchanged(self):
        instance = make_loop_instance(error_slo=0.025)
        state = inactive_state(instance)
        evaluation = evaluate_gh_phase2_query_candidates(instance, state, QUERY)

        first = process_gh_phase2_query_candidates(instance, state, QUERY, evaluation)
        second = process_gh_phase2_query_candidates(instance, state, QUERY, evaluation)

        self.assertEqual(first, second)
        self.assertEqual(state.routing_fraction_by_candidate, {})
        self.assertEqual(evaluation, evaluate_gh_phase2_query_candidates(instance, state, QUERY))
        with self.assertRaises(FrozenInstanceError):
            first.initial_state = state
        with self.assertRaises(FrozenInstanceError):
            first.attempts[0].ranked_candidate_index = 99



if __name__ == "__main__":
    unittest.main()
