"""Hand-derived tests for the pure GH Phase-1 greedy activation loop."""

import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    GHPhase1Config,
    GHPhase1State,
    GHPhase1TerminationReason,
    apply_gh_phase1_activation,
    evaluate_gh_phase1_candidates,
    initialize_gh_phase1_state,
    rank_phase1_candidates,
    run_gh_phase1,
)
from llm_resource_allocation.algorithms import gh_phase1 as gh_phase1_module
from llm_resource_allocation.data.tiny_instances import make_minimal_tiny_instance


def make_loop_instance(
    *,
    query_ids: tuple[str, ...] = ("q1", "q2", "q3"),
    model_ids: tuple[str, ...] = ("model_a", "model_b", "model_c"),
    cost_budget_usd: float = 10.0,
    error_overrides: dict[tuple[str, str], float] | None = None,
):
    """Build a transparent set-cover fixture with scores 2, 1/2, and 1/3.

    ``model_a`` covers q1/q2 using one GPU, ``model_b`` covers q1 using two,
    and ``model_c`` covers q3 using three.  All costs use one hour at $1/GPU-h.
    """

    base = make_minimal_tiny_instance()
    query_template = base.queries[0]
    queries = tuple(
        replace(
            query_template,
            id=query_id,
            delay_slo_seconds=1.0,
            error_slo=0.10,
        )
        for query_id in query_ids
    )
    weight_by_model = {
        "model_a": 1.0,
        "model_b": 2.0,
        "model_c": 3.0,
        "model_d": 1.0,
    }
    models = tuple(
        replace(
            base.models[0],
            id=model_id,
            weight_gb=weight_by_model[model_id],
        )
        for model_id in model_ids
    )
    tier = replace(
        base.gpu_tiers[0],
        id="tier",
        memory_capacity_gb=1.0,
        hourly_rental_usd=1.0,
        tensor_parallel_degrees=(1, 2, 3),
        pipeline_parallel_depths=(1,),
    )
    passing = {
        ("q1", "model_a"),
        ("q2", "model_a"),
        ("q1", "model_b"),
        ("q3", "model_c"),
        ("q1", "model_d"),
    }
    errors = {
        (query.id, model.id): (
            0.05 if (query.id, model.id) in passing else 0.20
        )
        for query in queries
        for model in models
    }
    if error_overrides:
        errors.update(error_overrides)
    triples = tuple(
        (query.id, model.id, tier.id)
        for query in queries
        for model in models
    )
    return replace(
        base,
        id="gh_phase1_loop_fixture",
        queries=queries,
        models=models,
        gpu_tiers=(tier,),
        base_error_rate=errors,
        compute_delay_seconds_per_token={key: 0.01 for key in triples},
        communication_delay_seconds_per_token={key: 0.0 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=cost_budget_usd,
        horizon_hours=1.0,
    )


class GHPhase1InitializationTests(unittest.TestCase):
    """Protect the exact Phase-1 initial state and explicit beta parameter."""

    def test_initializer_sets_iunc_all_queries_empty_y_and_beta_delta_cap(self):
        instance = make_loop_instance(cost_budget_usd=10.0)

        state = initialize_gh_phase1_state(instance, GHPhase1Config(0.8))

        self.assertEqual(state.uncovered_query_ids, ("q1", "q2", "q3"))
        self.assertEqual(state.active_deployments, ())
        self.assertEqual(state.accumulated_rental_cost_usd, 0.0)
        self.assertEqual(state.rental_cap_usd, 8.0)
        self.assertEqual(state.iteration_count, 0)
        self.assertNotIn("iteration_count", {item.name for item in fields(state)})

    def test_config_is_explicit_immutable_and_validated(self):
        config = GHPhase1Config(0.8)

        with self.assertRaises(FrozenInstanceError):
            config.budget_fraction = 0.5
        for invalid in (-0.1, 1.1, float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GHPhase1Config(invalid)

    def test_public_candidate_boundary_checks_supplied_state_accounting(self):
        instance = make_loop_instance()
        state = initialize_gh_phase1_state(instance, GHPhase1Config(0.8))
        for changed, message in (
            (replace(state, accumulated_rental_cost_usd=1.0), "activation costs"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_gh_phase1_candidates(instance, changed)


class GHPhase1LoopHandCases(unittest.TestCase):
    """Exercise one and multiple greedy set-cover activations by hand."""

    def test_one_iteration_covers_all_and_records_exact_activation(self):
        instance = make_loop_instance(query_ids=("q1", "q2"))

        result = run_gh_phase1(instance, GHPhase1Config(0.8))

        self.assertEqual(result.termination_reason, GHPhase1TerminationReason.ALL_QUERIES_COVERED)
        self.assertEqual(result.selected_pair_sequence, (("model_a", "tier"),))
        self.assertEqual(len(result.iterations), 1)
        deployment = result.final_state.active_deployments[0]
        self.assertEqual(deployment.covered_query_ids, ("q1", "q2"))
        self.assertEqual(deployment.gpu_count, 1)
        self.assertEqual(deployment.activation_cost_usd, 1.0)
        self.assertEqual(result.final_state.uncovered_query_ids, ())

    def test_multiple_iterations_recompute_candidates_from_current_iunc(self):
        instance = make_loop_instance()

        result = run_gh_phase1(instance, GHPhase1Config(1.0))

        self.assertEqual(
            result.selected_pair_sequence,
            (("model_a", "tier"), ("model_c", "tier")),
        )
        self.assertEqual(result.iterations[0].state_before.uncovered_query_ids, ("q1", "q2", "q3"))
        self.assertEqual(result.iterations[0].state_after.uncovered_query_ids, ("q3",))
        second = result.iterations[1]
        self.assertEqual(second.state_before, result.iterations[0].state_after)
        self.assertEqual(
            tuple(evaluation.coverage.uncovered_query_ids for evaluation in second.candidate_evaluations),
            (("q3",), ("q3",), ("q3",)),
        )
        self.assertEqual(second.candidate_evaluations[0].coverage.covered_query_ids, ())
        self.assertEqual(second.candidate_evaluations[1].coverage.covered_query_ids, ())
        self.assertEqual(second.selected_candidate.model_id, "model_c")
        self.assertEqual(result.final_state.accumulated_rental_cost_usd, 4.0)
        self.assertEqual(tuple(d.gpu_count for d in result.final_state.active_deployments), (1, 3))

    def test_transition_removes_exact_coverage_and_preserves_remaining_order(self):
        instance = make_loop_instance(
            error_overrides={
                ("q1", "model_b"): 0.20,
                ("q2", "model_b"): 0.05,
            }
        )
        state = initialize_gh_phase1_state(instance, GHPhase1Config(1.0))
        model_b = evaluate_gh_phase1_candidates(instance, state)[1]

        updated = apply_gh_phase1_activation(instance, state, model_b)

        self.assertEqual(model_b.coverage.covered_query_ids, ("q2",))
        self.assertEqual(updated.uncovered_query_ids, ("q1", "q3"))
        self.assertEqual(state.uncovered_query_ids, ("q1", "q2", "q3"))

    def test_all_jk_candidates_are_recomputed_in_semantic_order(self):
        instance = make_loop_instance()
        state = initialize_gh_phase1_state(instance, GHPhase1Config(1.0))

        evaluations = evaluate_gh_phase1_candidates(instance, state)

        self.assertEqual(
            tuple((item.model_id, item.tier_id) for item in evaluations),
            (("model_a", "tier"), ("model_b", "tier"), ("model_c", "tier")),
        )


class GHPhase1TerminationTests(unittest.TestCase):
    """Pin literal pre-iteration cap behavior and safe empty-argmax handling."""

    def test_no_rankable_candidate_stops_without_activation(self):
        instance = make_loop_instance(
            query_ids=("q3",), model_ids=("model_a", "model_b")
        )

        result = run_gh_phase1(instance, GHPhase1Config(1.0))

        self.assertEqual(result.termination_reason, GHPhase1TerminationReason.NO_RANKABLE_CANDIDATE)
        self.assertEqual(result.iterations, ())
        self.assertEqual(result.final_state.uncovered_query_ids, ("q3",))
        self.assertEqual(len(result.terminal_candidate_evaluations), 2)
        self.assertFalse(
            any(
                candidate.is_rankable
                for candidate in result.terminal_candidate_evaluations
            )
        )

    def test_activation_reaching_cap_is_committed_then_cap_stops_loop(self):
        instance = make_loop_instance(cost_budget_usd=2.0)

        result = run_gh_phase1(instance, GHPhase1Config(0.5))

        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(result.final_state.accumulated_rental_cost_usd, 1.0)
        self.assertEqual(result.final_state.rental_cap_usd, 1.0)
        self.assertEqual(result.final_state.uncovered_query_ids, ("q3",))
        self.assertEqual(result.termination_reason, GHPhase1TerminationReason.RENTAL_CAP_REACHED)

    def test_activation_overshooting_cap_is_committed_without_prospective_check(self):
        instance = make_loop_instance(cost_budget_usd=1.0)

        result = run_gh_phase1(instance, GHPhase1Config(0.5))

        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(result.final_state.accumulated_rental_cost_usd, 1.0)
        self.assertEqual(result.final_state.rental_cap_usd, 0.5)
        self.assertEqual(result.termination_reason, GHPhase1TerminationReason.RENTAL_CAP_REACHED)

    def test_zero_cap_stops_before_candidate_evaluation(self):
        instance = make_loop_instance()
        with patch.object(
            gh_phase1_module,
            "evaluate_gh_phase1_candidates",
            side_effect=AssertionError("guard should stop before evaluation"),
        ):
            result = run_gh_phase1(instance, GHPhase1Config(0.0))

        self.assertEqual(result.iterations, ())
        self.assertEqual(result.termination_reason, GHPhase1TerminationReason.RENTAL_CAP_REACHED)

    def test_all_covered_reason_precedes_cap_when_both_become_true(self):
        instance = make_loop_instance(query_ids=("q1", "q2"), cost_budget_usd=2.0)

        result = run_gh_phase1(instance, GHPhase1Config(0.5))

        self.assertEqual(result.final_state.accumulated_rental_cost_usd, 1.0)
        self.assertEqual(result.final_state.rental_cap_usd, 1.0)
        self.assertEqual(result.termination_reason, GHPhase1TerminationReason.ALL_QUERIES_COVERED)


class GHPhase1DeterminismAndIsolationTests(unittest.TestCase):
    """Keep Phase 1 deterministic, auditable, immutable, and solver-free."""

    def test_exact_tie_uses_instance_semantic_order(self):
        forward = make_loop_instance(
            query_ids=("q1",), model_ids=("model_a", "model_d")
        )
        reverse = make_loop_instance(
            query_ids=("q1",), model_ids=("model_d", "model_a")
        )

        self.assertEqual(run_gh_phase1(forward, GHPhase1Config(1.0)).selected_pair_sequence, (("model_a", "tier"),))
        self.assertEqual(run_gh_phase1(reverse, GHPhase1Config(1.0)).selected_pair_sequence, (("model_d", "tier"),))

    def test_repeated_runs_are_equal_and_inputs_remain_unchanged(self):
        instance = make_loop_instance()
        config = GHPhase1Config(1.0)
        original_queries = instance.query_ids

        first = run_gh_phase1(instance, config)
        second = run_gh_phase1(instance, config)

        self.assertEqual(first, second)
        self.assertEqual(instance.query_ids, original_queries)
        self.assertEqual(config, GHPhase1Config(1.0))
        with self.assertRaises(FrozenInstanceError):
            first.final_state.uncovered_query_ids = ()
        with self.assertRaises(FrozenInstanceError):
            first.iterations[0].state_after = GHPhase1State((), (), 0.0, 0.0)

    def test_loop_reuses_5d_stable_ranking_once_per_activation_attempt(self):
        instance = make_loop_instance()
        with patch.object(
            gh_phase1_module,
            "rank_phase1_candidates",
            wraps=rank_phase1_candidates,
        ) as ranking:
            result = run_gh_phase1(instance, GHPhase1Config(1.0))

        self.assertEqual(ranking.call_count, len(result.iterations))

    def test_trace_reconstructs_every_state_transition(self):
        result = run_gh_phase1(make_loop_instance(), GHPhase1Config(1.0))

        state = result.initial_state
        for index, trace in enumerate(result.iterations):
            self.assertEqual(trace.iteration_index, index)
            self.assertEqual(trace.state_before, state)
            self.assertEqual(trace.selected_candidate, trace.ranked_candidates[0])
            self.assertIn(trace.selected_candidate, trace.candidate_evaluations)
            state = trace.state_after
        self.assertEqual(state, result.final_state)
        self.assertIs(result.final_state, result.iterations[-1].state_after)
        self.assertNotIn("final_state", {item.name for item in fields(result)})
        self.assertNotIn(
            "selected_candidate", {item.name for item in fields(result.iterations[0])}
        )
        self.assertNotIn(
            "iteration_index", {item.name for item in fields(result.iterations[0])}
        )



if __name__ == "__main__":
    unittest.main()
