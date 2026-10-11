"""Hand-derived tests for the GH Phase-2 outer query loop."""

import sys
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    GHPhase1Config,
    GHPhase2ConfigurationSource,
    GHPhase2QueryTerminationReason,
    GHPhase2RunCompletionStatus,
    initialize_gh_phase2_state,
    process_gh_phase2_query_candidates,
    run_gh_phase1,
    run_gh_phase2,
)
from llm_resource_allocation.algorithms import gh_phase2_run as run_module
from llm_resource_allocation.algorithms import gh_phase2_query as query_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig


TIER = "tier"
MODEL = "model"


def make_outer_instance(
    query_specs=(
        ("q_high", 30.0, 1.0, 0.10),
        ("q_mid", 20.0, 0.2, 0.10),
        ("q_low", 10.0, 1.0, 0.10),
    ),
    *,
    cost_budget_usd=20.0,
):
    """Return a transparent multi-query outer-loop fixture.

    Query specs are ``(id, lambda, delay SLO, error SLO)``.  TP1 has a
    0.31-second delay and TP2 has a 0.16-second delay, so the default first
    query activates TP1 and the second query must upgrade that same pair.
    """

    base = make_minimal_tiny_instance()
    template = base.queries[0]
    queries = tuple(
        replace(
            template,
            id=query_id,
            arrival_rate_per_hour=arrival,
            delay_slo_seconds=delay_slo,
            error_slo=error_slo,
        )
        for query_id, arrival, delay_slo, error_slo in query_specs
    )
    model = replace(base.models[0], id=MODEL)
    tier = replace(
        base.gpu_tiers[0],
        id=TIER,
        memory_capacity_gb=8.0,
        hourly_rental_usd=1.0,
        tensor_parallel_degrees=(1, 2),
        pipeline_parallel_depths=(1,),
    )
    triples = tuple((query.id, MODEL, TIER) for query in queries)
    return replace(
        base,
        id="gh_phase2_outer_loop_fixture",
        queries=queries,
        models=(model,),
        gpu_tiers=(tier,),
        base_error_rate={(query.id, MODEL): 0.05 for query in queries},
        compute_delay_seconds_per_token={key: 0.10 for key in triples},
        communication_delay_seconds_per_token={key: 0.01 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=cost_budget_usd,
    )


def initialized_state(instance):
    """Enter Phase 2 from an already completed zero-cap Phase 1."""

    phase1 = run_gh_phase1(instance, GHPhase1Config(0.0))
    return initialize_gh_phase2_state(instance, phase1)


def sole_candidate(trace):
    """Return the fixture's sole model-tier candidate evaluation."""

    return trace.query_evaluation.candidate_evaluations[0]


class GHPhase2OuterOrderAndCompositionTests(unittest.TestCase):
    """Protect the exact one-pass order and 5F-to-5H composition."""

    def test_descending_lambda_order_processes_every_query_exactly_once(self):
        instance = make_outer_instance()
        state = initialized_state(instance)

        result = run_gh_phase2(instance, state)

        self.assertEqual(result.processed_query_ids, state.query_order)
        self.assertEqual(result.processed_query_ids, ("q_high", "q_mid", "q_low"))
        self.assertEqual(len(result.query_traces), len(instance.queries))
        self.assertEqual(len(set(result.processed_query_ids)), len(instance.queries))

    def test_equal_lambda_ties_preserve_instance_order_not_lexical_ids(self):
        instance = make_outer_instance(
            (
                ("q_z", 30.0, 1.0, 0.10),
                ("q_a", 30.0, 1.0, 0.10),
                ("q_last", 10.0, 1.0, 0.10),
            )
        )
        state = initialized_state(instance)

        result = run_gh_phase2(instance, state)

        self.assertEqual(result.processed_query_ids, ("q_z", "q_a", "q_last"))

    def test_each_query_evaluates_once_across_aliases_and_has_the_exact_handoff(self):
        instance = make_outer_instance()
        state = initialized_state(instance)
        evaluated = []
        evaluator_code = query_module._evaluate_gh_phase2_query_candidates.__code__
        previous_profile = sys.getprofile()

        def profile(frame, event, arg):
            if event == "call" and frame.f_code is evaluator_code:
                evaluated.append((frame.f_locals["state"], frame.f_locals["query_id"]))

        with patch.object(
            run_module,
            "_process_gh_phase2_query",
            wraps=run_module._process_gh_phase2_query,
        ) as process:
            sys.setprofile(profile)
            try:
                result = run_gh_phase2(instance, state)
            finally:
                sys.setprofile(previous_profile)

        self.assertEqual(len(evaluated), len(instance.queries))
        self.assertEqual(process.call_count, len(instance.queries))
        for index, trace in enumerate(result.query_traces):
            self.assertIs(evaluated[index][0], trace.state_at_query_entry)
            self.assertEqual(evaluated[index][1], trace.query_id)
            self.assertIs(process.call_args_list[index].args[1], trace.state_at_query_entry)
            self.assertEqual(process.call_args_list[index].args[2], trace.query_id)
            self.assertIs(trace.query_process_result.query_evaluation, trace.query_evaluation)

    def test_each_trace_matches_direct_5h_on_the_same_entry_and_evaluation(self):
        instance = make_outer_instance()
        result = run_gh_phase2(instance, initialized_state(instance))

        for trace in result.query_traces:
            direct = process_gh_phase2_query_candidates(
                instance,
                trace.state_at_query_entry,
                trace.query_id,
                trace.query_evaluation,
            )
            self.assertEqual(trace.query_process_result, direct)

    def test_trace_positions_arrivals_and_state_chain_are_exact(self):
        instance = make_outer_instance()
        state = initialized_state(instance)
        result = run_gh_phase2(instance, state)

        current = state
        expected_arrivals = (30.0, 20.0, 10.0)
        for position, (trace, arrival) in enumerate(
            zip(result.query_traces, expected_arrivals)
        ):
            self.assertEqual(trace.query_position, position)
            self.assertEqual(trace.arrival_rate_per_hour, arrival)
            self.assertIs(trace.state_at_query_entry, current)
            self.assertIs(trace.query_process_result.initial_state, current)
            self.assertIs(
                trace.query_process_result.query_evaluation,
                trace.query_evaluation,
            )
            self.assertIs(
                trace.state_after_query,
                trace.query_process_result.final_state,
            )
            current = trace.state_after_query
        self.assertIs(result.final_state, current)


class GHPhase2OuterStatePersistenceTests(unittest.TestCase):
    """Verify shared and query-indexed state across query boundaries."""

    def test_first_query_route_deployment_gpu_and_budget_persist_to_second(self):
        instance = make_outer_instance()
        result = run_gh_phase2(instance, initialized_state(instance))
        first, second = result.query_traces[:2]
        route = ("q_high", MODEL, TIER)

        self.assertEqual(first.state_after_query.routing_fraction_by_candidate[route], 1.0)
        self.assertEqual(first.state_after_query.deployments[0].configuration, ParallelismConfig(1, 1))
        self.assertGreater(first.state_after_query.accumulated_budget_cost_usd, 1.0)
        self.assertIs(second.state_at_query_entry, first.state_after_query)
        self.assertEqual(second.state_at_query_entry.routing_fraction_by_candidate[route], 1.0)
        self.assertEqual(second.state_at_query_entry.deployments[0].gpu_count, 1)
        self.assertEqual(
            second.state_at_query_entry.accumulated_budget_cost_usd,
            first.state_after_query.accumulated_budget_cost_usd,
        )

    def test_untouched_later_query_dynamic_state_remains_at_entry_values(self):
        instance = make_outer_instance()
        result = run_gh_phase2(instance, initialized_state(instance))
        after_first = result.query_traces[0].state_after_query

        self.assertEqual(after_first.remaining_demand_by_query["q_high"], 0.0)
        self.assertEqual(after_first.unmet_fraction_by_query["q_high"], 0.0)
        self.assertGreater(after_first.cumulative_error_by_query["q_high"], 0.0)
        self.assertGreater(after_first.cumulative_delay_seconds_by_query["q_high"], 0.0)
        for query_id in ("q_mid", "q_low"):
            self.assertEqual(after_first.remaining_demand_by_query[query_id], 1.0)
            self.assertEqual(after_first.unmet_fraction_by_query[query_id], 1.0)
            self.assertEqual(after_first.cumulative_error_by_query[query_id], 0.0)
            self.assertEqual(after_first.cumulative_delay_seconds_by_query[query_id], 0.0)

    def test_fresh_second_query_evaluation_observes_first_query_shared_state(self):
        instance = make_outer_instance()
        result = run_gh_phase2(instance, initialized_state(instance))
        second = result.query_traces[1]
        candidate = sole_candidate(second)

        self.assertTrue(candidate.is_active_pair)
        self.assertEqual(candidate.current_gpu_count, 1)
        self.assertEqual(candidate.m2_inputs.current_gpu_count, 1)
        self.assertEqual(
            candidate.m3_upgrade.remaining_budget_usd,
            second.state_at_query_entry.remaining_budget_usd,
        )
        self.assertLess(candidate.m3_upgrade.remaining_budget_usd, instance.cost_budget_usd)

    def test_first_query_activation_triggers_second_query_m3_upgrade(self):
        instance = make_outer_instance()
        result = run_gh_phase2(instance, initialized_state(instance))
        first, second = result.query_traces[:2]
        first_candidate = sole_candidate(first)
        second_candidate = sole_candidate(second)

        self.assertEqual(
            first_candidate.candidate_configuration.source,
            GHPhase2ConfigurationSource.M1_INACTIVE_PAIR,
        )
        self.assertEqual(
            second_candidate.candidate_configuration.source,
            GHPhase2ConfigurationSource.M3_ACTIVE_PAIR_UPGRADE,
        )
        self.assertEqual(second_candidate.current_configuration, ParallelismConfig(1, 1))
        self.assertEqual(second_candidate.effective_configuration, ParallelismConfig(2, 1))
        self.assertEqual(second.state_after_query.deployments[0].configuration, ParallelismConfig(2, 1))

    def test_routes_are_not_recommitted_or_updated_twice_between_queries(self):
        instance = make_outer_instance()
        result = run_gh_phase2(instance, initialized_state(instance))

        self.assertEqual(
            result.final_state.routing_fraction_by_candidate,
            {
                ("q_high", MODEL, TIER): 1.0,
                ("q_mid", MODEL, TIER): 1.0,
                ("q_low", MODEL, TIER): 1.0,
            },
        )
        self.assertEqual(
            tuple(
                trace.query_process_result.committed_attempt_count
                for trace in result.query_traces
            ),
            (1, 1, 1),
        )


class GHPhase2OuterContinuationAndCompletionTests(unittest.TestCase):
    """Protect residual/no-candidate continuation and narrow completion."""

    def test_residual_demand_in_first_query_does_not_stop_later_queries(self):
        instance = make_outer_instance(
            (
                ("q_partial", 30.0, 1.0, 0.025),
                ("q_later", 20.0, 1.0, 0.10),
            )
        )
        result = run_gh_phase2(instance, initialized_state(instance))

        first, second = result.query_traces
        self.assertEqual(
            first.query_process_result.termination_reason,
            GHPhase2QueryTerminationReason.CANDIDATES_EXHAUSTED_WITH_RESIDUAL_DEMAND,
        )
        self.assertEqual(first.state_after_query.remaining_demand_by_query["q_partial"], 0.5)
        self.assertEqual(second.query_id, "q_later")
        self.assertEqual(second.state_after_query.remaining_demand_by_query["q_later"], 0.0)
        self.assertEqual(result.processed_query_ids, ("q_partial", "q_later"))

    def test_no_rankable_candidate_for_one_query_does_not_affect_next(self):
        instance = make_outer_instance(
            (
                ("q_none", 30.0, 1.0, 0.0),
                ("q_later", 20.0, 1.0, 0.10),
            )
        )
        result = run_gh_phase2(instance, initialized_state(instance))

        first, second = result.query_traces
        self.assertEqual(first.query_evaluation.ranked_candidates, ())
        self.assertEqual(first.query_process_result.attempts, ())
        self.assertIs(first.state_after_query, first.state_at_query_entry)
        self.assertEqual(second.state_after_query.remaining_demand_by_query["q_later"], 0.0)

    def test_result_exposes_per_query_residual_demand_and_only_completion_status(self):
        instance = make_outer_instance(
            (
                ("q_partial", 30.0, 1.0, 0.025),
                ("q_full", 20.0, 1.0, 0.10),
            )
        )
        result = run_gh_phase2(instance, initialized_state(instance))

        self.assertEqual(
            result.residual_demand_by_query,
            {"q_partial": 0.5, "q_full": 0.0},
        )
        self.assertEqual(
            result.completion_status,
            GHPhase2RunCompletionStatus.ALL_QUERIES_PROCESSED,
        )
        self.assertTrue(result.all_queries_processed)
        self.assertEqual(tuple(GHPhase2RunCompletionStatus), (GHPhase2RunCompletionStatus.ALL_QUERIES_PROCESSED,))


class GHPhase2OuterBoundaryTests(unittest.TestCase):
    """Protect determinism, immutability, stale-state rejection, and scope."""

    def test_results_are_deterministic_and_input_state_is_unchanged(self):
        instance = make_outer_instance()
        state = initialized_state(instance)

        first = run_gh_phase2(instance, state)
        second = run_gh_phase2(instance, state)

        self.assertEqual(first, second)
        self.assertEqual(state.routing_fraction_by_candidate, {})
        self.assertEqual(state.remaining_demand_by_query, {query.id: 1.0 for query in instance.queries})
        self.assertEqual(state.deployments, ())

    def test_run_result_trace_and_residual_mapping_are_immutable(self):
        instance = make_outer_instance()
        result = run_gh_phase2(instance, initialized_state(instance))

        with self.assertRaises(FrozenInstanceError):
            result.initial_state = result.final_state
        with self.assertRaises(FrozenInstanceError):
            result.query_traces[0].query_position = 99
        with self.assertRaises(TypeError):
            result.residual_demand_by_query["q_high"] = 1.0
        self.assertIs(
            result.residual_demand_by_query, result.final_state.remaining_demand_by_query
        )
        self.assertTrue(
            {"final_state", "processed_query_ids", "residual_demand_by_query"}.isdisjoint(
                item.name for item in fields(result)
            )
        )
        self.assertTrue(
            {"query_id", "state_at_query_entry", "query_evaluation", "state_after_query"}.isdisjoint(
                item.name for item in fields(result.query_traces[0])
            )
        )

    def test_stale_or_reordered_phase2_state_is_rejected_before_5f_or_5h(self):
        instance = make_outer_instance()
        state = initialized_state(instance)
        stale = replace(state, query_order=tuple(reversed(state.query_order)))

        with patch.object(
            query_module, "_evaluate_gh_phase2_query_candidates"
        ) as evaluate, patch.object(
            run_module, "_process_gh_phase2_query"
        ) as process:
            with self.assertRaisesRegex(ValueError, "descending arrival rate"):
                run_gh_phase2(instance, stale)

        evaluate.assert_not_called()
        process.assert_not_called()



if __name__ == "__main__":
    unittest.main()
