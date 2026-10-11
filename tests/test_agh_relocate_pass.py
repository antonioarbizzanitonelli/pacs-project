"""Hand-derived tests for deterministic AGH relocate pass orchestration."""

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    AGH_RELOCATE_MAX_PASSES,
    AGHRelocateRejectionReason,
    AGHRelocateRunTerminationReason,
    AGHRelocateSourceStatus,
    evaluate_agh_relocate_move,
    run_agh_relocate,
    run_agh_relocate_pass,
)
from llm_resource_allocation.algorithms import agh_relocate_pass as pass_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import FoundationModel
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.validation.placement import PlacementValidator
from llm_resource_allocation.algorithms._agh_placement_ops import (
    _AGHPlacementEvaluation,
    _build_agh_search_context,
)


TIER = "gpu_basic_fp16"


def make_scan_instance(
    model_weights,
    *,
    query_ids=("q_0",),
):
    """Build a transparent equal-delay/error lattice with explicit weights."""

    base = make_minimal_tiny_instance()
    queries = tuple(
        replace(
            base.queries[0],
            id=query_id,
            arrival_rate_per_hour=1.0,
            max_unmet_fraction=0.0,
        )
        for query_id in query_ids
    )
    models = tuple(
        FoundationModel(model_id, weight, 2.0e-6)
        for model_id, weight in model_weights
    )
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=20.0,
        tensor_parallel_degrees=(1,),
        pipeline_parallel_depths=(1,),
    )
    triples = tuple(
        (query.id, model.id, TIER)
        for query in queries
        for model in models
    )
    return replace(
        base,
        id="agh_relocate_pass_hand_fixture",
        queries=queries,
        models=models,
        gpu_tiers=(tier,),
        base_error_rate={
            (query.id, model.id): 0.02
            for query in queries
            for model in models
        },
        compute_delay_seconds_per_token={key: 0.01 for key in triples},
        communication_delay_seconds_per_token={key: 0.001 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
    )


def make_scan_state(instance, assignment_by_query):
    """Activate every pair and route each query fully to its supplied model."""

    route_keys = tuple(
        (query_id, model_id, TIER)
        for query_id in instance.query_ids
        for model_id in instance.model_ids
    )
    x = {
        key: float(assignment_by_query[key[0]] == key[1])
        for key in route_keys
    }
    z = {key: int(value > 0.0) for key, value in x.items()}
    pairs = tuple((model_id, TIER) for model_id in instance.model_ids)
    q = {pair: 1 for pair in pairs}
    u = {query_id: 0.0 for query_id in instance.query_ids}
    w = {(*pair, 1, 1): 1 for pair in pairs}
    y = {pair: 1 for pair in pairs}
    routing = RoutingDecisionState(x, z, q, u)
    parallelism = ParallelismDecisionState(routing, w, y)
    v = {(*key, 1, 1): value for key, value in x.items()}
    return PlacementDecisionState(parallelism, v)


def route(query_id, model_id):
    """Return one route key from the fixed hand-test tier."""

    return (query_id, model_id, TIER)


class AGHRelocatePassTraversalTests(unittest.TestCase):
    """Pin semantic traversal, first improvement, and refreshed state."""

    def test_one_source_first_destination_accepted_and_later_not_evaluated(self):
        instance = make_scan_instance((("source", 4.0), ("first", 3.0), ("later", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        result = run_agh_relocate_pass(instance, before)
        source_trace = result.source_traces[0]

        self.assertEqual(
            source_trace.eligible_destinations,
            (("first", TIER), ("later", TIER)),
        )
        self.assertEqual(
            tuple(item.destination for item in source_trace.attempts),
            (("first", TIER),),
        )
        self.assertEqual(source_trace.accepted_destination, ("first", TIER))
        self.assertEqual(result.placement_after.routing_fractions[route("q_0", "first")], 1.0)
        self.assertEqual(result.accepted_move_count, 1)

    def test_first_destination_rejected_second_accepted_with_full_evidence(self):
        instance = make_scan_instance((("source", 4.0), ("worse", 5.0), ("better", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        result = run_agh_relocate_pass(instance, before)
        attempts = result.source_traces[0].attempts

        self.assertEqual(
            tuple(item.destination for item in attempts),
            (("worse", TIER), ("better", TIER)),
        )
        self.assertFalse(attempts[0].accepted)
        self.assertEqual(
            attempts[0].move_result.rejection_reasons,
            (AGHRelocateRejectionReason.NOT_STRICTLY_IMPROVING,),
        )
        self.assertTrue(attempts[1].accepted)
        self.assertIs(attempts[0].placement_after, attempts[0].placement_before)

    def test_no_improving_destination_exhausts_order_and_preserves_source(self):
        instance = make_scan_instance((("source", 1.0), ("higher", 2.0), ("highest", 3.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        result = run_agh_relocate_pass(instance, before)

        self.assertFalse(result.changed)
        self.assertIs(result.placement_after, before)
        self.assertEqual(
            result.source_traces[0].status,
            AGHRelocateSourceStatus.NO_IMPROVING_DESTINATION,
        )
        self.assertEqual(len(result.source_traces[0].attempts), 2)

    def test_first_improvement_is_not_replaced_by_best_improvement(self):
        instance = make_scan_instance((("source", 4.0), ("first", 3.0), ("best", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        result = run_agh_relocate_pass(instance, before)

        # By hand, C2 falls by 0.001 for ``first`` but would fall by 0.003
        # for ``best``.  The approved policy deliberately takes the former.
        self.assertAlmostEqual(result.total_objective_improvement, 0.001)
        self.assertEqual(
            result.source_traces[0].accepted_destination,
            ("first", TIER),
        )
        self.assertEqual(result.placement_after.routing_fractions[route("q_0", "best")], 0.0)

    def test_source_snapshot_uses_query_model_tier_semantic_lattice_order(self):
        instance = make_scan_instance(
            (("z_model", 1.0), ("a_model", 1.0)),
            query_ids=("q_z", "q_a"),
        )
        before = make_scan_state(instance, {"q_z": "a_model", "q_a": "z_model"})

        result = run_agh_relocate_pass(instance, before)

        self.assertEqual(
            result.source_snapshot,
            (route("q_z", "a_model"), route("q_a", "z_model")),
        )

    def test_destination_order_is_model_tier_semantic_not_lexical(self):
        instance = make_scan_instance((("source", 1.0), ("z_dest", 2.0), ("a_dest", 3.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        source_trace = run_agh_relocate_pass(instance, before).source_traces[0]

        self.assertEqual(
            source_trace.eligible_destinations,
            (("z_dest", TIER), ("a_dest", TIER)),
        )
        self.assertEqual(
            tuple(item.destination for item in source_trace.attempts),
            source_trace.eligible_destinations,
        )

    def test_multiple_sources_accept_multiple_moves_using_refreshed_state(self):
        instance = make_scan_instance(
            (("model_0", 3.0), ("model_1", 2.0), ("model_2", 1.0)),
            query_ids=("q_0", "q_1"),
        )
        before = make_scan_state(instance, {"q_0": "model_0", "q_1": "model_1"})

        result = run_agh_relocate_pass(instance, before)
        accepted = tuple(item for item in result.attempt_traces if item.accepted)

        self.assertEqual(result.accepted_move_count, 2)
        self.assertEqual(len(accepted), 2)
        self.assertIs(accepted[1].placement_before, accepted[0].placement_after)
        self.assertEqual(
            accepted[1].placement_before.routing_fractions[
                route("q_0", "model_1")
            ],
            1.0,
        )
        self.assertEqual(
            result.placement_after.routing_fractions[route("q_1", "model_2")],
            1.0,
        )
        self.assertAlmostEqual(result.total_objective_improvement, 0.002)

    def test_destination_evaluation_reads_the_current_refreshed_placement(self):
        instance = make_scan_instance(
            (("model_0", 3.0), ("model_1", 2.0), ("model_2", 1.0)),
            query_ids=("q_0", "q_1"),
        )
        before = make_scan_state(instance, {"q_0": "model_0", "q_1": "model_1"})

        result = run_agh_relocate_pass(instance, before)
        second_source = result.source_traces[1]

        self.assertIs(
            second_source.placement_before,
            result.source_traces[0].placement_after,
        )
        self.assertIs(
            second_source.attempts[0].move_result.placement_before,
            second_source.placement_before,
        )
        self.assertEqual(
            second_source.placement_before.routing_fractions[
                route("q_0", "model_1")
            ],
            1.0,
        )

    def test_frozen_source_that_is_no_longer_positive_is_skipped(self):
        instance = make_scan_instance((("source", 3.0), ("destination", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})
        accepted = evaluate_agh_relocate_move(
            instance,
            before,
            route("q_0", "source"),
            ("destination", TIER),
        )

        trace = pass_module._scan_relocate_source(
            _AGHPlacementEvaluation(
                _build_agh_search_context(
                    instance, accepted.validation_report.paper_version, None
                ),
                accepted.accepted_placement,
                accepted.validation_report,
            ),
            route("q_0", "source"),
            pass_index=0,
            source_position=1,
        )

        self.assertEqual(trace.status, AGHRelocateSourceStatus.SOURCE_NO_LONGER_ACTIVE)
        self.assertEqual(trace.attempts, ())
        self.assertIs(trace.placement_after, accepted.accepted_placement)

    def test_pass_delegates_each_scientific_attempt_to_shared_move_core(self):
        instance = make_scan_instance((("source", 4.0), ("destination", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})
        delegated_results = []

        def delegate(*args, **kwargs):
            result = original(*args, **kwargs)
            delegated_results.append(result)
            return result

        original = pass_module._evaluate_agh_relocate_move
        with patch.object(
            pass_module,
            "_evaluate_agh_relocate_move",
            side_effect=delegate,
        ) as delegated:
            result = run_agh_relocate_pass(instance, before)

        delegated.assert_called_once()
        self.assertIs(delegated.call_args.args[0].placement, before)
        self.assertIs(
            delegated.call_args.args[0].validation, result.validation_before
        )
        self.assertIs(result.attempt_traces[0].move_result, delegated_results[0])


class AGHRelocateRunTests(unittest.TestCase):
    """Exercise no-change convergence, pass refresh, and the L=3 bound."""

    def test_validation_checks_each_new_proposal_once_and_carries_current_report(self):
        instance = make_scan_instance((("model_0", 4.0), ("model_1", 3.0), ("model_2", 1.0)))
        before = make_scan_state(instance, {"q_0": "model_0"})
        validated = []
        original = PlacementValidator.validate

        def record(validator, placement):
            validated.append(placement)
            return original(validator, placement)

        with patch.object(PlacementValidator, "validate", autospec=True, side_effect=record):
            result = run_agh_relocate(instance, before)

        proposals = [
            attempt.move_result.prospective_placement
            for pass_result in result.pass_results
            for attempt in pass_result.attempt_traces
            if attempt.move_result.prospective_placement is not None
        ]
        self.assertEqual(len(validated), 1 + len(proposals))
        self.assertIs(validated[0], before)
        self.assertEqual([id(item) for item in validated[1:]], [id(item) for item in proposals])
        current_report = result.pass_results[0].validation_before
        for pass_result in result.pass_results:
            self.assertIs(pass_result.validation_before, current_report)
            for attempt in pass_result.attempt_traces:
                self.assertIs(attempt.move_result.validation_before, current_report)
                if attempt.accepted:
                    current_report = attempt.move_result.validation_report
            self.assertIs(pass_result.validation_after, current_report)

    def test_rejected_destinations_keep_one_current_report(self):
        instance = make_scan_instance((("source", 1.0), ("higher", 2.0), ("highest", 3.0)))
        before = make_scan_state(instance, {"q_0": "source"})
        result = run_agh_relocate_pass(instance, before)

        self.assertEqual(len(result.attempt_traces), 2)
        for attempt in result.attempt_traces:
            self.assertIs(attempt.move_result.validation_before, result.validation_before)
            self.assertIs(attempt.placement_after, before)
        self.assertIs(result.validation_after, result.validation_before)

    def test_one_pass_api_never_executes_a_second_pass(self):
        instance = make_scan_instance((("model_0", 4.0), ("model_1", 3.0), ("model_2", 1.0)))
        before = make_scan_state(instance, {"q_0": "model_0"})

        result = run_agh_relocate_pass(instance, before)

        self.assertEqual(result.pass_index, 0)
        self.assertEqual(result.placement_after.routing_fractions[route("q_0", "model_1")], 1.0)
        self.assertEqual(result.placement_after.routing_fractions[route("q_0", "model_2")], 0.0)

    def test_changed_pass_has_positive_count_and_strict_objective_drop(self):
        instance = make_scan_instance((("source", 4.0), ("destination", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        result = run_agh_relocate_pass(instance, before)

        self.assertTrue(result.changed)
        self.assertGreater(result.accepted_move_count, 0)
        self.assertLess(result.objective_breakdown_after.total, result.objective_breakdown_before.total)
        self.assertGreater(result.total_objective_improvement, 0.0)

    def test_no_change_pass_has_zero_count_and_same_state(self):
        instance = make_scan_instance((("source", 1.0), ("destination", 2.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        result = run_agh_relocate_pass(instance, before)

        self.assertEqual(result.accepted_move_count, 0)
        self.assertFalse(result.changed)
        self.assertIs(result.placement_after, before)
        self.assertEqual(result.total_objective_improvement, 0.0)

    def test_run_stops_immediately_after_first_no_change_pass(self):
        instance = make_scan_instance((("model_0", 3.0), ("model_1", 2.0)))
        before = make_scan_state(instance, {"q_0": "model_0"})

        result = run_agh_relocate(instance, before)

        self.assertEqual(result.passes_executed, 2)
        self.assertTrue(result.pass_results[0].changed)
        self.assertFalse(result.pass_results[1].changed)
        self.assertEqual(result.termination_reason, AGHRelocateRunTerminationReason.NO_CHANGE_PASS)

    def test_three_changed_passes_stop_at_paper_maximum(self):
        instance = make_scan_instance(
            (("model_0", 4.0), ("model_1", 3.0), ("model_2", 2.0), ("model_3", 1.0))
        )
        before = make_scan_state(instance, {"q_0": "model_0"})

        result = run_agh_relocate(instance, before)

        self.assertEqual(result.passes_executed, 3)
        self.assertTrue(all(item.changed for item in result.pass_results))
        self.assertEqual(result.accepted_move_count, 3)
        self.assertEqual(
            result.termination_reason,
            AGHRelocateRunTerminationReason.MAX_PASSES_REACHED,
        )
        self.assertEqual(result.placement_after.routing_fractions[route("q_0", "model_3")], 1.0)

    def test_source_snapshot_is_refreshed_between_passes(self):
        instance = make_scan_instance(
            (
                ("model_0", 4.0),
                ("model_1", 3.0),
                ("model_2", 2.0),
                ("model_3", 1.0),
            )
        )
        before = make_scan_state(instance, {"q_0": "model_0"})

        result = run_agh_relocate(instance, before)

        self.assertEqual(
            tuple(item.source_snapshot for item in result.pass_results),
            ((route("q_0", "model_0"),), (route("q_0", "model_1"),), (route("q_0", "model_2"),)),
        )

    def test_every_accepted_move_and_changed_pass_is_strictly_monotone(self):
        instance = make_scan_instance(
            (("model_0", 4.0), ("model_1", 3.0), ("model_2", 2.0), ("model_3", 1.0))
        )
        before = make_scan_state(instance, {"q_0": "model_0"})

        result = run_agh_relocate(instance, before)
        accepted = tuple(
            attempt
            for pass_result in result.pass_results
            for attempt in pass_result.attempt_traces
            if attempt.accepted
        )

        self.assertTrue(
            all(
                item.move_result.objective_breakdown_after.total
                < item.move_result.objective_breakdown_before.total
                for item in accepted
            )
        )
        self.assertTrue(
            all(
                item.objective_breakdown_after.total < item.objective_breakdown_before.total
                for item in result.pass_results
                if item.changed
            )
        )
        self.assertLess(result.objective_breakdown_after.total, result.objective_breakdown_before.total)

    def test_reference_max_passes_accepts_only_one_through_three(self):
        instance = make_scan_instance((("source", 1.0), ("destination", 2.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        self.assertEqual(AGH_RELOCATE_MAX_PASSES, 3)
        for bad in (0, 4, -1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_agh_relocate(instance, before, max_passes=bad)

    def test_zero_traffic_deployments_remain_active_without_consolidation(self):
        instance = make_scan_instance(
            (("model_0", 4.0), ("model_1", 3.0), ("model_2", 2.0), ("model_3", 1.0))
        )
        before = make_scan_state(instance, {"q_0": "model_0"})

        result = run_agh_relocate(instance, before)

        self.assertEqual(
            result.placement_after.deployment_indicators,
            before.deployment_indicators,
        )
        self.assertEqual(
            result.placement_after.configuration_selectors,
            before.configuration_selectors,
        )
        self.assertEqual(result.placement_after.gpu_counts, before.gpu_counts)
        self.assertEqual(result.placement_after.routing_fractions[route("q_0", "model_0")], 0.0)


class AGHRelocatePassEvidenceTests(unittest.TestCase):
    """Protect transaction evidence, immutability, determinism, and isolation."""

    def test_rejected_attempt_retains_same_accepted_state_object(self):
        instance = make_scan_instance((("source", 2.0), ("worse", 3.0), ("better", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})

        first_attempt = run_agh_relocate_pass(instance, before).attempt_traces[0]

        self.assertFalse(first_attempt.accepted)
        self.assertIs(first_attempt.placement_after, first_attempt.placement_before)
        self.assertIs(first_attempt.move_result.accepted_placement, first_attempt.placement_before)

    def test_input_is_deeply_unchanged_and_result_is_frozen(self):
        instance = make_scan_instance((("source", 3.0), ("destination", 1.0)))
        before = make_scan_state(instance, {"q_0": "source"})
        snapshot = tuple(
            dict(mapping)
            for mapping in (
                before.routing_fractions,
                before.admission_indicators,
                before.deployment_indicators,
                before.unmet_fractions,
                before.configuration_selectors,
                before.gpu_counts,
                before.auxiliary_products,
            )
        )

        result = run_agh_relocate(instance, before)

        self.assertEqual(
            snapshot,
            tuple(
                dict(mapping)
                for mapping in (
                    before.routing_fractions,
                    before.admission_indicators,
                    before.deployment_indicators,
                    before.unmet_fractions,
                    before.configuration_selectors,
                    before.gpu_counts,
                    before.auxiliary_products,
                )
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            result.max_passes = 1
        with self.assertRaises(FrozenInstanceError):
            result.pass_results[0].pass_index = 1
        with self.assertRaises(FrozenInstanceError):
            result.pass_results[0].attempt_traces[0].destination_position = 1
        with self.assertRaises(TypeError):
            result.placement_after.routing_fractions[route("q_0", "destination")] = 0.0

    def test_identical_inputs_produce_identical_pass_and_run_traces(self):
        instance = make_scan_instance((("model_0", 3.0), ("model_1", 2.0), ("model_2", 1.0)))
        before = make_scan_state(instance, {"q_0": "model_0"})

        self.assertEqual(
            run_agh_relocate_pass(instance, before),
            run_agh_relocate_pass(instance, before),
        )
        self.assertEqual(
            run_agh_relocate(instance, before),
            run_agh_relocate(instance, before),
        )

    def test_module_has_no_solver_mechanism_or_complete_agh_orchestration(self):
        source = inspect.getsource(pass_module)
        tree = ast.parse(source)
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        self.assertFalse(any("pyomo" in name for name in imports))
        self.assertFalse(any(name.endswith(".solver") for name in imports))
        self.assertFalse(any(
            name.rsplit(".", 1)[-1] in ("mechanisms", "m1", "m2", "m3")
            for name in imports
        ))
        self.assertNotIn("run_gh(", source)
        self.assertNotIn("build_all_deterministic_agh_orderings(", source)
        self.assertNotIn("build_seeded_random_agh_ordering(", source)
        self.assertNotIn("incumbent", source.lower())
        self.assertFalse(
            any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and "consolid" in node.name.lower()
                for node in ast.walk(tree)
            )
        )


if __name__ == "__main__":
    unittest.main()
