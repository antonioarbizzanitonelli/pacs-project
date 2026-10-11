"""Hand-derived tests for Milestone-6D AGH consolidation."""

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    AGHConsolidationRejectionReason,
    AGHConsolidationRouteStatus,
    AGHConsolidationSourceScanStatus,
    evaluate_agh_consolidation_source,
    run_agh_consolidation_scan,
)
from llm_resource_allocation.algorithms import agh_consolidation as consolidation_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import FoundationModel
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import DecisionDomainError, RoutingDecisionState
from llm_resource_allocation.validation.placement import (
    ConstraintFamily,
    PlacementValidator,
    validate_placement,
)


TIER = "gpu_basic_fp16"


def make_consolidation_instance(
    model_ids,
    *,
    query_ids=("q_0",),
    errors=None,
    delays=None,
    compute=None,
):
    """Build a transparent 1-TP/1-PP consolidation lattice."""

    base = make_minimal_tiny_instance()
    queries = tuple(
        replace(
            base.queries[0],
            id=query_id,
            arrival_rate_per_hour=1000.0,
            input_tokens=1.0,
            output_tokens=0.0,
            delay_slo_seconds=500.0,
            error_slo=0.1,
            max_unmet_fraction=1.0,
        )
        for query_id in query_ids
    )
    models = tuple(FoundationModel(model_id, 1.0, 2.0e-6) for model_id in model_ids)
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=100.0,
        compute_throughput_tflops=10.0,
        hourly_rental_usd=1.0,
        tensor_parallel_degrees=(1,),
        pipeline_parallel_depths=(1,),
    )
    triples = tuple(
        (query.id, model.id, TIER)
        for query in queries
        for model in models
    )
    errors = errors or {}
    delays = delays or {}
    compute = compute or {}
    return replace(
        base,
        id="agh_consolidation_hand_fixture",
        queries=queries,
        models=models,
        gpu_tiers=(tier,),
        base_error_rate={
            (query.id, model.id): errors.get((query.id, model.id), 0.02)
            for query in queries
            for model in models
        },
        compute_delay_seconds_per_token={
            key: delays.get((key[0], key[1]), 0.1) for key in triples
        },
        communication_delay_seconds_per_token={key: 0.01 for key in triples},
        compute_gflop_per_token={
            key: compute.get((key[0], key[1]), 1.0) for key in triples
        },
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=1000.0,
        storage_capacity_gb=1000.0,
        horizon_hours=1.0,
        compute_utilization=1.0,
        seconds_per_hour=1.0,
    )


def make_consolidation_state(
    instance,
    route_values,
    *,
    active_models=None,
    unmet=None,
):
    """Create one dense canonical placement with selected active pairs."""

    active_models = tuple(active_models or instance.model_ids)
    x = {
        (query_id, model_id, TIER): float(
            route_values.get((query_id, model_id), 0.0)
        )
        for query_id in instance.query_ids
        for model_id in instance.model_ids
    }
    z = {key: int(value > 0.0) for key, value in x.items()}
    q = {
        (model_id, TIER): int(model_id in active_models)
        for model_id in instance.model_ids
    }
    y = dict(q)
    w = {
        (model_id, TIER, 1, 1): int(model_id in active_models)
        for model_id in instance.model_ids
    }
    if unmet is None:
        unmet = {
            query_id: 1.0
            - sum(
                x[(query_id, model_id, TIER)]
                for model_id in instance.model_ids
            )
            for query_id in instance.query_ids
        }
    routing = RoutingDecisionState(x, z, q, unmet)
    parallelism = ParallelismDecisionState(routing, w, y)
    v = {
        (query_id, model_id, TIER, 1, 1): (
            x[(query_id, model_id, TIER)] * w[(model_id, TIER, 1, 1)]
        )
        for query_id in instance.query_ids
        for model_id in instance.model_ids
    }
    return PlacementDecisionState(parallelism, v)


def pair(model_id):
    """Return one model-tier pair."""

    return (model_id, TIER)


def route(query_id, model_id):
    """Return one query-model-tier route."""

    return (query_id, model_id, TIER)


class AGHConsolidationLoadTests(unittest.TestCase):
    """Freeze Eq. (8g) utilization arithmetic and ascending semantic order."""

    def test_load_ratio_zero_rule_and_exact_tie_order(self):
        instance = make_consolidation_instance(
            ("z_zero", "a_zero", "z_tie", "a_tie"),
            query_ids=("q_z", "q_a"),
        )
        before = make_consolidation_state(
            instance,
            {("q_z", "z_tie"): 0.5, ("q_a", "a_tie"): 0.5},
        )

        result = run_agh_consolidation_scan(instance, before)

        self.assertEqual(
            tuple(item.pair for item in result.source_snapshot),
            (
                pair("z_zero"),
                pair("a_zero"),
                pair("z_tie"),
                pair("a_tie"),
            ),
        )
        self.assertEqual(
            tuple(item.required_compute_tflops_per_hour for item in result.source_snapshot),
            (0.0, 0.0, 0.5, 0.5),
        )
        self.assertEqual(
            tuple(item.available_compute_tflops_per_hour for item in result.source_snapshot),
            (10.0, 10.0, 10.0, 10.0),
        )
        self.assertEqual(
            tuple(item.utilization for item in result.source_snapshot),
            (0.0, 0.0, 0.05, 0.05),
        )

    def test_source_snapshot_contains_only_coherent_active_pairs(self):
        instance = make_consolidation_instance(("active", "inactive"))
        before = make_consolidation_state(
            instance,
            {("q_0", "active"): 1.0},
            active_models=("active",),
        )

        result = run_agh_consolidation_scan(instance, before)

        self.assertEqual(
            tuple(item.pair for item in result.source_snapshot),
            (pair("active"),),
        )


class AGHConsolidationTransactionTests(unittest.TestCase):
    """Exercise atomic redistribution, deactivation, and rollback."""

    def test_public_atomic_evaluator_rejects_invalid_domains_without_a_proposal(self):
        instance = make_consolidation_instance(("source", "destination"))
        before = make_consolidation_state(instance, {("q_0", "source"): 1.1})
        result = evaluate_agh_consolidation_source(instance, before, pair("source"))

        self.assertEqual(
            result.rejection_reasons,
            (AGHConsolidationRejectionReason.INVALID_PLACEMENT_DOMAINS,),
        )
        self.assertIs(result.accepted_placement, before)
        self.assertIsNone(result.prospective_placement)
        self.assertFalse(result.validation_before.domain_diagnostic.is_valid)

    def test_public_scan_preserves_domain_error_for_invalid_input(self):
        instance = make_consolidation_instance(("source", "destination"))
        before = make_consolidation_state(instance, {("q_0", "source"): 1.1})
        with self.assertRaises(DecisionDomainError):
            run_agh_consolidation_scan(instance, before)

    def test_zero_traffic_source_is_deactivated_and_strictly_improves(self):
        instance = make_consolidation_instance(
            ("empty", "destination"),
            query_ids=("q_0",),
        )
        before = make_consolidation_state(
            instance,
            {("q_0", "destination"): 1.0},
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("empty")
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.source_routes, ())
        self.assertEqual(result.route_traces, ())
        self.assertAlmostEqual(result.objective_delta, -1.0)
        self.assertEqual(result.accepted_placement.deployment_indicators[pair("empty")], 0)
        self.assertEqual(result.accepted_placement.gpu_counts[pair("empty")], 0)
        self.assertEqual(
            result.accepted_placement.configuration_selectors[
                (*pair("empty"), 1, 1)
            ],
            0,
        )

    def test_one_route_is_moved_whole_to_first_feasible_destination(self):
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_move", "q_anchor"),
        )
        before = make_consolidation_state(
            instance,
            {("q_move", "source"): 0.4, ("q_anchor", "destination"): 1.0},
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.source_routes, (route("q_move", "source"),))
        self.assertEqual(result.route_traces[0].moved_fraction, 0.4)
        self.assertEqual(
            result.route_traces[0].selected_destination,
            pair("destination"),
        )
        self.assertEqual(
            result.accepted_placement.routing_fractions[
                route("q_move", "destination")
            ],
            0.4,
        )

    def test_multiple_source_routes_follow_authoritative_query_order(self):
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_z", "q_a", "q_anchor"),
        )
        before = make_consolidation_state(
            instance,
            {
                ("q_z", "source"): 0.4,
                ("q_a", "source"): 0.6,
                ("q_anchor", "destination"): 1.0,
            },
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.source_routes,
            (route("q_z", "source"), route("q_a", "source")),
        )
        self.assertEqual(
            tuple(item.selected_destination for item in result.route_traces),
            (pair("destination"), pair("destination")),
        )

    def test_different_queries_can_use_different_first_fit_destinations(self):
        # 0.3 * 0.4 = 0.12 exceeds the exact 0.1 error SLO.
        errors = {("q_second", "d_first"): 0.3}
        instance = make_consolidation_instance(
            ("source", "d_first", "d_second"),
            query_ids=("q_first", "q_second", "anchor_1", "anchor_2"),
            errors=errors,
        )
        before = make_consolidation_state(
            instance,
            {
                ("q_first", "source"): 0.4,
                ("q_second", "source"): 0.4,
                ("anchor_1", "d_first"): 1.0,
                ("anchor_2", "d_second"): 1.0,
            },
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            tuple(item.selected_destination for item in result.route_traces),
            (pair("d_first"), pair("d_second")),
        )
        second_attempts = result.route_traces[1].attempts
        self.assertEqual(
            tuple(item.destination for item in second_attempts),
            (pair("d_first"), pair("d_second")),
        )
        self.assertFalse(second_attempts[0].feasible)
        self.assertTrue(second_attempts[1].feasible)

    def test_first_fit_stops_before_a_later_feasible_destination(self):
        instance = make_consolidation_instance(
            ("source", "first", "later"),
            query_ids=("q_move", "anchor_1", "anchor_2"),
        )
        before = make_consolidation_state(
            instance,
            {
                ("q_move", "source"): 0.5,
                ("anchor_1", "first"): 1.0,
                ("anchor_2", "later"): 1.0,
            },
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        trace = result.route_traces[0]
        self.assertTrue(result.accepted)
        self.assertEqual(trace.eligible_destinations, (pair("first"), pair("later")))
        self.assertEqual(
            tuple(item.destination for item in trace.attempts),
            (pair("first"),),
        )
        self.assertEqual(trace.selected_destination, pair("first"))

    def test_full_route_is_not_split_across_individually_insufficient_pairs(self):
        compute = {("q_move", "d_1"): 15.0, ("q_move", "d_2"): 15.0}
        instance = make_consolidation_instance(
            ("source", "d_1", "d_2"),
            query_ids=("q_move", "anchor_1", "anchor_2"),
            compute=compute,
        )
        before = make_consolidation_state(
            instance,
            {
                ("q_move", "source"): 1.0,
                ("anchor_1", "d_1"): 0.1,
                ("anchor_2", "d_2"): 0.1,
            },
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.rejection_reasons,
            (AGHConsolidationRejectionReason.REDISTRIBUTION_FAILED,),
        )
        trace = result.route_traces[0]
        self.assertEqual(trace.moved_fraction, 1.0)
        self.assertEqual(len(trace.attempts), 2)
        self.assertTrue(all(not item.feasible for item in trace.attempts))
        self.assertIs(result.accepted_placement, before)

    def test_partial_provisional_work_rolls_back_when_a_later_route_fails(self):
        # 0.3 * 0.4 = 0.12 exceeds the exact 0.1 error SLO.
        errors = {("q_fail", "destination"): 0.3}
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_ok", "q_fail", "anchor"),
            errors=errors,
        )
        before = make_consolidation_state(
            instance,
            {
                ("q_ok", "source"): 0.4,
                ("q_fail", "source"): 0.4,
                ("anchor", "destination"): 1.0,
            },
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        self.assertFalse(result.accepted)
        self.assertEqual(len(result.route_traces), 2)
        self.assertEqual(
            result.route_traces[0].status,
            AGHConsolidationRouteStatus.REDISTRIBUTED,
        )
        self.assertEqual(
            result.route_traces[1].status,
            AGHConsolidationRouteStatus.NO_FEASIBLE_DESTINATION,
        )
        self.assertEqual(
            result.redistributed_placement.routing_fractions[
                route("q_ok", "destination")
            ],
            0.4,
        )
        self.assertIs(result.accepted_placement, before)
        self.assertEqual(before.routing_fractions[route("q_ok", "source")], 0.4)

    def test_final_validator_infeasibility_rolls_back_complete_transaction(self):
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_move", "anchor"),
        )
        before = make_consolidation_state(
            instance,
            {("q_move", "source"): 0.4, ("anchor", "destination"): 1.0},
        )
        original = consolidation_module._deactivate_source_pair

        def invalid_final(problem_data, redistributed, source):
            valid = original(problem_data, redistributed, source)
            q = dict(valid.deployment_indicators)
            q[pair("destination")] = 0
            routing = RoutingDecisionState(
                valid.routing_fractions,
                valid.admission_indicators,
                q,
                valid.unmet_fractions,
            )
            parallelism = ParallelismDecisionState(
                routing,
                valid.configuration_selectors,
                valid.gpu_counts,
            )
            return PlacementDecisionState(parallelism, valid.auxiliary_products)

        with patch.object(
            consolidation_module,
            "_deactivate_source_pair",
            side_effect=invalid_final,
        ):
            result = evaluate_agh_consolidation_source(
                instance, before, pair("source")
            )

        self.assertFalse(result.accepted)
        self.assertIn(
            AGHConsolidationRejectionReason.INFEASIBLE,
            result.rejection_reasons,
        )
        self.assertIn(
            ConstraintFamily.ROUTING_CONSISTENCY,
            result.validation_report.violated_families,
        )
        self.assertIs(result.accepted_placement, before)

    def test_feasible_but_non_improving_final_objective_rolls_back(self):
        delays = {("q_move", "destination"): 300.0}
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_move", "anchor"),
            delays=delays,
        )
        before = make_consolidation_state(
            instance,
            {("q_move", "source"): 0.5, ("anchor", "destination"): 1.0},
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.validation_report.is_feasible)
        self.assertGreater(result.objective_delta, 0.0)
        self.assertEqual(
            result.rejection_reasons,
            (AGHConsolidationRejectionReason.NOT_STRICTLY_IMPROVING,),
        )
        self.assertIs(result.accepted_placement, before)

    def test_route_can_worsen_objective_before_final_deactivation_saves_cost(self):
        delays = {("q_move", "destination"): 50.0}
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_move", "anchor"),
            delays=delays,
        )
        before = make_consolidation_state(
            instance,
            {("q_move", "source"): 0.5, ("anchor", "destination"): 1.0},
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )

        attempt = result.route_traces[0].attempts[0]
        self.assertTrue(attempt.feasible)
        self.assertGreater(
            attempt.validation_report.objective_breakdown.total,
            result.objective_breakdown_before.total,
        )
        self.assertTrue(result.accepted)
        self.assertLess(result.objective_breakdown_after.total, result.objective_breakdown_before.total)

    def test_exact_deactivation_fixed_destination_and_x_z_v_u_invariants(self):
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_move", "anchor"),
        )
        before = make_consolidation_state(
            instance,
            {("q_move", "source"): 0.4, ("anchor", "destination"): 1.0},
        )

        result = evaluate_agh_consolidation_source(
            instance, before, pair("source")
        )
        after = result.accepted_placement

        self.assertTrue(result.accepted)
        self.assertEqual(after.deployment_indicators[pair("source")], 0)
        self.assertEqual(after.gpu_counts[pair("source")], 0)
        self.assertEqual(after.configuration_selectors[(*pair("source"), 1, 1)], 0)
        self.assertEqual(after.routing_fractions[route("q_move", "source")], 0.0)
        self.assertEqual(after.admission_indicators[route("q_move", "source")], 0)
        self.assertEqual(
            after.auxiliary_products[(*route("q_move", "source"), 1, 1)],
            0.0,
        )
        self.assertEqual(after.routing_fractions[route("q_move", "destination")], 0.4)
        self.assertEqual(after.admission_indicators[route("q_move", "destination")], 1)
        self.assertEqual(
            after.auxiliary_products[(*route("q_move", "destination"), 1, 1)],
            0.4,
        )
        self.assertEqual(
            after.deployment_indicators[pair("destination")],
            before.deployment_indicators[pair("destination")],
        )
        self.assertEqual(
            after.configuration_selectors[(*pair("destination"), 1, 1)],
            before.configuration_selectors[(*pair("destination"), 1, 1)],
        )
        self.assertEqual(after.gpu_counts[pair("destination")], 1)
        self.assertEqual(after.unmet_fractions, before.unmet_fractions)
        self.assertTrue(all(item.is_consistent for item in result.demand_evidence))


class AGHConsolidationScanTests(unittest.TestCase):
    """Pin one frozen scan, state refresh, and multiple consolidations."""

    def test_scan_adopts_accepted_state_and_can_consolidate_multiple_sources(self):
        instance = make_consolidation_instance(
            ("source_a", "source_b", "destination"),
            query_ids=("q_a", "q_b", "q_destination"),
        )
        before = make_consolidation_state(
            instance,
            {
                ("q_a", "source_a"): 0.2,
                ("q_b", "source_b"): 0.4,
                ("q_destination", "destination"): 1.0,
            },
        )

        result = run_agh_consolidation_scan(instance, before)

        self.assertEqual(
            tuple(item.pair for item in result.source_snapshot),
            (pair("source_a"), pair("source_b"), pair("destination")),
        )
        self.assertEqual(result.accepted_source_count, 2)
        self.assertEqual(
            tuple(item.status for item in result.source_traces),
            (
                AGHConsolidationSourceScanStatus.ACCEPTED,
                AGHConsolidationSourceScanStatus.ACCEPTED,
                AGHConsolidationSourceScanStatus.REJECTED,
            ),
        )
        self.assertIs(
            result.source_traces[1].placement_before,
            result.source_traces[0].placement_after,
        )
        self.assertEqual(
            result.placement_after.routing_fractions[route("q_a", "destination")],
            0.2,
        )
        self.assertEqual(
            result.placement_after.routing_fractions[route("q_b", "destination")],
            0.4,
        )
        self.assertEqual(result.placement_after.deployment_indicators[pair("source_a")], 0)
        self.assertEqual(result.placement_after.deployment_indicators[pair("source_b")], 0)
        self.assertEqual(result.placement_after.deployment_indicators[pair("destination")], 1)
        self.assertEqual(result.total_objective_improvement, 2.0)

    def test_scan_skips_a_frozen_source_that_is_no_longer_active(self):
        instance = make_consolidation_instance(
            ("empty_a", "empty_b", "destination"),
            query_ids=("anchor",),
        )
        before = make_consolidation_state(
            instance,
            {("anchor", "destination"): 1.0},
        )
        original_evaluator = consolidation_module._evaluate_agh_consolidation_source
        original_deactivate = consolidation_module._deactivate_source_pair

        def evaluator(current, source_arg):
            instance_arg = current.context.problem_data.instance
            placement_arg = current.placement
            if source_arg == pair("empty_a"):
                problem_data = current.context.problem_data
                after_a = original_deactivate(
                    problem_data, placement_arg, pair("empty_a")
                )
                after_b = original_deactivate(
                    problem_data, after_a, pair("empty_b")
                )
                report = validate_placement(
                    instance_arg,
                    after_b,
                    current.context.validator.paper_version,
                    current.context.validator.tolerances,
                )
                return SimpleNamespace(
                    source=source_arg,
                    placement_before=placement_arg,
                    accepted=True,
                    accepted_placement=after_b,
                    validation_report=report,
                )
            return original_evaluator(current, source_arg)

        with patch.object(
            consolidation_module,
            "_evaluate_agh_consolidation_source",
            side_effect=evaluator,
        ):
            result = run_agh_consolidation_scan(instance, before)

        self.assertEqual(
            result.source_traces[1].status,
            AGHConsolidationSourceScanStatus.SOURCE_NO_LONGER_ACTIVE,
        )
        self.assertIsNone(result.source_traces[1].source_result)
        self.assertIs(
            result.source_traces[1].placement_after,
            result.source_traces[1].placement_before,
        )

    def test_scan_delegates_each_non_stale_source_to_transaction_evaluator(self):
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_source", "q_destination"),
        )
        before = make_consolidation_state(
            instance,
            {("q_source", "source"): 0.2, ("q_destination", "destination"): 1.0},
        )
        delegated_results = []
        original = consolidation_module._evaluate_agh_consolidation_source

        def delegate(*args, **kwargs):
            item = original(*args, **kwargs)
            delegated_results.append(item)
            return item

        with patch.object(
            consolidation_module,
            "_evaluate_agh_consolidation_source",
            side_effect=delegate,
        ) as delegated:
            result = run_agh_consolidation_scan(instance, before)

        self.assertEqual(delegated.call_count, 2)
        self.assertIs(
            result.source_traces[0].source_result,
            delegated_results[0],
        )
        self.assertIs(
            result.source_traces[1].source_result,
            delegated_results[1],
        )


class AGHConsolidationEvidenceAndIsolationTests(unittest.TestCase):
    """Protect immutable evidence, repeatability, and the 6D boundary."""

    def test_scan_validates_provisional_and_deactivated_states_and_reuses_reports(self):
        instance = make_consolidation_instance(
            ("source", "destination"), query_ids=("q_source", "q_destination")
        )
        before = make_consolidation_state(
            instance, {("q_source", "source"): 0.2, ("q_destination", "destination"): 1.0}
        )
        validated = []
        original = PlacementValidator.validate

        def record(validator, placement):
            validated.append(placement)
            return original(validator, placement)

        with patch.object(PlacementValidator, "validate", autospec=True, side_effect=record):
            result = run_agh_consolidation_scan(instance, before)

        self.assertEqual(result.accepted_source_count, 1)
        transaction = result.source_traces[0].source_result
        provisional = transaction.route_traces[0].attempts[0].prospective_placement
        self.assertEqual([id(item) for item in validated], [id(before), id(provisional), id(transaction.prospective_placement)])
        self.assertIs(transaction.validation_before, result.validation_before)
        self.assertIs(
            result.source_traces[1].source_result.validation_before, transaction.validation_report
        )
        self.assertIs(result.validation_after, transaction.validation_report)

    def test_input_is_deeply_unchanged_and_evidence_is_frozen(self):
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_move", "anchor"),
        )
        before = make_consolidation_state(
            instance,
            {("q_move", "source"): 0.4, ("anchor", "destination"): 1.0},
        )
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

        result = run_agh_consolidation_scan(instance, before)

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
            result.source_traces = ()
        with self.assertRaises(FrozenInstanceError):
            result.source_snapshot[0].required_compute_tflops_per_hour = 5.0
        with self.assertRaises(FrozenInstanceError):
            result.source_traces[0].source_result.rejection_reasons = ()
        with self.assertRaises(TypeError):
            result.placement_after.routing_fractions[route("q_move", "destination")] = 0.0

    def test_identical_inputs_produce_identical_transaction_and_scan_evidence(self):
        instance = make_consolidation_instance(
            ("source", "destination"),
            query_ids=("q_move", "anchor"),
        )
        before = make_consolidation_state(
            instance,
            {("q_move", "source"): 0.4, ("anchor", "destination"): 1.0},
        )

        self.assertEqual(
            evaluate_agh_consolidation_source(instance, before, pair("source")),
            evaluate_agh_consolidation_source(instance, before, pair("source")),
        )
        self.assertEqual(
            run_agh_consolidation_scan(instance, before),
            run_agh_consolidation_scan(instance, before),
        )

    def test_module_excludes_relocate_acceptance_m1_m3_solver_and_complete_agh(self):
        source = inspect.getsource(consolidation_module)
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

        self.assertNotIn("evaluate_agh_relocate_move", source)
        self.assertFalse(any("pyomo" in name for name in imports))
        self.assertFalse(any(name.endswith(".solver") for name in imports))
        self.assertFalse(any(
            name.rsplit(".", 1)[-1] in ("mechanisms", "m1", "m2", "m3")
            for name in imports
        ))
        self.assertNotIn("select_minimum_feasible_parallelism", source)
        self.assertNotIn("select_m3_parallelism_upgrade", source)
        self.assertNotIn("run_gh(", source)
        self.assertNotIn("run_agh_relocate(", source)
        self.assertNotIn("build_all_deterministic_agh_orderings(", source)
        self.assertNotIn("build_seeded_random_agh_ordering(", source)
        self.assertNotIn("incumbent", source.lower())


if __name__ == "__main__":
    unittest.main()
