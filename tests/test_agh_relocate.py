"""Hand-derived tests for one immutable prospective AGH relocate move."""

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, replace

from llm_resource_allocation.algorithms import (
    AGHRelocateRejectionReason,
    evaluate_agh_relocate_move,
)
from llm_resource_allocation.algorithms import agh_relocate as relocate_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import FoundationModel
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.validation.placement import (
    ConstraintFamily,
    validate_placement,
)


QUERY = "q_text"
TIER = "gpu_basic_fp16"
SOURCE_PAIR = ("model_source", TIER)
DESTINATION_PAIR = ("model_destination", TIER)
SOURCE = (QUERY, *SOURCE_PAIR)
DESTINATION_ROUTE = (QUERY, *DESTINATION_PAIR)


def make_relocate_instance(
    *,
    source_weight=4.0,
    destination_weight=1.0,
    source_delay=0.10,
    destination_delay=0.10,
    source_error=0.02,
    destination_error=0.02,
    destination_compute=1.0,
    include_third=False,
):
    """Build a transparent one-query relocation lattice from the tiny fixture."""

    base = make_minimal_tiny_instance()
    tier = replace(
        base.gpu_tiers[0],
        tensor_parallel_degrees=(1,),
        pipeline_parallel_depths=(1,),
    )
    models = [
        FoundationModel("model_source", source_weight, 2.0e-6),
        FoundationModel("model_destination", destination_weight, 2.0e-6),
    ]
    if include_third:
        models.append(FoundationModel("model_unrelated", 2.0, 2.0e-6))
    delay_by_model = {
        "model_source": source_delay,
        "model_destination": destination_delay,
        "model_unrelated": 0.10,
    }
    error_by_model = {
        "model_source": source_error,
        "model_destination": destination_error,
        "model_unrelated": 0.02,
    }
    compute_by_model = {
        "model_source": 1.0,
        "model_destination": destination_compute,
        "model_unrelated": 1.0,
    }
    triples = tuple((QUERY, model.id, TIER) for model in models)
    return replace(
        base,
        id="agh_relocate_hand_fixture",
        models=tuple(models),
        gpu_tiers=(tier,),
        base_error_rate={(QUERY, m.id): error_by_model[m.id] for m in models},
        compute_delay_seconds_per_token={
            key: delay_by_model[key[1]] for key in triples
        },
        communication_delay_seconds_per_token={key: 0.01 for key in triples},
        compute_gflop_per_token={
            key: compute_by_model[key[1]] for key in triples
        },
        kv_cache_residency_seconds={key: 1.0 for key in triples},
    )


def make_relocate_state(
    instance,
    *,
    source_x=1.0,
    destination_x=0.0,
    third_x=0.0,
    unmet=0.0,
    destination_active=True,
    destination_selected=True,
    destination_y=1,
    source_z=None,
):
    """Create a complete state with caller-controlled structural edge cases."""

    route_values = {
        (QUERY, model.id, TIER): 0.0 for model in instance.models
    }
    route_values[SOURCE] = source_x
    route_values[DESTINATION_ROUTE] = destination_x
    if "model_unrelated" in instance.model_ids:
        route_values[(QUERY, "model_unrelated", TIER)] = third_x
    z = {key: int(value > 0.0) for key, value in route_values.items()}
    if source_z is not None:
        z[SOURCE] = source_z

    q = {SOURCE_PAIR: 1, DESTINATION_PAIR: int(destination_active)}
    y = {SOURCE_PAIR: 1, DESTINATION_PAIR: destination_y if destination_active else 0}
    w = {
        (*SOURCE_PAIR, 1, 1): 1,
        (*DESTINATION_PAIR, 1, 1): int(
            destination_active and destination_selected
        ),
    }
    if "model_unrelated" in instance.model_ids:
        third_pair = ("model_unrelated", TIER)
        q[third_pair] = 1
        y[third_pair] = 1
        w[(*third_pair, 1, 1)] = 1
    routing = RoutingDecisionState(
        routing_fractions=route_values,
        admission_indicators=z,
        deployment_indicators=q,
        unmet_fractions={QUERY: unmet},
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors=w,
        gpu_counts=y,
    )
    v = {
        (query_id, model_id, tier_id, n, m): (
            route_values[(query_id, model_id, tier_id)]
            * w[(model_id, tier_id, n, m)]
        )
        for query_id, model_id, tier_id in route_values
        for n, m in ((1, 1),)
    }
    return PlacementDecisionState(parallelism, v)


class AGHRelocateAcceptanceTests(unittest.TestCase):
    """Exercise full-objective acceptance and transactional rejection."""

    def test_valid_strictly_improving_full_move_is_accepted(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.rejection_reasons, ())
        self.assertIs(result.accepted_placement, result.prospective_placement)
        self.assertTrue(result.validation_report.is_feasible)
        self.assertAlmostEqual(result.objective_breakdown_before.total, 2.00710003)
        self.assertAlmostEqual(result.objective_breakdown_after.total, 2.00410003)
        self.assertAlmostEqual(result.objective_delta, -0.003)

    def test_feasible_but_worse_is_rejected_transactionally(self):
        instance = make_relocate_instance(
            source_weight=1.0, destination_weight=4.0
        )
        before = make_relocate_state(instance)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.validation_report.is_feasible)
        self.assertGreater(result.objective_delta, 0.0)
        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.NOT_STRICTLY_IMPROVING,),
        )
        self.assertIs(result.accepted_placement, before)

    def test_feasible_equal_objective_is_rejected_by_strict_comparison(self):
        instance = make_relocate_instance(
            source_weight=1.0, destination_weight=1.0
        )
        before = make_relocate_state(instance)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertTrue(result.validation_report.is_feasible)
        self.assertEqual(result.objective_delta, 0.0)
        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.NOT_STRICTLY_IMPROVING,),
        )

    def test_compute_infeasible_destination_is_rejected_without_upgrade(self):
        instance = make_relocate_instance(destination_compute=2_000_000.0)
        before = make_relocate_state(instance)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertFalse(result.accepted)
        self.assertIn(AGHRelocateRejectionReason.INFEASIBLE, result.rejection_reasons)
        self.assertIn(
            ConstraintFamily.COMPUTE_CAPACITY,
            result.validation_report.violated_families,
        )
        self.assertEqual(result.prospective_placement.gpu_counts, before.gpu_counts)

    def test_error_infeasibility_retains_exact_validator_evidence(self):
        instance = make_relocate_instance(destination_error=0.20)
        before = make_relocate_state(instance)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertFalse(result.accepted)
        self.assertIn(ConstraintFamily.ERROR_SLO, result.validation_report.violated_families)
        diagnostic = result.validation_report.error_slo_diagnostics[0]
        self.assertAlmostEqual(diagnostic.weighted_error, 0.20)
        self.assertAlmostEqual(diagnostic.error_limit, 0.10)

    def test_multiple_validation_failures_are_not_collapsed(self):
        instance = make_relocate_instance(
            destination_delay=4.0,
            destination_error=0.20,
            destination_compute=2_000_000.0,
        )
        before = make_relocate_state(instance)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        families = set(result.validation_report.violated_families)
        self.assertTrue(
            {
                ConstraintFamily.COMPUTE_CAPACITY,
                ConstraintFamily.DELAY_SLO,
                ConstraintFamily.ERROR_SLO,
            }.issubset(families)
        )
        self.assertGreaterEqual(result.validation_report.violation_count, 3)


class AGHRelocateMappingTests(unittest.TestCase):
    """Protect the exact approved x/z/q/u/w/y/v mapping."""

    def test_complete_fraction_moves_and_no_partial_source_remains(self):
        instance = make_relocate_instance()
        before = make_relocate_state(
            instance, source_x=0.8, destination_x=0.0, unmet=0.2
        )

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertEqual(result.moved_fraction, 0.8)
        self.assertEqual(result.prospective_placement.routing_fractions[SOURCE], 0.0)
        self.assertEqual(
            result.prospective_placement.routing_fractions[DESTINATION_ROUTE], 0.8
        )

    def test_existing_destination_route_receives_exact_addition(self):
        instance = make_relocate_instance()
        before = make_relocate_state(
            instance, source_x=0.4, destination_x=0.4, unmet=0.2
        )

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertEqual(result.moved_fraction, 0.4)
        self.assertEqual(
            result.prospective_placement.routing_fractions[DESTINATION_ROUTE], 0.8
        )

    def test_routed_demand_and_unmet_fraction_are_preserved(self):
        instance = make_relocate_instance()
        before = make_relocate_state(
            instance, source_x=0.4, destination_x=0.4, unmet=0.2
        )

        evidence = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        ).demand_consistency

        self.assertEqual(evidence.routed_before, 0.8)
        self.assertEqual(evidence.routed_after, 0.8)
        self.assertEqual(evidence.unmet_before, 0.2)
        self.assertEqual(evidence.unmet_after, 0.2)
        self.assertTrue(evidence.is_consistent)

    def test_unrelated_route_is_bitwise_preserved(self):
        instance = make_relocate_instance(include_third=True)
        before = make_relocate_state(
            instance,
            source_x=0.25,
            destination_x=0.25,
            third_x=0.5,
        )
        key = (QUERY, "model_unrelated", TIER)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertEqual(
            result.prospective_placement.routing_fractions[key],
            before.routing_fractions[key],
        )

    def test_source_deployment_is_retained_after_its_route_becomes_zero(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )
        after = result.prospective_placement

        self.assertEqual(after.deployment_indicators[SOURCE_PAIR], 1)
        self.assertEqual(after.configuration_selectors[(*SOURCE_PAIR, 1, 1)], 1)
        self.assertEqual(after.gpu_counts[SOURCE_PAIR], 1)
        self.assertEqual(result.objective_breakdown_before.gpu_rental_cost, 2.0)
        self.assertEqual(result.objective_breakdown_after.gpu_rental_cost, 2.0)

    def test_destination_configuration_is_retained_exactly(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        after = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        ).prospective_placement

        self.assertEqual(after.deployment_indicators, before.deployment_indicators)
        self.assertEqual(after.configuration_selectors, before.configuration_selectors)
        self.assertEqual(after.gpu_counts, before.gpu_counts)

    def test_z_uses_only_established_positive_route_mapping(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        after = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        ).prospective_placement

        self.assertEqual(after.admission_indicators[SOURCE], 0)
        self.assertEqual(after.admission_indicators[DESTINATION_ROUTE], 1)

    def test_affected_v_entries_are_exact_x_times_fixed_w(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        after = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        ).prospective_placement

        self.assertEqual(after.auxiliary_products[(*SOURCE, 1, 1)], 0.0)
        self.assertEqual(
            after.auxiliary_products[(*DESTINATION_ROUTE, 1, 1)], 1.0
        )

    def test_paper_version_is_forwarded_to_both_full_objective_reports(self):
        instance = make_relocate_instance()
        before = make_relocate_state(
            instance, source_x=0.8, destination_x=0.0, unmet=0.2
        )

        v1 = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR, paper_version=PaperVersion.V1
        )
        v2 = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR, paper_version=PaperVersion.V2
        )

        self.assertEqual(v1.validation_before.paper_version, PaperVersion.V1)
        self.assertEqual(v2.validation_report.paper_version, PaperVersion.V2)
        self.assertEqual(
            v1.objective_breakdown_before.unmet_demand_penalty,
            v2.objective_breakdown_before.unmet_demand_penalty,
        )


class AGHRelocateEligibilityTests(unittest.TestCase):
    """Exercise structural source/destination rejection before construction."""

    def test_inactive_destination_is_rejected_without_fallback(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance, destination_active=False)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.DESTINATION_NOT_ACTIVE,),
        )
        self.assertIsNone(result.prospective_placement)

    def test_zero_source_route_is_rejected_explicitly(self):
        instance = make_relocate_instance()
        before = make_relocate_state(
            instance, source_x=0.0, destination_x=1.0
        )

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.NO_SOURCE_ROUTING,),
        )
        self.assertEqual(result.moved_fraction, 0.0)

    def test_unknown_source_is_rejected_explicitly(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        result = evaluate_agh_relocate_move(
            instance, before, (QUERY, "missing", TIER), DESTINATION_PAIR
        )
        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.INVALID_SOURCE,),
        )
        self.assertIs(result.accepted_placement, before)
        self.assertIs(
            result.objective_breakdown_before,
            result.validation_before.objective_breakdown,
        )
        self.assertIsNone(result.objective_breakdown_after)
        self.assertIsNone(result.objective_delta)

    def test_unknown_destination_is_rejected_explicitly(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, ("missing", TIER)
        )
        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.INVALID_DESTINATION,),
        )

    def test_source_and_destination_pair_must_be_distinct(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, SOURCE_PAIR
        )
        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.SAME_SOURCE_AND_DESTINATION,),
        )

    def test_malformed_destination_configuration_is_rejected(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance, destination_selected=False)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.INCOHERENT_DESTINATION_DEPLOYMENT,),
        )
        self.assertFalse(result.destination_evidence.is_coherent_active_deployment)

    def test_malformed_source_admission_is_rejected(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance, source_z=0)
        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )
        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.INCOHERENT_SOURCE_DEPLOYMENT,),
        )

    def test_domain_invalid_input_is_not_clipped_or_normalized(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance, source_x=1.1)

        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )

        self.assertEqual(
            result.rejection_reasons,
            (AGHRelocateRejectionReason.INVALID_PLACEMENT_DOMAINS,),
        )
        self.assertEqual(before.routing_fractions[SOURCE], 1.1)
        self.assertIsNone(result.prospective_placement)


class AGHRelocatePurityTests(unittest.TestCase):
    """Protect deterministic immutability and the bounded implementation scope."""

    def test_identical_input_produces_identical_complete_evidence(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        first = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )
        second = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )
        self.assertEqual(first, second)

    def test_original_nested_state_remains_unchanged(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        snapshot = (
            dict(before.routing_fractions),
            dict(before.admission_indicators),
            dict(before.deployment_indicators),
            dict(before.unmet_fractions),
            dict(before.configuration_selectors),
            dict(before.gpu_counts),
            dict(before.auxiliary_products),
        )

        evaluate_agh_relocate_move(instance, before, SOURCE, DESTINATION_PAIR)

        self.assertEqual(
            snapshot,
            (
                dict(before.routing_fractions),
                dict(before.admission_indicators),
                dict(before.deployment_indicators),
                dict(before.unmet_fractions),
                dict(before.configuration_selectors),
                dict(before.gpu_counts),
                dict(before.auxiliary_products),
            ),
        )
        with self.assertRaises(TypeError):
            before.routing_fractions[SOURCE] = 0.0

    def test_result_and_evidence_are_frozen(self):
        result = evaluate_agh_relocate_move(
            make_relocate_instance(),
            make_relocate_state(make_relocate_instance()),
            SOURCE,
            DESTINATION_PAIR,
        )
        with self.assertRaises(FrozenInstanceError):
            result.rejection_reasons = ()
        with self.assertRaises(FrozenInstanceError):
            result.demand_consistency.routed_after = 0.0
        with self.assertRaisesRegex(ValueError, "requires a prospective placement"):
            replace(result, prospective_placement=None)

    def test_source_imports_exclude_m1_m3_pyomo_and_solvers(self):
        tree = ast.parse(inspect.getsource(relocate_module))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertFalse(any("pyomo" in name for name in imported))
        self.assertFalse(any(
            name.rsplit(".", 1)[-1] in ("mechanisms", "m1", "m2", "m3")
            for name in imported
        ))
        self.assertFalse(any(name.endswith(".solver") for name in imported))
        source = inspect.getsource(relocate_module)
        self.assertNotIn("select_minimum_feasible_parallelism(", source)
        self.assertNotIn("select_m3_parallelism_upgrade(", source)

    def test_module_exposes_one_public_operation_and_no_pass_or_scan(self):
        tree = ast.parse(inspect.getsource(relocate_module))
        public_functions = tuple(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        )
        self.assertEqual(public_functions, ("evaluate_agh_relocate_move",))
        evaluate_node = next(
            node for node in tree.body if getattr(node, "name", None) == public_functions[0]
        )
        self.assertFalse(
            any(
                isinstance(node, (ast.For, ast.While))
                for node in ast.walk(evaluate_node)
            )
        )

    def test_prospective_objective_is_exact_validator_objective(self):
        instance = make_relocate_instance()
        before = make_relocate_state(instance)
        result = evaluate_agh_relocate_move(
            instance, before, SOURCE, DESTINATION_PAIR
        )
        direct = validate_placement(
            instance, result.prospective_placement, PaperVersion.V2
        )
        self.assertEqual(result.objective_breakdown_after, direct.objective_breakdown)
        self.assertEqual(result.validation_report, direct)
        self.assertIs(
            result.objective_breakdown_before,
            result.validation_before.objective_breakdown,
        )
        self.assertIs(
            result.objective_breakdown_after,
            result.validation_report.objective_breakdown,
        )


if __name__ == "__main__":
    unittest.main()
