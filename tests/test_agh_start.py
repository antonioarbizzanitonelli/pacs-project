"""Focused tests for one complete supplied-ordering AGH start."""

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    DETERMINISTIC_AGH_ORDERING_SPECS,
    AGH_RELOCATE_MAX_PASSES,
    GHPhase1Config,
    build_deterministic_agh_ordering,
    run_gh,
)
from llm_resource_allocation.algorithms import (
    AGHStartCompletionStatus,
    AGHStartFailureReason,
    run_agh_start,
)
from llm_resource_allocation.algorithms import agh_start as start_module
from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.validation.placement import (
    ConstraintFamily,
    validate_placement,
)


QUERY = "q_text"
TIER = "gpu_basic_fp16"


def make_infeasible_gh_instance():
    """Build the hand-derived D-0039 error-SLO failure fixture."""

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        error_slo=0.025,
        delay_slo_seconds=1.0,
        max_unmet_fraction=0.2,
    )
    models = tuple(
        replace(base.models[0], id=model_id, weight_gb=1.0)
        for model_id in ("model_a", "model_b")
    )
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=8.0,
        tensor_parallel_degrees=(1, 2),
        pipeline_parallel_depths=(1,),
    )
    triples = tuple((QUERY, model.id, TIER) for model in models)
    return replace(
        base,
        id="agh_start_d0039_fixture",
        queries=(query,),
        models=models,
        gpu_tiers=(tier,),
        base_error_rate={(QUERY, model.id): 0.05 for model in models},
        compute_delay_seconds_per_token={key: 0.10 for key in triples},
        communication_delay_seconds_per_token={key: 0.01 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=20.0,
        storage_capacity_gb=100.0,
    )


def first_ordering(instance):
    """Return one existing 6A2 ordering without constructing a schedule."""

    return build_deterministic_agh_ordering(
        instance, DETERMINISTIC_AGH_ORDERING_SPECS[0]
    )


class AGHStartSuccessfulLifecycleTests(unittest.TestCase):
    """Prove the real GH-relocate-consolidation-final lifecycle."""

    def test_hand_checkable_tiny_case_completes_every_stage(self):
        instance = make_minimal_tiny_instance()
        ordering = first_ordering(instance)

        result = run_agh_start(instance, GHPhase1Config(0.8), ordering)

        self.assertTrue(result.completed)
        self.assertEqual(
            result.completion_status, AGHStartCompletionStatus.COMPLETED
        )
        self.assertEqual(result.failure_reasons, ())
        self.assertEqual(
            result.gh_result.phase2_result.processed_query_ids,
            ordering.query_ids,
        )
        self.assertEqual(result.relocate_result.passes_executed, 1)
        self.assertEqual(result.consolidation_result.accepted_source_count, 0)
        self.assertTrue(result.final_validation.is_feasible)
        self.assertAlmostEqual(result.final_objective_breakdown.total, 1.00410003)

    def test_exact_placement_identity_chain_is_preserved(self):
        instance = make_minimal_tiny_instance()
        result = run_agh_start(
            instance, GHPhase1Config(0.8), first_ordering(instance)
        )

        self.assertIs(result.gh_placement, result.gh_result.placement_decisions)
        self.assertIs(result.relocate_result.placement_before, result.gh_placement)
        self.assertIs(
            result.relocate_placement, result.relocate_result.placement_after
        )
        self.assertIs(
            result.consolidation_result.placement_before,
            result.relocate_placement,
        )
        self.assertIs(
            result.consolidation_placement,
            result.consolidation_result.placement_after,
        )
        self.assertIs(result.final_placement, result.consolidation_placement)

    def test_supplied_ordering_reaches_only_the_gh_override(self):
        instance = make_heterogeneous_tiny_instance()
        ordering = first_ordering(instance)
        config = GHPhase1Config(0.0)
        with patch.object(
            start_module, "_run_gh_with_phase1", wraps=start_module._run_gh_with_phase1
        ) as gh_call:
            result = run_agh_start(instance, config, ordering)

        gh_call.assert_called_once()
        self.assertEqual(gh_call.call_args.args[:2], (instance, config))
        self.assertIs(gh_call.call_args.args[2], result.gh_result.phase1_result)
        self.assertEqual(gh_call.call_args.kwargs["query_order_override"], ordering.query_ids)
        self.assertIs(gh_call.call_args.kwargs["problem_data"].instance, instance)
        self.assertIs(result.ordering, ordering)
        self.assertEqual(
            result.gh_result.initialized_phase2_state.query_order,
            ordering.query_ids,
        )

    def test_gh_result_equals_the_unchanged_direct_authority(self):
        instance = make_minimal_tiny_instance()
        config = GHPhase1Config(0.8)
        ordering = first_ordering(instance)

        composed = run_agh_start(instance, config, ordering)
        direct = run_gh(
            instance, config, query_order_override=ordering.query_ids
        )

        self.assertEqual(composed.gh_result, direct)

    def test_relocate_and_consolidation_are_each_invoked_once(self):
        instance = make_minimal_tiny_instance()
        ordering = first_ordering(instance)
        with patch.object(
            start_module,
            "_run_agh_relocate",
            wraps=start_module._run_agh_relocate,
        ) as relocate, patch.object(
            start_module,
            "_run_agh_consolidation_scan",
            wraps=start_module._run_agh_consolidation_scan,
        ) as consolidate:
            result = run_agh_start(instance, GHPhase1Config(0.8), ordering)

        relocate.assert_called_once()
        consolidate.assert_called_once()
        self.assertIs(relocate.call_args.args[0].placement, result.gh_placement)
        self.assertEqual(relocate.call_args.kwargs, {"max_passes": AGH_RELOCATE_MAX_PASSES})
        self.assertIs(consolidate.call_args.args[0].placement, result.relocate_placement)
        self.assertIs(
            relocate.call_args.args[0].context, consolidate.call_args.args[0].context
        )

    def test_exact_stage_reports_are_reused_and_objective_is_their_evidence(self):
        instance = make_minimal_tiny_instance()
        result = run_agh_start(
            instance, GHPhase1Config(0.8), first_ordering(instance)
        )
        self.assertIs(
            result.relocate_result.pass_results[0].validation_before,
            result.gh_result.validation_report,
        )
        self.assertIs(
            result.consolidation_result.validation_before,
            result.relocate_result.pass_results[-1].validation_after,
        )
        self.assertIs(result.final_validation, result.consolidation_result.validation_after)
        self.assertIs(
            result.final_objective_breakdown, result.final_validation.objective_breakdown
        )
        self.assertEqual(
            result.final_validation,
            validate_placement(
                instance,
                result.final_placement,
                result.gh_result.identity.paper_version,
                None,
            ),
        )

    def test_repeated_execution_is_deterministic(self):
        instance = make_minimal_tiny_instance()
        ordering = first_ordering(instance)
        config = GHPhase1Config(0.8)

        first = run_agh_start(instance, config, ordering)
        second = run_agh_start(instance, config, ordering)

        self.assertEqual(first, second)

    def test_result_and_nested_inputs_are_deeply_immutable(self):
        instance = make_minimal_tiny_instance()
        ordering = first_ordering(instance)
        result = run_agh_start(instance, GHPhase1Config(0.8), ordering)

        with self.assertRaises(FrozenInstanceError):
            result.completion_status = AGHStartCompletionStatus.COMPLETED
        with self.assertRaises(FrozenInstanceError):
            ordering.query_ids = ("changed",)
        with self.assertRaises(TypeError):
            result.final_placement.unmet_fractions[QUERY] = 1.0


class AGHStartFailureTests(unittest.TestCase):
    """Keep incomplete-start behavior typed, auditable, and non-repairing."""

    def test_infeasible_gh_stops_before_all_local_search(self):
        instance = make_infeasible_gh_instance()
        ordering = first_ordering(instance)
        with patch.object(start_module, "_run_agh_relocate") as relocate, patch.object(
            start_module, "_run_agh_consolidation_scan"
        ) as consolidate:
            result = run_agh_start(instance, GHPhase1Config(0.0), ordering)

        self.assertEqual(
            result.completion_status,
            AGHStartCompletionStatus.STOPPED_AFTER_GH,
        )
        self.assertIn(
            AGHStartFailureReason.GH_INDEPENDENTLY_INFEASIBLE,
            result.failure_reasons,
        )
        self.assertIn(
            ConstraintFamily.ERROR_SLO,
            result.gh_result.validation_report.violated_families,
        )
        self.assertIs(result.gh_placement, result.gh_result.placement_decisions)
        self.assertIsNone(result.relocate_result)
        self.assertIsNone(result.consolidation_result)
        self.assertIsNone(result.final_placement)
        relocate.assert_not_called()
        consolidate.assert_not_called()

    def test_invalid_non_ordering_input_is_rejected_before_gh(self):
        instance = make_minimal_tiny_instance()
        with patch.object(start_module, "_run_gh_with_phase1") as gh_call, self.assertRaisesRegex(
            ValueError, "AGHQueryOrdering"
        ):
            run_agh_start(instance, GHPhase1Config(0.8), instance.query_ids)
        gh_call.assert_not_called()

    def test_final_infeasibility_is_a_typed_noncomplete_result(self):
        instance = make_minimal_tiny_instance()
        invalid_instance = make_infeasible_gh_instance()
        invalid_gh = run_gh(invalid_instance, GHPhase1Config(0.0))
        original = start_module._run_agh_consolidation_scan

        def infeasible_scan(current):
            result = original(current)
            report = replace(
                result.validation_after,
                error_slo_diagnostics=invalid_gh.validation_report.error_slo_diagnostics,
            )
            return replace(result, validation_after=report)

        with patch.object(
            start_module,
            "_run_agh_consolidation_scan",
            side_effect=infeasible_scan,
        ):
            result = run_agh_start(
                instance, GHPhase1Config(0.8), first_ordering(instance)
            )

        self.assertEqual(
            result.completion_status,
            AGHStartCompletionStatus.FINAL_EVALUATION_FAILED,
        )
        self.assertEqual(
            result.failure_reasons,
            (AGHStartFailureReason.FINAL_INDEPENDENTLY_INFEASIBLE,),
        )
        self.assertIsNotNone(result.gh_result)
        self.assertIsNotNone(result.relocate_result)
        self.assertIsNotNone(result.consolidation_result)
        self.assertIs(result.final_placement, result.consolidation_placement)
        self.assertFalse(result.completed)


class AGHStartScopeTests(unittest.TestCase):
    """Prove the orchestrator delegates construction and local search."""

    def test_orchestrator_calls_each_scientific_stage_exactly_once(self):
        function_tree = ast.parse(inspect.getsource(start_module._run_agh_start))
        calls = tuple(
            node.func.id
            for node in ast.walk(function_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )

        self.assertEqual(calls.count("_run_gh_with_phase1"), 1)
        self.assertEqual(calls.count("_run_agh_relocate"), 1)
        self.assertEqual(calls.count("_run_agh_consolidation_scan"), 1)
        self.assertEqual(calls.count("validate_placement"), 0)

    def test_orchestrator_contains_no_schedule_or_backend_behavior(self):
        source = inspect.getsource(start_module.run_agh_start).lower()
        for forbidden in (
            "build_all_deterministic",
            "build_seeded_random",
            "compute_adaptive",
            "incumbent",
            "five_nonimproving",
            "pyomo",
            "solver",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_only_one_ordering_object_is_retained(self):
        instance = make_minimal_tiny_instance()
        ordering = first_ordering(instance)
        result = run_agh_start(instance, GHPhase1Config(0.8), ordering)

        self.assertIs(result.ordering, ordering)
        self.assertFalse(hasattr(result, "orderings"))
        self.assertFalse(hasattr(result, "incumbent"))
        self.assertFalse(hasattr(result, "non_improving_count"))


if __name__ == "__main__":
    unittest.main()
