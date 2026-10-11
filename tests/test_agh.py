"""Focused tests for final public AGH Algorithm-2 closure."""

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    AGHFinalCheckFailureReason,
    AGHCompletionStatus,
    AGHMultiStartTerminationReason,
    AGHResult,
    GHPhase1Config,
    build_all_deterministic_agh_orderings,
    run_agh,
    run_agh_start,
)
from llm_resource_allocation.algorithms import agh as agh_module
from llm_resource_allocation.algorithms import agh_multistart as multistart_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.validation.placement import (
    ValidationTolerances,
    PlacementValidator,
    validate_placement,
)


def _completed_start(instance, config):
    ordering = build_all_deterministic_agh_orderings(instance)[0]
    return run_agh_start(instance, config, ordering)


def _strictly_improving_start_factory(base_start, start_count):
    call_index = 0

    def execute(context, config, ordering, phase1_result):
        nonlocal call_index
        decrease_remaining = start_count - call_index - 1
        objective = replace(
            base_start.final_objective_breakdown,
            gpu_rental_cost=(
                base_start.final_objective_breakdown.gpu_rental_cost + decrease_remaining
            ),
        )
        validation = replace(
            base_start.final_validation,
            objective_breakdown=objective,
        )
        call_index += 1
        return replace(
            base_start,
            ordering=ordering,
            consolidation_result=replace(
                base_start.consolidation_result, validation_after=validation
            ),
        )

    return execute


class AGHFinalClosureEndToEndTests(unittest.TestCase):
    """Exercise the real complete pipeline on hand-checkable synthetic data."""

    def setUp(self):
        self.instance = make_minimal_tiny_instance()
        self.config = GHPhase1Config(0.8)

    def test_minimal_run_validates_each_start_once_and_fresh_final_incumbent(self):
        validated = []
        original = PlacementValidator.validate

        def record(validator, placement):
            validated.append(placement)
            return original(validator, placement)

        with patch.object(PlacementValidator, "validate", autospec=True, side_effect=record):
            result = run_agh(self.instance, self.config, random_seed=37)

        starts = result.multistart_result.start_results
        self.assertEqual(len(validated), len(starts) + 1)
        self.assertEqual(
            [id(item) for item in validated[:-1]], [id(start.gh_placement) for start in starts]
        )
        self.assertIs(validated[-1], result.final_placement)
        self.assertIsNot(result.final_validation, result.final_incumbent.final_validation)

    def test_real_minimal_case_closes_algorithm_2(self):
        result = run_agh(self.instance, self.config, random_seed=37)

        self.assertIsInstance(result, AGHResult)
        self.assertTrue(result.completed)
        self.assertEqual(result.completion_status, AGHCompletionStatus.COMPLETED)
        self.assertEqual(result.failure_reasons, ())
        self.assertTrue(result.final_validation.is_feasible)
        self.assertAlmostEqual(result.final_objective_breakdown.total, 1.00410003)
        self.assertEqual(result.scheduled_random_start_count, 20)
        self.assertEqual(result.random_seed, 37)

    def test_final_identity_and_full_objective_match_incumbent(self):
        result = run_agh(self.instance, self.config, random_seed=41)

        self.assertIs(
            result.final_incumbent,
            result.multistart_result.final_incumbent,
        )
        self.assertIs(
            result.final_placement,
            result.final_incumbent.final_placement,
        )
        self.assertEqual(
            result.final_objective_breakdown,
            result.final_incumbent.final_objective_breakdown,
        )
        self.assertIs(
            result.final_objective_breakdown,
            result.final_validation.objective_breakdown,
        )
        self.assertIsNot(
            result.final_validation,
            result.final_incumbent.final_validation,
        )

    def test_real_early_stop_evidence_remains_visible(self):
        result = run_agh(self.instance, self.config, random_seed=43)

        self.assertEqual(
            result.termination_reason,
            AGHMultiStartTerminationReason.FIVE_CONSECUTIVE_NONIMPROVING,
        )
        self.assertEqual(result.executed_start_count, 6)
        self.assertEqual(result.unexecuted_start_count, 22)

    def test_fresh_validator_is_called_once_on_exact_incumbent_placement(self):
        tolerances = ValidationTolerances(demand_balance=1e-9)
        with patch.object(
            agh_module,
            "validate_placement",
            wraps=agh_module.validate_placement,
        ) as validator:
            result = run_agh(
                self.instance,
                self.config,
                random_seed=47,
                paper_version=PaperVersion.V1,
                validation_tolerances=tolerances,
            )

        validator.assert_called_once_with(
            self.instance,
            result.final_incumbent.final_placement,
            PaperVersion.V1,
            tolerances,
        )
        self.assertTrue(result.completed)

    def test_repeated_execution_with_same_seed_is_deterministic(self):
        first = run_agh(self.instance, self.config, random_seed=53)
        second = run_agh(self.instance, self.config, random_seed=53)

        self.assertEqual(first, second)

    def test_input_instance_and_configuration_are_unchanged(self):
        expected_instance = make_minimal_tiny_instance()
        expected_config = GHPhase1Config(0.8)

        run_agh(self.instance, self.config, random_seed=59)

        self.assertEqual(self.instance, expected_instance)
        self.assertEqual(self.config, expected_config)


class AGHFinalClosureBoundaryTests(unittest.TestCase):
    """Prove thin delegation, typed absence, and closure inconsistencies."""

    @classmethod
    def setUpClass(cls):
        cls.instance = make_minimal_tiny_instance()
        cls.config = GHPhase1Config(0.8)
        cls.base_start = _completed_start(cls.instance, cls.config)

    def test_multistart_is_invoked_exactly_once_and_preserved(self):
        returned_results = []

        def execute_multistart(*args, **kwargs):
            result = multistart_module.run_agh_multistart(*args, **kwargs)
            returned_results.append(result)
            return result

        with patch.object(
            agh_module,
            "run_agh_multistart",
            side_effect=execute_multistart,
        ) as multistart:
            result = run_agh(self.instance, self.config, random_seed=61)

        multistart.assert_called_once_with(
            self.instance,
            self.config,
            random_seed=61,
            paper_version=PaperVersion.V2,
            validation_tolerances=None,
        )
        self.assertIs(result.multistart_result, returned_results[0])

    def test_schedule_exhaustion_is_preserved_by_public_result(self):
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_strictly_improving_start_factory(self.base_start, 28),
        ) as one_start:
            result = run_agh(self.instance, self.config, random_seed=67)

        self.assertTrue(result.completed)
        self.assertEqual(one_start.call_count, 28)
        self.assertEqual(
            result.termination_reason,
            AGHMultiStartTerminationReason.SCHEDULE_EXHAUSTED,
        )
        self.assertEqual(result.executed_start_count, 28)
        self.assertEqual(result.unexecuted_start_count, 0)
        self.assertEqual(
            result.final_objective_breakdown,
            self.base_start.final_objective_breakdown,
        )

    def test_real_no_valid_incumbent_is_typed_and_not_revalidated(self):
        infeasible_instance = replace(
            self.instance,
            queries=(replace(self.instance.queries[0], error_slo=0.0),),
        )
        with patch.object(agh_module, "validate_placement") as validator:
            result = run_agh(
                infeasible_instance,
                self.config,
                random_seed=71,
            )

        validator.assert_not_called()
        self.assertFalse(result.completed)
        self.assertEqual(
            result.completion_status,
            AGHCompletionStatus.NO_FEASIBLE_INCUMBENT,
        )
        self.assertEqual(
            result.failure_reasons,
            (AGHFinalCheckFailureReason.NO_FEASIBLE_INCUMBENT,),
        )
        self.assertIsNone(result.final_incumbent)
        self.assertIsNone(result.final_placement)
        self.assertIsNone(result.final_validation)
        self.assertIsNone(result.final_objective_breakdown)
        self.assertEqual(result.executed_start_count, 5)

    def test_fresh_objective_mismatch_returns_typed_failure_without_repair(self):
        accepted = run_agh(self.instance, self.config, random_seed=73)
        mismatched_objective = replace(
            accepted.final_objective_breakdown,
            gpu_rental_cost=accepted.final_objective_breakdown.gpu_rental_cost + 1.0,
        )
        mismatched_validation = replace(
            accepted.final_validation,
            objective_breakdown=mismatched_objective,
        )
        with patch.object(
            agh_module,
            "run_agh_multistart",
            return_value=accepted.multistart_result,
        ), patch.object(
            agh_module,
            "validate_placement",
            return_value=mismatched_validation,
        ):
            result = run_agh(self.instance, self.config, random_seed=73)

        self.assertEqual(
            result.completion_status,
            AGHCompletionStatus.FINAL_CONSISTENCY_FAILED,
        )
        self.assertEqual(
            result.failure_reasons,
            (AGHFinalCheckFailureReason.FINAL_OBJECTIVE_MISMATCH,),
        )
        self.assertIs(result.final_objective_breakdown, mismatched_objective)
        self.assertIs(result.final_placement, accepted.final_placement)

    def test_fresh_infeasibility_returns_typed_failure_without_repair(self):
        accepted = run_agh(self.instance, self.config, random_seed=79)
        tighter_instance = replace(
            self.instance,
            queries=(replace(self.instance.queries[0], error_slo=0.0),),
        )
        infeasible_validation = validate_placement(
            tighter_instance,
            accepted.final_placement,
            PaperVersion.V2,
        )
        self.assertFalse(infeasible_validation.is_feasible)
        with patch.object(
            agh_module,
            "run_agh_multistart",
            return_value=accepted.multistart_result,
        ), patch.object(
            agh_module,
            "validate_placement",
            return_value=infeasible_validation,
        ):
            result = run_agh(self.instance, self.config, random_seed=79)

        self.assertEqual(
            result.completion_status,
            AGHCompletionStatus.FINAL_CONSISTENCY_FAILED,
        )
        self.assertEqual(
            result.failure_reasons,
            (AGHFinalCheckFailureReason.FINAL_VALIDATION_INFEASIBLE,),
        )
        self.assertIs(result.final_placement, accepted.final_placement)

    def test_result_evidence_is_frozen(self):
        result = run_agh(self.instance, self.config, random_seed=83)

        with self.assertRaises(FrozenInstanceError):
            result.completion_status = AGHCompletionStatus.NO_FEASIBLE_INCUMBENT
        self.assertIsInstance(result.failure_reasons, tuple)

    def test_producer_retains_incumbent_and_fresh_objective_evidence(self):
        result = run_agh(self.instance, self.config, random_seed=89)

        self.assertIs(result.final_incumbent, result.multistart_result.final_incumbent)
        self.assertIs(result.final_placement, result.final_incumbent.final_placement)
        self.assertIs(
            result.final_objective_breakdown,
            result.final_validation.objective_breakdown,
        )
        self.assertIsNot(result.final_validation, result.final_incumbent.final_validation)
        self.assertTrue(result.final_validation.is_feasible)
        self.assertEqual(
            result.final_objective_breakdown,
            result.final_incumbent.final_objective_breakdown,
        )

    def test_closure_source_contains_no_internal_agh_reimplementation(self):
        source = inspect.getsource(agh_module.run_agh)
        tree = ast.parse(source)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        self.assertEqual(
            sum(name == "run_agh_multistart" for name in called_names),
            1,
        )
        self.assertEqual(sum(name == "validate_placement" for name in called_names), 1)
        self.assertTrue(
            {
                "_run_agh_start",
                "run_gh",
                "run_agh_relocate",
                "run_agh_consolidation_scan",
            }.isdisjoint(called_names)
        )


if __name__ == "__main__":
    unittest.main()
