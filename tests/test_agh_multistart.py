"""Focused tests for complete AGH multi-start orchestration."""

import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    DETERMINISTIC_AGH_ORDERING_SPECS,
    AGH_MAX_CONSECUTIVE_NONIMPROVING_STARTS,
    AGHMultiStartIncumbentStatus,
    AGHMultiStartTerminationReason,
    AGHOrderingSource,
    AGHStartComparisonOutcome,
    GHPhase1Config,
    compute_adaptive_agh_random_start_count,
    build_all_deterministic_agh_orderings,
    run_agh_multistart,
    run_agh_start,
)
from llm_resource_allocation.algorithms import agh_multistart as multistart_module
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.math.objective import (
    ObjectiveBreakdown,
    PaperVersion,
)
from llm_resource_allocation.validation.placement import ValidationTolerances


def _base_start_results():
    instance = make_minimal_tiny_instance()
    ordering = build_all_deterministic_agh_orderings(instance)[0]
    completed = run_agh_start(instance, GHPhase1Config(0.8), ordering)
    infeasible_instance = replace(
        instance,
        queries=(replace(instance.queries[0], error_slo=0.0),),
    )
    infeasible_ordering = build_all_deterministic_agh_orderings(
        infeasible_instance
    )[0]
    incomplete = run_agh_start(
        infeasible_instance, GHPhase1Config(0.8), infeasible_ordering
    )
    return instance, completed, incomplete


def _completed_with_objective(base, ordering, total):
    objective = ObjectiveBreakdown(total, 0.0, 0.0, 0.0, 0.0)
    validation = replace(base.final_validation, objective_breakdown=objective)
    return replace(
        base,
        ordering=ordering,
        consolidation_result=replace(base.consolidation_result, validation_after=validation),
    )


def _side_effect_from(base, outcomes, incomplete_base=None):
    outcome_iterator = iter(outcomes)

    def execute(context, config, ordering, phase1_result):
        outcome = next(outcome_iterator)
        if outcome is None:
            if incomplete_base is None:
                raise AssertionError("test requires incomplete AGH evidence")
            return replace(incomplete_base, ordering=ordering)
        return _completed_with_objective(base, ordering, outcome)

    return execute


class AGHMultiStartRealExecutionTests(unittest.TestCase):
    """Exercise the public orchestration with the real one-start authority."""

    def test_all_starts_share_phase1_but_keep_distinct_gh_placements(self):
        instance = make_minimal_tiny_instance()
        with patch.object(
            multistart_module, "run_gh_phase1", wraps=multistart_module.run_gh_phase1
        ) as phase1:
            result = run_agh_multistart(instance, GHPhase1Config(0.8), random_seed=37)

        phase1.assert_called_once()
        starts = result.start_results
        shared_phase1 = starts[0].gh_result.phase1_result
        self.assertTrue(all(start.gh_result.phase1_result is shared_phase1 for start in starts))
        self.assertEqual(len({id(start.gh_placement) for start in starts}), len(starts))
        self.assertEqual(len(starts), 6)

    def test_minimal_case_establishes_then_stops_after_five_ties(self):
        instance = make_minimal_tiny_instance()

        result = run_agh_multistart(
            instance, GHPhase1Config(0.8), random_seed=37
        )

        self.assertTrue(result.has_feasible_incumbent)
        self.assertEqual(
            result.incumbent_status, AGHMultiStartIncumbentStatus.AVAILABLE
        )
        self.assertEqual(result.scheduled_start_count, 28)
        self.assertEqual(result.executed_start_count, 6)
        self.assertEqual(result.unexecuted_start_count, 22)
        self.assertEqual(result.consecutive_nonimprovements, 5)
        self.assertEqual(
            result.termination_reason,
            AGHMultiStartTerminationReason.FIVE_CONSECUTIVE_NONIMPROVING,
        )
        self.assertAlmostEqual(result.final_incumbent.final_objective_breakdown.total, 1.00410003)
        self.assertEqual(
            result.start_traces[0].comparison_outcome,
            AGHStartComparisonOutcome.INITIAL_INCUMBENT_ESTABLISHED,
        )
        self.assertEqual(
            [trace.consecutive_nonimprovements_after for trace in result.start_traces],
            [0, 1, 2, 3, 4, 5],
        )
        self.assertTrue(
            all(
                trace.ordering.source is AGHOrderingSource.DETERMINISTIC
                for trace in result.start_traces
            )
        )

    def test_repeated_real_execution_is_deterministic(self):
        instance = make_minimal_tiny_instance()
        first = run_agh_multistart(instance, GHPhase1Config(0.8), random_seed=11)
        second = run_agh_multistart(instance, GHPhase1Config(0.8), random_seed=11)

        self.assertEqual(first, second)


class AGHMultiStartScheduleTests(unittest.TestCase):
    """Prove exact ordering, random provenance, and schedule completion."""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.base, cls.incomplete = _base_start_results()
        cls.config = GHPhase1Config(0.8)

    def test_full_schedule_is_eight_deterministic_then_exactly_r_random(self):
        objectives = tuple(float(100 - index) for index in range(28))
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(self.base, objectives),
        ) as one_start:
            result = run_agh_multistart(
                self.instance, self.config, random_seed=1234
            )

        self.assertEqual(result.scale.random_start_count, 20)
        self.assertEqual(result.scheduled_start_count, 28)
        self.assertEqual(result.executed_start_count, 28)
        self.assertEqual(result.unexecuted_start_count, 0)
        self.assertEqual(
            result.termination_reason,
            AGHMultiStartTerminationReason.SCHEDULE_EXHAUSTED,
        )
        self.assertEqual(one_start.call_count, 28)
        self.assertEqual(
            tuple(
                trace.ordering.deterministic_spec
                for trace in result.start_traces[:8]
            ),
            DETERMINISTIC_AGH_ORDERING_SPECS,
        )
        for random_index, trace in enumerate(result.start_traces[8:]):
            self.assertIs(trace.ordering.source, AGHOrderingSource.SEEDED_RANDOM)
            self.assertEqual(trace.ordering.seed, 1234)
            self.assertEqual(trace.ordering.random_start_index, random_index)

    def test_duplicate_permutations_are_not_removed(self):
        objectives = tuple(float(100 - index) for index in range(28))
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(self.base, objectives),
        ):
            result = run_agh_multistart(self.instance, self.config, random_seed=5)

        self.assertEqual(len({trace.ordering.query_ids for trace in result.start_traces}), 1)
        self.assertEqual(result.executed_start_count, 28)

    def test_same_seed_reproduces_random_schedule(self):
        objectives = tuple(float(100 - index) for index in range(28))
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(self.base, objectives),
        ):
            first = run_agh_multistart(self.instance, self.config, random_seed=91)
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(self.base, objectives),
        ):
            second = run_agh_multistart(self.instance, self.config, random_seed=91)

        self.assertEqual(
            tuple(trace.ordering for trace in first.start_traces[8:]),
            tuple(trace.ordering for trace in second.start_traces[8:]),
        )

    def test_adaptive_random_count_boundaries_remain_exact(self):
        self.assertEqual(compute_adaptive_agh_random_start_count(500, 1, 1), 20)
        self.assertEqual(compute_adaptive_agh_random_start_count(501, 1, 1), 10)
        self.assertEqual(compute_adaptive_agh_random_start_count(2000, 1, 1), 10)
        self.assertEqual(compute_adaptive_agh_random_start_count(2001, 1, 1), 5)
        self.assertEqual(compute_adaptive_agh_random_start_count(5000, 1, 1), 5)
        self.assertEqual(compute_adaptive_agh_random_start_count(5001, 1, 1), 3)


class AGHMultiStartIncumbentTests(unittest.TestCase):
    """Prove D-0045 incumbent and counter transitions."""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.base, cls.incomplete = _base_start_results()
        cls.config = GHPhase1Config(0.8)

    def _run(self, outcomes, seed=7):
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(self.base, outcomes, self.incomplete),
        ) as one_start:
            result = run_agh_multistart(
                self.instance, self.config, random_seed=seed
            )
        return result, one_start

    def test_ties_and_worse_candidates_do_not_replace_incumbent(self):
        result, _ = self._run((100.0, 100.0, 101.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0))

        first, tie, worse, improved = result.start_traces[:4]
        self.assertIs(tie.incumbent_after, first.start_result)
        self.assertIs(worse.incumbent_after, first.start_result)
        self.assertEqual(tie.comparison_outcome, AGHStartComparisonOutcome.NON_IMPROVING)
        self.assertEqual(worse.comparison_outcome, AGHStartComparisonOutcome.NON_IMPROVING)
        self.assertIs(improved.incumbent_after, improved.start_result)
        self.assertEqual(improved.comparison_outcome, AGHStartComparisonOutcome.STRICT_IMPROVEMENT)
        self.assertIs(result.final_incumbent, improved.start_result)

    def test_strict_improvement_resets_counter(self):
        result, _ = self._run((100.0, 100.0, 101.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0))

        self.assertEqual(
            [trace.consecutive_nonimprovements_after for trace in result.start_traces[:4]],
            [0, 1, 2, 0],
        )
        self.assertTrue(result.start_traces[3].strict_improvement)

    def test_initial_incumbent_after_failures_resets_counter_without_improvement(self):
        result, _ = self._run((None, None, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0))

        self.assertEqual(
            [trace.consecutive_nonimprovements_after for trace in result.start_traces[:3]],
            [1, 2, 0],
        )
        established = result.start_traces[2]
        self.assertEqual(
            established.comparison_outcome,
            AGHStartComparisonOutcome.INITIAL_INCUMBENT_ESTABLISHED,
        )
        self.assertFalse(established.strict_improvement)
        self.assertTrue(established.incumbent_changed)

    def test_five_incomplete_starts_stop_with_no_incumbent(self):
        result, one_start = self._run((None, None, None, None, None))

        self.assertEqual(one_start.call_count, 5)
        self.assertIsNone(result.final_incumbent)
        self.assertFalse(result.has_feasible_incumbent)
        self.assertEqual(
            result.incumbent_status,
            AGHMultiStartIncumbentStatus.NO_FEASIBLE_INCUMBENT,
        )
        self.assertEqual(result.consecutive_nonimprovements, 5)

    def test_counter_spans_deterministic_random_boundary(self):
        objectives = (
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            90.0,
            90.0,
            90.0,
            90.0,
            80.0,
            79.0,
            78.0,
            77.0,
            76.0,
            75.0,
            74.0,
            73.0,
            72.0,
            71.0,
            70.0,
            69.0,
            68.0,
            67.0,
            66.0,
            65.0,
            64.0,
            63.0,
            62.0,
        )
        result, _ = self._run(objectives)

        last_deterministic = result.start_traces[7]
        first_random = result.start_traces[8]
        self.assertEqual(last_deterministic.consecutive_nonimprovements_after, 2)
        self.assertEqual(first_random.consecutive_nonimprovements_before, 2)
        self.assertEqual(first_random.consecutive_nonimprovements_after, 3)

    def test_stop_is_immediate_and_random_orderings_are_not_built(self):
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(
                self.base, (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
            ),
        ) as one_start, patch.object(
            multistart_module,
            "build_seeded_random_agh_ordering",
            wraps=multistart_module.build_seeded_random_agh_ordering,
        ) as random_builder:
            result = run_agh_multistart(
                self.instance, self.config, random_seed=17
            )

        self.assertEqual(one_start.call_count, 6)
        random_builder.assert_not_called()
        self.assertEqual(
            result.consecutive_nonimprovements,
            AGH_MAX_CONSECUTIVE_NONIMPROVING_STARTS,
        )

    def test_one_start_receives_public_configuration_unchanged(self):
        tolerances = ValidationTolerances(demand_balance=1e-8)
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(
                self.base, (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
            ),
        ) as one_start:
            result = run_agh_multistart(
                self.instance,
                self.config,
                random_seed=19,
                paper_version=PaperVersion.V1,
                validation_tolerances=tolerances,
            )

        for call, trace in zip(one_start.call_args_list, result.start_traces, strict=True):
            context, config, ordering, phase1_result = call.args
            self.assertIs(context.problem_data.instance, self.instance)
            self.assertIs(config, self.config)
            self.assertIs(ordering, trace.ordering)
            self.assertIs(context.validator.paper_version, PaperVersion.V1)
            self.assertIs(context.validator.tolerances, tolerances)
            self.assertIs(context, one_start.call_args_list[0].args[0])
            self.assertIs(phase1_result, one_start.call_args_list[0].args[3])
            self.assertEqual(call.kwargs, {})

    def test_invalid_seed_fails_before_any_start(self):
        with patch.object(multistart_module, "_run_agh_start") as one_start:
            for invalid in (True, 1.5, "7", None):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "random_seed"):
                        run_agh_multistart(
                            self.instance, self.config, random_seed=invalid
                        )
        one_start.assert_not_called()


class AGHMultiStartEvidenceTests(unittest.TestCase):
    """Protect immutable evidence and the 6F composition boundary."""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.base, cls.incomplete = _base_start_results()
        cls.config = GHPhase1Config(0.8)

    def test_result_trace_and_scale_are_immutable(self):
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(
                self.base, (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
            ),
        ):
            result = run_agh_multistart(
                self.instance, self.config, random_seed=29
            )

        for target, attribute, value in (
            (result, "random_seed", 30),
            (result.scale, "query_type_count", 2),
            (result.start_traces[0], "schedule_index", 4),
        ):
            with self.subTest(target=type(target).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(target, attribute, value)
        self.assertIsInstance(result.start_traces, tuple)
        self.assertEqual(result.start_results, tuple(t.start_result for t in result.start_traces))

    def test_orchestration_does_not_mutate_inputs(self):
        expected_instance = make_minimal_tiny_instance()
        expected_config = GHPhase1Config(0.8)
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(
                self.base, (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
            ),
        ):
            run_agh_multistart(self.instance, self.config, random_seed=30)

        self.assertEqual(self.instance, expected_instance)
        self.assertEqual(self.config, expected_config)

    def test_final_incumbent_is_exact_executed_start_object(self):
        with patch.object(
            multistart_module,
            "_run_agh_start",
            side_effect=_side_effect_from(
                self.base, (100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0)
            ),
        ):
            result = run_agh_multistart(
                self.instance, self.config, random_seed=31
            )

        self.assertIs(result.final_incumbent, result.start_traces[1].start_result)

    def test_orchestrator_calls_only_the_one_start_algorithm_authority(self):
        source = inspect.getsource(multistart_module.run_agh_multistart)
        tree = ast.parse(source)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        self.assertEqual(
            sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_run_agh_start"
            ),
            1,
        )
        self.assertTrue(
            {"run_gh", "run_agh_relocate", "run_agh_consolidation_scan"}.isdisjoint(
                called_names
            )
        )


if __name__ == "__main__":
    unittest.main()
