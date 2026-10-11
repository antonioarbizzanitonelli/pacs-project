"""Tests for the isolated Milestone-4D solver and objective cross-check."""

import unittest
from dataclasses import FrozenInstanceError, replace
from math import inf, nan
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pyomo.contrib.appsi.base import TerminationCondition

from llm_resource_allocation.data.tiny_instances import make_minimal_tiny_instance
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.optimization.solver import (
    NoSolverIncumbentError,
    SolverBackend,
    SolverBackendAvailability,
    SolverBackendInfo,
    SolverBackendUnavailableError,
    SolverConfig,
    SolverStatus,
    UnsupportedSolverBackendError,
    probe_solver_backend,
    require_solver_incumbent,
    solve_pyomo_model,
    translate_gurobi_options,
    translate_highs_options,
)
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.pyomo_pdm import build_pdm_pyomo_model
from llm_resource_allocation.optimization.extraction import (
    extract_raw_pdm_solution_from_pyomo,
    normalize_raw_pdm_solution,
)
from llm_resource_allocation.optimization import solver as solver_module
from llm_resource_allocation.validation.objective_comparison import (
    DEFAULT_OBJECTIVE_COMPARISON_ABSOLUTE_TOLERANCE,
    ObjectiveComparison,
)
from llm_resource_allocation.validation.placement import validate_placement


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ("m_small", "gpu_basic_fp16")
W_11 = (*PAIR, 1, 1)
W_21 = (*PAIR, 2, 1)
V_11 = (*ROUTE, 1, 1)
V_21 = (*ROUTE, 2, 1)


def solve_and_independently_validate_minimal():
    """Run the explicit 4D pipeline without hiding any lower-level API."""

    instance = make_minimal_tiny_instance()
    problem_data = PdmProblemData(instance)
    model = build_pdm_pyomo_model(problem_data)
    solver_result = solve_pyomo_model(model, SolverConfig(backend=SolverBackend.HIGHS))
    require_solver_incumbent(solver_result)
    raw_solution = extract_raw_pdm_solution_from_pyomo(model, problem_data)
    decision_state = normalize_raw_pdm_solution(raw_solution, problem_data)
    validation_report = validate_placement(
        instance,
        decision_state,
        PaperVersion.V2,
    )
    objective_comparison = ObjectiveComparison(
        solver_result.objective_value,
        validation_report.objective_breakdown.total,
    )
    return (
        solver_result,
        raw_solution,
        decision_state,
        validation_report,
        objective_comparison,
    )


class SolverConfigurationTests(unittest.TestCase):
    """Protect the Gurobi default, common options, and HiGHS translation."""

    def test_defaults_are_explicit_unspecified_and_immutable(self) -> None:
        config = SolverConfig()

        self.assertIs(config.backend, SolverBackend.GUROBI)
        self.assertIsNone(config.time_limit_seconds)
        self.assertIsNone(config.relative_mip_gap)
        self.assertIsNone(config.absolute_mip_gap)
        self.assertFalse(config.tee)
        self.assertEqual(dict(translate_gurobi_options(config)), {})
        with self.assertRaises(FrozenInstanceError):
            config.tee = True

    def test_requested_options_map_exactly_to_highs(self) -> None:
        config = SolverConfig(
            backend=SolverBackend.HIGHS,
            time_limit_seconds=12.5,
            relative_mip_gap=0.02,
            absolute_mip_gap=0.25,
            tee=True,
        )

        options = translate_highs_options(config)

        self.assertEqual(
            dict(options),
            {
                "time_limit": 12.5,
                "mip_rel_gap": 0.02,
                "mip_abs_gap": 0.25,
            },
        )
        with self.assertRaises(TypeError):
            options["time_limit"] = 1.0

    def test_requested_supported_options_execute_on_highs(self) -> None:
        model = build_pdm_pyomo_model(
            PdmProblemData(make_minimal_tiny_instance())
        )

        result = solve_pyomo_model(
            model,
            SolverConfig(
                backend=SolverBackend.HIGHS,
                time_limit_seconds=5.0,
                relative_mip_gap=0.0,
                absolute_mip_gap=0.0,
            ),
        )

        self.assertIs(result.status, SolverStatus.OPTIMAL)
        self.assertTrue(result.has_incumbent)

    def test_invalid_numeric_options_are_rejected(self) -> None:
        for field_name in (
            "time_limit_seconds",
            "relative_mip_gap",
            "absolute_mip_gap",
        ):
            for invalid in (-1.0, inf, -inf, nan, True, "1"):
                with self.subTest(field=field_name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        SolverConfig(**{field_name: invalid})

    def test_invalid_backend_and_logging_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SolverConfig(backend="highs")
        with self.assertRaises(ValueError):
            SolverConfig(tee=1)

    def test_highs_translation_rejects_gurobi_config(self) -> None:
        with self.assertRaises(UnsupportedSolverBackendError):
            translate_highs_options(
                SolverConfig(backend=SolverBackend.GUROBI)
            )


class SolverAvailabilityAndStatusTests(unittest.TestCase):
    """Check the required backend and termination normalization without timing."""

    def test_highs_backend_is_available_with_a_version(self) -> None:
        info = probe_solver_backend(SolverBackend.HIGHS)

        self.assertIs(info.availability, SolverBackendAvailability.AVAILABLE)
        self.assertIsNotNone(info.version)
        self.assertRegex(info.version, r"^\d+\.\d+\.\d+$")
        self.assertIn("APPSI", info.detail)

    def test_gurobi_probe_returns_a_normalized_availability(self) -> None:
        info = probe_solver_backend(SolverBackend.GUROBI)

        self.assertIn(info.availability, tuple(SolverBackendAvailability))
        self.assertTrue(info.detail)

    def test_time_limit_status_depends_on_actual_incumbent(self) -> None:
        self.assertIs(
            solver_module._normalize_appsi_status(
                TerminationCondition.maxTimeLimit,
                True,
            ),
            SolverStatus.FEASIBLE,
        )
        self.assertIs(
            solver_module._normalize_appsi_status(
                TerminationCondition.maxTimeLimit,
                False,
            ),
            SolverStatus.NO_SOLUTION,
        )


class SolverFailureTransportTests(unittest.TestCase):
    """Preserve unusable-output evidence and guard incumbent loading."""

    def setUp(self) -> None:
        self.backend_info = SolverBackendInfo(
            SolverBackend.HIGHS, SolverBackendAvailability.AVAILABLE,
            "test-version", "synthetic backend",
        )
        self.solver = Mock()
        self.solver.highs_options = {}
        self.solver.available.return_value = solver_module.Highs.Availability.FullLicense
        self.solver.version.return_value = ("test-version",)
        self.model = object()
        self.loader = Mock()

    def solve(self, appsi_result=None, error=None):
        self.solver.solve.return_value = appsi_result
        self.solver.solve.side_effect = error
        with patch.object(
            solver_module, "probe_solver_backend",
            side_effect=AssertionError("solve must use its actual solver interface"),
        ), patch.object(
            solver_module, "Highs", return_value=self.solver
        ), patch.object(solver_module, "perf_counter", side_effect=(10.0, 10.25)):
            return solve_pyomo_model(self.model, SolverConfig(backend=SolverBackend.HIGHS))

    def test_execution_failure_retains_exception_without_incumbent(self) -> None:
        result = self.solve(error=RuntimeError("solve failed"))

        self.assertIs(result.status, SolverStatus.ERROR)
        self.assertEqual(result.termination_condition, "execution_error")
        self.assertFalse(result.has_incumbent)
        self.assertIsNone(result.objective_value)
        self.assertIsNone(result.best_bound)
        self.assertIsNone(result.relative_gap)
        self.assertEqual(result.solve_time_seconds, 0.25)
        self.assertEqual(result.raw_status_metadata["exception_type"], "RuntimeError")
        self.assertEqual(result.raw_status_metadata["exception_message"], "solve failed")
        self.assertEqual(result.raw_status_metadata["solver_version"], "test-version")
        self.assertNotIn("pyomo_termination_condition", result.raw_status_metadata)
        with self.assertRaises(NoSolverIncumbentError):
            require_solver_incumbent(result)

    def test_load_failure_preserves_bound_and_rejects_incumbent(self) -> None:
        self.loader.load_vars.side_effect = RuntimeError("load failed")
        appsi_result = SimpleNamespace(
            best_feasible_objective=5.0,
            best_objective_bound=4.0,
            termination_condition=TerminationCondition.maxTimeLimit,
            solution_loader=self.loader,
        )
        result = self.solve(appsi_result)

        self.assertIs(result.status, SolverStatus.ERROR)
        self.assertEqual(result.termination_condition, "incumbent_load_error")
        self.assertFalse(result.has_incumbent)
        self.assertIsNone(result.objective_value)
        self.assertEqual(result.best_bound, 4.0)
        self.assertIsNone(result.relative_gap)
        self.assertEqual(result.solve_time_seconds, 0.25)
        self.assertEqual(result.raw_status_metadata["exception_message"], "load failed")
        self.assertEqual(result.raw_status_metadata["pyomo_termination_condition"], "maxTimeLimit")
        self.loader.load_vars.assert_called_once_with()
        with self.assertRaises(NoSolverIncumbentError):
            require_solver_incumbent(result)

    def test_limit_without_finite_objective_never_loads_variables(self) -> None:
        appsi_result = SimpleNamespace(
            best_feasible_objective=nan,
            best_objective_bound=4.0,
            termination_condition=TerminationCondition.maxTimeLimit,
            solution_loader=self.loader,
        )
        result = self.solve(appsi_result)

        self.assertIs(result.status, SolverStatus.NO_SOLUTION)
        self.assertEqual(result.termination_condition, "maxTimeLimit")
        self.assertFalse(result.has_incumbent)
        self.assertIsNone(result.objective_value)
        self.assertEqual(result.best_bound, 4.0)
        self.assertIsNone(result.relative_gap)
        self.loader.load_vars.assert_not_called()
        self.assertFalse(self.solver.config.load_solution)

    def test_solve_prepares_only_the_interface_it_uses(self) -> None:
        appsi_result = SimpleNamespace(
            best_feasible_objective=None,
            best_objective_bound=None,
            termination_condition=TerminationCondition.maxTimeLimit,
        )
        self.solver.solve.return_value = appsi_result
        with (
            patch.object(solver_module, "Highs", return_value=self.solver) as constructor,
            patch.object(solver_module, "probe_solver_backend",
                         side_effect=AssertionError("explicit probe is not part of solve")) as probe,
        ):
            result = solve_pyomo_model(
                self.model,
                SolverConfig(backend=SolverBackend.HIGHS, time_limit_seconds=1.0),
            )
        constructor.assert_called_once_with()
        self.solver.available.assert_called_once_with()
        self.solver.version.assert_called_once_with()
        self.solver.solve.assert_called_once_with(self.model)
        probe.assert_not_called()
        self.assertEqual(result.raw_status_metadata["solver_version"], "test-version")
        self.assertEqual(result.raw_status_metadata["backend_availability"], "available")
        self.assertEqual(self.solver.highs_options, {"time_limit": 1.0})

    def test_unavailable_or_unlicensed_actual_interface_rejects_before_solve(self) -> None:
        cases = (
            (SolverBackend.HIGHS, "Highs", solver_module.Highs.Availability.NotFound),
            (SolverBackend.GUROBI, "Gurobi", solver_module.Gurobi.Availability.BadLicense),
        )
        for backend, name, availability in cases:
            with self.subTest(backend=backend):
                solver = Mock()
                solver.highs_options = {}
                solver.gurobi_options = {}
                solver.available.return_value = availability
                with patch.object(solver_module, name, return_value=solver):
                    with self.assertRaisesRegex(SolverBackendUnavailableError, availability.name):
                        solve_pyomo_model(self.model, SolverConfig(backend=backend))
                solver.version.assert_not_called()
                solver.solve.assert_not_called()

    def test_backend_setup_exception_is_typed_and_retains_its_cause(self) -> None:
        error = RuntimeError("license environment failed")
        self.solver.available.side_effect = error
        with patch.object(solver_module, "Gurobi", return_value=self.solver):
            with self.assertRaisesRegex(SolverBackendUnavailableError, "license environment failed") as raised:
                solve_pyomo_model(self.model, SolverConfig(backend=SolverBackend.GUROBI))
        self.assertIs(raised.exception.__cause__, error)
        self.solver.solve.assert_not_called()


class ObjectiveComparisonTests(unittest.TestCase):
    """Keep solver/reference agreement explicit and solver-independent."""

    def test_default_absolute_tolerance_and_difference_are_visible(self) -> None:
        comparison = ObjectiveComparison(1.0, 1.0 + 5e-9)

        self.assertEqual(
            comparison.absolute_tolerance,
            DEFAULT_OBJECTIVE_COMPARISON_ABSOLUTE_TOLERANCE,
        )
        self.assertAlmostEqual(comparison.absolute_difference, 5e-9)
        self.assertTrue(comparison.agrees)
        self.assertFalse(ObjectiveComparison(1.0, 1.1).agrees)

    def test_invalid_objective_comparison_inputs_are_rejected(self) -> None:
        for invalid in (nan, inf, -inf, True, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ObjectiveComparison(invalid, 1.0)
        with self.assertRaises(ValueError):
            ObjectiveComparison(1.0, 1.0, absolute_tolerance=-1.0)


class PdmHighsIntegrationTests(unittest.TestCase):
    """Provide the first real, independently validated ``P_DM`` solves."""

    def test_optimal_tiny_solve_passes_full_independent_pipeline(self) -> None:
        (
            result,
            raw,
            state,
            report,
            comparison,
        ) = solve_and_independently_validate_minimal()

        self.assertIs(result.status, SolverStatus.OPTIMAL)
        self.assertEqual(result.termination_condition, "optimal")
        self.assertTrue(result.has_incumbent)
        self.assertAlmostEqual(result.objective_value, 1.00410003)
        self.assertAlmostEqual(result.best_bound, 1.00410003)
        self.assertEqual(result.relative_gap, 0.0)
        self.assertGreaterEqual(result.solve_time_seconds, 0.0)
        self.assertEqual(
            result.raw_status_metadata["interface"],
            "pyomo.contrib.appsi.solvers.Highs",
        )
        with self.assertRaises(TypeError):
            result.raw_status_metadata["interface"] = "changed"
        with self.assertRaises(FrozenInstanceError):
            result.status = SolverStatus.ERROR

        # The solver must choose the cheaper one-GPU TP=1 configuration and
        # full service; these are raw incumbent values, not a fixed assignment.
        self.assertEqual(raw.x[ROUTE], 1.0)
        self.assertEqual(raw.z[ROUTE], 1.0)
        self.assertEqual(raw.q[PAIR], 1.0)
        self.assertEqual(raw.u["q_text"], 0.0)
        self.assertEqual(raw.w[W_11], 1.0)
        self.assertEqual(raw.w[W_21], 0.0)
        self.assertEqual(raw.y[PAIR], 1.0)
        self.assertEqual(raw.v[V_11], 1.0)
        self.assertEqual(raw.v[V_21], 0.0)
        self.assertEqual(state.gpu_counts[PAIR], 1)

        self.assertTrue(report.is_feasible)
        self.assertEqual(report.violations, ())
        self.assertTrue(comparison.agrees)
        self.assertAlmostEqual(comparison.solver_objective, 1.00410003)
        self.assertAlmostEqual(comparison.independent_objective, 1.00410003)
        self.assertEqual(comparison.absolute_difference, 0.0)

    def test_infeasible_tiny_solve_has_no_incumbent_and_guard_stops_extraction(self) -> None:
        # With u<=0.2, at least 0.8 routing is mandatory.  Then x<=z<=q
        # activates a configuration and y>=1, whose rental alone violates
        # the exact zero cost budget.
        instance = replace(make_minimal_tiny_instance(), cost_budget_usd=0.0)
        problem_data = PdmProblemData(instance)
        model = build_pdm_pyomo_model(problem_data)

        result = solve_pyomo_model(model, SolverConfig(backend=SolverBackend.HIGHS))

        self.assertIs(result.status, SolverStatus.INFEASIBLE)
        self.assertEqual(result.termination_condition, "infeasible")
        self.assertFalse(result.has_incumbent)
        self.assertIsNone(result.objective_value)
        self.assertIsNone(result.best_bound)
        self.assertIsNone(result.relative_gap)
        with self.assertRaises(NoSolverIncumbentError):
            require_solver_incumbent(result)
        self.assertTrue(all(variable.value is None for variable in model.x.values()))

    def test_repeated_solve_is_reproducible_in_status_objective_and_state(self) -> None:
        first = solve_and_independently_validate_minimal()
        second = solve_and_independently_validate_minimal()

        first_result, _, first_state, first_report, first_comparison = first
        second_result, _, second_state, second_report, second_comparison = second
        self.assertIs(first_result.status, second_result.status)
        self.assertEqual(first_result.objective_value, second_result.objective_value)
        self.assertEqual(first_state, second_state)
        self.assertTrue(first_report.is_feasible)
        self.assertTrue(second_report.is_feasible)
        self.assertTrue(first_comparison.agrees)
        self.assertTrue(second_comparison.agrees)


if __name__ == "__main__":
    unittest.main()
