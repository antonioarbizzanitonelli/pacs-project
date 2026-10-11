"""Optional Gurobi execution, incumbent safety, and independent cross-checks."""

import unittest
from dataclasses import replace
from math import nan
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pyomo.contrib.appsi.base import TerminationCondition

from llm_resource_allocation.data import (
    make_heterogeneous_solve_tiny_instance,
    make_minimal_tiny_instance,
    make_parallelism_tradeoff_tiny_instance,
)
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.optimization import solver as solver_module
from llm_resource_allocation.optimization.pdm_run import execute_and_record_pdm_run
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.pyomo_pdm import build_pdm_pyomo_model
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
)
from llm_resource_allocation.validation.objective_comparison import ObjectiveComparison
from llm_resource_allocation.verification import enumerate_tiny_pdm


class GurobiConfigurationTests(unittest.TestCase):
    """Keep native options and optional dependency/license failures explicit."""

    def test_defaults_and_requested_options(self) -> None:
        self.assertEqual(
            dict(translate_gurobi_options(SolverConfig())),
            {},
        )
        options = translate_gurobi_options(SolverConfig(
            backend=SolverBackend.GUROBI,
            time_limit_seconds=12.5,
            relative_mip_gap=0.02,
            absolute_mip_gap=0.25,
        ))
        self.assertEqual(
            dict(options), {"TimeLimit": 12.5, "MIPGap": 0.02, "MIPGapAbs": 0.25}
        )
        with self.assertRaises(TypeError):
            options["TimeLimit"] = 1.0
        zero_options = translate_gurobi_options(SolverConfig(
            backend=SolverBackend.GUROBI, time_limit_seconds=0.0,
            relative_mip_gap=0.0, absolute_mip_gap=0.0,
        ))
        self.assertEqual(dict(zero_options), {
            "TimeLimit": 0.0, "MIPGap": 0.0, "MIPGapAbs": 0.0,
        })

    def test_translation_rejects_wrong_type_and_backend(self) -> None:
        with self.assertRaises(TypeError):
            translate_gurobi_options(None)
        with self.assertRaises(UnsupportedSolverBackendError):
            translate_gurobi_options(SolverConfig(backend=SolverBackend.HIGHS))

    def test_missing_package_is_unavailable_without_importing_it(self) -> None:
        with patch.object(solver_module, "find_spec", return_value=None), patch.object(
            solver_module, "import_module"
        ) as importer:
            info = probe_solver_backend(SolverBackend.GUROBI)
        self.assertIs(info.availability, SolverBackendAvailability.UNAVAILABLE)
        importer.assert_not_called()

    def test_license_failure_is_distinct_from_absence_and_environment_is_disposed(self) -> None:
        gurobipy = Mock()
        gurobipy.Env.return_value.start.side_effect = RuntimeError("license rejected")
        with patch.object(solver_module, "find_spec", return_value=object()), patch.object(
            solver_module, "import_module", return_value=gurobipy
        ):
            info = probe_solver_backend(SolverBackend.GUROBI)
        self.assertIs(info.availability, SolverBackendAvailability.INSTALLED_BUT_UNUSABLE)
        self.assertIn("license rejected", info.detail)
        gurobipy.Env.return_value.dispose.assert_called_once_with()

    def test_unavailable_backend_stops_before_execution(self) -> None:
        for availability in (
            solver_module.Gurobi.Availability.NotFound,
            solver_module.Gurobi.Availability.BadLicense,
        ):
            with self.subTest(availability=availability):
                solver = Mock()
                solver.gurobi_options = {}
                solver.available.return_value = availability
                with patch.object(solver_module, "Gurobi", return_value=solver) as factory:
                    with self.assertRaises(SolverBackendUnavailableError):
                        solve_pyomo_model(object(), SolverConfig(backend=SolverBackend.GUROBI))
                    factory.assert_called_once_with()
                    solver.solve.assert_not_called()
                    solver.version.assert_not_called()


class GurobiTransportTests(unittest.TestCase):
    """Exercise Gurobi dispatch without relying on solver timings or a license."""

    def setUp(self) -> None:
        self.config = SolverConfig(
            backend=SolverBackend.GUROBI, tee=True, time_limit_seconds=5.0,
            relative_mip_gap=0.0, absolute_mip_gap=0.0,
        )
        self.solver = Mock()
        self.solver.gurobi_options = {}
        self.solver.available.return_value = solver_module.Gurobi.Availability.FullLicense
        self.solver.version.return_value = ("test-version",)
        self.loader = Mock()
        self.info = SolverBackendInfo(
            SolverBackend.GUROBI, SolverBackendAvailability.AVAILABLE,
            "test-version", "synthetic backend",
        )

    def solve(self, objective=None, termination=TerminationCondition.maxTimeLimit,
              execution_error=None):
        self.solver.solve.return_value = SimpleNamespace(
            best_feasible_objective=objective, best_objective_bound=4.0,
            termination_condition=termination, solution_loader=self.loader,
        )
        self.solver.solve.side_effect = execution_error
        with patch.object(solver_module, "probe_solver_backend",
                          side_effect=AssertionError("solve must use its actual interface")), patch.object(
            solver_module, "Gurobi", return_value=self.solver
        ), patch.object(solver_module, "Highs") as highs, patch.object(
            solver_module, "perf_counter", side_effect=(10.0, 10.25)
        ):
            result = solve_pyomo_model(object(), self.config)
            highs.assert_not_called()
        self.assertFalse(self.solver.config.load_solution)
        self.assertTrue(self.solver.config.stream_solver)
        self.assertEqual(self.solver.gurobi_options, {
            "TimeLimit": 5.0, "MIPGap": 0.0, "MIPGapAbs": 0.0,
        })
        self.assertIs(result.backend, SolverBackend.GUROBI)
        self.assertEqual(result.raw_status_metadata["interface"], "pyomo.contrib.appsi.solvers.Gurobi")
        self.assertEqual(result.raw_status_metadata["solver_version"], "test-version")
        self.assertEqual(result.solve_time_seconds, 0.25)
        return result

    def test_limit_with_incumbent_is_feasible_and_loads_values(self) -> None:
        result = self.solve(objective=5.0)
        self.assertIs(result.status, SolverStatus.FEASIBLE)
        self.assertTrue(result.has_incumbent)
        self.assertEqual(result.relative_gap, 0.2)
        self.loader.load_vars.assert_called_once_with()

    def test_limit_without_finite_objective_cannot_load_values(self) -> None:
        result = self.solve(objective=nan)
        self.assertIs(result.status, SolverStatus.NO_SOLUTION)
        self.assertFalse(result.has_incumbent)
        self.loader.load_vars.assert_not_called()
        with self.assertRaises(NoSolverIncumbentError):
            require_solver_incumbent(result)

    def test_optimal_without_incumbent_is_an_error(self) -> None:
        result = self.solve(termination=TerminationCondition.optimal)
        self.assertIs(result.status, SolverStatus.ERROR)
        self.assertFalse(result.has_incumbent)
        self.loader.load_vars.assert_not_called()

    def test_execution_failure_has_no_incumbent(self) -> None:
        result = self.solve(execution_error=RuntimeError("solve failed"))
        self.assertIs(result.status, SolverStatus.ERROR)
        self.assertEqual(result.termination_condition, "execution_error")
        self.assertFalse(result.has_incumbent)
        self.loader.load_vars.assert_not_called()

    def test_load_failure_retains_bound_and_discards_incumbent(self) -> None:
        self.loader.load_vars.side_effect = RuntimeError("load failed")
        result = self.solve(objective=5.0)
        self.assertIs(result.status, SolverStatus.ERROR)
        self.assertEqual(result.termination_condition, "incumbent_load_error")
        self.assertFalse(result.has_incumbent)
        self.assertEqual(result.best_bound, 4.0)
        self.assertIsNone(result.objective_value)
        self.assertEqual(result.raw_status_metadata["exception_message"], "load failed")


class GurobiIntegrationTests(unittest.TestCase):
    """Cross-solve synthetic cases; skip only real solves when Gurobi is unusable."""

    @classmethod
    def setUpClass(cls) -> None:
        info = probe_solver_backend(SolverBackend.GUROBI)
        if info.availability is not SolverBackendAvailability.AVAILABLE:
            raise unittest.SkipTest(info.detail)
        cls.config = SolverConfig(
            backend=SolverBackend.GUROBI, time_limit_seconds=5.0,
            relative_mip_gap=0.0, absolute_mip_gap=0.0,
        )

    def test_tiny_optima_match_independent_oracle_and_highs_for_both_versions(self) -> None:
        for factory in (make_minimal_tiny_instance, make_parallelism_tradeoff_tiny_instance):
            for version in PaperVersion:
                with self.subTest(factory=factory.__name__, version=version):
                    data = PdmProblemData(factory())
                    oracle = enumerate_tiny_pdm(data, paper_version=version)
                    gurobi = execute_and_record_pdm_run(data, self.config, version)
                    highs = execute_and_record_pdm_run(
                        data, SolverConfig(backend=SolverBackend.HIGHS), version
                    )
                    self.assertTrue(gurobi.accepted_run)
                    self.assertTrue(highs.accepted_run)
                    self.assertIs(gurobi.solver_evidence.result.status, SolverStatus.OPTIMAL)
                    self.assertTrue(ObjectiveComparison(
                        gurobi.solver_evidence.result.objective_value, oracle.best_objective
                    ).agrees)
                    self.assertEqual(gurobi.solution, oracle.best_state)
                    self.assertEqual(gurobi.solution, highs.solution)

    def test_zero_budget_has_no_incumbent_and_does_not_initialize_variables(self) -> None:
        data = PdmProblemData(replace(make_minimal_tiny_instance(), cost_budget_usd=0.0))
        for version in PaperVersion:
            with self.subTest(version=version):
                model = build_pdm_pyomo_model(data, version)
                result = solve_pyomo_model(model, self.config)
                self.assertIs(result.status, SolverStatus.INFEASIBLE)
                self.assertFalse(result.has_incumbent)
                with self.assertRaises(NoSolverIncumbentError):
                    require_solver_incumbent(result)
                self.assertTrue(all(variable.value is None for variable in model.x.values()))

    def test_heterogeneous_objectives_agree_after_independent_validation(self) -> None:
        instance = make_heterogeneous_solve_tiny_instance()
        reordered = replace(
            instance, queries=tuple(reversed(instance.queries)),
            models=tuple(reversed(instance.models)),
            gpu_tiers=tuple(reversed(instance.gpu_tiers)),
        )
        for version in PaperVersion:
            results = []
            for candidate in (instance, reordered):
                with self.subTest(version=version, order=candidate.query_ids):
                    data = PdmProblemData(candidate)
                    gurobi = execute_and_record_pdm_run(data, self.config, version)
                    highs = execute_and_record_pdm_run(
                        data, SolverConfig(backend=SolverBackend.HIGHS), version
                    )
                    self.assertTrue(gurobi.accepted_run)
                    self.assertTrue(highs.accepted_run)
                    self.assertTrue(ObjectiveComparison(
                        gurobi.solver_evidence.result.objective_value,
                        highs.solver_evidence.result.objective_value,
                    ).agrees)
                    results.append(gurobi.solver_evidence.result.objective_value)
            self.assertTrue(ObjectiveComparison(*results).agrees)

    def test_repeated_minimal_solve_preserves_unique_decisions(self) -> None:
        data = PdmProblemData(make_minimal_tiny_instance())
        first = execute_and_record_pdm_run(data, self.config)
        second = execute_and_record_pdm_run(data, self.config)
        self.assertTrue(first.accepted_run)
        self.assertTrue(second.accepted_run)
        self.assertEqual(first.solution, second.solution)
        self.assertEqual(first.solver_evidence.result.objective_value,
                         second.solver_evidence.result.objective_value)



if __name__ == "__main__":
    unittest.main()
