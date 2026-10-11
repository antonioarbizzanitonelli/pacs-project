"""Independent tiny-oracle contracts, hand-derived cases, and HiGHS cross-checks."""

import unittest
from dataclasses import FrozenInstanceError, replace
from math import inf, nan

from llm_resource_allocation.data.tiny_instances import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
    make_parallelism_tradeoff_tiny_instance,
)
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.optimization.solver import (
    SolverBackend,
    SolverConfig,
    SolverStatus,
    require_solver_incumbent,
    solve_pyomo_model,
)
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.pyomo_pdm import build_pdm_pyomo_model
from llm_resource_allocation.optimization.extraction import (
    extract_raw_pdm_solution_from_pyomo,
    normalize_raw_pdm_solution,
)
from llm_resource_allocation.validation.objective_comparison import ObjectiveComparison
from llm_resource_allocation.validation.placement import validate_placement
from llm_resource_allocation.verification import (
    TinyEnumerationConfig,
    TinyEnumerationLimitError,
    UnsupportedTinyOracleInstanceError,
    enumerate_tiny_pdm,
    estimate_tiny_discrete_search_space,
)


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ("m_small", "gpu_basic_fp16")
W_11 = (*PAIR, 1, 1)
W_21 = (*PAIR, 2, 1)


def solve_with_highs(instance):
    """Run the explicit solver pipeline used only for independent comparison."""

    problem_data = PdmProblemData(instance)
    model = build_pdm_pyomo_model(problem_data)
    solver_result = solve_pyomo_model(model, SolverConfig(backend=SolverBackend.HIGHS))
    if solver_result.has_incumbent:
        require_solver_incumbent(solver_result)
        raw = extract_raw_pdm_solution_from_pyomo(model, problem_data)
        state = normalize_raw_pdm_solution(raw, problem_data)
        report = validate_placement(instance, state, PaperVersion.V2)
    else:
        state = None
        report = None
    return solver_result, state, report


class TinyEnumerationContractTests(unittest.TestCase):
    """Protect scope, configuration, search accounting, and isolation."""


    def test_config_is_validated_and_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            TinyEnumerationConfig().max_discrete_combinations = 1
        for invalid in (0, -1, True, 1.5, "16"):
            with self.subTest(maximum=invalid):
                with self.assertRaises(ValueError):
                    TinyEnumerationConfig(max_discrete_combinations=invalid)
        for invalid in (-1.0, inf, -inf, nan, True, "1e-8"):
            with self.subTest(tolerance=invalid):
                with self.assertRaises(ValueError):
                    TinyEnumerationConfig(objective_tolerance=invalid)

    def test_search_formula_and_pre_enumeration_limit_are_explicit(self) -> None:
        problem_data = PdmProblemData(make_minimal_tiny_instance())

        self.assertEqual(estimate_tiny_discrete_search_space(problem_data), 16)
        with self.assertRaises(TinyEnumerationLimitError) as caught:
            enumerate_tiny_pdm(
                problem_data,
                config=TinyEnumerationConfig(max_discrete_combinations=15),
            )
        self.assertEqual(caught.exception.search_space_size, 16)
        self.assertEqual(caught.exception.maximum, 15)

    def test_unsupported_shape_fails_with_structured_error(self) -> None:
        problem_data = PdmProblemData(make_heterogeneous_tiny_instance())

        with self.assertRaises(UnsupportedTinyOracleInstanceError):
            enumerate_tiny_pdm(
                problem_data,
                config=TinyEnumerationConfig(
                    max_discrete_combinations=20_000_000
                ),
            )


class PrimaryTinyOracleTests(unittest.TestCase):
    """Establish the hand-worked, oracle, and HiGHS three-way agreement."""

    def test_primary_oracle_matches_hand_solution_and_independent_validator(self) -> None:
        instance = make_minimal_tiny_instance()
        result = enumerate_tiny_pdm(PdmProblemData(instance))

        # Hand derivation, independent of HiGHS:
        # TP1 objective = 6.001 - 4.99689997*x on [0.8, 1], hence
        # x=1,u=0 and 1.00410003. TP2 reaches 2.00260003 at x=1.
        self.assertEqual(result.search_space_size, 16)
        self.assertEqual(result.candidates_examined, 5)
        self.assertEqual(result.feasible_candidates, 2)
        self.assertEqual(len(result.diagnostics), 5)
        self.assertAlmostEqual(result.best_objective, 1.00410003)
        self.assertEqual(result.tie_count, 1)
        self.assertTrue(result.is_unique_optimum)
        self.assertEqual(len(result.optimal_states), 1)

        state = result.best_state
        self.assertIs(state, result.optimal_states[0])
        self.assertIsNotNone(state)
        self.assertEqual(state.routing_fractions[ROUTE], 1.0)
        self.assertEqual(state.unmet_fractions["q_text"], 0.0)
        self.assertEqual(state.admission_indicators[ROUTE], 1)
        self.assertEqual(state.deployment_indicators[PAIR], 1)
        self.assertEqual(state.configuration_selectors[W_11], 1)
        self.assertEqual(state.configuration_selectors[W_21], 0)
        self.assertEqual(state.gpu_counts[PAIR], 1)
        report = validate_placement(instance, state, PaperVersion.V2)
        self.assertTrue(report.is_feasible)
        self.assertEqual(report.violations, ())
        self.assertEqual(report.objective_breakdown.total, result.best_objective)

    def test_primary_oracle_and_highs_have_identical_unique_state(self) -> None:
        instance = make_minimal_tiny_instance()
        oracle = enumerate_tiny_pdm(PdmProblemData(instance))
        solver_result, solver_state, solver_report = solve_with_highs(instance)

        self.assertIs(solver_result.status, SolverStatus.OPTIMAL)
        self.assertTrue(solver_report.is_feasible)
        self.assertTrue(
            ObjectiveComparison(
                solver_result.objective_value,
                oracle.best_objective,
            ).agrees
        )
        self.assertTrue(oracle.is_unique_optimum)
        self.assertEqual(solver_state, oracle.best_state)


class TradeoffAndInfeasibilityTests(unittest.TestCase):
    """Exercise a non-first configuration and an incumbent-free problem."""

    def test_second_case_requires_tp2_and_matches_highs(self) -> None:
        instance = make_parallelism_tradeoff_tiny_instance()
        oracle = enumerate_tiny_pdm(PdmProblemData(instance))
        solver_result, solver_state, solver_report = solve_with_highs(instance)

        self.assertEqual(oracle.search_space_size, 16)
        self.assertEqual(oracle.candidates_examined, 5)
        self.assertEqual(oracle.feasible_candidates, 1)
        self.assertAlmostEqual(oracle.best_objective, 2.00260003)
        self.assertTrue(oracle.is_unique_optimum)
        self.assertEqual(oracle.best_state.configuration_selectors[W_11], 0)
        self.assertEqual(oracle.best_state.configuration_selectors[W_21], 1)
        self.assertEqual(oracle.best_state.gpu_counts[PAIR], 2)

        self.assertIs(solver_result.status, SolverStatus.OPTIMAL)
        self.assertTrue(solver_report.is_feasible)
        self.assertTrue(
            ObjectiveComparison(
                solver_result.objective_value,
                oracle.best_objective,
            ).agrees
        )
        self.assertEqual(solver_state, oracle.best_state)

    def test_zero_budget_is_infeasible_for_oracle_and_highs(self) -> None:
        instance = replace(make_minimal_tiny_instance(), cost_budget_usd=0.0)
        oracle = enumerate_tiny_pdm(PdmProblemData(instance))
        solver_result, solver_state, solver_report = solve_with_highs(instance)

        self.assertEqual(oracle.feasible_candidates, 0)
        self.assertIsNone(oracle.best_state)
        self.assertIsNone(oracle.best_objective)
        self.assertEqual(oracle.tie_count, 0)
        self.assertFalse(oracle.is_unique_optimum)
        self.assertIs(solver_result.status, SolverStatus.INFEASIBLE)
        self.assertFalse(solver_result.has_incumbent)
        self.assertIsNone(solver_state)
        self.assertIsNone(solver_report)

    def test_repeated_oracle_runs_are_exactly_deterministic(self) -> None:
        problem_data = PdmProblemData(
            make_parallelism_tradeoff_tiny_instance()
        )

        first = enumerate_tiny_pdm(problem_data)
        second = enumerate_tiny_pdm(problem_data)

        self.assertEqual(first, second)

    def test_equal_objectives_retain_ties_in_stable_4a_order(self) -> None:
        base = make_minimal_tiny_instance()
        query = replace(
            base.queries[0],
            delay_penalty_usd_per_millisecond_query=0.0,
            unmet_penalty_usd_per_hour=0.0,
        )
        tier = replace(base.gpu_tiers[0], hourly_rental_usd=0.0)
        instance = replace(
            base,
            queries=(query,),
            gpu_tiers=(tier,),
            storage_price_usd_per_gb_hour=0.0,
        )

        result = enumerate_tiny_pdm(PdmProblemData(instance))

        # Two configurations times two tied interval endpoints are all optimal.
        self.assertEqual(result.best_objective, 0.0)
        self.assertEqual(result.feasible_candidates, 2)
        self.assertEqual(result.tie_count, 4)
        self.assertEqual(len(result.optimal_states), 4)
        self.assertFalse(result.is_unique_optimum)
        self.assertEqual(result.best_state.configuration_selectors[W_11], 1)
        self.assertEqual(result.best_state.routing_fractions[ROUTE], 0.8)

    def test_tp_pp_candidates_and_endpoint_ties_follow_complete_lattice_order(self) -> None:
        base = make_minimal_tiny_instance()
        query = replace(
            base.queries[0],
            delay_penalty_usd_per_millisecond_query=0.0,
            unmet_penalty_usd_per_hour=0.0,
        )
        tier = replace(
            base.gpu_tiers[0],
            hourly_rental_usd=0.0,
            pipeline_parallel_depths=(1, 2),
        )
        instance = replace(
            base,
            queries=(query,),
            gpu_tiers=(tier,),
            storage_price_usd_per_gb_hour=0.0,
        )
        result = enumerate_tiny_pdm(
            PdmProblemData(instance),
            config=TinyEnumerationConfig(max_discrete_combinations=64),
        )

        # Four w bits plus q,z give 64 raw states. Exact pruning leaves
        # one undeployed state and four configurations for each admission bit.
        w_12 = (*PAIR, 1, 2)
        w_22 = (*PAIR, 2, 2)
        configurations = (W_11, w_12, W_21, w_22)
        expected_candidates = (
            (0, 0, 0, (0, 0, 0, 0), None),
            (1, 1, 0, (1, 0, 0, 0), W_11),
            (2, 1, 0, (0, 1, 0, 0), w_12),
            (3, 1, 0, (0, 0, 1, 0), W_21),
            (4, 1, 0, (0, 0, 0, 1), w_22),
            (5, 1, 1, (1, 0, 0, 0), W_11),
            (6, 1, 1, (0, 1, 0, 0), w_12),
            (7, 1, 1, (0, 0, 1, 0), W_21),
            (8, 1, 1, (0, 0, 0, 1), w_22),
        )
        observed_candidates = tuple(
            (
                diagnostic.ordinal,
                diagnostic.deployment_indicator,
                diagnostic.admission_indicator,
                diagnostic.configuration_selectors,
                diagnostic.selected_configuration,
            )
            for diagnostic in result.diagnostics
        )
        self.assertEqual(observed_candidates, expected_candidates)
        self.assertEqual(result.search_space_size, 64)
        self.assertEqual(result.candidates_examined, 9)
        self.assertEqual(result.feasible_candidates, 4)
        self.assertEqual(result.tie_count, 8)
        self.assertEqual(result.best_objective, 0.0)
        self.assertFalse(result.is_unique_optimum)
        self.assertEqual(
            tuple(diagnostic.is_feasible for diagnostic in result.diagnostics),
            (False, False, False, False, False, True, True, True, True),
        )

        # Every feasible interval is [0.8,1] with zero objective throughout.
        expected_optima = tuple(
            (configuration, endpoint, gpu_count)
            for configuration, gpu_count in zip(
                configurations, (1, 2, 2, 4), strict=True
            )
            for endpoint in (0.8, 1.0)
        )
        observed_optima = tuple(
            (
                next(
                    key
                    for key, selected in state.configuration_selectors.items()
                    if selected
                ),
                state.routing_fractions[ROUTE],
                state.gpu_counts[PAIR],
            )
            for state in result.optimal_states
        )
        self.assertEqual(observed_optima, expected_optima)
        self.assertEqual(result.best_state, result.optimal_states[0])
        for diagnostic in result.diagnostics[5:]:
            self.assertEqual(diagnostic.routing_lower_bound, 0.8)
            self.assertEqual(diagnostic.routing_upper_bound, 1.0)
            self.assertTrue(diagnostic.endpoint_tie)
        for state in result.optimal_states:
            self.assertTrue(
                validate_placement(instance, state, PaperVersion.V2).is_feasible
            )

    def test_single_point_interval_has_one_unique_endpoint(self) -> None:
        base = make_minimal_tiny_instance()
        instance = replace(
            base,
            queries=(replace(base.queries[0], max_unmet_fraction=0.0),),
            gpu_tiers=(
                replace(base.gpu_tiers[0], tensor_parallel_degrees=(1,)),
            ),
        )
        result = enumerate_tiny_pdm(PdmProblemData(instance))

        # u=0 and x+u=1 force x=1; the sole TP1 objective is 1.00410003.
        self.assertEqual(result.search_space_size, 8)
        self.assertEqual(result.candidates_examined, 3)
        self.assertEqual(result.feasible_candidates, 1)
        self.assertEqual(result.tie_count, 1)
        self.assertTrue(result.is_unique_optimum)
        self.assertAlmostEqual(result.best_objective, 1.00410003)
        self.assertEqual(result.optimal_states, (result.best_state,))
        diagnostic = result.diagnostics[-1]
        self.assertEqual(diagnostic.routing_lower_bound, 1.0)
        self.assertEqual(diagnostic.routing_upper_bound, 1.0)
        self.assertEqual(diagnostic.selected_routing_fraction, 1.0)
        self.assertFalse(diagnostic.endpoint_tie)
        self.assertTrue(
            validate_placement(instance, result.best_state, PaperVersion.V2).is_feasible
        )

    def test_tolerance_retains_stable_representative_above_strict_minimum(self) -> None:
        base = make_minimal_tiny_instance()
        query = replace(
            base.queries[0],
            delay_penalty_usd_per_millisecond_query=0.0,
            unmet_penalty_usd_per_hour=2e-8,
        )
        instance = replace(
            base,
            queries=(query,),
            gpu_tiers=(replace(base.gpu_tiers[0], hourly_rental_usd=0.0),),
            storage_price_usd_per_gb_hour=0.0,
        )
        problem_data = PdmProblemData(instance)
        tolerant = enumerate_tiny_pdm(
            problem_data, config=TinyEnumerationConfig(objective_tolerance=1e-8)
        )
        exact = enumerate_tiny_pdm(
            problem_data, config=TinyEnumerationConfig(objective_tolerance=0.0)
        )

        # Only C5 remains: 2e-8*(1-x). The lower endpoint costs 4e-9;
        # x=1 costs zero, but both endpoints tie under the chosen tolerance.
        self.assertAlmostEqual(tolerant.best_objective, 4e-9, delta=1e-20)
        self.assertEqual(tolerant.best_state.routing_fractions[ROUTE], 0.8)
        self.assertEqual(tolerant.best_state.configuration_selectors[W_11], 1)
        self.assertEqual(tolerant.tie_count, 4)
        self.assertEqual(
            tuple(
                state.routing_fractions[ROUTE] for state in tolerant.optimal_states
            ),
            (0.8, 1.0, 0.8, 1.0),
        )
        self.assertEqual(exact.best_objective, 0.0)
        self.assertEqual(exact.best_state.routing_fractions[ROUTE], 1.0)
        self.assertEqual(exact.tie_count, 2)
        self.assertGreater(tolerant.best_objective, exact.best_objective)
        for state in tolerant.optimal_states:
            report = validate_placement(instance, state, PaperVersion.V2)
            self.assertTrue(report.is_feasible)
        best_report = validate_placement(
            instance, tolerant.best_state, PaperVersion.V2
        )
        self.assertEqual(
            best_report.objective_breakdown.total, tolerant.best_objective
        )


if __name__ == "__main__":
    unittest.main()
