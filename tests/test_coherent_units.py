"""Hand-derived dimensional checks for the explicit coherent-unit convention."""

from dataclasses import replace
import unittest

from pyomo.environ import value

from llm_resource_allocation.algorithms.agh import run_agh
from llm_resource_allocation.algorithms.gh import run_gh
from llm_resource_allocation.algorithms.gh_phase1 import GHPhase1Config
from llm_resource_allocation.algorithms.m1 import select_minimum_feasible_parallelism
from llm_resource_allocation.algorithms.m2 import M2EvaluationInputs, compute_m2_marginal_cost
from llm_resource_allocation.data.tiny_instances import make_coherent_tiny_instance, make_minimal_tiny_instance
from llm_resource_allocation.domain import InstanceValidationError, UnitConvention, validate_problem_instance
from llm_resource_allocation.math.gpu_memory import compute_gpu_memory_residuals
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.storage_capacity import compute_storage_capacity_residuals
from llm_resource_allocation.optimization.pdm_run import execute_and_record_pdm_run
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.pyomo_pdm import build_pdm_pyomo_model
from llm_resource_allocation.optimization.solver import SolverBackend, SolverConfig
from llm_resource_allocation.validation.placement import validate_placement
from llm_resource_allocation.verification.tiny_enumeration import enumerate_tiny_pdm

from placement_support import PAIR, ROUTE, make_minimal_placement_state


def dimensional_instance():
    base = make_coherent_tiny_instance()
    query = replace(base.queries[0], arrival_rate_per_hour=6.0,
                    input_tokens=60.0, output_tokens=40.0,
                    storage_gb_per_token=0.002, max_unmet_fraction=0.5,
                    delay_penalty_usd_per_millisecond_query=0.02)
    return replace(base, queries=(query,),
                   models=(replace(base.models[0], weight_gb=4.0, kv_cache_gb_per_token=0.001),),
                   compute_delay_seconds_per_token={ROUTE: 0.001},
                   communication_delay_seconds_per_token={ROUTE: 0.0},
                   kv_cache_residency_seconds={ROUTE: 120.0},
                   horizon_hours=3.0, data_residency_hours=2.0,
                   storage_price_usd_per_gb_hour=0.25)


class CoherentDimensionalTests(unittest.TestCase):
    def test_gb_properties_do_not_relabel_historical_mixed_units(self):
        instance = make_minimal_tiny_instance()
        state = make_minimal_placement_state()
        memory = compute_gpu_memory_residuals(instance, state)[0]
        storage = compute_storage_capacity_residuals(instance, state)[0]
        self.assertGreater(memory.printed_memory_lhs, 0.0)
        self.assertGreater(storage.printed_storage_lhs, 0.0)
        with self.assertRaisesRegex(ValueError, "literal"):
            _ = memory.kv_memory_usage_gb
        with self.assertRaisesRegex(ValueError, "literal"):
            _ = storage.data_storage_usage_gb

    def test_hand_derived_stock_costs_and_policy_evidence(self):
        instance = dimensional_instance()
        report = validate_placement(instance, make_minimal_placement_state(x=0.5, u=0.5), PaperVersion.V2)
        self.assertTrue(report.is_feasible)
        self.assertIs(report.unit_convention, UnitConvention.COHERENT)
        self.assertEqual(report.data_residency_hours, 2.0)
        # 0.001 GB/token * 100 tokens * 6 requests/h * 120 s /3600 *0.5.
        self.assertAlmostEqual(report.gpu_memory_diagnostics[0].kv_memory_printed_term, 0.01)
        # 4 GB weights + (0.002*100*6*0.5) GB/h *2 h.
        storage = report.storage_capacity_diagnostics[0]
        self.assertAlmostEqual(storage.data_storage_usage_gb, 1.2)
        self.assertAlmostEqual(storage.storage_usage_gb, 5.2)
        costs = report.objective_breakdown
        for actual, expected in zip(
            (costs.gpu_rental_cost, costs.model_storage_cost, costs.data_storage_cost,
             costs.delay_penalty, costs.unmet_demand_penalty, costs.total),
            (3.0, 3.0, 0.9, 18.0, 7.5, 32.4), strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(report.budget_diagnostic.total_budget_use, 6.9)

    def test_arrival_and_residency_scaling_with_zero_arrivals(self):
        instance = dimensional_instance()
        state = make_minimal_placement_state(x=0.5, u=0.5)
        for arrival, expected in ((0.0, 0.0), (6.0, 0.01), (12.0, 0.02)):
            changed = replace(instance, queries=(replace(instance.queries[0], arrival_rate_per_hour=arrival),))
            self.assertAlmostEqual(compute_gpu_memory_residuals(changed, state)[0].kv_memory_printed_term, expected)
        twice = replace(instance, kv_cache_residency_seconds={ROUTE: 240.0})
        self.assertAlmostEqual(compute_gpu_memory_residuals(twice, state)[0].kv_memory_printed_term, 0.02)
        twice = replace(instance, data_residency_hours=4.0)
        self.assertAlmostEqual(compute_storage_capacity_residuals(twice, state)[0].data_storage_usage_gb, 2.4)

    def test_pyomo_independent_transcription_matches_hand_values(self):
        instance = dimensional_instance()
        state = make_minimal_placement_state(x=0.5, u=0.5)
        model = build_pdm_pyomo_model(PdmProblemData(instance))
        for family, values in (
            ("x", state.routing_fractions), ("z", state.admission_indicators),
            ("q", state.deployment_indicators), ("u", state.unmet_fractions),
            ("w", state.configuration_selectors), ("y", state.gpu_counts),
            ("v", state.auxiliary_products),
        ):
            component = getattr(model, family)
            for key in component:
                component[key].set_value(values.get(key, 0.0))
        self.assertAlmostEqual(value(model.kv_memory_printed_term[PAIR]), 0.01)
        self.assertAlmostEqual(value(model.printed_storage_lhs["q_text"]), 5.2)
        self.assertAlmostEqual(value(model.cost_data_storage), 0.9)
        self.assertAlmostEqual(value(model.delay_penalty), 18.0)
        self.assertAlmostEqual(value(model.total_objective), 32.4)

    def test_m2_cost_includes_resident_data_and_period_delay(self):
        instance = dimensional_instance()
        selection = select_minimum_feasible_parallelism(instance, *ROUTE)
        costs = compute_m2_marginal_cost(instance, selection, M2EvaluationInputs(1.0, 0.0, 0.0, 0))
        # TP=1: GPU 3; storage 3*.25*(4+.002*100*6*2)=4.8;
        # delay .02 USD/ms/request *100 ms/request *18 requests=36.
        self.assertEqual(selection.selected_configuration.tensor_parallel_degree, 1)
        self.assertAlmostEqual(costs.incremental_gpu_cost, 3.0)
        self.assertAlmostEqual(costs.model_and_data_storage_cost, 4.8)
        self.assertAlmostEqual(costs.delay_penalty_cost, 36.0)

    def test_coherent_v1_is_explicitly_unsupported(self):
        instance = dimensional_instance()
        with self.assertRaisesRegex(ValueError, "[Cc]oherent"):
            build_pdm_pyomo_model(PdmProblemData(instance), PaperVersion.V1)
        with self.assertRaisesRegex(ValueError, "coherent"):
            validate_placement(instance, make_minimal_placement_state(), PaperVersion.V1)

    def test_invalid_unit_and_residency_inputs_fail_at_domain_boundary(self):
        for changes in ({"data_residency_hours": -1}, {"data_residency_hours": True},
                        {"data_residency_hours": float("nan")}, {"unit_convention": "coherent"},
                        {"seconds_per_hour": 1.0}):
            with self.subTest(changes=changes), self.assertRaises(InstanceValidationError):
                validate_problem_instance(replace(dimensional_instance(), **changes))

    def test_tiny_optimum_is_hand_derived_and_oracle_validated(self):
        instance = make_coherent_tiny_instance()
        problem = PdmProblemData(instance)
        oracle = enumerate_tiny_pdm(problem)
        # TP=2, x=.8: rental2 + model .001 + data .000000024
        # + (.01*1000*.16*10*.8)=12.8 + unmet1.
        expected = 15.801000024
        self.assertAlmostEqual(oracle.best_objective, expected, places=9)
        record = execute_and_record_pdm_run(problem, SolverConfig(backend=SolverBackend.HIGHS))
        self.assertTrue(record.accepted_run)
        self.assertAlmostEqual(record.scientific_evidence.objective_breakdown.total, expected, places=9)
        self.assertAlmostEqual(record.solution.routing_fractions[ROUTE], 0.8)

    def test_both_heuristics_return_fresh_validated_coherent_placements(self):
        instance = make_coherent_tiny_instance()
        config = GHPhase1Config(budget_fraction=0.8)
        gh = run_gh(instance, config)
        agh = run_agh(instance, config, random_seed=37)
        self.assertTrue(gh.algorithm_completed)
        self.assertTrue(gh.independently_feasible)
        self.assertTrue(agh.completed)
        self.assertIs(gh.validation_report.unit_convention, UnitConvention.COHERENT)
        self.assertIs(agh.final_validation.unit_convention, UnitConvention.COHERENT)


if __name__ == "__main__":
    unittest.main()
