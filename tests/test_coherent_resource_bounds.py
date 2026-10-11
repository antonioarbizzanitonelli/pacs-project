"""Hand-derived coherent resource limits checked by oracle and independent solvers."""

from dataclasses import replace
import unittest

from llm_resource_allocation.data.tiny_instances import make_coherent_tiny_instance
from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.optimization.pdm_run import execute_and_record_pdm_run
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.solver import (
    SolverBackend,
    SolverBackendAvailability,
    SolverConfig,
    SolverStatus,
    probe_solver_backend,
)
from llm_resource_allocation.validation.placement import ConstraintFamily, validate_placement
from llm_resource_allocation.verification.tiny_enumeration import enumerate_tiny_pdm

from placement_support import ROUTE, make_minimal_placement_state


def _storage_cost_instance() -> ProblemInstance:
    """Return one TP1 case with data rate 10 GB/h and two-hour residency.

    With 4 GB of weights, data stock is 20*x GB. The three-hour cost has
    C1=3, C2=3, C3=15*x and C5=60*(1-x), hence objective 66-45*x.
    A unique optimum therefore serves the largest feasible x.
    """

    base = make_coherent_tiny_instance()
    query = replace(
        base.queries[0],
        arrival_rate_per_hour=10.0,
        input_tokens=60.0,
        output_tokens=40.0,
        storage_gb_per_token=0.01,
        delay_penalty_usd_per_millisecond_query=0.0,
        unmet_penalty_usd_per_hour=20.0,
        max_unmet_fraction=0.8,
    )
    return replace(
        base,
        id="coherent_resource_stock",
        queries=(query,),
        models=(replace(base.models[0], weight_gb=4.0, kv_cache_gb_per_token=0.001),),
        gpu_tiers=(replace(base.gpu_tiers[0], tensor_parallel_degrees=(1,)),),
        compute_delay_seconds_per_token={ROUTE: 0.001},
        communication_delay_seconds_per_token={ROUTE: 0.0},
        kv_cache_residency_seconds={ROUTE: 120.0},
        horizon_hours=3.0,
        data_residency_hours=2.0,
        storage_price_usd_per_gb_hour=0.25,
        storage_capacity_gb=100.0,
        cost_budget_usd=100.0,
    )


def _memory_limited_instance() -> ProblemInstance:
    """Return 4+2*x <=5 GB, hence x<=0.5 and objective 202-200*x.

    KV slope = .01 GB/token *100 tokens *3600 requests/hour *2 s /3600.
    With free storage, two-hour rental is 2 USD and unmet cost is 200*(1-x).
    """

    base = _storage_cost_instance()
    return replace(
        base,
        id="coherent_memory_limited",
        queries=(
            replace(base.queries[0], arrival_rate_per_hour=3600.0,
                    unmet_penalty_usd_per_hour=100.0),
        ),
        models=(replace(base.models[0], kv_cache_gb_per_token=0.01),),
        gpu_tiers=(replace(base.gpu_tiers[0], memory_capacity_gb=5.0),),
        kv_cache_residency_seconds={ROUTE: 2.0},
        horizon_hours=2.0,
        storage_price_usd_per_gb_hour=0.0,
        storage_capacity_gb=10_000.0,
    )


class CoherentResourceBoundTests(unittest.TestCase):
    def assert_hand_optimum(self, instance, expected_objective):
        """Compare both implementations with the independently derived x=.5."""

        problem = PdmProblemData(instance)
        oracle = enumerate_tiny_pdm(problem)
        self.assertTrue(oracle.is_unique_optimum)
        self.assertAlmostEqual(oracle.best_objective, expected_objective)
        self.assertEqual(oracle.best_state.routing_fractions[ROUTE], 0.5)
        admitted_diagnostic = next(
            diagnostic for diagnostic in oracle.diagnostics
            if diagnostic.deployment_indicator and diagnostic.admission_indicator
        )
        self.assertEqual(admitted_diagnostic.routing_upper_bound, 0.5)
        for state in oracle.optimal_states:
            self.assertTrue(validate_placement(instance, state, PaperVersion.V2).is_feasible)

        record = execute_and_record_pdm_run(
            problem, SolverConfig(backend=SolverBackend.HIGHS)
        )
        self.assertIs(record.solver_evidence.result.status, SolverStatus.OPTIMAL)
        self.assertTrue(record.accepted_run)
        self.assertTrue(record.scientific_evidence.validation_report.is_feasible)
        self.assertAlmostEqual(record.solver_evidence.result.objective_value, expected_objective)
        self.assertAlmostEqual(record.scientific_evidence.objective_breakdown.total, expected_objective)
        self.assertAlmostEqual(record.solution.routing_fractions[ROUTE], 0.5)
        return record.scientific_evidence.validation_report

    def assert_only_violation(self, instance, expected_family):
        report = validate_placement(
            instance, make_minimal_placement_state(x=0.75, u=0.25), PaperVersion.V2
        )
        self.assertFalse(report.is_feasible)
        self.assertEqual({violation.family for violation in report.violations}, {expected_family})
        return report

    def test_mean_kv_occupancy_bounds_service_fraction(self):
        instance = _memory_limited_instance()
        # 4+2*x <=5 permits x=.5; cost 202-200*.5 =102 USD.
        report = self.assert_hand_optimum(instance, 102.0)
        memory = report.gpu_memory_diagnostics[0]
        self.assertEqual(memory.kv_memory_usage_gb, 1.0)
        self.assertEqual(memory.memory_usage_gb, 5.0)
        rejected = self.assert_only_violation(instance, ConstraintFamily.GPU_MEMORY)
        self.assertEqual(rejected.gpu_memory_diagnostics[0].violation, 0.5)

    def test_resident_data_storage_bounds_service_fraction(self):
        instance = replace(_storage_cost_instance(), storage_capacity_gb=14.0)
        # 4+20*x <=14 permits x=.5; cost 66-45*.5 =43.5 USD.
        report = self.assert_hand_optimum(instance, 43.5)
        storage = report.storage_capacity_diagnostics[0]
        self.assertEqual(storage.data_storage_usage_gb, 10.0)
        self.assertEqual(storage.storage_usage_gb, 14.0)
        self.assertEqual(report.objective_breakdown.data_storage_cost, 7.5)
        rejected = self.assert_only_violation(instance, ConstraintFamily.STORAGE_CAPACITY)
        self.assertEqual(rejected.storage_capacity_diagnostics[0].violation, 5.0)

    def test_resident_data_cost_bounds_primary_budget(self):
        instance = replace(_storage_cost_instance(), cost_budget_usd=13.5)
        # Budget C1+C2+C3 =6+15*x <=13.5 permits x=.5 independently of storage.
        report = self.assert_hand_optimum(instance, 43.5)
        self.assertEqual(report.budget_diagnostic.total_budget_use, 13.5)
        self.assertEqual(report.budget_diagnostic.data_storage_cost, 7.5)
        rejected = self.assert_only_violation(instance, ConstraintFamily.BUDGET)
        self.assertEqual(rejected.budget_diagnostic.violation, 3.75)

    def test_optional_gurobi_matches_highs_on_coherent_storage_bound(self):
        backend = probe_solver_backend(SolverBackend.GUROBI)
        if backend.availability is not SolverBackendAvailability.AVAILABLE:
            self.skipTest(f"licensed Gurobi unavailable: {backend.detail}")
        instance = replace(_storage_cost_instance(), storage_capacity_gb=14.0)
        problem = PdmProblemData(instance)
        records = [
            execute_and_record_pdm_run(problem, SolverConfig(backend=selected))
            for selected in (SolverBackend.HIGHS, SolverBackend.GUROBI)
        ]
        for record in records:
            self.assertTrue(record.accepted_run)
            self.assertIs(record.solver_evidence.result.status, SolverStatus.OPTIMAL)
            self.assertTrue(record.scientific_evidence.validation_report.is_feasible)
            self.assertAlmostEqual(record.solver_evidence.result.objective_value, 43.5)
            self.assertAlmostEqual(record.solution.routing_fractions[ROUTE], 0.5)
        self.assertEqual(records[0].solution, records[1].solution)


if __name__ == "__main__":
    unittest.main()
