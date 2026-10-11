"""Hand-derived boundaries and scale checks for independent acceptance policy."""

import unittest
from dataclasses import fields, replace
from math import inf, nan, nextafter
from unittest.mock import patch

from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.math.budget import BudgetDiagnostic
from llm_resource_allocation.math.compute_capacity import ComputeCapacityDiagnostic
from llm_resource_allocation.math.error_slo import ErrorSloDiagnostic
from llm_resource_allocation.math.gpu_memory import GpuMemoryDiagnostic
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.processing_delay import DelaySloDiagnostic
from llm_resource_allocation.math.routing import RoutingConsistencyDiagnostic
from llm_resource_allocation.math.storage_capacity import QueryStorageCapacityDiagnostic
from llm_resource_allocation.optimization.pdm_run import execute_and_record_pdm_run
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.solver import (
    SolverBackend, SolverBackendAvailability, SolverConfig, probe_solver_backend,
)
from llm_resource_allocation.validation.placement import (
    ConstraintFamily, ValidationTolerances, validate_placement,
)

from placement_support import make_minimal_placement_state


PAIR = ("m_small", "gpu_basic_fp16")
QUERY = "q_text"
ROUTE = (QUERY, *PAIR)


# Each injected diagnostic has lhs=defect and rhs=0; expected floors are
# independently stated here, rather than imported from production constants.
SCALED_ROWS = (
    (ConstraintFamily.BUDGET, "budget_diagnostic", "budget", 1e-8,
     lambda lhs, rhs: BudgetDiagnostic(lhs, 0.0, 0.0, rhs)),
    (ConstraintFamily.GPU_MEMORY, "gpu_memory_diagnostics", "gpu_memory", 1e-9,
     lambda lhs, rhs: GpuMemoryDiagnostic(*PAIR, 0.0, lhs, rhs)),
    (ConstraintFamily.COMPUTE_CAPACITY, "compute_capacity_diagnostics", "compute_capacity", 1e-8,
     lambda lhs, rhs: ComputeCapacityDiagnostic(*PAIR, lhs, rhs)),
    (ConstraintFamily.STORAGE_CAPACITY, "storage_capacity_diagnostics", "storage_capacity", 1e-9,
     lambda lhs, rhs: QueryStorageCapacityDiagnostic(QUERY, 0.0, lhs, rhs)),
    (ConstraintFamily.DELAY_SLO, "delay_slo_diagnostics", "delay_slo", 1e-10,
     lambda lhs, rhs: DelaySloDiagnostic(QUERY, lhs, rhs)),
    (ConstraintFamily.ERROR_SLO, "error_slo_diagnostics", "error_slo", 1e-12,
     lambda lhs, rhs: ErrorSloDiagnostic(QUERY, lhs, rhs)),
)


def with_row(report, field, diagnostic, **changes):
    """Inject plain hand-specified evidence to exercise report classification."""

    return replace(report, **{
        field: diagnostic if field == "budget_diagnostic" else (diagnostic,),
        **changes,
    })


class PlacementToleranceTests(unittest.TestCase):
    def setUp(self):
        self.base = validate_placement(
            make_minimal_tiny_instance(), make_minimal_placement_state(), PaperVersion.V2,
        )

    def test_absolute_thresholds_are_inclusive_for_every_scaled_family(self):
        for family, field, policy_field, floor, make in SCALED_ROWS:
            policy = replace(self.base.tolerances, **{policy_field + "_relative": 0.0})
            for defect, accepted in (
                (nextafter(floor, 0.0), True), (floor, True),
                (nextafter(floor, inf), False),
            ):
                with self.subTest(family=family, defect=defect):
                    report = with_row(self.base, field, make(defect, 0.0), tolerances=policy)
                    self.assertEqual(report.is_feasible, accepted)
                    thresholds = [tau for (f, _), tau in report.applied_tolerances.items() if f == family]
                    self.assertEqual(thresholds, [floor])
                    if not accepted:
                        self.assertEqual(report.violated_families, (family,))
                        self.assertEqual(report.violations[0].raw_violation, defect)

    def test_defaults_accept_small_zero_capacity_residuals_without_repair(self):
        for family, field, _, floor, make in SCALED_ROWS:
            with self.subTest(family=family):
                diagnostic = make(floor / 2, 0.0)
                report = with_row(self.base, field, diagnostic)
                exact = replace(report, tolerances=ValidationTolerances.exact())
                self.assertTrue(report.is_feasible)
                self.assertEqual(exact.violated_families, (family,))
                self.assertEqual(getattr(report, field), getattr(exact, field))
                self.assertEqual(report.objective_breakdown, exact.objective_breakdown)
                self.assertEqual(report.processing_delays_seconds, exact.processing_delays_seconds)
                excessive = with_row(self.base, field, make(2 * floor, 0.0))
                self.assertEqual(excessive.violated_families, (family,))

    def test_relative_threshold_boundary_uses_both_row_sides(self):
        # With atol=0, rtol=.5, lhs=2, rhs=1: violation=1 and tau=1.
        # A larger lhs increases the violation twice as fast as the threshold.
        for family, field, policy_field, _, make in SCALED_ROWS:
            policy = replace(self.base.tolerances, **{
                policy_field: 0.0, policy_field + "_relative": 0.5,
            })
            for lhs, accepted in ((nextafter(2.0, 0.0), True), (2.0, True),
                                  (nextafter(2.0, inf), False)):
                with self.subTest(family=family, lhs=lhs):
                    report = with_row(self.base, field, make(lhs, 1.0), tolerances=policy)
                    self.assertEqual(report.is_feasible, accepted)

    def test_compute_threshold_at_reference_and_large_scales(self):
        for capacity, defect, expected in ((32400.0, 2e-8, True),
                                           (32400.0, 1e-6, False),
                                           (1e12, 0.25, True), (1e12, 2.0, False)):
            with self.subTest(capacity=capacity, defect=defect):
                report = with_row(self.base, "compute_capacity_diagnostics",
                                  ComputeCapacityDiagnostic(*PAIR, capacity + defect, capacity))
                self.assertEqual(report.is_feasible, expected)
                exact = replace(report, tolerances=ValidationTolerances.exact())
                self.assertEqual(exact.violated_families, (ConstraintFamily.COMPUTE_CAPACITY,))
        report = with_row(self.base, "compute_capacity_diagnostics",
                          ComputeCapacityDiagnostic(*PAIR, 32400.0, 32400.0))
        self.assertAlmostEqual(
            report.applied_tolerances[ConstraintFamily.COMPUTE_CAPACITY, PAIR],
            4.24e-8, delta=1e-22,
        )
        rhs_dominates = with_row(self.base, "compute_capacity_diagnostics",
                                 ComputeCapacityDiagnostic(*PAIR, 1.0, 1e12))
        self.assertEqual(rhs_dominates.applied_tolerances[ConstraintFamily.COMPUTE_CAPACITY, PAIR],
                         1.00000001)

    def test_placement_routing_boundaries_are_inclusive(self):
        for fraction, expected in ((nextafter(1e-10, 0.0), True), (1e-10, True),
                                   (nextafter(1e-10, inf), False)):
            diagnostic = RoutingConsistencyDiagnostic(*ROUTE, fraction, 0.0, 0.0)
            report = with_row(self.base, "routing_consistency_diagnostics", diagnostic)
            self.assertEqual(report.is_feasible, expected)
            self.assertEqual(report.applied_tolerances[
                ConstraintFamily.ROUTING_CONSISTENCY, (*ROUTE, "admission_slack")
            ], 1e-10)
            exact = replace(report, tolerances=ValidationTolerances.exact())
            self.assertEqual(exact.violated_families, (ConstraintFamily.ROUTING_CONSISTENCY,))

    def test_exact_policy_and_threshold_evidence_are_immutable(self):
        self.assertTrue(all(getattr(ValidationTolerances.exact(), f.name) == 0.0
                            for f in fields(ValidationTolerances)))
        self.assertEqual(self.base.applied_tolerances[ConstraintFamily.CONFIGURATION_SELECTION, PAIR], 0.0)
        self.assertEqual(self.base.applied_tolerances[ConstraintFamily.GPU_COUNT_LINKING, PAIR], 0.0)
        self.assertEqual(set(family for family, _ in self.base.applied_tolerances),
                         set(ConstraintFamily) - {ConstraintFamily.DECISION_DOMAINS})
        with self.assertRaises(TypeError):
            self.base.applied_tolerances[ConstraintFamily.BUDGET, ("global",)] = 1.0
        invalid = validate_placement(make_minimal_tiny_instance(),
                                     make_minimal_placement_state(x=1.0 + 2**-40), PaperVersion.V2)
        self.assertFalse(invalid.is_feasible)
        self.assertEqual(dict(invalid.applied_tolerances), {})

    def test_relative_fields_reject_invalid_values_and_overflow(self):
        for policy_type in (ValidationTolerances,):
            for descriptor in fields(policy_type):
                if descriptor.name.endswith("_relative"):
                    for value in (-1.0, inf, nan, True, "1e-12"):
                        with self.subTest(policy=policy_type, field=descriptor.name, value=value):
                            with self.assertRaises(ValueError):
                                policy_type(**{descriptor.name: value})
        with self.assertRaisesRegex(ValueError, "must be finite"):
            with_row(self.base, "compute_capacity_diagnostics",
                     ComputeCapacityDiagnostic(*PAIR, 1e308, 1e308),
                     tolerances=replace(self.base.tolerances, compute_capacity_relative=2.0))

    def test_real_delay_error_budget_and_capacity_boundaries(self):
        # Hand arithmetic: C1+C2+C3=1+.001+.00000003; memory=1+.000006;
        # storage=1+.00003; compute=1*3*10/1000=.03; delay=.31; error=.05.
        base = make_minimal_tiny_instance()
        query, tier = base.queries[0], base.gpu_tiers[0]
        cases = (
            (replace(base, cost_budget_usd=1.00100003 - 2**-29), ConstraintFamily.BUDGET),
            (replace(base, gpu_tiers=(replace(tier, memory_capacity_gb=1.000006 - 2**-31),)),
             ConstraintFamily.GPU_MEMORY),
            (replace(base, storage_capacity_gb=1.00003 - 2**-31), ConstraintFamily.STORAGE_CAPACITY),
            (replace(base, gpu_tiers=(replace(tier, compute_throughput_tflops=(.03 - 2**-29)/3240),)),
             ConstraintFamily.COMPUTE_CAPACITY),
            (replace(base, queries=(replace(query, delay_slo_seconds=.31 - 2**-35),)), ConstraintFamily.DELAY_SLO),
            (replace(base, queries=(replace(query, error_slo=.05 - 2**-41),)), ConstraintFamily.ERROR_SLO),
        )
        for instance, family in cases:
            with self.subTest(family=family):
                state = make_minimal_placement_state()
                tolerant = validate_placement(instance, state, PaperVersion.V2)
                exact = validate_placement(instance, state, PaperVersion.V2, ValidationTolerances.exact())
                self.assertTrue(tolerant.is_feasible)
                self.assertEqual(exact.violated_families, (family,))
                self.assertEqual(tolerant.objective_breakdown, exact.objective_breakdown)

    def test_real_demand_defects_on_both_sides_preserve_decisions(self):
        instance = make_minimal_tiny_instance()
        for state in (make_minimal_placement_state(x=1.0 - 2**-35),
                      make_minimal_placement_state(u=2**-35)):
            report = validate_placement(instance, state, PaperVersion.V2)
            exact = validate_placement(instance, state, PaperVersion.V2, ValidationTolerances.exact())
            self.assertTrue(report.is_feasible)
            self.assertEqual(exact.violated_families, (ConstraintFamily.DEMAND_BALANCE,))
            self.assertEqual(report.demand_balance_diagnostics, exact.demand_balance_diagnostics)
            self.assertEqual(report.mccormick_diagnostics, exact.mccormick_diagnostics)
            self.assertEqual(report.objective_breakdown, exact.objective_breakdown)




class PdmRunnerToleranceTests(unittest.TestCase):
    def test_runner_retains_explicit_policy_with_both_backends(self):
        data = PdmProblemData(make_minimal_tiny_instance())
        policy = ValidationTolerances.exact()
        for backend in (SolverBackend.HIGHS, SolverBackend.GUROBI):
            if (
                backend is SolverBackend.GUROBI
                and probe_solver_backend(backend).availability is not SolverBackendAvailability.AVAILABLE
            ):
                continue
            with self.subTest(backend=backend):
                record = execute_and_record_pdm_run(
                    data, SolverConfig(backend=backend), validation_tolerances=policy,
                )
                self.assertTrue(record.accepted_run)
                report = record.scientific_evidence.validation_report
                self.assertIs(report.tolerances, policy)
                self.assertTrue(all(tau == 0.0 for tau in report.applied_tolerances.values()))
                self.assertEqual(record.solution.routing_fractions[ROUTE], 1.0)

    def test_invalid_runner_policy_fails_before_solving(self):
        data = PdmProblemData(make_minimal_tiny_instance())
        with patch("llm_resource_allocation.optimization.pdm_run.solve_pyomo_model") as solve:
            for invalid in (False, 0.0, {}, "exact"):
                with self.subTest(policy=invalid), self.assertRaises(TypeError):
                    execute_and_record_pdm_run(data, SolverConfig(), validation_tolerances=invalid)
            solve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
