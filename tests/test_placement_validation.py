"""Integration tests for complete solver-independent placement validation."""

import unittest
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from math import inf, nan, nextafter
import sys

from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.math.objective import (
    ObjectiveState,
    PaperVersion,
    compute_objective_breakdown,
)
from llm_resource_allocation.validation.placement import (
    ConstraintFamily,
    ValidationTolerances,
    validate_placement,
)
from llm_resource_allocation.math.compute_capacity import compute_compute_capacity_diagnostics
from llm_resource_allocation.math.gpu_memory import compute_gpu_memory_residuals
from llm_resource_allocation.math.placement import validate_placement_decision_domains
from llm_resource_allocation.math.storage_capacity import compute_storage_capacity_residuals

if __package__:
    from .placement_support import make_minimal_placement_state as make_placement_state
else:
    from placement_support import make_minimal_placement_state as make_placement_state


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ("m_small", "gpu_basic_fp16")
W_11 = (*PAIR, 1, 1)
W_21 = (*PAIR, 2, 1)
V_11 = (*ROUTE, 1, 1)
V_21 = (*ROUTE, 2, 1)


def observe_library_calls(action):
    """Count actual function executions, including calls through imported aliases."""

    calls = Counter()

    def profile(frame, event, argument):
        if event == "call" and frame.f_globals.get("__name__", "").startswith(
            "llm_resource_allocation"
        ):
            calls[frame.f_code.co_qualname] += 1

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = action()
    finally:
        sys.setprofile(previous_profile)
    return result, calls


class PlacementValidationTests(unittest.TestCase):
    """Exercise complete reports, orchestration, and independent evidence."""

    def test_full_report_checks_domains_and_computes_shared_terms_once(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state()
        report, calls = observe_library_calls(
            lambda: validate_placement(instance, state, PaperVersion.V2)
        )

        self.assertTrue(report.is_feasible)
        for name in (
            "validate_problem_instance",
            "validate_placement_decision_domains",
            "_append_deployment_indicator_errors",
            "_compute_processing_delays_seconds",
            "compute_gpu_rental_cost",
            "compute_model_storage_cost",
            "compute_data_storage_cost",
        ):
            with self.subTest(function=name):
                self.assertEqual(calls[name], 1)
        self.assertEqual(calls["compute_configuration_delay_seconds"], 2)

    def test_standalone_composites_check_instance_and_shared_q_once(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state()
        for function, decisions in (
            (validate_placement_decision_domains, state),
            (compute_gpu_memory_residuals, state),
            (compute_storage_capacity_residuals, state),
            (compute_compute_capacity_diagnostics, state.parallelism_decisions),
        ):
            with self.subTest(function=function.__name__):
                _, calls = observe_library_calls(lambda: function(instance, decisions))
                self.assertEqual(calls["validate_problem_instance"], 1)
                self.assertEqual(calls["_append_deployment_indicator_errors"], 1)

    def test_invalid_domains_stop_before_any_scientific_calculation(self) -> None:
        instance = make_minimal_tiny_instance()
        report, calls = observe_library_calls(
            lambda: validate_placement(
                instance, make_placement_state(x=1.1), PaperVersion.V2
            )
        )
        self.assertFalse(report.domain_diagnostic.is_valid)
        self.assertEqual(calls["validate_problem_instance"], 1)
        for name in (
            "_compute_processing_delays_seconds",
            "compute_objective_breakdown",
            "_compute_gpu_memory_residuals",
            "_compute_mccormick_consistency_diagnostics",
        ):
            self.assertEqual(calls[name], 0)

    def test_report_summaries_reuse_owned_immutable_violations(self) -> None:
        report = validate_placement(
            make_minimal_tiny_instance(),
            make_placement_state(v={V_11: 0.75, V_21: 0.25}),
            PaperVersion.V2,
        )
        violations = report.violations
        _, calls = observe_library_calls(
            lambda: (
                report.is_feasible,
                report.violation_count,
                report.violated_families,
                report.max_violation_by_family,
            )
        )
        self.assertIs(report.violations, violations)
        self.assertEqual(calls["PlacementValidationReport._collect_violations"], 0)
        self.assertEqual(calls["compute_mccormick_residuals"], 0)
        changed = replace(report, tolerances=ValidationTolerances(mccormick=0.25))
        self.assertEqual(changed.violations, ())
        self.assertEqual(report.violation_count, 2)

    def test_default_mccormick_accepts_small_residuals_without_repair(self) -> None:
        instance = make_minimal_tiny_instance()
        delta = 2.0**-34  # Exactly representable; about 5.82e-11, below 1e-10.
        states = (
            make_placement_state(v={V_11: 1.0 - delta}),
            make_placement_state(x=0.875, u=0.125, v={V_11: 0.875 + delta}),
            make_placement_state(v={V_11: 1.0, V_21: delta}),
        )
        for state in states:
            with self.subTest(auxiliaries=state.auxiliary_products):
                report = validate_placement(instance, state, PaperVersion.V2)
                exact = validate_placement(
                    instance, state, PaperVersion.V2, ValidationTolerances(mccormick=0.0)
                )
                self.assertTrue(report.is_feasible)
                self.assertEqual(report.tolerances.mccormick, 1e-10)
                self.assertEqual(exact.violated_families, (ConstraintFamily.MCCORMICK,))
                self.assertEqual(report.mccormick_diagnostics, exact.mccormick_diagnostics)
                self.assertEqual(report.objective_breakdown, exact.objective_breakdown)
                self.assertEqual(report.processing_delays_seconds, exact.processing_delays_seconds)
                self.assertEqual(report.gpu_memory_diagnostics, exact.gpu_memory_diagnostics)
                self.assertEqual(exact.max_violation_by_family[ConstraintFamily.MCCORMICK], delta)

    def test_default_mccormick_rejects_residuals_above_threshold(self) -> None:
        instance = make_minimal_tiny_instance()
        delta = 2.0**-33  # Exactly representable; about 1.16e-10, above 1e-10.
        states = (
            make_placement_state(v={V_11: 1.0 - delta}),
            make_placement_state(x=0.875, u=0.125, v={V_11: 0.875 + delta}),
            make_placement_state(v={V_11: 1.0, V_21: delta}),
        )
        for state in states:
            with self.subTest(auxiliaries=state.auxiliary_products):
                report = validate_placement(instance, state, PaperVersion.V2)
                self.assertFalse(report.is_feasible)
                self.assertEqual(report.violated_families, (ConstraintFamily.MCCORMICK,))
                self.assertEqual(report.max_violation_by_family[ConstraintFamily.MCCORMICK], delta)

    def test_default_mccormick_threshold_is_inclusive(self) -> None:
        instance = make_minimal_tiny_instance()
        for value, expected in (
            (nextafter(1e-10, 0.0), True),
            (1e-10, True),
            (nextafter(1e-10, inf), False),
        ):
            with self.subTest(value=value):
                state = make_placement_state(v={V_11: 1.0, V_21: value})
                report = validate_placement(instance, state, PaperVersion.V2)
                self.assertEqual(report.is_feasible, expected)
                diagnostic = next(
                    item for item in report.mccormick_diagnostics
                    if item.tensor_parallel_degree == 2
                )
                self.assertEqual(diagnostic.residuals.w_minus_v, -value)

    def test_tolerant_defaults_do_not_relax_domains_and_exact_is_available(self) -> None:
        instance = make_minimal_tiny_instance()
        delta = 2.0**-40
        for state in (
            make_placement_state(x=1.0 + delta),
            make_placement_state(w={W_11: 1.0 - delta}),
            make_placement_state(v={V_11: -delta}),
        ):
            with self.subTest(state=state):
                report = validate_placement(instance, state, PaperVersion.V2)
                self.assertEqual(report.violated_families, (ConstraintFamily.DECISION_DOMAINS,))
        report = validate_placement(
            instance, make_placement_state(u=delta), PaperVersion.V2
        )
        exact = validate_placement(
            instance, make_placement_state(u=delta), PaperVersion.V2,
            ValidationTolerances.exact(),
        )
        self.assertTrue(report.is_feasible)
        self.assertEqual(exact.violated_families, (ConstraintFamily.DEMAND_BALANCE,))
        self.assertEqual(report.demand_balance_diagnostics, exact.demand_balance_diagnostics)

    def test_fully_feasible_tiny_state_reports_objective_and_all_families(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state()

        report = validate_placement(instance, state, PaperVersion.V2)

        self.assertTrue(report.domain_diagnostic.is_valid)
        self.assertTrue(report.is_feasible)
        self.assertEqual(report.violation_count, 0)
        self.assertEqual(report.violated_families, ())
        self.assertEqual(dict(report.max_violation_by_family), {})
        self.assertAlmostEqual(report.processing_delays_seconds["q_text"], 0.31)
        self.assertIsNotNone(report.objective_breakdown)
        self.assertAlmostEqual(report.objective_breakdown.gpu_rental_cost, 1.0)
        self.assertAlmostEqual(report.objective_breakdown.model_storage_cost, 0.001)
        self.assertAlmostEqual(report.objective_breakdown.data_storage_cost, 3e-8)
        self.assertAlmostEqual(report.objective_breakdown.delay_penalty, 0.0031)
        self.assertAlmostEqual(report.objective_breakdown.total, 1.00410003)

        self.assertEqual(len(report.mccormick_diagnostics), 2)
        self.assertEqual(len(report.demand_balance_diagnostics), 1)
        self.assertIsNotNone(report.budget_diagnostic)
        self.assertEqual(len(report.configuration_selection_diagnostics), 1)
        self.assertEqual(len(report.gpu_count_linking_diagnostics), 1)
        self.assertEqual(len(report.gpu_memory_diagnostics), 1)
        self.assertEqual(len(report.compute_capacity_diagnostics), 1)
        self.assertEqual(len(report.storage_capacity_diagnostics), 1)
        self.assertEqual(len(report.delay_slo_diagnostics), 1)
        self.assertEqual(len(report.error_slo_diagnostics), 1)
        self.assertEqual(len(report.routing_consistency_diagnostics), 1)

    def test_report_objective_equals_direct_authoritative_composition(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state()
        report = validate_placement(instance, state, PaperVersion.V2)

        direct = compute_objective_breakdown(
            instance,
            ObjectiveState(
                gpu_counts=state.gpu_counts,
                admission_indicators=state.admission_indicators,
                routing_fractions=state.routing_fractions,
                unmet_fractions=state.unmet_fractions,
                processing_delays_seconds=report.processing_delays_seconds,
            ),
            PaperVersion.V2,
        )

        self.assertEqual(report.objective_breakdown, direct)

    def test_domain_valid_state_preserves_all_simultaneous_family_failures(self) -> None:
        base = make_minimal_tiny_instance()
        strict_query = replace(
            base.queries[0], delay_slo_seconds=0.1, error_slo=0.01
        )
        instance = replace(
            base,
            queries=(strict_query,),
            cost_budget_usd=0.0,
            storage_capacity_gb=1e-6,
        )
        state = make_placement_state(
            x=0.5,
            z=0.0,
            q=0.0,
            u=0.0,
            y=0.0,
            v={V_11: 1.0},
        )

        report = validate_placement(instance, state, PaperVersion.V2)

        expected = {
            ConstraintFamily.MCCORMICK,
            ConstraintFamily.DEMAND_BALANCE,
            ConstraintFamily.BUDGET,
            ConstraintFamily.CONFIGURATION_SELECTION,
            ConstraintFamily.GPU_COUNT_LINKING,
            ConstraintFamily.GPU_MEMORY,
            ConstraintFamily.COMPUTE_CAPACITY,
            ConstraintFamily.STORAGE_CAPACITY,
            ConstraintFamily.DELAY_SLO,
            ConstraintFamily.ERROR_SLO,
            ConstraintFamily.ROUTING_CONSISTENCY,
        }
        self.assertTrue(report.domain_diagnostic.is_valid)
        self.assertFalse(report.is_feasible)
        self.assertEqual(set(report.violated_families), expected)
        self.assertGreaterEqual(report.violation_count, len(expected))
        self.assertEqual(set(report.max_violation_by_family), expected)
        self.assertEqual(
            [(item.family, item.semantic_location) for item in report.violations],
            [
                (ConstraintFamily.MCCORMICK, (*V_11, "x_minus_v")),
                (ConstraintFamily.DEMAND_BALANCE, (ROUTE[0],)),
                (ConstraintFamily.BUDGET, ("global",)),
                (ConstraintFamily.CONFIGURATION_SELECTION, PAIR),
                (ConstraintFamily.GPU_COUNT_LINKING, PAIR),
                (ConstraintFamily.GPU_MEMORY, PAIR),
                (ConstraintFamily.COMPUTE_CAPACITY, PAIR),
                (ConstraintFamily.STORAGE_CAPACITY, (ROUTE[0],)),
                (ConstraintFamily.DELAY_SLO, (ROUTE[0],)),
                (ConstraintFamily.ERROR_SLO, (ROUTE[0],)),
                (ConstraintFamily.ROUTING_CONSISTENCY, (*ROUTE, "admission_slack")),
            ],
        )
        for violation in report.violations:
            self.assertTrue(violation.semantic_location)
            self.assertIsNotNone(violation.raw_violation)

    def test_named_mccormick_violations_preserve_order_and_tolerance_boundary(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state(v={V_11: 0.75, V_21: 0.25})

        exact = validate_placement(instance, state, PaperVersion.V2)
        self.assertEqual(
            [
                (item.semantic_location, item.slack)
                for item in exact.violations
                if item.family is ConstraintFamily.MCCORMICK
            ],
            [
                ((*V_11, "v_minus_x_minus_w_plus_one"), -0.25),
                ((*V_21, "w_minus_v"), -0.25),
            ],
        )

        at_boundary = validate_placement(
            instance, state, PaperVersion.V2, ValidationTolerances(mccormick=0.25)
        )
        below_boundary = validate_placement(
            instance, state, PaperVersion.V2, ValidationTolerances(mccormick=0.125)
        )
        self.assertNotIn(ConstraintFamily.MCCORMICK, at_boundary.violated_families)
        self.assertEqual(below_boundary.violations, exact.violations)
        self.assertEqual(at_boundary.mccormick_diagnostics, exact.mccormick_diagnostics)
        self.assertEqual(at_boundary.objective_breakdown, exact.objective_breakdown)

    def test_each_constraint_family_can_be_isolated_on_tiny_state(self) -> None:
        base = make_minimal_tiny_instance()
        cases = {
            ConstraintFamily.MCCORMICK: (
                base,
                make_placement_state(v={V_11: 0.9}),
            ),
            ConstraintFamily.DEMAND_BALANCE: (
                base,
                make_placement_state(u=0.1),
            ),
            ConstraintFamily.BUDGET: (
                replace(base, cost_budget_usd=1.0),
                make_placement_state(),
            ),
            ConstraintFamily.CONFIGURATION_SELECTION: (
                base,
                make_placement_state(
                    w={W_11: 1.0, W_21: 1.0},
                    y=3.0,
                    v={V_11: 1.0, V_21: 1.0},
                ),
            ),
            ConstraintFamily.GPU_COUNT_LINKING: (
                base,
                make_placement_state(y=2.0),
            ),
            ConstraintFamily.GPU_MEMORY: (
                replace(
                    base,
                    gpu_tiers=(replace(base.gpu_tiers[0], memory_capacity_gb=1.0),),
                ),
                make_placement_state(),
            ),
            ConstraintFamily.COMPUTE_CAPACITY: (
                replace(
                    base,
                    gpu_tiers=(
                        replace(
                            base.gpu_tiers[0], compute_throughput_tflops=1e-6
                        ),
                    ),
                ),
                make_placement_state(),
            ),
            ConstraintFamily.STORAGE_CAPACITY: (
                replace(base, storage_capacity_gb=1.0),
                make_placement_state(),
            ),
            ConstraintFamily.DELAY_SLO: (
                replace(
                    base,
                    queries=(replace(base.queries[0], delay_slo_seconds=0.3),),
                ),
                make_placement_state(),
            ),
            ConstraintFamily.ERROR_SLO: (
                replace(
                    base,
                    queries=(replace(base.queries[0], error_slo=0.04),),
                ),
                make_placement_state(),
            ),
            ConstraintFamily.ROUTING_CONSISTENCY: (
                base,
                make_placement_state(z=0.0),
            ),
        }

        for expected_family, (instance, state) in cases.items():
            with self.subTest(family=expected_family.value):
                report = validate_placement(instance, state, PaperVersion.V2)
                self.assertEqual(report.violated_families, (expected_family,))

    def test_inconsistent_v_is_not_repaired_for_delay_memory_or_objective(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state(v={V_11: 0.9})

        report = validate_placement(instance, state, PaperVersion.V2)

        self.assertEqual(report.violated_families, (ConstraintFamily.MCCORMICK,))
        self.assertAlmostEqual(report.processing_delays_seconds["q_text"], 0.279)
        self.assertAlmostEqual(
            report.gpu_memory_diagnostics[0].printed_memory_lhs, 1.0000054
        )
        self.assertAlmostEqual(report.objective_breakdown.delay_penalty, 0.00279)

    def test_v1_and_v2_reports_differ_only_by_published_c5_scaling(self) -> None:
        instance = replace(make_minimal_tiny_instance(), horizon_hours=2.0)
        state = make_placement_state(x=0.8, u=0.2, v={V_11: 0.8})

        v1 = validate_placement(instance, state, PaperVersion.V1)
        v2 = validate_placement(instance, state, PaperVersion.V2)

        self.assertTrue(v1.is_feasible)
        self.assertTrue(v2.is_feasible)
        self.assertEqual(v1.violations, v2.violations)
        self.assertAlmostEqual(v1.objective_breakdown.unmet_demand_penalty, 1.0)
        self.assertAlmostEqual(v2.objective_breakdown.unmet_demand_penalty, 2.0)
        self.assertAlmostEqual(
            v2.objective_breakdown.total - v1.objective_breakdown.total, 1.0
        )

    def test_invalid_domain_returns_structured_short_circuit_report(self) -> None:
        report = validate_placement(
            make_minimal_tiny_instance(),
            make_placement_state(x=1.1),
            PaperVersion.V2,
        )

        self.assertFalse(report.domain_diagnostic.is_valid)
        self.assertEqual(report.domain_diagnostic.error_type, "DecisionDomainError")
        self.assertIn("routing_fractions", report.domain_diagnostic.message)
        self.assertIsNone(report.objective_breakdown)
        self.assertEqual(report.processing_delays_seconds, {})
        self.assertEqual(report.mccormick_diagnostics, ())
        self.assertEqual(report.violation_count, 1)
        self.assertEqual(
            report.violated_families, (ConstraintFamily.DECISION_DOMAINS,)
        )
        self.assertEqual(dict(report.max_violation_by_family), {})

    def test_complete_entry_point_checks_every_extracted_decision_domain(self) -> None:
        """Prove x,z,q,u,w,y,v all cross the exact shared domain boundary."""

        invalid_states = {
            "x": make_placement_state(x=1.1),
            "z": make_placement_state(z=0.5),
            "q": make_placement_state(q=0.5),
            "u": make_placement_state(u=0.3),
            "w": make_placement_state(w={W_11: 0.5}),
            "y": make_placement_state(y=1.5),
            "v": make_placement_state(v={V_11: 1.1}),
        }

        for variable_name, state in invalid_states.items():
            with self.subTest(variable=variable_name):
                report = validate_placement(
                    make_minimal_tiny_instance(), state, PaperVersion.V2
                )
                self.assertFalse(report.domain_diagnostic.is_valid)
                self.assertIsNone(report.objective_breakdown)
                self.assertEqual(
                    report.violated_families,
                    (ConstraintFamily.DECISION_DOMAINS,),
                )

    def test_per_family_tolerance_changes_only_classification(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state(u=1e-9)

        exact = validate_placement(instance, state, PaperVersion.V2)
        tolerant = validate_placement(
            instance,
            state,
            PaperVersion.V2,
            ValidationTolerances(demand_balance=1e-8),
        )

        self.assertEqual(
            exact.violated_families, (ConstraintFamily.DEMAND_BALANCE,)
        )
        self.assertTrue(tolerant.is_feasible)
        self.assertEqual(
            exact.demand_balance_diagnostics,
            tolerant.demand_balance_diagnostics,
        )

    def test_tolerances_and_report_are_immutable_and_validated(self) -> None:
        for invalid in (-1.0, nan, inf, True, "0"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ValidationTolerances(delay_slo=invalid)

        report = validate_placement(
            make_minimal_tiny_instance(), make_placement_state(), PaperVersion.V2
        )
        with self.assertRaises(FrozenInstanceError):
            report.paper_version = PaperVersion.V1
        with self.assertRaises(TypeError):
            report.processing_delays_seconds["q_text"] = 0.0

    def test_invalid_paper_version_is_rejected_before_domain_short_circuit(self) -> None:
        instance = make_minimal_tiny_instance()
        for x in (1.0, 1.1):
            for invalid in (None, "v2", True, object()):
                with self.subTest(x=x, paper_version=invalid):
                    with self.assertRaisesRegex(ValueError, "paper_version must be"):
                        validate_placement(
                            instance, make_placement_state(x=x), invalid
                        )

    def test_wrong_tolerance_type_is_rejected_before_domain_short_circuit(self) -> None:
        instance = make_minimal_tiny_instance()
        for x in (1.0, 1.1):
            for invalid in (False, 0, 0.0, "", {}, [], True, object()):
                with self.subTest(x=x, tolerances=invalid):
                    with self.assertRaisesRegex(TypeError, "tolerances must be"):
                        validate_placement(
                            instance, make_placement_state(x=x), PaperVersion.V2, invalid
                        )

    def test_none_uses_defaults_and_explicit_tolerances_are_retained(self) -> None:
        instance = make_minimal_tiny_instance()
        for x in (1.0, 1.1):
            with self.subTest(x=x):
                state = make_placement_state(x=x)
                default = validate_placement(instance, state, PaperVersion.V2)
                explicit_none = validate_placement(instance, state, PaperVersion.V2, None)
                self.assertEqual(default, explicit_none)
                self.assertEqual(default.tolerances, ValidationTolerances())

                tolerances = ValidationTolerances(demand_balance=1e-8)
                configured = validate_placement(
                    instance, state, PaperVersion.V2, tolerances
                )
                self.assertIs(configured.tolerances, tolerances)


if __name__ == "__main__":
    unittest.main()
