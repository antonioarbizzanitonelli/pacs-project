"""Reusable validation configuration and immutable evidence ownership."""

from dataclasses import FrozenInstanceError, fields, replace
from math import inf, nan
import unittest

from llm_resource_allocation.data import (
    make_minimal_tiny_instance,
)
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.validation.objective_comparison import ObjectiveComparison
from llm_resource_allocation.validation.placement import (
    ConstraintFamily,
    PlacementDomainDiagnostic,
    PlacementValidator,
    ValidationTolerances,
    ValidationViolation,
    validate_placement,
)


def make_state(instance, *, x=1.0, u=0.0):
    """Supply exact decisions for the minimal one-route, one-GPU case."""

    route = (instance.queries[0].id, instance.models[0].id, instance.gpu_tiers[0].id)
    configuration = (*route[1:], 1, 1)
    routing = RoutingDecisionState(
        {route: x}, {route: 1.0}, {route[1:]: 1.0}, {route[0]: u}
    )
    return PlacementDecisionState(
        ParallelismDecisionState(routing, {configuration: 1.0}, {route[1:]: 1.0}),
        {(*route, 1, 1): x},
    )


class ConfiguredValidatorTests(unittest.TestCase):
    def test_placement_validator_reuses_policy_without_retaining_candidates(self):
        instance = make_minimal_tiny_instance()
        tolerances = ValidationTolerances(demand_balance=1e-8)
        validator = PlacementValidator(instance, PaperVersion.V2, tolerances)

        invalid = validator.validate(make_state(instance, x=1.1))
        full = validator.validate(make_state(instance))
        partial_state = make_state(instance, x=0.8, u=0.2)
        partial = validator.validate(partial_state)

        self.assertFalse(invalid.domain_diagnostic.is_valid)
        self.assertIsNone(invalid.objective_breakdown)
        self.assertTrue(full.is_feasible)
        self.assertAlmostEqual(full.objective_breakdown.total, 1.00410003)
        self.assertTrue(partial.is_feasible)
        self.assertAlmostEqual(partial.processing_delays_seconds[instance.query_ids[0]], 0.248)
        self.assertAlmostEqual(partial.objective_breakdown.total, 2.003480024)
        self.assertEqual(
            partial, validate_placement(instance, partial_state, PaperVersion.V2, tolerances)
        )
        self.assertIs(validator.instance, instance)
        self.assertIs(partial.tolerances, tolerances)
        self.assertEqual(validator.validate(make_state(instance)), full)
        with self.assertRaises(FrozenInstanceError):
            validator.paper_version = PaperVersion.V1

    def test_invalid_instance_remains_a_structured_scientific_diagnostic(self):
        instance = replace(make_minimal_tiny_instance(), cost_budget_usd=-1.0)
        validator = PlacementValidator(instance, PaperVersion.V2)
        report = validator.validate(make_state(instance))

        self.assertFalse(report.domain_diagnostic.is_valid)
        self.assertEqual(report.domain_diagnostic.error_type, "InstanceValidationError")
        self.assertIsNone(report.objective_breakdown)

    def test_validator_constructors_reject_invalid_configuration(self):
        instance = make_minimal_tiny_instance()
        for invalid in (False, {}, object()):
            with self.subTest(tolerances=invalid):
                with self.assertRaises(TypeError):
                    PlacementValidator(instance, PaperVersion.V2, invalid)
        with self.assertRaises(ValueError):
            PlacementValidator(instance, "v2")



class EvidenceInvariantTests(unittest.TestCase):
    def test_validators_produce_coherent_boundary_diagnostics(self):
        instance = make_minimal_tiny_instance()
        placement_validator = PlacementValidator(instance, PaperVersion.V2)
        for diagnostic_type, valid, invalid in (
            (
                PlacementDomainDiagnostic,
                placement_validator.validate(make_state(instance)).domain_diagnostic,
                placement_validator.validate(make_state(instance, x=1.1)).domain_diagnostic,
            ),
        ):
            with self.subTest(diagnostic_type=diagnostic_type):
                self.assertIsInstance(valid, diagnostic_type)
                self.assertTrue(valid.is_valid)
                self.assertIsNone(valid.error_type)
                self.assertIsNone(valid.message)
                self.assertFalse(invalid.is_valid)
                self.assertEqual(invalid.error_type, "DecisionDomainError")
                self.assertIsInstance(invalid.message, str)
                self.assertTrue(invalid.message)
                with self.assertRaises(FrozenInstanceError):
                    valid.is_valid = False

    def test_reports_copy_caller_owned_diagnostic_sequences(self):
        instance = make_minimal_tiny_instance()
        placement_report = PlacementValidator(instance, PaperVersion.V2).validate(
            make_state(instance)
        )

        for report in (placement_report,):
            for descriptor in fields(report):
                if not descriptor.init:
                    continue
                original = getattr(report, descriptor.name)
                if isinstance(original, tuple):
                    with self.subTest(report=type(report).__name__, field=descriptor.name):
                        caller_sequence = list(original)
                        copied = replace(report, **{descriptor.name: caller_sequence})
                        caller_sequence.clear()
                        self.assertEqual(getattr(copied, descriptor.name), original)
                        self.assertTrue(copied.is_feasible)

    def test_violations_copy_their_semantic_locations(self):
        for violation_type, family in (
            (ValidationViolation, ConstraintFamily.DEMAND_BALANCE),
        ):
            with self.subTest(violation_type=violation_type):
                location = ["query"]
                violation = violation_type(family, location, -0.1)
                location.clear()
                self.assertEqual(violation.semantic_location, ("query",))
                self.assertEqual(violation.raw_violation, 0.1)

    def test_objective_comparison_enforces_invariants_on_direct_construction(self):
        comparison = ObjectiveComparison(1, 1.0 + 5e-9, 1e-8)
        self.assertTrue(comparison.agrees)
        self.assertIsInstance(comparison.solver_objective, float)
        for name in ("solver_objective", "independent_objective", "absolute_tolerance"):
            for invalid in (inf, nan, True, "1"):
                with self.subTest(field=name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        replace(comparison, **{name: invalid})
        with self.assertRaises(ValueError):
            replace(comparison, absolute_tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
