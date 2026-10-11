"""Non-solving tests for raw ``P_DM`` extraction and normalization."""

import unittest
from dataclasses import FrozenInstanceError, replace
from math import inf, nan

from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.placement import (
    PlacementDecisionState,
    validate_placement_decision_domains,
)
from llm_resource_allocation.optimization.extraction import (
    PdmModelLatticeMismatchError,
    PdmSolutionNormalizationError,
    PdmVariableValueExtractionError,
    RawPdmSolution,
    SolutionNormalizationTolerances,
    extract_raw_pdm_solution_from_pyomo,
    normalize_raw_pdm_solution,
)
from llm_resource_allocation.optimization.problem_data import (
    PdmVariableFamily,
    PdmProblemData,
)
from llm_resource_allocation.optimization.pyomo_pdm import build_pdm_pyomo_model
from llm_resource_allocation.validation.placement import (
    ConstraintFamily,
    validate_placement,
)

if __package__:
    from .placement_support import make_minimal_placement_state as make_placement_state
else:
    from placement_support import make_minimal_placement_state as make_placement_state


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ("m_small", "gpu_basic_fp16")
W_11 = (*PAIR, 1, 1)
V_11 = (*ROUTE, 1, 1)


def assign_state(model, state: PlacementDecisionState) -> None:
    """Populate a Pyomo model from sparse plain values without solving."""

    mappings = {
        "x": state.routing_fractions,
        "z": state.admission_indicators,
        "q": state.deployment_indicators,
        "u": state.unmet_fractions,
        "w": state.configuration_selectors,
        "y": state.gpu_counts,
        "v": state.auxiliary_products,
    }
    for family_name, values in mappings.items():
        component = getattr(model, family_name)
        for key in component:
            component[key].set_value(values.get(key, 0.0), skip_validation=True)


def replace_raw_family(
    raw: RawPdmSolution,
    family_name: str,
    key,
    value_to_set: float,
) -> RawPdmSolution:
    """Return raw evidence with one deliberately altered value."""

    fields = {
        name: dict(getattr(raw, name)) for name in ("x", "z", "q", "u", "w", "y", "v")
    }
    fields[family_name][key] = value_to_set
    return RawPdmSolution(**fields)


class RawPdmExtractionTests(unittest.TestCase):
    """Protect dense 4A traversal and raw-value evidence."""

    def test_exact_feasible_state_extracts_dense_immutable_lattices(self) -> None:
        data = PdmProblemData(make_minimal_tiny_instance())
        model = build_pdm_pyomo_model(data)
        assign_state(model, make_placement_state())

        raw = extract_raw_pdm_solution_from_pyomo(model, data)
        lattices = data.variable_lattices

        self.assertEqual(tuple(raw.x), lattices.route_keys)
        self.assertEqual(tuple(raw.z), lattices.route_keys)
        self.assertEqual(tuple(raw.q), lattices.model_tier_keys)
        self.assertEqual(tuple(raw.u), data.index_sets.query_ids)
        self.assertEqual(tuple(raw.w), lattices.configuration_keys)
        self.assertEqual(tuple(raw.y), lattices.model_tier_keys)
        self.assertEqual(tuple(raw.v), lattices.auxiliary_product_keys)
        self.assertEqual(raw.v[V_11], 1.0)
        self.assertTrue(
            all(
                isinstance(value, float)
                for family_name in ("x", "z", "q", "u", "w", "y", "v")
                for value in getattr(raw, family_name).values()
            )
        )
        with self.assertRaises(TypeError):
            raw.x[ROUTE] = 0.0
        with self.assertRaises(FrozenInstanceError):
            raw.x = {}

    def test_uninitialized_value_fails_with_family_and_key(self) -> None:
        data = PdmProblemData(make_minimal_tiny_instance())
        model = build_pdm_pyomo_model(data)
        assign_state(model, make_placement_state())
        model.x[ROUTE].set_value(None)

        with self.assertRaises(PdmVariableValueExtractionError) as caught:
            extract_raw_pdm_solution_from_pyomo(model, data)

        self.assertEqual(caught.exception.family, PdmVariableFamily.X)
        self.assertEqual(caught.exception.semantic_key, ROUTE)
        self.assertIsNone(caught.exception.raw_value)
        self.assertEqual(caught.exception.reason, "value is uninitialized")

    def test_nonfinite_value_fails_during_raw_extraction(self) -> None:
        data = PdmProblemData(make_minimal_tiny_instance())
        model = build_pdm_pyomo_model(data)
        assign_state(model, make_placement_state())

        for nonfinite in (nan, inf, -inf):
            with self.subTest(nonfinite=nonfinite):
                model.v[V_11].set_value(nonfinite, skip_validation=True)
                with self.assertRaises(PdmVariableValueExtractionError) as caught:
                    extract_raw_pdm_solution_from_pyomo(model, data)
                self.assertEqual(caught.exception.family, PdmVariableFamily.V)
                self.assertEqual(caught.exception.semantic_key, V_11)
                self.assertEqual(caught.exception.reason, "value is not finite")

    def test_missing_model_key_fails_instead_of_extracting_intersection(self) -> None:
        data = PdmProblemData(make_minimal_tiny_instance())
        model = build_pdm_pyomo_model(data)
        assign_state(model, make_placement_state())
        del model.x[ROUTE]

        with self.assertRaises(PdmModelLatticeMismatchError) as caught:
            extract_raw_pdm_solution_from_pyomo(model, data)

        self.assertEqual(caught.exception.family, PdmVariableFamily.X)
        self.assertEqual(caught.exception.missing_keys, (ROUTE,))
        self.assertEqual(caught.exception.unexpected_keys, ())

    def test_incompatible_problem_data_is_rejected(self) -> None:
        source_data = PdmProblemData(make_minimal_tiny_instance())
        wrong_data = PdmProblemData(make_heterogeneous_tiny_instance())
        model = build_pdm_pyomo_model(source_data)
        assign_state(model, make_placement_state())

        with self.assertRaises(PdmModelLatticeMismatchError):
            extract_raw_pdm_solution_from_pyomo(model, wrong_data)


class PdmNormalizationTests(unittest.TestCase):
    """Exercise domain-only cleanup independently from scientific validation."""

    def setUp(self) -> None:
        self.instance = make_minimal_tiny_instance()
        self.data = PdmProblemData(self.instance)
        self.model = build_pdm_pyomo_model(self.data)
        assign_state(self.model, make_placement_state())
        self.raw = extract_raw_pdm_solution_from_pyomo(self.model, self.data)

    def test_defaults_are_explicit_conservative_and_immutable(self) -> None:
        tolerances = SolutionNormalizationTolerances()
        self.assertEqual(tolerances.binary_integrality, 1e-7)
        self.assertEqual(tolerances.integer_integrality, 1e-7)
        self.assertEqual(tolerances.bound_clipping, 1e-9)
        with self.assertRaises(FrozenInstanceError):
            tolerances.bound_clipping = 1e-6

        for field_name in (
            "binary_integrality",
            "integer_integrality",
            "bound_clipping",
        ):
            for invalid in (-1.0, inf, nan, True, "1e-7"):
                with self.subTest(field=field_name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        SolutionNormalizationTolerances(**{field_name: invalid})
        with self.assertRaises(ValueError):
            SolutionNormalizationTolerances(binary_integrality=0.5)
        with self.assertRaises(ValueError):
            SolutionNormalizationTolerances(integer_integrality=0.5)

    def test_out_of_tolerance_values_fail_with_structured_evidence(self) -> None:
        cases = (
            ("z", ROUTE, 0.1, PdmVariableFamily.Z, 1e-7),
            ("y", PAIR, 3.2, PdmVariableFamily.Y, 1e-7),
            ("y", PAIR, -1.0, PdmVariableFamily.Y, 1e-7),
            ("x", ROUTE, -0.01, PdmVariableFamily.X, 1e-9),
            ("v", V_11, 1.01, PdmVariableFamily.V, 1e-9),
            ("u", "q_text", 0.21, PdmVariableFamily.U, 1e-9),
        )
        for family_name, key, raw_value, family, tolerance in cases:
            with self.subTest(family=family_name):
                altered = replace_raw_family(
                    self.raw, family_name, key, raw_value
                )
                with self.assertRaises(PdmSolutionNormalizationError) as caught:
                    normalize_raw_pdm_solution(altered, self.data)
                self.assertEqual(caught.exception.family, family)
                self.assertEqual(caught.exception.semantic_key, key)
                self.assertEqual(caught.exception.raw_value, raw_value)
                self.assertEqual(caught.exception.tolerance, tolerance)
                self.assertTrue(caught.exception.requirement)

    def test_continuous_lower_and_upper_noise_is_clipped_only_at_bounds(self) -> None:
        lower_noise = replace_raw_family(self.raw, "x", ROUTE, -1e-10)
        lower_state = normalize_raw_pdm_solution(lower_noise, self.data)
        self.assertEqual(lower_state.routing_fractions[ROUTE], 0.0)
        validate_placement_decision_domains(self.instance, lower_state)

        upper_noise = replace_raw_family(self.raw, "v", V_11, 1.0 + 1e-10)
        upper_state = normalize_raw_pdm_solution(upper_noise, self.data)
        self.assertEqual(upper_state.auxiliary_products[V_11], 1.0)
        validate_placement_decision_domains(self.instance, upper_state)

        unmet_noise = replace_raw_family(self.raw, "u", "q_text", 0.2 + 1e-10)
        unmet_state = normalize_raw_pdm_solution(unmet_noise, self.data)
        self.assertEqual(unmet_state.unmet_fractions["q_text"], 0.2)
        validate_placement_decision_domains(self.instance, unmet_state)

    def test_custom_tolerances_are_applied_by_domain_category_only(self) -> None:
        binary_noise = replace_raw_family(self.raw, "q", PAIR, 1.0 - 1e-6)
        with self.assertRaises(PdmSolutionNormalizationError):
            normalize_raw_pdm_solution(binary_noise, self.data)

        normalized = normalize_raw_pdm_solution(
            binary_noise,
            self.data,
            SolutionNormalizationTolerances(binary_integrality=1e-5),
        )
        self.assertEqual(normalized.deployment_indicators[PAIR], 1)

    def test_manually_created_raw_lattice_mismatch_fails_normalization(self) -> None:
        fields = {
            name: dict(getattr(self.raw, name))
            for name in ("x", "z", "q", "u", "w", "y", "v")
        }
        del fields["v"][V_11]
        incomplete = RawPdmSolution(**fields)

        with self.assertRaises(PdmSolutionNormalizationError) as caught:
            normalize_raw_pdm_solution(incomplete, self.data)

        self.assertEqual(caught.exception.family, PdmVariableFamily.V)
        self.assertIsNone(caught.exception.semantic_key)


class PdmExtractionRoundTripTests(unittest.TestCase):
    """Verify extraction-normalization-validation composition without a solver."""

    def test_exact_feasible_state_round_trips_and_remains_feasible(self) -> None:
        instance = make_minimal_tiny_instance()
        original = make_placement_state()
        data = PdmProblemData(instance)
        model = build_pdm_pyomo_model(data)
        assign_state(model, original)

        raw = extract_raw_pdm_solution_from_pyomo(model, data)
        normalized = normalize_raw_pdm_solution(raw, data)
        report = validate_placement(instance, normalized, PaperVersion.V2)

        self.assertTrue(report.is_feasible)
        for key in data.variable_lattices.route_keys:
            self.assertEqual(normalized.routing_fractions[key], original.routing_fractions.get(key, 0.0))
            self.assertEqual(normalized.admission_indicators[key], original.admission_indicators.get(key, 0.0))
        for key in data.variable_lattices.model_tier_keys:
            self.assertEqual(normalized.deployment_indicators[key], original.deployment_indicators.get(key, 0.0))
            self.assertEqual(normalized.gpu_counts[key], original.gpu_counts.get(key, 0.0))
        for query_id in data.index_sets.query_ids:
            self.assertEqual(
                normalized.unmet_fractions[query_id],
                original.unmet_fractions.get(query_id, 0.0),
            )
        for key in data.variable_lattices.configuration_keys:
            self.assertEqual(normalized.configuration_selectors[key], original.configuration_selectors.get(key, 0.0))
        for key in data.variable_lattices.auxiliary_product_keys:
            self.assertEqual(normalized.auxiliary_products[key], original.auxiliary_products.get(key, 0.0))
        self.assertAlmostEqual(report.objective_breakdown.total, 1.00410003)

    def test_multi_violation_round_trip_preserves_all_eleven_families(self) -> None:
        base = make_minimal_tiny_instance()
        instance = replace(
            base,
            queries=(replace(base.queries[0], delay_slo_seconds=0.1, error_slo=0.01),),
            cost_budget_usd=0.0,
            storage_capacity_gb=1e-6,
        )
        original = make_placement_state(
            x=0.5, z=0.0, q=0.0, u=0.0, y=0.0, v={V_11: 1.0}
        )
        data = PdmProblemData(instance)
        model = build_pdm_pyomo_model(data)
        assign_state(model, original)

        normalized = normalize_raw_pdm_solution(
            extract_raw_pdm_solution_from_pyomo(model, data), data
        )
        original_report = validate_placement(instance, original, PaperVersion.V2)
        report = validate_placement(instance, normalized, PaperVersion.V2)

        self.assertFalse(report.is_feasible)
        self.assertEqual(len(set(report.violated_families)), 11)
        self.assertEqual(report.violated_families, original_report.violated_families)

    def test_direct_v_defect_is_preserved_through_every_layer(self) -> None:
        instance = make_minimal_tiny_instance()
        data = PdmProblemData(instance)
        model = build_pdm_pyomo_model(data)
        assign_state(model, make_placement_state(v={V_11: 0.9}))

        raw = extract_raw_pdm_solution_from_pyomo(model, data)
        normalized = normalize_raw_pdm_solution(raw, data)
        report = validate_placement(instance, normalized, PaperVersion.V2)

        self.assertEqual(raw.v[V_11], 0.9)
        self.assertEqual(normalized.auxiliary_products[V_11], 0.9)
        self.assertFalse(report.is_feasible)
        self.assertEqual(report.violated_families, (ConstraintFamily.MCCORMICK,))
        self.assertAlmostEqual(report.processing_delays_seconds["q_text"], 0.279)

    def test_solver_noise_is_preserved_raw_then_domain_normalized(self) -> None:
        base = make_heterogeneous_tiny_instance()
        fast_tier = replace(
            base.gpu_tiers[1],
            tensor_parallel_degrees=(1, 2, 4),
            pipeline_parallel_depths=(1, 2),
        )
        instance = replace(base, gpu_tiers=(base.gpu_tiers[0], fast_tier))
        data = PdmProblemData(instance)
        model = build_pdm_pyomo_model(data)

        active_pair = ("m_accurate", "gpu_fast_fp16")
        active_configuration = (*active_pair, 2, 2)
        active_routes = {
            (query_id, *active_pair) for query_id in instance.query_ids
        }
        active_auxiliaries = {
            (*route_key, 2, 2) for route_key in active_routes
        }
        for key in model.x:
            model.x[key].set_value(
                1.0 + 1e-10 if key in active_routes else -1e-10,
                skip_validation=True,
            )
            model.z[key].set_value(1.0 if key in active_routes else 0.0)
        for key in model.q:
            model.q[key].set_value(
                1.0 - 1e-10 if key == active_pair else 0.0,
                skip_validation=True,
            )
            model.y[key].set_value(
                4.0 - 1e-10 if key == active_pair else 0.0,
                skip_validation=True,
            )
        for key in model.u:
            model.u[key].set_value(0.0)
        for key in model.w:
            model.w[key].set_value(
                1.0 + 1e-10 if key == active_configuration else 0.0,
                skip_validation=True,
            )
        for key in model.v:
            model.v[key].set_value(
                1.0 + 1e-10 if key in active_auxiliaries else -1e-10,
                skip_validation=True,
            )

        raw = extract_raw_pdm_solution_from_pyomo(model, data)
        self.assertEqual(raw.q[active_pair], 1.0 - 1e-10)
        self.assertEqual(raw.w[active_configuration], 1.0 + 1e-10)
        self.assertEqual(raw.y[active_pair], 4.0 - 1e-10)
        self.assertEqual(raw.x[next(iter(active_routes))], 1.0 + 1e-10)
        self.assertTrue(any(value == -1e-10 for value in raw.x.values()))

        normalized = normalize_raw_pdm_solution(raw, data)
        self.assertEqual(normalized.deployment_indicators[active_pair], 1)
        self.assertEqual(normalized.configuration_selectors[active_configuration], 1)
        self.assertEqual(normalized.gpu_counts[active_pair], 4)
        self.assertTrue(all(value in (0.0, 1.0) for value in normalized.routing_fractions.values()))
        self.assertTrue(all(value in (0.0, 1.0) for value in normalized.auxiliary_products.values()))
        validate_placement_decision_domains(instance, normalized)

        report = validate_placement(instance, normalized, PaperVersion.V2)
        self.assertTrue(report.is_feasible)


if __name__ == "__main__":
    unittest.main()
