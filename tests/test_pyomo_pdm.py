"""Non-solving tests for the symbolic Pyomo ``P_DM`` reference model."""

import unittest
from dataclasses import replace

from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import Constraint, Var, value

from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.optimization.problem_data import (
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


def make_minimal_4b_problem_data() -> PdmProblemData:
    """Return the hand-counted 1x1x1, four-configuration 4A fixture."""

    base = make_minimal_tiny_instance()
    tier = replace(base.gpu_tiers[0], pipeline_parallel_depths=(1, 2))
    return PdmProblemData(replace(base, gpu_tiers=(tier,)))


def make_heterogeneous_4b_problem_data() -> PdmProblemData:
    """Return the 4A heterogeneous fixture with tier-specific lattices."""

    base = make_heterogeneous_tiny_instance()
    fast_tier = replace(
        base.gpu_tiers[1],
        tensor_parallel_degrees=(1, 2, 4),
        pipeline_parallel_depths=(1, 2),
    )
    return PdmProblemData(
        replace(base, gpu_tiers=(base.gpu_tiers[0], fast_tier))
    )


def assign_state(model, state: PlacementDecisionState) -> None:
    """Assign every Pyomo variable directly from a sparse plain state."""

    mappings = {
        "x": state.routing_fractions,
        "z": state.admission_indicators,
        "q": state.deployment_indicators,
        "u": state.unmet_fractions,
        "w": state.configuration_selectors,
        "y": state.gpu_counts,
        "v": state.auxiliary_products,
    }
    for component_name, state_values in mappings.items():
        component = getattr(model, component_name)
        for key in component:
            component[key].set_value(state_values.get(key, 0.0))


def constraint_is_satisfied(constraint, tolerance: float = 1e-12) -> bool:
    """Evaluate one assigned Pyomo row without a solver."""

    body = value(constraint.body)
    if constraint.lower is not None and body < value(constraint.lower) - tolerance:
        return False
    if constraint.upper is not None and body > value(constraint.upper) + tolerance:
        return False
    return True


CONSTRAINT_COMPONENT_FAMILIES = {
    "mccormick_v_le_x": ConstraintFamily.MCCORMICK,
    "mccormick_v_le_w": ConstraintFamily.MCCORMICK,
    "mccormick_v_ge_x_plus_w_minus_1": ConstraintFamily.MCCORMICK,
    "mccormick_v_nonnegative": ConstraintFamily.MCCORMICK,
    "demand_balance": ConstraintFamily.DEMAND_BALANCE,
    "budget": ConstraintFamily.BUDGET,
    "configuration_selection": ConstraintFamily.CONFIGURATION_SELECTION,
    "gpu_count_linking": ConstraintFamily.GPU_COUNT_LINKING,
    "gpu_memory": ConstraintFamily.GPU_MEMORY,
    "compute_capacity": ConstraintFamily.COMPUTE_CAPACITY,
    "storage_capacity": ConstraintFamily.STORAGE_CAPACITY,
    "delay_slo": ConstraintFamily.DELAY_SLO,
    "error_slo": ConstraintFamily.ERROR_SLO,
    "routing_x_le_z": ConstraintFamily.ROUTING_CONSISTENCY,
    "routing_z_le_q": ConstraintFamily.ROUTING_CONSISTENCY,
}


def violated_pyomo_families(model) -> set[ConstraintFamily]:
    """Classify failed assigned rows by their shared scientific family."""

    families: set[ConstraintFamily] = set()
    for component_name, family in CONSTRAINT_COMPONENT_FAMILIES.items():
        component = getattr(model, component_name)
        if any(not constraint_is_satisfied(row) for row in component.values()):
            families.add(family)
    return families


class PdmPyomoStructureTests(unittest.TestCase):
    """Protect exact 4A indexing, domains, and unpresolved row counts."""

    def assert_model_counts_match_problem_data(self, data: PdmProblemData) -> None:
        """Compare each concrete Pyomo component with 4A metadata."""

        model = build_pdm_pyomo_model(data)
        counts = data.variable_counts
        self.assertEqual(
            (len(model.x), len(model.z), len(model.q), len(model.u)),
            (counts.n_x, counts.n_z, counts.n_q, counts.n_u),
        )
        self.assertEqual(
            (len(model.w), len(model.y), len(model.v)),
            (counts.n_w, counts.n_y, counts.n_v),
        )
        self.assertEqual(
            sum(1 for _ in model.component_data_objects(Var)),
            counts.total_decision_variable_count,
        )

        expected = data.constraint_counts
        self.assertEqual(len(model.demand_balance), expected.eq_8b)
        self.assertEqual(len(model.budget), expected.eq_8c)
        self.assertEqual(len(model.configuration_selection), expected.eq_8d)
        self.assertEqual(len(model.gpu_count_linking), expected.eq_8e)
        self.assertEqual(len(model.gpu_memory), expected.eq_8f)
        self.assertEqual(len(model.compute_capacity), expected.eq_8g)
        self.assertEqual(len(model.storage_capacity), expected.eq_8h_d_0016)
        self.assertEqual(len(model.delay_slo), expected.eq_8i)
        self.assertEqual(len(model.error_slo), expected.eq_8j)
        self.assertEqual(
            len(model.routing_x_le_z) + len(model.routing_z_le_q),
            expected.eq_8k_explicit_linking_rows,
        )
        self.assertEqual(
            len(model.mccormick_v_le_x)
            + len(model.mccormick_v_le_w)
            + len(model.mccormick_v_ge_x_plus_w_minus_1)
            + len(model.mccormick_v_nonnegative),
            expected.mccormick_inequalities,
        )
        self.assertEqual(
            sum(1 for _ in model.component_data_objects(Constraint)),
            expected.total_explicit_reference_rows,
        )

    def test_hand_checkable_1x1x1_counts_are_13_variables_and_27_rows(self) -> None:
        data = make_minimal_4b_problem_data()
        self.assertEqual(data.variable_counts.total_decision_variable_count, 13)
        self.assertEqual(data.constraint_counts.total_explicit_reference_rows, 27)
        self.assert_model_counts_match_problem_data(data)

    def test_heterogeneous_counts_are_74_variables_and_169_rows(self) -> None:
        data = make_heterogeneous_4b_problem_data()
        self.assertEqual(data.variable_counts.total_decision_variable_count, 74)
        self.assertEqual(data.constraint_counts.total_explicit_reference_rows, 169)
        self.assert_model_counts_match_problem_data(data)

    def test_model_sets_are_exact_4a_lattices_and_exclude_unsupported_config(self) -> None:
        data = make_heterogeneous_4b_problem_data()
        model = build_pdm_pyomo_model(data)
        lattices = data.variable_lattices

        self.assertEqual(tuple(model.query_ids), data.index_sets.query_ids)
        self.assertEqual(tuple(model.model_tier_keys), lattices.model_tier_keys)
        self.assertEqual(tuple(model.route_keys), lattices.route_keys)
        self.assertEqual(tuple(model.configuration_keys), lattices.configuration_keys)
        self.assertEqual(tuple(model.auxiliary_product_keys), lattices.auxiliary_product_keys)
        unsupported_w = ("m_compact", "gpu_budget_int8", 2, 2)
        unsupported_v = ("q_short", *unsupported_w)
        self.assertNotIn(unsupported_w, model.w)
        self.assertNotIn(unsupported_v, model.v)

    def test_all_seven_variable_domains_and_query_specific_u_bound(self) -> None:
        data = make_heterogeneous_4b_problem_data()
        model = build_pdm_pyomo_model(data)

        for variable in model.x.values():
            self.assertTrue(variable.is_continuous())
            self.assertEqual((variable.lb, variable.ub), (0.0, 1.0))
        for component in (model.z, model.q, model.w):
            self.assertTrue(all(variable.is_binary() for variable in component.values()))
        for variable in model.y.values():
            self.assertTrue(variable.is_integer())
            self.assertEqual(variable.lb, 0.0)
            self.assertIsNone(variable.ub)
        for query in data.instance.queries:
            self.assertTrue(model.u[query.id].is_continuous())
            self.assertEqual(
                (model.u[query.id].lb, model.u[query.id].ub),
                (0.0, query.max_unmet_fraction),
            )
        for variable in model.v.values():
            self.assertTrue(variable.is_continuous())
            self.assertEqual((variable.lb, variable.ub), (0.0, 1.0))

    def test_budget_expression_structurally_contains_only_x_z_and_y(self) -> None:
        model = build_pdm_pyomo_model(PdmProblemData(make_minimal_tiny_instance()))
        variable_families = {
            variable.parent_component().name
            for variable in identify_variables(model.budget.body)
        }
        self.assertEqual(variable_families, {"x", "z", "y"})

    def test_builder_rejects_invalid_api_values(self) -> None:
        data = PdmProblemData(make_minimal_tiny_instance())
        with self.assertRaises(TypeError):
            build_pdm_pyomo_model(data.instance)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_pdm_pyomo_model(data, "v2")  # type: ignore[arg-type]


class PdmPyomoOracleAgreementTests(unittest.TestCase):
    """Cross-check symbolic evaluation against the independent validator."""

    def assert_symbolic_values_match_report(self, model, report) -> None:
        """Compare every requested named expression with numeric evidence."""

        objective = report.objective_breakdown
        self.assertIsNotNone(objective)
        self.assertAlmostEqual(value(model.cost_gpu), objective.gpu_rental_cost)
        self.assertAlmostEqual(
            value(model.cost_model_storage), objective.model_storage_cost
        )
        self.assertAlmostEqual(
            value(model.cost_data_storage), objective.data_storage_cost
        )
        self.assertAlmostEqual(value(model.delay_penalty), objective.delay_penalty)
        self.assertAlmostEqual(
            value(model.unmet_penalty), objective.unmet_demand_penalty
        )
        self.assertAlmostEqual(value(model.total_objective), objective.total)
        self.assertAlmostEqual(value(model.objective.expr), objective.total)
        self.assertAlmostEqual(
            value(model.budget_use), report.budget_diagnostic.total_budget_use
        )

        for diagnostic in report.delay_slo_diagnostics:
            self.assertAlmostEqual(
                value(model.processing_delay[diagnostic.query_id]),
                diagnostic.processing_delay,
            )
        for diagnostic in report.error_slo_diagnostics:
            self.assertAlmostEqual(
                value(model.weighted_error[diagnostic.query_id]),
                diagnostic.weighted_error,
            )
        for diagnostic in report.compute_capacity_diagnostics:
            pair = (diagnostic.model_id, diagnostic.gpu_tier_id)
            self.assertAlmostEqual(value(model.required_compute[pair]), diagnostic.required_compute)
            self.assertAlmostEqual(value(model.available_compute[pair]), diagnostic.available_compute)
        for diagnostic in report.gpu_memory_diagnostics:
            pair = (diagnostic.model_id, diagnostic.gpu_tier_id)
            self.assertAlmostEqual(
                value(model.model_memory_usage[pair]), diagnostic.model_memory_usage_gb
            )
            self.assertAlmostEqual(
                value(model.kv_memory_printed_term[pair]),
                diagnostic.kv_memory_printed_term,
            )
            self.assertAlmostEqual(
                value(model.printed_memory_lhs[pair]), diagnostic.printed_memory_lhs
            )
            self.assertAlmostEqual(
                value(model.available_memory_rhs[pair]),
                diagnostic.available_memory_rhs_gb,
            )
        for diagnostic in report.storage_capacity_diagnostics:
            self.assertAlmostEqual(
                value(model.model_storage_usage[diagnostic.query_id]),
                diagnostic.model_storage_usage_gb,
            )
            self.assertAlmostEqual(
                value(model.data_storage_rate[diagnostic.query_id]),
                diagnostic.data_storage_rate_gb_per_hour,
            )
            self.assertAlmostEqual(
                value(model.printed_storage_lhs[diagnostic.query_id]),
                diagnostic.printed_storage_lhs,
            )

    def test_feasible_milestone_3l_state_satisfies_all_rows_and_matches_oracle(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state()
        model = build_pdm_pyomo_model(PdmProblemData(instance))
        assign_state(model, state)
        report = validate_placement(instance, state, PaperVersion.V2)

        self.assertTrue(report.is_feasible)
        self.assertEqual(violated_pyomo_families(model), set())
        self.assertTrue(
            all(
                constraint_is_satisfied(row)
                for row in model.component_data_objects(Constraint)
            )
        )
        self.assert_symbolic_values_match_report(model, report)
        self.assertAlmostEqual(value(model.total_objective), 1.00410003)

    def test_multi_violation_state_agrees_with_all_validator_families(self) -> None:
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
        model = build_pdm_pyomo_model(PdmProblemData(instance))
        assign_state(model, state)
        report = validate_placement(instance, state, PaperVersion.V2)

        self.assertFalse(report.is_feasible)
        self.assertEqual(
            violated_pyomo_families(model), set(report.violated_families)
        )
        self.assertEqual(len(set(report.violated_families)), 11)
        self.assert_symbolic_values_match_report(model, report)

    def test_direct_v_defect_is_not_repaired_by_either_implementation(self) -> None:
        instance = make_minimal_tiny_instance()
        state = make_placement_state(v={V_11: 0.9})
        model = build_pdm_pyomo_model(PdmProblemData(instance))
        assign_state(model, state)
        report = validate_placement(instance, state, PaperVersion.V2)

        self.assertEqual(violated_pyomo_families(model), {ConstraintFamily.MCCORMICK})
        self.assertEqual(report.violated_families, (ConstraintFamily.MCCORMICK,))
        self.assertFalse(
            constraint_is_satisfied(
                model.mccormick_v_ge_x_plus_w_minus_1[V_11]
            )
        )
        self.assertAlmostEqual(value(model.processing_delay["q_text"]), 0.279)
        self.assertAlmostEqual(value(model.printed_memory_lhs[PAIR]), 1.0000054)
        self.assertAlmostEqual(value(model.delay_penalty), 0.00279)
        self.assert_symbolic_values_match_report(model, report)

    def test_v1_and_v2_differ_only_in_published_c5_horizon_factor(self) -> None:
        instance = replace(make_minimal_tiny_instance(), horizon_hours=2.0)
        state = make_placement_state(x=0.8, u=0.2, v={V_11: 0.8})
        models = {
            version: build_pdm_pyomo_model(PdmProblemData(instance), version)
            for version in (PaperVersion.V1, PaperVersion.V2)
        }
        reports = {
            version: validate_placement(instance, state, version)
            for version in (PaperVersion.V1, PaperVersion.V2)
        }
        for version, model in models.items():
            assign_state(model, state)
            self.assert_symbolic_values_match_report(model, reports[version])

        same_components = (
            "cost_gpu",
            "cost_model_storage",
            "cost_data_storage",
            "delay_penalty",
        )
        for component_name in same_components:
            self.assertAlmostEqual(
                value(getattr(models[PaperVersion.V1], component_name)),
                value(getattr(models[PaperVersion.V2], component_name)),
            )
        self.assertAlmostEqual(value(models[PaperVersion.V1].unmet_penalty), 1.0)
        self.assertAlmostEqual(value(models[PaperVersion.V2].unmet_penalty), 2.0)


if __name__ == "__main__":
    unittest.main()
