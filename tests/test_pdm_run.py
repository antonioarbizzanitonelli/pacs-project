"""Heterogeneous end-to-end tests for the Milestone-4F run record."""

import unittest
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from math import ulp

from pyomo.core.base.component import Component

from llm_resource_allocation.data.tiny_instances import (
    make_heterogeneous_solve_tiny_instance,
    make_heterogeneous_tiny_instance,
)
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.optimization.solver import (
    NoSolverIncumbentError,
    SolverBackend,
    SolverConfig,
    SolverStatus,
)
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.pyomo_pdm import build_pdm_pyomo_model
from llm_resource_allocation.optimization.pdm_run import execute_and_record_pdm_run


UNSUPPORTED_CONFIGURATION = ("m_compact", "gpu_budget_int8", 2, 2)
UNSUPPORTED_AUXILIARY = (
    "q_short",
    "m_compact",
    "gpu_budget_int8",
    2,
    2,
)


class HeterogeneousSolveFixtureTests(unittest.TestCase):
    """Protect the narrow synthetic adjustment and full tier-aware lattice."""

    def test_solve_fixture_changes_only_id_and_two_arrival_rates(self) -> None:
        base = make_heterogeneous_tiny_instance()
        solve_fixture = make_heterogeneous_solve_tiny_instance()

        self.assertEqual(solve_fixture.id, "tiny_2x2x2_solve")
        self.assertEqual(base.models, solve_fixture.models)
        self.assertEqual(base.gpu_tiers, solve_fixture.gpu_tiers)
        for field_name in (
            "base_error_rate",
            "compute_delay_seconds_per_token",
            "communication_delay_seconds_per_token",
            "compute_gflop_per_token",
            "kv_cache_residency_seconds",
            "storage_price_usd_per_gb_hour",
            "cost_budget_usd",
            "storage_capacity_gb",
            "horizon_hours",
            "compute_utilization",
        ):
            self.assertEqual(
                getattr(base, field_name),
                getattr(solve_fixture, field_name),
            )

        expected_rates = {"q_short": 36.0, "q_long": 12.0}
        base_by_id = {query.id: query for query in base.queries}
        for query in solve_fixture.queries:
            self.assertEqual(query.arrival_rate_per_hour, expected_rates[query.id])
            self.assertEqual(
                replace(
                    query,
                    arrival_rate_per_hour=(
                        base_by_id[query.id].arrival_rate_per_hour
                    ),
                ),
                base_by_id[query.id],
            )

    def test_structural_counts_and_unsupported_configuration_are_exact(self) -> None:
        problem_data = PdmProblemData(
            make_heterogeneous_solve_tiny_instance()
        )
        variable_counts = problem_data.variable_counts
        constraint_counts = problem_data.constraint_counts

        self.assertEqual(
            (
                variable_counts.n_x,
                variable_counts.n_z,
                variable_counts.n_q,
                variable_counts.n_u,
                variable_counts.n_w,
                variable_counts.n_y,
                variable_counts.n_v,
            ),
            (8, 8, 4, 2, 12, 4, 24),
        )
        self.assertEqual(variable_counts.total_decision_variable_count, 62)
        self.assertEqual(
            (
                constraint_counts.eq_8b,
                constraint_counts.eq_8c,
                constraint_counts.eq_8d,
                constraint_counts.eq_8e,
                constraint_counts.eq_8f,
                constraint_counts.eq_8g,
                constraint_counts.eq_8h_d_0016,
                constraint_counts.eq_8i,
                constraint_counts.eq_8j,
                constraint_counts.eq_8k_paper_semantic_inequalities,
                constraint_counts.eq_8k_explicit_linking_rows,
                constraint_counts.eq_8k_x_lower_bounds,
                constraint_counts.mccormick_inequalities,
            ),
            (2, 1, 4, 4, 4, 4, 2, 2, 2, 24, 16, 8, 96),
        )
        self.assertEqual(constraint_counts.total_explicit_reference_rows, 137)
        self.assertNotIn(
            UNSUPPORTED_CONFIGURATION,
            problem_data.variable_lattices.configuration_keys,
        )
        self.assertNotIn(
            UNSUPPORTED_AUXILIARY,
            problem_data.variable_lattices.auxiliary_product_keys,
        )
        model = build_pdm_pyomo_model(problem_data)
        self.assertNotIn(UNSUPPORTED_CONFIGURATION, model.w)
        self.assertNotIn(UNSUPPORTED_AUXILIARY, model.v)


class HeterogeneousPdmRunTests(unittest.TestCase):
    """Exercise solve-through-validation evidence on the 2x2x2 instance."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = make_heterogeneous_solve_tiny_instance()
        cls.problem_data = PdmProblemData(cls.instance)
        cls.config = SolverConfig(backend=SolverBackend.HIGHS)
        cls.record = execute_and_record_pdm_run(cls.problem_data, cls.config)

    def test_end_to_end_record_is_accepted_and_independently_valid(self) -> None:
        record = self.record
        result = record.solver_evidence.result
        evidence = record.scientific_evidence

        self.assertIs(result.status, SolverStatus.OPTIMAL)
        self.assertTrue(result.has_incumbent)
        self.assertTrue(record.has_incumbent)
        self.assertTrue(record.independently_feasible)
        self.assertEqual(evidence.validation_report.violations, ())
        self.assertTrue(record.objective_agrees)
        self.assertTrue(record.accepted_run)
        self.assertAlmostEqual(result.objective_value, 1.111001008, places=11)
        self.assertAlmostEqual(
            evidence.objective_comparison.absolute_difference,
            0.0,
            places=11,
        )

    def test_original_unscaled_instance_passes_default_validation(self) -> None:
        problem_data = PdmProblemData(make_heterogeneous_tiny_instance())
        record = execute_and_record_pdm_run(problem_data, self.config)
        self.assertTrue(record.accepted_run)
        self.assertEqual(record.scientific_evidence.validation_report.tolerances.mccormick, 1e-10)
        self.assertEqual(record.scientific_evidence.validation_report.violations, ())
        self.assertEqual(record.identity.instance_id, "tiny_2x2x2")

    def test_objective_breakdown_and_meaningful_state_are_persisted(self) -> None:
        breakdown = self.record.scientific_evidence.objective_breakdown
        state = self.record.solution

        self.assertIs(
            breakdown,
            self.record.scientific_evidence.validation_report.objective_breakdown,
        )
        self.assertEqual(breakdown.gpu_rental_cost, 1.0)
        self.assertEqual(breakdown.model_storage_cost, 0.024)
        self.assertEqual(breakdown.data_storage_cost, 0.000001008)
        self.assertAlmostEqual(breakdown.delay_penalty, 0.087, delta=ulp(0.087))
        self.assertEqual(breakdown.unmet_demand_penalty, 0.0)
        self.assertEqual(breakdown.total, 1.111001008)
        for query_id in self.record.identity.query_ids:
            routed = sum(
                value
                for (candidate_query, _, _), value in state.routing_fractions.items()
                if candidate_query == query_id
            )
            self.assertEqual(routed, 1.0)
            self.assertEqual(state.unmet_fractions[query_id], 0.0)

        active_pairs = {
            pair
            for pair, indicator in state.deployment_indicators.items()
            if indicator == 1
        }
        self.assertTrue(active_pairs)
        for pair in active_pairs:
            selected = [
                configuration
                for configuration, indicator in (
                    state.configuration_selectors.items()
                )
                if configuration[:2] == pair and indicator == 1
            ]
            self.assertEqual(len(selected), 1)
            _, _, tensor_parallel, pipeline_parallel = selected[0]
            self.assertEqual(
                state.gpu_counts[pair],
                tensor_parallel * pipeline_parallel,
            )
        for route, fraction in state.routing_fractions.items():
            if fraction <= 0.0:
                continue
            query_id, model_id, tier_id = route
            pair = (model_id, tier_id)
            self.assertEqual(state.admission_indicators[route], 1)
            self.assertEqual(state.deployment_indicators[pair], 1)
            selected = next(
                configuration
                for configuration, indicator in (
                    state.configuration_selectors.items()
                )
                if configuration[:2] == pair and indicator == 1
            )
            auxiliary = (query_id, *selected)
            self.assertEqual(state.auxiliary_products[auxiliary], fraction)

    def test_repeat_run_preserves_status_objective_and_scientific_validity(self) -> None:
        repeated = execute_and_record_pdm_run(self.problem_data, self.config)

        self.assertIs(repeated.solver_evidence.result.status, SolverStatus.OPTIMAL)
        self.assertEqual(
            repeated.solver_evidence.result.objective_value,
            self.record.solver_evidence.result.objective_value,
        )
        self.assertTrue(repeated.independently_feasible)
        self.assertTrue(repeated.objective_agrees)
        self.assertTrue(repeated.accepted_run)

    def test_reordered_entities_preserve_semantic_result(self) -> None:
        reordered_instance = replace(
            self.instance,
            id="tiny_2x2x2_solve_reordered",
            queries=tuple(reversed(self.instance.queries)),
            models=tuple(reversed(self.instance.models)),
            gpu_tiers=tuple(reversed(self.instance.gpu_tiers)),
        )
        reordered = execute_and_record_pdm_run(
            PdmProblemData(reordered_instance),
            self.config,
        )

        self.assertEqual(
            reordered.identity.query_ids,
            tuple(reversed(self.record.identity.query_ids)),
        )
        self.assertEqual(
            reordered.identity.model_ids,
            tuple(reversed(self.record.identity.model_ids)),
        )
        self.assertEqual(
            reordered.identity.tier_ids,
            tuple(reversed(self.record.identity.tier_ids)),
        )
        self.assertTrue(reordered.accepted_run)
        self.assertEqual(
            reordered.scientific_evidence.objective_breakdown,
            self.record.scientific_evidence.objective_breakdown,
        )
        for field_name in (
            "routing_fractions",
            "admission_indicators",
            "deployment_indicators",
            "unmet_fractions",
            "configuration_selectors",
            "gpu_counts",
            "auxiliary_products",
        ):
            self.assertEqual(
                set(getattr(reordered.solution, field_name)),
                set(getattr(self.record.solution, field_name)),
            )
        for query_id in self.record.identity.query_ids:
            original_service = sum(
                value
                for (candidate_query, _, _), value in (
                    self.record.solution.routing_fractions.items()
                )
                if candidate_query == query_id
            )
            reordered_service = sum(
                value
                for (candidate_query, _, _), value in (
                    reordered.solution.routing_fractions.items()
                )
                if candidate_query == query_id
            )
            self.assertEqual(reordered_service, original_service)
            self.assertEqual(
                reordered.solution.unmet_fractions[query_id],
                self.record.solution.unmet_fractions[query_id],
            )

    def test_completed_record_is_immutable_and_contains_no_pyomo_component(self) -> None:
        self.assertFalse(hasattr(self.record, "solver_result"))
        self.assertIs(
            self.record.scientific_evidence.validation_report.paper_version,
            PaperVersion.V2,
        )
        with self.assertRaises(FrozenInstanceError):
            self.record.identity.instance_id = "changed"
        with self.assertRaises(TypeError):
            self.record.solution.unmet_fractions["q_short"] = 1.0
        with self.assertRaises(TypeError):
            self.record.solver_evidence.result.raw_status_metadata["new"] = "value"

        seen: set[int] = set()

        def assert_plain_snapshot(value) -> None:
            self.assertNotIsInstance(value, Component)
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            if is_dataclass(value) and not isinstance(value, type):
                for field in fields(value):
                    assert_plain_snapshot(getattr(value, field.name))
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    assert_plain_snapshot(key)
                    assert_plain_snapshot(item)
            elif isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    assert_plain_snapshot(item)

        assert_plain_snapshot(self.record)

    def test_no_incumbent_raises_before_record_construction(self) -> None:
        infeasible_instance = replace(self.instance, cost_budget_usd=0.0)

        with self.assertRaises(NoSolverIncumbentError):
            execute_and_record_pdm_run(
                PdmProblemData(infeasible_instance),
                self.config,
                PaperVersion.V2,
            )


if __name__ == "__main__":
    unittest.main()
