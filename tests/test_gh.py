"""End-to-end tests for exact GH closure and independent validation."""

import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    GHCompletionStatus,
    GHPhase1Config,
    GHPhase2Deployment,
    canonicalize_gh_phase2_state,
    initialize_gh_phase2_state,
    run_gh,
    run_gh_phase1,
)
from llm_resource_allocation.algorithms import gh as gh_module
from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.domain import ParallelismConfig
from llm_resource_allocation.math.objective import PaperVersion
from llm_resource_allocation.validation.placement import ConstraintFamily


QUERY = "q_text"
TIER = "gpu_basic_fp16"


def make_two_candidate_instance(*, error_slo=0.025, max_unmet_fraction=0.2):
    """Build the D-0039 two-identical-candidate hand fixture."""

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        error_slo=error_slo,
        delay_slo_seconds=1.0,
        max_unmet_fraction=max_unmet_fraction,
    )
    models = tuple(
        replace(base.models[0], id=model_id, weight_gb=1.0)
        for model_id in ("model_a", "model_b")
    )
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=8.0,
        tensor_parallel_degrees=(1, 2),
        pipeline_parallel_depths=(1,),
    )
    triples = tuple((QUERY, model.id, TIER) for model in models)
    return replace(
        base,
        id="gh_complete_d0039_fixture",
        queries=(query,),
        models=models,
        gpu_tiers=(tier,),
        base_error_rate={(QUERY, model.id): 0.05 for model in models},
        compute_delay_seconds_per_token={key: 0.10 for key in triples},
        communication_delay_seconds_per_token={key: 0.01 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        cost_budget_usd=20.0,
        storage_capacity_gb=100.0,
    )


def make_residual_instance():
    """Build one candidate that can serve exactly half the query."""

    instance = make_two_candidate_instance(max_unmet_fraction=0.5)
    model = instance.models[0]
    key = (QUERY, model.id, TIER)
    return replace(
        instance,
        id="gh_complete_residual_fixture",
        models=(model,),
        base_error_rate={(QUERY, model.id): 0.05},
        compute_delay_seconds_per_token={key: 0.10},
        communication_delay_seconds_per_token={key: 0.01},
        compute_gflop_per_token={key: 1.0},
        kv_cache_residency_seconds={key: 1.0},
    )


class GHCompositionAndFeasibilityTests(unittest.TestCase):
    """Exercise line-23 composition and deliberately separate outcomes."""

    def test_each_run_freshly_validates_its_exact_canonical_placement(self):
        instance = make_minimal_tiny_instance()
        config = GHPhase1Config(0.8)
        first = run_gh(instance, config)
        with patch.object(
            gh_module, "validate_placement", wraps=gh_module.validate_placement
        ) as validate:
            second = run_gh(instance, config)

        validate.assert_called_once_with(
            instance, second.placement_decisions, PaperVersion.V2, None
        )
        self.assertIs(validate.call_args.args[1], second.placement_decisions)
        self.assertIsNot(first.validation_report, second.validation_report)
        self.assertEqual(first.validation_report, second.validation_report)
        self.assertIs(second.objective_breakdown, second.validation_report.objective_breakdown)

    def test_run_composes_each_frozen_phase_once_with_exact_handoffs(self):
        instance = make_minimal_tiny_instance()
        config = GHPhase1Config(0.8)
        with patch.object(
            gh_module, "_run_gh_phase1", wraps=gh_module._run_gh_phase1
        ) as phase1, patch.object(
            gh_module,
            "_initialize_gh_phase2_state",
            wraps=gh_module._initialize_gh_phase2_state,
        ) as initialize, patch.object(
            gh_module, "_run_gh_phase2", wraps=gh_module._run_gh_phase2
        ) as phase2:
            result = run_gh(instance, config)

        phase1.assert_called_once_with(instance, config)
        initialize.assert_called_once()
        phase2.assert_called_once()
        self.assertIs(initialize.call_args.args[1], result.phase1_result)
        self.assertIs(phase2.call_args.args[1], result.initialized_phase2_state)
        self.assertIs(result.phase2_result.final_state, result.literal_final_state)
        self.assertIs(result.canonicalization.source_state, result.literal_final_state)

    def test_fully_served_tiny_gh_is_independently_feasible(self):
        result = run_gh(
            make_minimal_tiny_instance(),
            GHPhase1Config(0.8),
        )

        self.assertTrue(result.algorithm_completed)
        self.assertTrue(result.canonicalization_completed)
        self.assertTrue(result.independently_feasible)
        self.assertEqual(result.validation_report.violated_families, ())
        self.assertIsNotNone(result.objective_breakdown)
        self.assertAlmostEqual(result.objective_breakdown.gpu_rental_cost, 1.0)
        self.assertAlmostEqual(result.objective_breakdown.model_storage_cost, 0.001)
        self.assertAlmostEqual(result.objective_breakdown.data_storage_cost, 3e-8)
        self.assertAlmostEqual(result.objective_breakdown.delay_penalty, 0.0031)
        self.assertEqual(result.objective_breakdown.unmet_demand_penalty, 0.0)
        self.assertAlmostEqual(result.objective_breakdown.total, 1.00410003)

    def test_d0039_literal_completion_remains_independently_infeasible(self):
        result = run_gh(
            make_two_candidate_instance(),
            GHPhase1Config(0.0),
        )

        self.assertTrue(result.algorithm_completed)
        self.assertTrue(result.canonicalization_completed)
        self.assertFalse(result.independently_feasible)
        self.assertIn(ConstraintFamily.ERROR_SLO, result.validation_report.violated_families)
        self.assertEqual(result.literal_final_state.cumulative_error_by_query[QUERY], 0.05)
        self.assertEqual(
            result.validation_report.error_slo_diagnostics[0].weighted_error,
            0.05,
        )
        self.assertEqual(result.validation_report.error_slo_diagnostics[0].error_limit, 0.025)
        self.assertEqual(result.residual_demand_by_query, {QUERY: 0.0})

    def test_residual_demand_can_still_be_independently_feasible(self):
        result = run_gh(make_residual_instance(), GHPhase1Config(0.0))

        self.assertTrue(result.algorithm_completed)
        self.assertTrue(result.independently_feasible)
        self.assertEqual(result.residual_demand_by_query, {QUERY: 0.5})
        self.assertEqual(result.placement_decisions.unmet_fractions[QUERY], 0.5)
        self.assertEqual(result.demand_consistency[0].lhs, 1.0)
        self.assertEqual(result.demand_consistency[0].signed_residual, 0.0)

    def test_public_completion_status_does_not_encode_feasibility(self):
        feasible = run_gh(make_minimal_tiny_instance(), GHPhase1Config(0.8))
        infeasible = run_gh(make_two_candidate_instance(), GHPhase1Config(0.0))

        self.assertEqual(tuple(GHCompletionStatus), (GHCompletionStatus.ALGORITHM_1_COMPLETED,))
        self.assertEqual(feasible.completion_status, infeasible.completion_status)
        self.assertNotEqual(feasible.independently_feasible, infeasible.independently_feasible)


class GHCanonicalizationTests(unittest.TestCase):
    """Protect every exact dense conversion rule and lattice authority."""

    def test_tiny_mapping_materializes_dense_x_z_q_u_w_y_v(self):
        result = run_gh(make_minimal_tiny_instance(), GHPhase1Config(0.8))
        decisions = result.placement_decisions
        route = (QUERY, "m_small", TIER)
        pair = ("m_small", TIER)

        self.assertEqual(decisions.routing_fractions, {route: 1.0})
        self.assertEqual(decisions.admission_indicators, {route: 1})
        self.assertEqual(decisions.deployment_indicators, {pair: 1})
        self.assertEqual(decisions.unmet_fractions, {QUERY: 0.0})
        self.assertEqual(
            decisions.configuration_selectors,
            {(*pair, 1, 1): 1, (*pair, 2, 1): 0},
        )
        self.assertEqual(decisions.gpu_counts, {pair: 1})
        self.assertEqual(
            decisions.auxiliary_products,
            {(*route, 1, 1): 1.0, (*route, 2, 1): 0.0},
        )
        self.assertEqual(result.canonicalization.mapping_inconsistencies, ())

    def test_phase1_only_deployment_retains_q_w_y_without_routing(self):
        instance = make_minimal_tiny_instance()
        phase1 = run_gh_phase1(instance, GHPhase1Config(0.8))
        initialized = initialize_gh_phase2_state(instance, phase1)

        report = canonicalize_gh_phase2_state(instance, initialized)
        decisions = report.placement_decisions

        self.assertEqual(sum(decisions.routing_fractions.values()), 0.0)
        self.assertEqual(sum(decisions.admission_indicators.values()), 0)
        self.assertEqual(sum(decisions.auxiliary_products.values()), 0.0)
        self.assertEqual(sum(decisions.deployment_indicators.values()), 1)
        self.assertEqual(sum(decisions.configuration_selectors.values()), 1)
        self.assertEqual(sum(decisions.gpu_counts.values()), 1)

    def test_heterogeneous_tier_lattices_exclude_unsupported_configurations(self):
        instance = make_heterogeneous_tiny_instance()
        phase1 = run_gh_phase1(instance, GHPhase1Config(0.8))
        state = initialize_gh_phase2_state(instance, phase1)

        report = canonicalize_gh_phase2_state(instance, state)
        decisions = report.placement_decisions

        self.assertEqual(report.variable_counts.n_x, 8)
        self.assertEqual(report.variable_counts.n_z, 8)
        self.assertEqual(report.variable_counts.n_q, 4)
        self.assertEqual(report.variable_counts.n_u, 2)
        self.assertEqual(report.variable_counts.n_w, 12)
        self.assertEqual(report.variable_counts.n_y, 4)
        self.assertEqual(report.variable_counts.n_v, 24)
        self.assertNotIn(
            ("m_compact", "gpu_budget_int8", 2, 2),
            decisions.configuration_selectors,
        )
        self.assertIn(
            ("m_compact", "gpu_fast_fp16", 2, 2),
            decisions.configuration_selectors,
        )

    def test_fractional_routes_and_products_are_copied_exactly(self):
        result = run_gh(make_two_candidate_instance(), GHPhase1Config(0.0))
        decisions = result.placement_decisions
        route_a = (QUERY, "model_a", TIER)
        route_b = (QUERY, "model_b", TIER)

        self.assertEqual(decisions.routing_fractions[route_a], 0.5)
        self.assertEqual(decisions.routing_fractions[route_b], 0.5)
        self.assertEqual(decisions.admission_indicators[route_a], 1)
        self.assertEqual(decisions.admission_indicators[route_b], 1)
        self.assertEqual(decisions.auxiliary_products[(*route_a, 1, 1)], 0.5)
        self.assertEqual(decisions.auxiliary_products[(*route_b, 1, 1)], 0.5)
        self.assertEqual(decisions.auxiliary_products[(*route_a, 2, 1)], 0.0)
        self.assertEqual(decisions.auxiliary_products[(*route_b, 2, 1)], 0.0)
        self.assertEqual(result.canonicalization.positive_route_count, 2)
        self.assertEqual(result.canonicalization.positive_auxiliary_count, 2)

    def test_binary_counts_and_route_totals_are_explicit(self):
        result = run_gh(make_two_candidate_instance(), GHPhase1Config(0.0))
        counts = result.canonicalization.binary_counts

        self.assertEqual(counts.admission_indicators_equal_one, 2)
        self.assertEqual(counts.deployment_indicators_equal_one, 2)
        self.assertEqual(counts.configuration_selectors_equal_one, 2)
        self.assertEqual(result.canonicalization.route_totals_by_query, {QUERY: 1.0})


class GHAuditAndBoundaryTests(unittest.TestCase):
    """Verify independent audits, determinism, immutability, and isolation."""

    def test_budget_and_demand_audits_reconcile_without_tolerance(self):
        result = run_gh(make_two_candidate_instance(), GHPhase1Config(0.0))

        self.assertTrue(result.budget_consistency.is_exactly_consistent)
        self.assertEqual(result.budget_consistency.absolute_difference_usd, 0.0)
        self.assertEqual(
            result.budget_consistency.recomputed_budget_cost_usd,
            result.budget_consistency.gpu_rental_cost
            + result.budget_consistency.model_storage_cost
            + result.budget_consistency.data_storage_cost,
        )
        self.assertEqual(result.demand_consistency[0].routed_fraction, 1.0)
        self.assertEqual(result.demand_consistency[0].unmet_fraction, 0.0)
        self.assertEqual(result.demand_consistency[0].absolute_residual, 0.0)
        self.assertEqual(
            result.demand_consistency[0].signed_residual,
            result.validation_report.demand_balance_diagnostics[0].signed_residual,
        )

    def test_paper_version_is_explicit_and_changes_only_published_c5(self):
        instance = make_residual_instance()
        v1 = run_gh(instance, GHPhase1Config(0.0), paper_version=PaperVersion.V1)
        v2 = run_gh(instance, GHPhase1Config(0.0), paper_version=PaperVersion.V2)

        self.assertEqual(v1.placement_decisions, v2.placement_decisions)
        self.assertEqual(v1.identity.paper_version, PaperVersion.V1)
        self.assertEqual(v2.identity.paper_version, PaperVersion.V2)
        self.assertEqual(
            v2.objective_breakdown.unmet_demand_penalty,
            instance.horizon_hours * v1.objective_breakdown.unmet_demand_penalty,
        )

    def test_repeated_results_are_deterministic_and_deeply_immutable(self):
        instance = make_minimal_tiny_instance()
        first = run_gh(instance, GHPhase1Config(0.8))
        second = run_gh(instance, GHPhase1Config(0.8))

        self.assertEqual(first, second)
        self.assertTrue(
            {"literal_final_state", "objective_breakdown", "residual_demand_by_query"}.isdisjoint(
                item.name for item in fields(first)
            )
        )
        self.assertIs(first.objective_breakdown, first.validation_report.objective_breakdown)
        self.assertIs(first.literal_final_state, first.phase2_result.final_state)
        self.assertIs(first.residual_demand_by_query, first.phase2_result.residual_demand_by_query)
        with self.assertRaises(FrozenInstanceError):
            first.completion_status = GHCompletionStatus.ALGORITHM_1_COMPLETED
        with self.assertRaises(TypeError):
            first.residual_demand_by_query[QUERY] = 1.0
        with self.assertRaises(TypeError):
            first.canonicalization.route_totals_by_query[QUERY] = 0.0
        with self.assertRaises(TypeError):
            first.placement_decisions.routing_fractions[(QUERY, "m_small", TIER)] = 0.0



if __name__ == "__main__":
    unittest.main()
