"""Hand-derived tests for one-candidate GH Phase-2 commitment."""

import unittest
from dataclasses import FrozenInstanceError, replace

from llm_resource_allocation.algorithms import (
    GHPhase1Config,
    GHPhase2CommitRejectionReason,
    GHPhase2ConfigurationSource,
    GHPhase2Deployment,
    GHPhase2State,
    commit_gh_phase2_candidate,
    compute_gh_phase2_candidate_allocation,
    evaluate_gh_phase2_query_candidates,
    initialize_gh_phase2_state,
    run_gh_phase1,
)
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ROUTE[1:]


def inactive_state(instance):
    """Enter Phase 2 without Phase-1 activations."""

    phase1 = run_gh_phase1(instance, GHPhase1Config(0.0))
    return phase1, initialize_gh_phase2_state(instance, phase1)


def sole_candidate(instance, state, model_id="m_small"):
    """Return one explicitly selected rankable candidate."""

    evaluation = evaluate_gh_phase2_query_candidates(instance, state, "q_text")
    return next(
        candidate
        for candidate in evaluation.ranked_candidates
        if candidate.model_id == model_id
    )


def make_two_model_instance():
    """Duplicate the minimal model with complete transparent coefficients."""

    base = make_minimal_tiny_instance()
    old_model = replace(base.models[0], id="m_old")
    new_model = replace(base.models[0], id="m_new")
    models = (old_model, new_model)
    triples = {
        ("q_text", model.id, "gpu_basic_fp16"): 0.10 for model in models
    }
    return replace(
        base,
        id="gh_phase2_commit_two_models",
        models=models,
        base_error_rate={
            ("q_text", model.id): 0.05 for model in models
        },
        compute_delay_seconds_per_token=triples,
        communication_delay_seconds_per_token={
            key: 0.01 for key in triples
        },
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
    )


class GHPhase2AllocationTests(unittest.TestCase):
    """Protect the literal Algorithm-1 line-18 allocation."""

    def test_public_boundaries_reject_mismatched_candidate_gpu_evidence(self):
        instance = make_minimal_tiny_instance()
        _, state = inactive_state(instance)
        candidate = sole_candidate(instance, state)
        changed = replace(
            candidate,
            m2_inputs=replace(candidate.m2_inputs, current_gpu_count=1),
        )
        with self.assertRaisesRegex(ValueError, "M2 GPU count"):
            compute_gh_phase2_candidate_allocation(state, changed)
        with self.assertRaisesRegex(ValueError, "declared state"):
            commit_gh_phase2_candidate(instance, state, changed)

    def test_proposal_is_minimum_of_unmet_and_effective_coverage(self):
        instance = make_minimal_tiny_instance()
        _, state = inactive_state(instance)
        limited_state = replace(
            state, unmet_fraction_by_query={"q_text": 0.4}
        )
        candidate = sole_candidate(instance, limited_state)

        allocation = compute_gh_phase2_candidate_allocation(
            limited_state, candidate
        )

        self.assertEqual(allocation.unserved_fraction_before, 0.4)
        self.assertEqual(allocation.candidate_coverage_fraction, 1.0)
        self.assertEqual(allocation.proposed_routing_fraction, 0.4)

    def test_zero_proposal_rejects_without_resource_checks(self):
        instance = make_minimal_tiny_instance()
        _, state = inactive_state(instance)
        exhausted = replace(
            state, unmet_fraction_by_query={"q_text": 0.0}
        )
        candidate = sole_candidate(instance, exhausted)

        result = commit_gh_phase2_candidate(instance, exhausted, candidate)

        self.assertFalse(result.committed)
        self.assertIs(result.state_after, exhausted)
        self.assertEqual(
            result.diagnostic.rejection_reasons,
            (GHPhase2CommitRejectionReason.ZERO_ALLOCATION,),
        )
        self.assertEqual(result.diagnostic.gpu_memory_diagnostics, ())
        self.assertEqual(result.diagnostic.compute_capacity_diagnostics, ())
        self.assertEqual(result.diagnostic.storage_capacity_diagnostics, ())
        self.assertIsNone(result.diagnostic.budget_comparison)


class GHPhase2SuccessfulCommitTests(unittest.TestCase):
    """Cover inactive, active-current, M3, and partial success branches."""

    def test_inactive_pair_success_has_hand_calculated_state_and_budget(self):
        instance = make_minimal_tiny_instance()
        phase1, state = inactive_state(instance)
        candidate = sole_candidate(instance, state)
        phase1_before = phase1
        candidate_before = candidate

        result = commit_gh_phase2_candidate(instance, state, candidate)

        self.assertTrue(result.committed)
        self.assertIsNot(result.state_after, state)
        self.assertEqual(result.state_after.routing_fraction_by_candidate, {ROUTE: 1.0})
        self.assertEqual(result.state_after.unmet_fraction_by_query["q_text"], 0.0)
        self.assertEqual(result.state_after.remaining_demand_by_query["q_text"], 0.0)
        self.assertEqual(result.state_after.cumulative_error_by_query["q_text"], 0.05)
        self.assertAlmostEqual(
            result.state_after.cumulative_delay_seconds_by_query["q_text"], 0.31
        )
        self.assertEqual(result.state_after.deployments[0].gpu_count, 1)
        self.assertAlmostEqual(
            result.state_after.accumulated_budget_cost_usd, 1.00100003
        )
        budget = result.diagnostic.budget_comparison
        self.assertIsNotNone(budget)
        self.assertEqual(budget.incremental_gpu_rental_cost_usd, 1.0)
        self.assertEqual(budget.incremental_model_storage_cost_usd, 0.001)
        self.assertAlmostEqual(budget.incremental_data_storage_cost_usd, 0.00000003)
        self.assertEqual(phase1, phase1_before)
        self.assertEqual(candidate, candidate_before)
        self.assertEqual(state.routing_fraction_by_candidate, {})

    def test_active_current_commit_does_not_change_deployment_or_gpus(self):
        instance = make_minimal_tiny_instance()
        phase1 = run_gh_phase1(instance, GHPhase1Config(0.5))
        state = initialize_gh_phase2_state(instance, phase1)
        candidate = sole_candidate(instance, state)

        result = commit_gh_phase2_candidate(instance, state, candidate)

        self.assertTrue(result.committed)
        deployment = result.diagnostic.prospective_deployment
        self.assertEqual(
            deployment.source,
            GHPhase2ConfigurationSource.CURRENT_ACTIVE_CONFIGURATION,
        )
        self.assertEqual(deployment.incremental_gpu_count, 0)
        self.assertEqual(result.state_after.deployments, state.deployments)
        self.assertEqual(
            result.diagnostic.budget_comparison.incremental_gpu_rental_cost_usd, 0.0
        )

    def test_m3_commit_upgrades_deployment_and_routes_in_one_transition(self):
        base = make_minimal_tiny_instance()
        query = replace(base.queries[0], delay_slo_seconds=0.2)
        instance = replace(base, queries=(query,))
        state = GHPhase2State(
            deployments=(
                GHPhase2Deployment(
                    "m_small", "gpu_basic_fp16", ParallelismConfig(1, 1)
                ),
            ),
            query_order=("q_text",),
            routing_fraction_by_candidate={},
            remaining_demand_by_query={"q_text": 1.0},
            unmet_fraction_by_query={"q_text": 1.0},
            cumulative_error_by_query={"q_text": 0.0},
            cumulative_delay_seconds_by_query={"q_text": 0.0},
            accumulated_budget_cost_usd=1.0,
            remaining_budget_usd=19.0,
        )
        candidate = sole_candidate(instance, state)

        result = commit_gh_phase2_candidate(instance, state, candidate)

        self.assertTrue(result.committed)
        self.assertEqual(
            candidate.candidate_configuration.source,
            GHPhase2ConfigurationSource.M3_ACTIVE_PAIR_UPGRADE,
        )
        self.assertEqual(result.state_after.deployments[0].configuration, ParallelismConfig(2, 1))
        self.assertEqual(result.state_after.deployments[0].gpu_count, 2)
        self.assertEqual(result.state_after.routing_fraction_by_candidate[ROUTE], 1.0)
        self.assertAlmostEqual(
            result.state_after.cumulative_delay_seconds_by_query["q_text"], 0.16
        )
        self.assertEqual(
            result.diagnostic.budget_comparison.incremental_gpu_rental_cost_usd, 1.0
        )

    def test_partial_coverage_commits_xbar_and_leaves_positive_unmet(self):
        base = make_minimal_tiny_instance()
        query = replace(base.queries[0], error_slo=0.025)
        instance = replace(base, queries=(query,))
        _, state = inactive_state(instance)
        candidate = sole_candidate(instance, state)

        result = commit_gh_phase2_candidate(instance, state, candidate)

        self.assertTrue(result.committed)
        self.assertEqual(
            candidate.m2_evaluation.coverage_diagnostic.effective_coverage, 0.5
        )
        self.assertEqual(result.state_after.routing_fraction_by_candidate[ROUTE], 0.5)
        self.assertEqual(result.state_after.unmet_fraction_by_query["q_text"], 0.5)
        self.assertEqual(result.state_after.remaining_demand_by_query["q_text"], 0.5)
        self.assertEqual(result.state_after.cumulative_error_by_query["q_text"], 0.025)
        self.assertAlmostEqual(
            result.state_after.cumulative_delay_seconds_by_query["q_text"], 0.155
        )

    def test_unmet_limited_commit_uses_u_not_larger_xbar(self):
        instance = make_minimal_tiny_instance()
        _, state = inactive_state(instance)
        state = replace(state, unmet_fraction_by_query={"q_text": 0.4})
        candidate = sole_candidate(instance, state)

        result = commit_gh_phase2_candidate(instance, state, candidate)

        self.assertTrue(result.committed)
        self.assertEqual(result.diagnostic.allocation.candidate_coverage_fraction, 1.0)
        self.assertEqual(result.diagnostic.allocation.proposed_routing_fraction, 0.4)
        self.assertEqual(result.state_after.unmet_fraction_by_query["q_text"], 0.0)
        self.assertEqual(result.state_after.remaining_demand_by_query["q_text"], 0.6)


class GHPhase2RejectionTests(unittest.TestCase):
    """Reject each line-16 family prospectively with full evidence."""

    def assert_rejected_unchanged(self, instance, expected_reason):
        """Evaluate the sole candidate and assert atomic rejection."""

        _, state = inactive_state(instance)
        candidate = sole_candidate(instance, state)
        result = commit_gh_phase2_candidate(instance, state, candidate)
        self.assertFalse(result.committed)
        self.assertIs(result.state_after, state)
        self.assertIn(expected_reason, result.diagnostic.rejection_reasons)
        self.assertEqual(state.routing_fraction_by_candidate, {})
        return result

    def test_gpu_memory_rejection(self):
        base = make_minimal_tiny_instance()
        model = replace(base.models[0], kv_cache_gb_per_token=1.0)
        tier = replace(base.gpu_tiers[0], memory_capacity_gb=1.0)
        instance = replace(base, models=(model,), gpu_tiers=(tier,))

        result = self.assert_rejected_unchanged(
            instance, GHPhase2CommitRejectionReason.GPU_MEMORY
        )

        residual = result.diagnostic.gpu_memory_diagnostics[0]
        self.assertEqual(residual.printed_memory_lhs, 4.0)
        self.assertEqual(residual.available_memory_rhs_gb, 1.0)
        self.assertEqual(residual.slack, -3.0)

    def test_compute_capacity_rejection(self):
        base = make_minimal_tiny_instance()
        tier = replace(base.gpu_tiers[0], compute_throughput_tflops=0.000001)
        instance = replace(base, gpu_tiers=(tier,))

        result = self.assert_rejected_unchanged(
            instance, GHPhase2CommitRejectionReason.COMPUTE_CAPACITY
        )

        residual = result.diagnostic.compute_capacity_diagnostics[0]
        self.assertEqual(residual.required_compute, 0.03)
        self.assertEqual(residual.available_compute, 0.00324)
        self.assertLess(residual.slack, 0.0)

    def test_storage_rejection(self):
        base = make_minimal_tiny_instance()
        instance = replace(base, storage_capacity_gb=0.5)

        result = self.assert_rejected_unchanged(
            instance, GHPhase2CommitRejectionReason.STORAGE
        )

        residual = result.diagnostic.storage_capacity_diagnostics[0]
        self.assertAlmostEqual(residual.printed_storage_lhs, 1.00003)
        self.assertEqual(residual.storage_limit_gb, 0.5)
        self.assertLess(residual.slack, 0.0)

    def test_global_budget_rejection_uses_c1_c2_c3(self):
        base = make_minimal_tiny_instance()
        instance = replace(base, cost_budget_usd=1.0005)

        result = self.assert_rejected_unchanged(
            instance, GHPhase2CommitRejectionReason.GLOBAL_BUDGET
        )

        budget = result.diagnostic.budget_comparison
        self.assertAlmostEqual(budget.prospective_budget_diagnostic.total_budget_use, 1.00100003)
        self.assertEqual(budget.prospective_budget_diagnostic.budget_limit, 1.0005)
        self.assertLess(budget.prospective_budget_diagnostic.slack, 0.0)

    def test_simultaneous_failures_are_all_retained(self):
        base = make_minimal_tiny_instance()
        model = replace(base.models[0], kv_cache_gb_per_token=1.0)
        tier = replace(
            base.gpu_tiers[0],
            memory_capacity_gb=1.0,
            compute_throughput_tflops=0.000001,
        )
        instance = replace(base, models=(model,), gpu_tiers=(tier,))

        result = self.assert_rejected_unchanged(
            instance, GHPhase2CommitRejectionReason.GPU_MEMORY
        )

        self.assertEqual(
            result.diagnostic.rejection_reasons,
            (
                GHPhase2CommitRejectionReason.GPU_MEMORY,
                GHPhase2CommitRejectionReason.COMPUTE_CAPACITY,
            ),
        )


class GHPhase2RoutingAndBoundaryTests(unittest.TestCase):
    """Protect sparse additivity, immutability, determinism, and isolation."""

    def test_previous_sparse_route_is_preserved_when_new_route_is_added(self):
        instance = make_two_model_instance()
        prior_cost = 1.0010000075
        state = GHPhase2State(
            deployments=(
                GHPhase2Deployment(
                    "m_old", "gpu_basic_fp16", ParallelismConfig(1, 1)
                ),
            ),
            query_order=("q_text",),
            routing_fraction_by_candidate={
                ("q_text", "m_old", "gpu_basic_fp16"): 0.25
            },
            remaining_demand_by_query={"q_text": 0.75},
            unmet_fraction_by_query={"q_text": 0.75},
            cumulative_error_by_query={"q_text": 0.0125},
            cumulative_delay_seconds_by_query={"q_text": 0.0775},
            accumulated_budget_cost_usd=prior_cost,
            remaining_budget_usd=instance.cost_budget_usd - prior_cost,
        )
        candidate = sole_candidate(instance, state, "m_new")

        result = commit_gh_phase2_candidate(instance, state, candidate)

        self.assertTrue(result.committed)
        self.assertEqual(
            result.state_after.routing_fraction_by_candidate,
            {
                ("q_text", "m_old", "gpu_basic_fp16"): 0.25,
                ("q_text", "m_new", "gpu_basic_fp16"): 0.75,
            },
        )
        self.assertEqual(
            state.routing_fraction_by_candidate,
            {("q_text", "m_old", "gpu_basic_fp16"): 0.25},
        )

    def test_success_result_and_nested_state_are_immutable(self):
        instance = make_minimal_tiny_instance()
        _, state = inactive_state(instance)
        result = commit_gh_phase2_candidate(
            instance, state, sole_candidate(instance, state)
        )

        with self.assertRaises(FrozenInstanceError):
            result.state_after = state
        with self.assertRaises(TypeError):
            result.state_after.routing_fraction_by_candidate[ROUTE] = 0.5

    def test_identical_calls_are_deterministic(self):
        instance = make_minimal_tiny_instance()
        _, state = inactive_state(instance)
        candidate = sole_candidate(instance, state)

        first = commit_gh_phase2_candidate(instance, state, candidate)
        second = commit_gh_phase2_candidate(instance, state, candidate)

        self.assertEqual(first, second)
        self.assertEqual(state.routing_fraction_by_candidate, {})



if __name__ == "__main__":
    unittest.main()
