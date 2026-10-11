"""Hand-derived tests for pure GH Phase-1 Eqs. (13)-(14) primitives."""

import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    M1ConfigurationDiagnostic,
    Phase1QueryExclusionReason,
    Phase1NonRankableReason,
    compute_phase1_activation_score,
    compute_phase1_pair_coverage,
    evaluate_phase1_candidate,
    rank_phase1_candidates,
    select_minimum_feasible_parallelism,
)
from llm_resource_allocation.algorithms import m1 as m1_module
from llm_resource_allocation.data.tiny_instances import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig


MODEL_ID = "m_small"


def make_phase1_instance(
    *,
    tier_ids_and_prices: tuple[tuple[str, float], ...] = (("tier_a", 3.0),),
    tensor_degrees: tuple[int, ...] = (1, 2),
    pipeline_depths: tuple[int, ...] = (1,),
    model_weight_gb: float = 1.0,
    memory_capacity_gb: float = 8.0,
):
    """Return a transparent 4-query fixture with query-specific M1/error cases."""

    base = make_minimal_tiny_instance()
    query_template = base.queries[0]
    queries = (
        replace(
            query_template,
            id="q_easy",
            delay_slo_seconds=0.5,
            error_slo=0.25,
        ),
        replace(
            query_template,
            id="q_same",
            delay_slo_seconds=0.5,
            error_slo=0.25,
        ),
        replace(
            query_template,
            id="q_tight",
            delay_slo_seconds=0.2,
            error_slo=0.25,
        ),
        replace(
            query_template,
            id="q_error",
            delay_slo_seconds=0.5,
            error_slo=0.10,
        ),
    )
    model = replace(base.models[0], weight_gb=model_weight_gb)
    tiers = tuple(
        replace(
            base.gpu_tiers[0],
            id=tier_id,
            hourly_rental_usd=price,
            memory_capacity_gb=memory_capacity_gb,
            tensor_parallel_degrees=tensor_degrees,
            pipeline_parallel_depths=pipeline_depths,
        )
        for tier_id, price in tier_ids_and_prices
    )
    pairs = {(query.id, MODEL_ID): 0.20 for query in queries}
    triples = tuple(
        (query.id, MODEL_ID, tier.id) for query in queries for tier in tiers
    )
    return replace(
        base,
        id="phase1_fixture",
        queries=queries,
        models=(model,),
        gpu_tiers=tiers,
        base_error_rate=pairs,
        compute_delay_seconds_per_token={key: 0.10 for key in triples},
        communication_delay_seconds_per_token={key: 0.01 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
        horizon_hours=2.0,
    )


def m1_evidence(instance, tier_id: str, query_ids: tuple[str, ...]):
    """Build the explicit M1 input required by the Phase-1 primitives."""

    return tuple(
        select_minimum_feasible_parallelism(
            instance,
            query_id=query_id,
            model_id=MODEL_ID,
            tier_id=tier_id,
        )
        for query_id in query_ids
    )


def evaluate(instance, tier_id: str, query_ids: tuple[str, ...]):
    """Evaluate one candidate using separately obtained M1 evidence."""

    return evaluate_phase1_candidate(
        instance,
        MODEL_ID,
        tier_id,
        query_ids,
        m1_evidence(instance, tier_id, query_ids),
    )


class Phase1CoverageHandCases(unittest.TestCase):
    """Verify exact Eq. (13) membership and uncovered-set semantics."""

    def test_single_query_covered_and_activation_value(self):
        """M1 selects (1,1); Cost=2*3*1=6 and benefit/cost=1/6."""

        result = evaluate(make_phase1_instance(), "tier_a", ("q_easy",))

        self.assertEqual(result.coverage.covered_query_ids, ("q_easy",))
        self.assertEqual(
            result.coverage.common_configuration, ParallelismConfig(1, 1)
        )
        self.assertEqual(result.activation.activation_cost_usd, 6.0)
        self.assertAlmostEqual(result.activation.coverage_per_cost, 1.0 / 6.0)

    def test_m1_infeasible_query_produces_empty_nonrankable_candidate(self):
        instance = make_phase1_instance(
            model_weight_gb=100.0,
            memory_capacity_gb=8.0,
        )
        result = evaluate(instance, "tier_a", ("q_easy",))

        self.assertEqual(result.coverage.covered_query_ids, ())
        self.assertEqual(
            result.coverage.query_diagnostics[0].exclusion_reasons,
            (Phase1QueryExclusionReason.M1_INFEASIBLE,),
        )
        self.assertEqual(result.activation.activation_cost_usd, None)
        self.assertEqual(
            result.activation.non_rankable_reasons,
            (Phase1NonRankableReason.EMPTY_COVERAGE,),
        )

    def test_two_queries_with_same_m1_configuration_are_both_covered(self):
        result = evaluate(
            make_phase1_instance(), "tier_a", ("q_easy", "q_same")
        )

        self.assertEqual(
            tuple(
                diagnostic.m1_selection.selected_configuration
                for diagnostic in result.coverage.query_diagnostics
            ),
            (ParallelismConfig(1, 1), ParallelismConfig(1, 1)),
        )
        self.assertEqual(
            result.coverage.covered_query_ids, ("q_easy", "q_same")
        )
        self.assertEqual(result.activation.activation_cost_usd, 6.0)
        self.assertAlmostEqual(result.activation.coverage_per_cost, 2.0 / 6.0)

    def test_different_m1_configurations_share_one_eq14_sized_shape(self):
        """q_easy needs 1 GPU, q_tight needs 2; common TP2 costs 12 dollars."""

        result = evaluate(
            make_phase1_instance(), "tier_a", ("q_easy", "q_tight")
        )

        self.assertEqual(
            tuple(
                diagnostic.m1_selection.selected_configuration
                for diagnostic in result.coverage.query_diagnostics
            ),
            (ParallelismConfig(1, 1), ParallelismConfig(2, 1)),
        )
        self.assertEqual(result.coverage.required_gpu_count, 2)
        self.assertEqual(
            result.coverage.common_configuration, ParallelismConfig(2, 1)
        )
        self.assertEqual(result.activation.activation_cost_usd, 12.0)
        self.assertAlmostEqual(result.activation.coverage_per_cost, 1.0 / 6.0)

    def test_partial_coverage_retains_exact_ids_and_order(self):
        result = evaluate(
            make_phase1_instance(),
            "tier_a",
            ("q_easy", "q_error", "q_tight"),
        )

        self.assertEqual(
            result.coverage.covered_query_ids, ("q_easy", "q_tight")
        )
        self.assertEqual(
            result.coverage.excluded_query_ids, ("q_error",)
        )
        rejected = result.coverage.query_diagnostics[1]
        self.assertEqual(rejected.effective_error, 0.20)
        self.assertEqual(rejected.error_slo, 0.10)
        self.assertEqual(
            rejected.exclusion_reasons,
            (Phase1QueryExclusionReason.ERROR_SLO_EXCEEDED,),
        )

    def test_already_covered_query_cannot_reappear(self):
        result = evaluate(
            make_phase1_instance(), "tier_a", ("q_tight", "q_error")
        )

        self.assertNotIn("q_easy", result.coverage.uncovered_query_ids)
        self.assertNotIn("q_easy", result.coverage.covered_query_ids)
        self.assertLessEqual(
            set(result.coverage.covered_query_ids),
            set(result.coverage.uncovered_query_ids),
        )

    def test_empty_uncovered_tuple_is_a_safe_empty_result(self):
        result = evaluate(make_phase1_instance(), "tier_a", ())

        self.assertTrue(result.coverage.is_empty)
        self.assertEqual(result.coverage.query_diagnostics, ())
        self.assertFalse(result.is_rankable)

    def test_duplicate_unknown_and_missing_evidence_fail_explicitly(self):
        instance = make_phase1_instance()
        with self.assertRaisesRegex(ValueError, "duplicate uncovered"):
            compute_phase1_pair_coverage(
                instance,
                MODEL_ID,
                "tier_a",
                ("q_easy", "q_easy"),
                (),
            )
        with self.assertRaisesRegex(ValueError, "unknown uncovered"):
            compute_phase1_pair_coverage(
                instance, MODEL_ID, "tier_a", ("q_unknown",), ()
            )
        with self.assertRaisesRegex(ValueError, "missing M1 evidence"):
            compute_phase1_pair_coverage(
                instance, MODEL_ID, "tier_a", ("q_easy",), ()
            )

    def test_public_coverage_boundaries_reject_incomplete_external_m1_diagnostics(self):
        instance = make_phase1_instance()
        selection = m1_evidence(instance, "tier_a", ("q_easy",))[0]
        incomplete = replace(selection, configuration_diagnostics=())
        for evaluate_boundary in (compute_phase1_pair_coverage, evaluate_phase1_candidate):
            with self.subTest(boundary=evaluate_boundary.__name__):
                with self.assertRaisesRegex(ValueError, "M1 diagnostics"):
                    evaluate_boundary(
                        instance, MODEL_ID, "tier_a", ("q_easy",), (incomplete,)
                    )


class Phase1CommonConfigurationAuditTests(unittest.TestCase):
    """Make the v2 prose/Eq. (14) shape closure executable and visible."""

    def test_common_shape_diagnostics_contain_only_tier_authority(self):
        instance = make_phase1_instance(tensor_degrees=(1, 4))
        result = evaluate(instance, "tier_a", ("q_easy",))

        diagnosed = tuple(
            diagnostic.configuration
            for diagnostic in result.coverage.common_configuration_diagnostics
        )
        self.assertEqual(
            diagnosed,
            (ParallelismConfig(1, 1), ParallelismConfig(4, 1)),
        )
        self.assertNotIn(ParallelismConfig(2, 1), diagnosed)

    def test_equal_product_common_shape_uses_tier_semantic_order(self):
        instance = make_phase1_instance(
            tensor_degrees=(1, 2),
            pipeline_depths=(2, 1),
            model_weight_gb=12.0,
            memory_capacity_gb=8.0,
        )
        result = evaluate(instance, "tier_a", ("q_easy", "q_same"))

        self.assertEqual(result.coverage.required_gpu_count, 2)
        self.assertEqual(
            result.coverage.common_configuration, ParallelismConfig(1, 2)
        )

    def test_reversing_tier_order_changes_only_equal_product_shape(self):
        common = dict(model_weight_gb=12.0, memory_capacity_gb=8.0)
        pp_first = evaluate(
            make_phase1_instance(
                tensor_degrees=(1, 2), pipeline_depths=(2, 1), **common
            ),
            "tier_a",
            ("q_easy", "q_same"),
        )
        tp_first = evaluate(
            make_phase1_instance(
                tensor_degrees=(2, 1), pipeline_depths=(1, 2), **common
            ),
            "tier_a",
            ("q_easy", "q_same"),
        )

        self.assertEqual(
            pp_first.coverage.common_configuration, ParallelismConfig(1, 2)
        )
        self.assertEqual(
            tp_first.coverage.common_configuration, ParallelismConfig(2, 1)
        )
        self.assertEqual(pp_first.coverage.covered_query_ids, tp_first.coverage.covered_query_ids)
        self.assertEqual(pp_first.activation, tp_first.activation)

    def test_inconsistent_external_m1_evidence_is_nonrankable_not_silently_shrunk(self):
        """Exercise the defensive policy for the paper's unspecified failure case."""

        instance = make_phase1_instance()
        selections = list(m1_evidence(instance, "tier_a", ("q_easy", "q_tight")))
        easy = selections[0]
        altered = tuple(
            replace(
                diagnostic,
                fixed_configuration_delay_seconds=(
                    diagnostic.delay_slo_seconds + 1.0
                    if diagnostic.configuration == ParallelismConfig(2, 1)
                    else diagnostic.fixed_configuration_delay_seconds
                ),
            )
            for diagnostic in easy.configuration_diagnostics
        )
        selections[0] = replace(easy, configuration_diagnostics=altered)

        result = evaluate_phase1_candidate(
            instance,
            MODEL_ID,
            "tier_a",
            ("q_easy", "q_tight"),
            tuple(selections),
        )

        self.assertEqual(
            result.coverage.covered_query_ids, ("q_easy", "q_tight")
        )
        self.assertIsNone(result.coverage.common_configuration)
        self.assertEqual(result.activation.activation_cost_usd, 12.0)
        self.assertEqual(
            result.activation.non_rankable_reasons,
            (
                Phase1NonRankableReason.NO_COMMON_CONFIGURATION_AT_EQ14_GPU_COUNT,
            ),
        )


class Phase1ScoringAndRankingTests(unittest.TestCase):
    """Protect Eq. (14), descending ranking, and stable caller-order ties."""

    def make_three_tier_evaluations(self, equal_prices: bool = False):
        prices = (1.0, 1.0, 1.0) if equal_prices else (3.0, 1.0, 2.0)
        instance = make_phase1_instance(
            tier_ids_and_prices=tuple(
                zip(("tier_a", "tier_b", "tier_c"), prices)
            )
        )
        query_ids = ("q_easy", "q_tight")
        evaluations = tuple(
            evaluate(instance, tier.id, query_ids) for tier in instance.gpu_tiers
        )
        return instance, evaluations

    def test_three_candidates_rank_by_descending_coverage_per_cost(self):
        _, evaluations = self.make_three_tier_evaluations()

        ranked = rank_phase1_candidates(evaluations)

        self.assertEqual(
            tuple(result.tier_id for result in ranked),
            ("tier_b", "tier_c", "tier_a"),
        )
        self.assertNotEqual(ranked[0], evaluations[0])

    def test_exact_tie_preserves_supplied_order(self):
        _, evaluations = self.make_three_tier_evaluations(equal_prices=True)
        supplied = (evaluations[2], evaluations[0], evaluations[1])

        self.assertEqual(rank_phase1_candidates(supplied), supplied)

    def test_reversing_tied_candidates_changes_only_representative_order(self):
        _, evaluations = self.make_three_tier_evaluations(equal_prices=True)
        forward = rank_phase1_candidates(evaluations)
        reverse = rank_phase1_candidates(tuple(reversed(evaluations)))

        self.assertEqual(forward, evaluations)
        self.assertEqual(reverse, tuple(reversed(evaluations)))
        self.assertEqual(
            {result.activation.coverage_per_cost for result in forward},
            {result.activation.coverage_per_cost for result in reverse},
        )

    def test_zero_price_nonempty_candidate_has_infinite_maximization_score(self):
        instance = make_phase1_instance(tier_ids_and_prices=(("tier_a", 0.0),))
        result = evaluate(instance, "tier_a", ("q_easy",))

        self.assertEqual(result.activation.activation_cost_usd, 0.0)
        self.assertEqual(result.activation.coverage_per_cost, float("inf"))
        self.assertTrue(result.is_rankable)

    def test_public_score_primitive_matches_composed_candidate(self):
        instance = make_phase1_instance()
        result = evaluate(instance, "tier_a", ("q_easy", "q_tight"))

        self.assertEqual(
            result.activation,
            compute_phase1_activation_score(instance, result.coverage),
        )


class Phase1IsolationAndImmutabilityTests(unittest.TestCase):
    """Keep the milestone outside M2, M3, solver, and state mutation."""

    def test_m1_evidence_is_reused_without_recomputation(self):
        instance = make_phase1_instance()
        query_ids = ("q_easy", "q_tight")
        evidence = m1_evidence(instance, "tier_a", query_ids)

        with patch.object(
            m1_module,
            "select_minimum_feasible_parallelism",
            side_effect=AssertionError("M1 was recomputed"),
        ):
            result = evaluate_phase1_candidate(
                instance, MODEL_ID, "tier_a", query_ids, evidence
            )

        self.assertEqual(result.coverage.covered_query_ids, query_ids)

    def test_results_are_immutable_deterministic_and_inputs_unchanged(self):
        instance = make_phase1_instance()
        query_ids = ("q_easy", "q_tight")
        evidence = m1_evidence(instance, "tier_a", query_ids)

        first = evaluate_phase1_candidate(
            instance, MODEL_ID, "tier_a", query_ids, evidence
        )
        second = evaluate_phase1_candidate(
            instance, MODEL_ID, "tier_a", query_ids, evidence
        )

        self.assertEqual(first, second)
        self.assertEqual(query_ids, ("q_easy", "q_tight"))
        self.assertEqual(evidence, m1_evidence(instance, "tier_a", query_ids))
        with self.assertRaises((FrozenInstanceError, TypeError)):
            first.coverage.covered_query_ids = ()



if __name__ == "__main__":
    unittest.main()
