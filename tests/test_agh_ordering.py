"""Hand-derived tests for the bounded Milestone-6A2 ordering layer."""

import random
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import (
    DETERMINISTIC_AGH_ORDERING_SPECS,
    AGHOrderingCriterion,
    AGHOrderingDirection,
    AGHOrderingSource,
    GHPhase1Config,
    build_all_deterministic_agh_orderings,
    build_seeded_random_agh_ordering,
    compute_adaptive_agh_random_start_count,
    initialize_gh_phase2_state,
    run_gh,
    run_gh_phase1,
    run_gh_phase2,
)
from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.algorithms import agh_ordering as ordering_module


def make_ordering_instance():
    """Return five semantic query IDs with hand-distinct ordering scalars."""

    base = make_minimal_tiny_instance()
    template = base.queries[0]
    query_values = (
        # id, lambda, phi, theta, epsilon
        ("q_a", 5.0, 20.0, 3e-6, 0.04),
        ("q_b", 1.0, 50.0, 5e-6, 0.02),
        ("q_c", 4.0, 10.0, 1e-6, 0.05),
        ("q_d", 1.0, 40.0, 4e-6, 0.01),
        ("q_e", 3.0, 30.0, 2e-6, 0.03),
    )
    queries = tuple(
        replace(
            template,
            id=query_id,
            arrival_rate_per_hour=arrival,
            unmet_penalty_usd_per_hour=penalty,
            storage_gb_per_token=storage,
            error_slo=error_slo,
        )
        for query_id, arrival, penalty, storage, error_slo in query_values
    )
    model = base.models[0]
    tier = base.gpu_tiers[0]
    triples = tuple((query.id, model.id, tier.id) for query in queries)
    return replace(
        base,
        id="agh_ordering_hand_fixture",
        queries=queries,
        base_error_rate={(query.id, model.id): 0.005 for query in queries},
        compute_delay_seconds_per_token={key: 0.01 for key in triples},
        communication_delay_seconds_per_token={key: 0.001 for key in triples},
        compute_gflop_per_token={key: 1.0 for key in triples},
        kv_cache_residency_seconds={key: 1.0 for key in triples},
    )


class DeterministicAGHOrderingTests(unittest.TestCase):
    def test_collection_validates_the_instance_once_for_all_eight_orderings(self):
        instance = make_ordering_instance()
        with patch.object(
            ordering_module,
            "validate_problem_instance",
            wraps=ordering_module.validate_problem_instance,
        ) as validate:
            result = build_all_deterministic_agh_orderings(instance)

        validate.assert_called_once_with(instance)
        self.assertEqual(len(result), 8)

    """Pin the eight-spec sequence and exact hand-derived orders."""

    def test_public_specs_have_exact_approved_sequence(self):
        observed = tuple(
            (spec.criterion, spec.direction, spec.label)
            for spec in DETERMINISTIC_AGH_ORDERING_SPECS
        )
        expected = (
            (AGHOrderingCriterion.ARRIVAL_RATE, AGHOrderingDirection.ASCENDING, "lambda_ascending"),
            (AGHOrderingCriterion.ARRIVAL_RATE, AGHOrderingDirection.DESCENDING, "lambda_descending"),
            (AGHOrderingCriterion.UNMET_PENALTY, AGHOrderingDirection.ASCENDING, "phi_ascending"),
            (AGHOrderingCriterion.UNMET_PENALTY, AGHOrderingDirection.DESCENDING, "phi_descending"),
            (AGHOrderingCriterion.STORAGE_PER_TOKEN, AGHOrderingDirection.ASCENDING, "theta_ascending"),
            (AGHOrderingCriterion.STORAGE_PER_TOKEN, AGHOrderingDirection.DESCENDING, "theta_descending"),
            (AGHOrderingCriterion.ERROR_SLO, AGHOrderingDirection.ASCENDING, "epsilon_ascending"),
            (AGHOrderingCriterion.ERROR_SLO, AGHOrderingDirection.DESCENDING, "epsilon_descending"),
        )
        self.assertEqual(observed, expected)

    def test_all_eight_orders_match_hand_derived_query_ids(self):
        orderings = build_all_deterministic_agh_orderings(make_ordering_instance())
        observed = {item.deterministic_spec.label: item.query_ids for item in orderings}
        expected = {
            "lambda_ascending": ("q_b", "q_d", "q_e", "q_c", "q_a"),
            "lambda_descending": ("q_a", "q_c", "q_e", "q_b", "q_d"),
            "phi_ascending": ("q_c", "q_a", "q_e", "q_d", "q_b"),
            "phi_descending": ("q_b", "q_d", "q_e", "q_a", "q_c"),
            "theta_ascending": ("q_c", "q_e", "q_a", "q_d", "q_b"),
            "theta_descending": ("q_b", "q_d", "q_a", "q_e", "q_c"),
            "epsilon_ascending": ("q_d", "q_b", "q_e", "q_a", "q_c"),
            "epsilon_descending": ("q_c", "q_a", "q_e", "q_b", "q_d"),
        }
        self.assertEqual(observed, expected)
        self.assertEqual(len(orderings), 8)

    def test_equal_lambda_ties_preserve_semantic_instance_order_both_ways(self):
        ascending, descending = build_all_deterministic_agh_orderings(
            make_ordering_instance()
        )[:2]

        self.assertLess(ascending.query_ids.index("q_b"), ascending.query_ids.index("q_d"))
        self.assertLess(descending.query_ids.index("q_b"), descending.query_ids.index("q_d"))

    def test_theta_order_is_not_lambda_or_phi_order(self):
        orderings = build_all_deterministic_agh_orderings(make_ordering_instance())

        self.assertNotEqual(orderings[4].query_ids, orderings[0].query_ids)
        self.assertNotEqual(orderings[4].query_ids, orderings[2].query_ids)
        self.assertEqual(orderings[4].deterministic_spec.label, "theta_ascending")

    def test_ordering_evidence_is_typed_and_immutable(self):
        ordering = build_all_deterministic_agh_orderings(make_ordering_instance())[0]

        self.assertIs(ordering.source, AGHOrderingSource.DETERMINISTIC)
        self.assertIs(ordering.deterministic_spec, DETERMINISTIC_AGH_ORDERING_SPECS[0])
        with self.assertRaises(FrozenInstanceError):
            ordering.query_ids = ("q_a",)


class SeededRandomAGHOrderingTests(unittest.TestCase):
    """Protect the local seeded primitive without constructing a schedule."""

    QUERY_IDS = ("q_a", "q_b", "q_c", "q_d", "q_e")

    def test_same_seed_context_reproduces_exact_order_and_evidence(self):
        first = build_seeded_random_agh_ordering(
            self.QUERY_IDS, seed=17, random_start_index=3
        )
        second = build_seeded_random_agh_ordering(
            self.QUERY_IDS, seed=17, random_start_index=3
        )

        self.assertEqual(first, second)
        self.assertEqual(first.seed, 17)
        self.assertEqual(first.random_start_index, 3)
        self.assertIs(first.source, AGHOrderingSource.SEEDED_RANDOM)

    def test_every_random_result_is_an_exact_permutation(self):
        for index in range(12):
            ordering = build_seeded_random_agh_ordering(
                self.QUERY_IDS, seed=29, random_start_index=index
            )
            self.assertEqual(set(ordering.query_ids), set(self.QUERY_IDS))
            self.assertEqual(len(ordering.query_ids), len(self.QUERY_IDS))

    def test_different_seed_context_changes_hand_fixture_order(self):
        first = build_seeded_random_agh_ordering(
            self.QUERY_IDS, seed=17, random_start_index=0
        )
        second = build_seeded_random_agh_ordering(
            self.QUERY_IDS, seed=18, random_start_index=0
        )

        self.assertNotEqual(first.query_ids, second.query_ids)

    def test_primitive_does_not_deduplicate_repeated_draw_requests(self):
        first = build_seeded_random_agh_ordering(
            ("q_a", "q_b"), seed=0, random_start_index=0
        )
        second = build_seeded_random_agh_ordering(
            ("q_a", "q_b"), seed=0, random_start_index=1
        )

        self.assertEqual(first.query_ids, ("q_a", "q_b"))
        self.assertEqual(second.query_ids, first.query_ids)
        self.assertNotEqual(first.random_start_index, second.random_start_index)

    def test_local_rng_does_not_mutate_global_random_state(self):
        before = random.getstate()
        build_seeded_random_agh_ordering(
            self.QUERY_IDS, seed=17, random_start_index=2
        )
        self.assertEqual(random.getstate(), before)

    def test_random_inputs_reject_duplicates_and_invalid_context(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_seeded_random_agh_ordering(
                ("q_a", "q_a"), seed=1, random_start_index=0
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            build_seeded_random_agh_ordering(
                self.QUERY_IDS, seed=True, random_start_index=0
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_seeded_random_agh_ordering(
                self.QUERY_IDS, seed=1, random_start_index=-1
            )


class AdaptiveRandomCountTests(unittest.TestCase):
    """Pin all six sides of the paper's strict adaptive-R thresholds."""

    def test_exact_threshold_boundaries(self):
        expected = {500: 20, 501: 10, 2000: 10, 2001: 5, 5000: 5, 5001: 3}
        for problem_scale, random_count in expected.items():
            with self.subTest(problem_scale=problem_scale):
                self.assertEqual(
                    compute_adaptive_agh_random_start_count(problem_scale, 1, 1),
                    random_count,
                )

    def test_count_is_random_only_not_eight_plus_r(self):
        self.assertEqual(compute_adaptive_agh_random_start_count(6, 6, 10), 20)

    def test_counts_must_be_positive_integers(self):
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compute_adaptive_agh_random_start_count(bad, 1, 1)


class GHOrderOverrideTests(unittest.TestCase):
    """Prove that only an explicit Phase-2 exact permutation changes GH."""

    def test_omitted_and_explicit_none_default_runs_are_identical(self):
        instance = make_ordering_instance()
        config = GHPhase1Config(0.0)

        omitted = run_gh(instance, config)
        explicit_none = run_gh(instance, config, query_order_override=None)

        self.assertEqual(omitted, explicit_none)
        self.assertEqual(
            omitted.phase2_result.processed_query_ids,
            ("q_a", "q_c", "q_e", "q_b", "q_d"),
        )

    def test_explicit_order_changes_only_phase2_traversal_not_phase1(self):
        instance = make_ordering_instance()
        config = GHPhase1Config(0.0)
        override = ("q_a", "q_e", "q_c", "q_b", "q_d")

        default = run_gh(instance, config)
        changed = run_gh(instance, config, query_order_override=override)

        self.assertEqual(default.phase1_result, changed.phase1_result)
        self.assertEqual(changed.initialized_phase2_state.query_order, override)
        self.assertEqual(changed.phase2_result.processed_query_ids, override)
        self.assertEqual(
            changed.phase2_result.query_traces[0].query_evaluation,
            default.phase2_result.query_traces[0].query_evaluation,
        )
        self.assertTrue(changed.canonicalization_completed)
        self.assertIs(
            changed.objective_breakdown,
            changed.validation_report.objective_breakdown,
        )

    def test_nondefault_state_requires_its_explicit_expected_order(self):
        instance = make_ordering_instance()
        phase1 = run_gh_phase1(instance, GHPhase1Config(0.0))
        override = tuple(reversed(instance.query_ids))
        state = initialize_gh_phase2_state(
            instance, phase1, query_order_override=override
        )

        with self.assertRaisesRegex(ValueError, "descending arrival"):
            run_gh_phase2(instance, state)
        result = run_gh_phase2(
            instance, state, expected_state_order=override
        )
        self.assertEqual(result.processed_query_ids, override)

    def test_malformed_overrides_are_rejected_before_phase2_execution(self):
        instance = make_ordering_instance()
        valid = instance.query_ids
        malformed = (
            valid[:-1],
            valid + ("q_extra",),
            (valid[0],) + valid[:-1],
            valid[:-1] + ("q_unknown",),
        )
        for order in malformed:
            with self.subTest(order=order), self.assertRaises(ValueError):
                run_gh(
                    instance,
                    GHPhase1Config(0.0),
                    query_order_override=order,
                )


if __name__ == "__main__":
    unittest.main()
