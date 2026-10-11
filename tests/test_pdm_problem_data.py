"""Tests for the solver-independent deterministic ``P_DM`` index boundary."""

import unittest
from dataclasses import FrozenInstanceError, replace

from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
from llm_resource_allocation.domain import InstanceValidationError
from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import (
    PlacementDecisionState,
    validate_placement_decision_domains,
)
from llm_resource_allocation.math.routing import RoutingDecisionState
from llm_resource_allocation.optimization.problem_data import (
    PdmProblemData,
    PdmVariableFamily,
    format_semantic_variable_name,
)


class MinimalPdmProblemDataTests(unittest.TestCase):
    """Hand-check the complete 1x1x1 structural boundary."""

    def setUp(self) -> None:
        """Use transparent N={1,2}, M={1,2} on the existing tiny instance."""

        base = make_minimal_tiny_instance()
        tier = replace(base.gpu_tiers[0], pipeline_parallel_depths=(1, 2))
        self.instance = replace(base, gpu_tiers=(tier,))
        self.data = PdmProblemData(self.instance)

    def test_exact_sets_and_lattices(self) -> None:
        """Verify I/J/K/N_k/M_k and every semantic tuple manually."""

        indices = self.data.index_sets
        lattices = self.data.variable_lattices

        self.assertEqual(indices.query_ids, ("q_text",))
        self.assertEqual(indices.model_ids, ("m_small",))
        self.assertEqual(indices.tier_ids, ("gpu_basic_fp16",))
        self.assertEqual(
            indices.tensor_parallel_degrees_by_tier["gpu_basic_fp16"],
            (1, 2),
        )
        self.assertEqual(
            indices.pipeline_parallel_depths_by_tier["gpu_basic_fp16"],
            (1, 2),
        )
        self.assertEqual(lattices.unmet_keys, ("q_text",))
        self.assertEqual(
            lattices.model_tier_keys, (("m_small", "gpu_basic_fp16"),)
        )
        self.assertEqual(
            lattices.route_keys,
            (("q_text", "m_small", "gpu_basic_fp16"),),
        )
        self.assertEqual(
            lattices.configuration_keys,
            (
                ("m_small", "gpu_basic_fp16", 1, 1),
                ("m_small", "gpu_basic_fp16", 1, 2),
                ("m_small", "gpu_basic_fp16", 2, 1),
                ("m_small", "gpu_basic_fp16", 2, 2),
            ),
        )
        self.assertEqual(
            lattices.auxiliary_product_keys,
            tuple(("q_text", *key) for key in lattices.configuration_keys),
        )

    def test_hand_calculated_variable_and_constraint_counts(self) -> None:
        """Check 13 decisions, Eq. (8k) split, and 16 McCormick rows."""

        variables = self.data.variable_counts
        self.assertEqual(
            (
                variables.n_x,
                variables.n_z,
                variables.n_q,
                variables.n_u,
                variables.n_w,
                variables.n_y,
                variables.n_v,
            ),
            (1, 1, 1, 1, 4, 1, 4),
        )
        self.assertEqual(variables.total_decision_variable_count, 13)

        constraints = self.data.constraint_counts
        self.assertEqual(
            (
                constraints.eq_8b,
                constraints.eq_8c,
                constraints.eq_8d,
                constraints.eq_8e,
                constraints.eq_8f,
                constraints.eq_8g,
                constraints.eq_8h_d_0016,
                constraints.eq_8i,
                constraints.eq_8j,
            ),
            (1, 1, 1, 1, 1, 1, 1, 1, 1),
        )
        self.assertEqual(constraints.eq_8k_paper_semantic_inequalities, 3)
        self.assertEqual(constraints.eq_8k_explicit_linking_rows, 2)
        self.assertEqual(constraints.eq_8k_x_lower_bounds, 1)
        self.assertEqual(constraints.mccormick_inequalities, 16)
        self.assertEqual(constraints.total_explicit_reference_rows, 27)

    def test_problem_data_retains_instance_authority_and_is_immutable(self) -> None:
        """Keep coefficients in the original instance and freeze metadata."""

        self.assertIs(self.data.instance, self.instance)
        with self.assertRaises(FrozenInstanceError):
            self.data.instance = make_minimal_tiny_instance()
        with self.assertRaises(TypeError):
            self.data.index_sets.tensor_parallel_degrees_by_tier[
                "gpu_basic_fp16"
            ] = (8,)


class HeterogeneousPdmProblemDataTests(unittest.TestCase):
    """Protect tier-specific configuration products and ordering."""

    def setUp(self) -> None:
        """Use tier A N={1,2},M={1}; tier B N={1,2,4},M={1,2}."""

        base = make_heterogeneous_tiny_instance()
        fast_tier = replace(
            base.gpu_tiers[1],
            tensor_parallel_degrees=(1, 2, 4),
            pipeline_parallel_depths=(1, 2),
        )
        self.instance = replace(
            base,
            gpu_tiers=(base.gpu_tiers[0], fast_tier),
        )
        self.data = PdmProblemData(self.instance)

    def test_configuration_and_auxiliary_lattices_are_tier_specific(self) -> None:
        """Count two configurations on tier A and six on tier B per model."""

        configurations = self.data.variable_lattices.configuration_keys
        budget_keys = tuple(
            key for key in configurations if key[1] == "gpu_budget_int8"
        )
        fast_keys = tuple(
            key for key in configurations if key[1] == "gpu_fast_fp16"
        )

        self.assertEqual(
            budget_keys,
            (
                ("m_compact", "gpu_budget_int8", 1, 1),
                ("m_compact", "gpu_budget_int8", 2, 1),
                ("m_accurate", "gpu_budget_int8", 1, 1),
                ("m_accurate", "gpu_budget_int8", 2, 1),
            ),
        )
        self.assertEqual(len(fast_keys), 12)
        self.assertEqual(self.data.variable_counts.n_w, 16)
        self.assertEqual(self.data.variable_counts.n_v, 32)
        self.assertEqual(self.data.variable_counts.total_decision_variable_count, 74)
        self.assertEqual(self.data.constraint_counts.mccormick_inequalities, 128)

    def test_unsupported_pipeline_depth_is_absent_from_w_and_v(self) -> None:
        """Forbid `(TP,PP)=(2,2)` on the tier whose M_k contains only 1."""

        unsupported_w = ("m_compact", "gpu_budget_int8", 2, 2)
        unsupported_v = ("q_short", *unsupported_w)
        self.assertNotIn(
            unsupported_w, self.data.variable_lattices.configuration_keys
        )
        self.assertNotIn(unsupported_v, self.data.variable_lattices.auxiliary_product_keys)

    def test_repeated_construction_and_nonlexicographic_order_are_deterministic(self) -> None:
        """Preserve caller-defined sequence order across repeated builds."""

        reordered_tiers = tuple(
            replace(
                tier,
                tensor_parallel_degrees=tuple(
                    reversed(tier.tensor_parallel_degrees)
                ),
                pipeline_parallel_depths=tuple(
                    reversed(tier.pipeline_parallel_depths)
                ),
            )
            for tier in reversed(self.instance.gpu_tiers)
        )
        reordered = replace(
            self.instance,
            queries=tuple(reversed(self.instance.queries)),
            models=tuple(reversed(self.instance.models)),
            gpu_tiers=reordered_tiers,
        )
        first = PdmProblemData(reordered)
        second = PdmProblemData(reordered)

        self.assertEqual(first, second)
        self.assertEqual(first.index_sets.query_ids, ("q_long", "q_short"))
        self.assertEqual(first.index_sets.model_ids, ("m_accurate", "m_compact"))
        self.assertEqual(
            first.index_sets.tier_ids,
            ("gpu_fast_fp16", "gpu_budget_int8"),
        )
        self.assertEqual(
            first.index_sets.tensor_parallel_degrees_by_tier["gpu_fast_fp16"],
            (4, 2, 1),
        )
        self.assertEqual(
            first.variable_lattices.route_keys[0],
            ("q_long", "m_accurate", "gpu_fast_fp16"),
        )
        self.assertEqual(first.variable_counts, second.variable_counts)
        self.assertEqual(first.constraint_counts, second.constraint_counts)

    def test_every_key_round_trips_into_milestone_3_decision_fields(self) -> None:
        """Show lattice keys are accepted unchanged by the independent validator."""

        lattices = self.data.variable_lattices
        routing = RoutingDecisionState(
            routing_fractions={key: 0.0 for key in lattices.route_keys},
            admission_indicators={key: 0.0 for key in lattices.route_keys},
            deployment_indicators={
                key: 0.0 for key in lattices.model_tier_keys
            },
            unmet_fractions={key: 0.0 for key in lattices.unmet_keys},
        )
        parallelism = ParallelismDecisionState(
            routing_decisions=routing,
            configuration_selectors={
                key: 0.0 for key in lattices.configuration_keys
            },
            gpu_counts={key: 0.0 for key in lattices.model_tier_keys},
        )
        state = PlacementDecisionState(
            parallelism_decisions=parallelism,
            auxiliary_products={key: 0.0 for key in lattices.auxiliary_product_keys},
        )

        validate_placement_decision_domains(self.instance, state)
        self.assertEqual(tuple(state.routing_fractions), lattices.route_keys)
        self.assertEqual(tuple(state.admission_indicators), lattices.route_keys)
        self.assertEqual(
            tuple(state.deployment_indicators), lattices.model_tier_keys
        )
        self.assertEqual(tuple(state.unmet_fractions), lattices.unmet_keys)
        self.assertEqual(
            tuple(state.configuration_selectors), lattices.configuration_keys
        )
        self.assertEqual(tuple(state.gpu_counts), lattices.model_tier_keys)
        self.assertEqual(tuple(state.auxiliary_products), lattices.auxiliary_product_keys)


class PdmProblemDataBoundaryTests(unittest.TestCase):
    """Check validation-first construction and semantic debug names."""

    def test_invalid_instance_is_rejected_before_lattice_construction(self) -> None:
        """Reuse `validate_problem_instance` instead of building partial data."""

        invalid = replace(make_minimal_tiny_instance(), cost_budget_usd=-1.0)
        with self.assertRaises(InstanceValidationError):
            PdmProblemData(invalid)

    def test_semantic_name_policy_is_stable_and_family_aware(self) -> None:
        """Format readable names while tuple keys remain extraction authority."""

        self.assertEqual(
            format_semantic_variable_name(
                PdmVariableFamily.X,
                ("q_text", "m_small", "gpu_basic_fp16"),
            ),
            "x[q_text,m_small,gpu_basic_fp16]",
        )
        self.assertEqual(
            format_semantic_variable_name(
                PdmVariableFamily.W,
                ("m_small", "gpu_basic_fp16", 2, 1),
            ),
            "w[m_small,gpu_basic_fp16,2,1]",
        )
        self.assertEqual(
            format_semantic_variable_name(PdmVariableFamily.U, "q_text"),
            "u[q_text]",
        )
        with self.assertRaises(ValueError):
            format_semantic_variable_name(
                PdmVariableFamily.Q,
                ("q_text", "m_small", "gpu_basic_fp16"),
            )


if __name__ == "__main__":
    unittest.main()
