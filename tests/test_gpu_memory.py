"""Hand-calculated and ambiguity-protection tests for Paper v2 Eq. (8f)."""

from dataclasses import replace
import math
import unittest

from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import InstanceValidationError
from llm_resource_allocation.math.routing import (
    DecisionDomainError,
    compute_demand_balance_residuals,
    compute_routing_consistency_residuals,
)
from llm_resource_allocation.math.parallelism import (
    compute_configuration_selection_residuals,
    compute_gpu_count_linking_residuals,
)
from llm_resource_allocation.math.placement import (
    compute_mccormick_consistency_diagnostics,
)
from llm_resource_allocation.math.budget import compute_budget_residual
from llm_resource_allocation.math.compute_capacity import compute_compute_capacity_diagnostics
from llm_resource_allocation.math.processing_delay import compute_delay_slo_residuals
from llm_resource_allocation.math.error_slo import compute_error_slo_residuals
from llm_resource_allocation.math.gpu_memory import (
    compute_gpu_memory_residuals,
    compute_kv_memory_printed_term_by_pair,
    compute_model_memory_usage_gb_by_pair,
)
from llm_resource_allocation.math.storage_capacity import compute_storage_capacity_residuals

if __package__:
    from .placement_support import make_sparse_placement_state as make_state
else:
    from placement_support import make_sparse_placement_state as make_state


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ROUTE[1:]
W_2_1 = (*PAIR, 2, 1)
W_2_2 = (*PAIR, 2, 2)
W_4_1 = (*PAIR, 4, 1)
V_2_1 = (*ROUTE, 2, 1)
V_2_2 = (*ROUTE, 2, 2)


def make_memory_instance(
    *,
    capacity_gb: float = 12.0,
    residency_seconds: float = 2.0,
    quantization_latency_scale: float = 1.0,
):
    """Build a one-query instance with transparent literal memory terms.

    The synthetic coefficients are ``B=40 GB``, ``beta=0.001 GB/token``, and
    ``r=100``. ``T_res`` is a stored diagnostic input, not a claim that the
    paper's Section 5.1 calibration produced the chosen value.
    """

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        arrival_rate_per_hour=10.0,
        input_tokens=60.0,
        output_tokens=40.0,
        storage_gb_per_token=0.001,
        delay_slo_seconds=10.0,
        error_slo=0.5,
        max_unmet_fraction=1.0,
    )
    model = replace(
        base.models[0],
        weight_gb=40.0,
        kv_cache_gb_per_token=0.001,
    )
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=capacity_gb,
        quantization_latency_scale=quantization_latency_scale,
        tensor_parallel_degrees=(1, 2, 4),
        pipeline_parallel_depths=(1, 2),
    )
    return replace(
        base,
        queries=(query,),
        models=(model,),
        gpu_tiers=(tier,),
        base_error_rate={(query.id, model.id): 0.01},
        compute_delay_seconds_per_token={ROUTE: 0.001},
        communication_delay_seconds_per_token={ROUTE: 0.0},
        compute_gflop_per_token={ROUTE: 1.0},
        kv_cache_residency_seconds={ROUTE: residency_seconds},
    )


def result_for_pair(results, pair=PAIR):
    """Select one deterministic model/tier diagnostic."""

    return next(
        result
        for result in results
        if (result.model_id, result.gpu_tier_id) == pair
    )


class GpuMemoryHandCalculationTests(unittest.TestCase):
    """Verify separate terms and the printed per-GPU capacity semantics."""

    def test_model_shard_is_ten_gb_and_y_does_not_multiply_it(self):
        """Check ``40/(2*2)=10 GB`` for both ``y=4`` and invalid ``y=40``."""

        instance = make_memory_instance()
        state_y4 = make_state(q={PAIR: 1}, w={W_2_2: 1}, y={PAIR: 4})
        state_y40 = make_state(q={PAIR: 1}, w={W_2_2: 1}, y={PAIR: 40})

        model_y4 = compute_model_memory_usage_gb_by_pair(instance, state_y4)
        model_y40 = compute_model_memory_usage_gb_by_pair(instance, state_y40)
        result_y4 = result_for_pair(
            compute_gpu_memory_residuals(instance, state_y4)
        )
        result_y40 = result_for_pair(
            compute_gpu_memory_residuals(instance, state_y40)
        )

        self.assertEqual(model_y4[PAIR], 10.0)
        self.assertEqual(model_y40[PAIR], 10.0)
        self.assertEqual(result_y4, result_y40)
        self.assertEqual(result_y4.available_memory_rhs_gb, 12.0)

    def test_multiple_selected_configurations_are_summed_literally(self):
        """Check invalid Eq. (8d) state gives ``40/2 + 40/4 = 30``."""

        instance = make_memory_instance(capacity_gb=40.0)
        state = make_state(
            q={PAIR: 1},
            w={W_2_1: 1, W_4_1: 1},
            y={PAIR: 6},
        )

        model = compute_model_memory_usage_gb_by_pair(instance, state)
        selection = result_for_pair(
            compute_configuration_selection_residuals(
                instance, state.parallelism_decisions
            )
        )

        self.assertEqual(model[PAIR], 30.0)
        self.assertFalse(selection.is_feasible())

    def test_kv_printed_term_is_five_hundredths(self):
        """Check ``0.001/2 * 100 * 2 * 0.5 = 0.05`` exactly."""

        instance = make_memory_instance()
        state = make_state(q={PAIR: 1}, w={W_2_1: 1}, v={V_2_1: 0.5})

        kv = compute_kv_memory_printed_term_by_pair(instance, state)

        self.assertAlmostEqual(kv[PAIR], 0.05)

    def test_feasible_violated_and_exact_boundary(self):
        """Check model-only LHS ``10`` against capacities ``12,8,10``."""

        state = make_state(q={PAIR: 1}, w={W_2_2: 1}, y={PAIR: 4})
        feasible = result_for_pair(
            compute_gpu_memory_residuals(
                make_memory_instance(capacity_gb=12.0), state
            )
        )
        violated = result_for_pair(
            compute_gpu_memory_residuals(
                make_memory_instance(capacity_gb=8.0), state
            )
        )
        boundary = result_for_pair(
            compute_gpu_memory_residuals(
                make_memory_instance(capacity_gb=10.0), state
            )
        )

        self.assertEqual(feasible.slack, 2.0)
        self.assertEqual(feasible.violation, 0.0)
        self.assertTrue(feasible.is_feasible())
        self.assertEqual(violated.slack, -2.0)
        self.assertEqual(violated.violation, 2.0)
        self.assertFalse(violated.is_feasible())
        self.assertEqual(boundary.slack, 0.0)
        self.assertEqual(boundary.violation, 0.0)
        self.assertTrue(boundary.is_feasible())


class GpuMemoryAmbiguityProtectionTests(unittest.TestCase):
    """Protect extracted-v, stored-T_res, and literal-no-nu semantics."""

    def test_inconsistent_extracted_v_is_used_without_repair(self):
        """Use ``v=.5`` although ``x=.6,w=1`` and expose McCormick failure."""

        instance = make_memory_instance()
        state = make_state(
            x={ROUTE: 0.6},
            q={PAIR: 1},
            w={W_2_1: 1},
            v={V_2_1: 0.5},
        )

        kv = compute_kv_memory_printed_term_by_pair(instance, state)
        mccormick = next(
            diagnostic
            for diagnostic in compute_mccormick_consistency_diagnostics(
                instance, state
            )
            if (
                diagnostic.query_id,
                diagnostic.model_id,
                diagnostic.gpu_tier_id,
                diagnostic.tensor_parallel_degree,
                diagnostic.pipeline_parallel_depth,
            )
            == V_2_1
        )

        self.assertAlmostEqual(kv[PAIR], 0.05)
        self.assertFalse(mccormick.is_feasible())

    def test_quantization_latency_scale_is_absent_from_literal_arithmetic(self):
        """Protect the printed omission of ``nu``, not a physical claim."""

        state = make_state(q={PAIR: 1}, w={W_2_1: 1}, v={V_2_1: 0.5})
        fp16 = result_for_pair(
            compute_gpu_memory_residuals(
                make_memory_instance(quantization_latency_scale=1.0), state
            )
        )
        int4_scale = result_for_pair(
            compute_gpu_memory_residuals(
                make_memory_instance(quantization_latency_scale=0.25), state
            )
        )

        self.assertEqual(fp16, int4_scale)

    def test_stored_residency_value_is_not_recomputed(self):
        """Check stored ``T_res=2`` gives `.05` and ``T_res=4`` gives `.10`."""

        state = make_state(q={PAIR: 1}, w={W_2_1: 1}, v={V_2_1: 0.5})
        term_two = compute_kv_memory_printed_term_by_pair(
            make_memory_instance(residency_seconds=2.0), state
        )[PAIR]
        term_four = compute_kv_memory_printed_term_by_pair(
            make_memory_instance(residency_seconds=4.0), state
        )[PAIR]

        self.assertAlmostEqual(term_two, 0.05)
        self.assertAlmostEqual(term_four, 0.10)


class GpuMemoryBoundaryAndIndependenceTests(unittest.TestCase):
    """Verify sparse behavior, invalid inputs, tolerance, and independence."""

    def test_inactive_sparse_pair_is_zero_boundary(self):
        """Check omitted ``w,v,q`` gives LHS=RHS=slack=zero and feasible."""

        result = result_for_pair(
            compute_gpu_memory_residuals(
                make_memory_instance(), make_state()
            )
        )

        self.assertEqual(result.model_memory_usage_gb, 0.0)
        self.assertEqual(result.kv_memory_printed_term, 0.0)
        self.assertEqual(result.printed_memory_lhs, 0.0)
        self.assertEqual(result.available_memory_rhs_gb, 0.0)
        self.assertEqual(result.slack, 0.0)
        self.assertTrue(result.is_feasible())

    def test_inactive_pair_with_selected_shard_reports_violation(self):
        """Check ``q=0,w[2,1]=1`` gives LHS=20, RHS=0, violation=20."""

        result = result_for_pair(
            compute_gpu_memory_residuals(
                make_memory_instance(), make_state(w={W_2_1: 1})
            )
        )

        self.assertEqual(result.printed_memory_lhs, 20.0)
        self.assertEqual(result.available_memory_rhs_gb, 0.0)
        self.assertEqual(result.slack, -20.0)
        self.assertEqual(result.violation, 20.0)
        self.assertFalse(result.is_feasible())

    def test_unknown_decisions_are_rejected(self):
        """Reject unknown ``w``, ``v``, and ``q`` keys instead of ignoring them."""

        instance = make_memory_instance()
        invalid_states = (
            make_state(w={("unknown", "gpu_basic_fp16", 2, 1): 1}),
            make_state(v={("unknown", *PAIR, 2, 1): 0.5}),
            make_state(q={("unknown", "gpu_basic_fp16"): 1}),
        )
        for state in invalid_states:
            with self.subTest(state=state), self.assertRaises(DecisionDomainError):
                compute_gpu_memory_residuals(instance, state)

    def test_missing_or_invalid_scientific_memory_data_is_rejected(self):
        """Keep sparse decisions distinct from required B, beta, T_res, and C_GPU."""

        instance = make_memory_instance()
        invalid_instances = (
            replace(
                instance,
                models=(replace(instance.models[0], weight_gb=0.0),),
            ),
            replace(
                instance,
                models=(
                    replace(instance.models[0], kv_cache_gb_per_token=-0.1),
                ),
            ),
            replace(instance, kv_cache_residency_seconds={}),
            replace(
                instance,
                gpu_tiers=(
                    replace(instance.gpu_tiers[0], memory_capacity_gb=0.0),
                ),
            ),
        )
        for invalid in invalid_instances:
            with self.subTest(instance=invalid), self.assertRaises(
                InstanceValidationError
            ):
                compute_gpu_memory_residuals(invalid, make_state())

    def test_tolerance_changes_only_classification(self):
        """Allow a tiny negative slack without modifying any raw diagnostic."""

        instance = make_memory_instance(capacity_gb=10.0 - 1.0e-9)
        state = make_state(q={PAIR: 1}, w={W_2_2: 1}, y={PAIR: 4})
        result = result_for_pair(compute_gpu_memory_residuals(instance, state))
        raw = (
            result.model_memory_usage_gb,
            result.kv_memory_printed_term,
            result.printed_memory_lhs,
            result.available_memory_rhs_gb,
            result.slack,
            result.violation,
        )

        self.assertFalse(result.is_feasible())
        self.assertTrue(result.is_feasible(tolerance=1.0e-8))
        self.assertEqual(
            raw,
            (
                result.model_memory_usage_gb,
                result.kv_memory_printed_term,
                result.printed_memory_lhs,
                result.available_memory_rhs_gb,
                result.slack,
                result.violation,
            ),
        )
        for invalid_tolerance in (-1.0, math.inf, math.nan, True):
            with self.subTest(tolerance=invalid_tolerance), self.assertRaises(
                ValueError
            ):
                result.is_feasible(invalid_tolerance)

    def test_memory_computes_while_every_other_constraint_family_fails(self):
        """Expose Eq. (8f) independently of all other implemented families."""

        instance = make_memory_instance(capacity_gb=100.0)
        query = replace(
            instance.queries[0],
            delay_slo_seconds=1.0,
            error_slo=0.05,
        )
        tier = replace(
            instance.gpu_tiers[0], compute_throughput_tflops=0.001
        )
        instance = replace(
            instance,
            queries=(query,),
            gpu_tiers=(tier,),
            base_error_rate={(query.id, instance.models[0].id): 0.2},
            compute_delay_seconds_per_token={ROUTE: 1.0},
            compute_gflop_per_token={ROUTE: 1000.0},
            cost_budget_usd=0.0,
            storage_capacity_gb=0.1,
        )
        state = make_state(
            x={ROUTE: 0.5},
            z={ROUTE: 0},
            q={PAIR: 1},
            u={"q_text": 0.0},
            y={PAIR: 1},
            v={V_2_1: 0.5},
        )

        memory = result_for_pair(compute_gpu_memory_residuals(instance, state))
        demand = compute_demand_balance_residuals(
            instance, state.routing_decisions
        )[0]
        routing = compute_routing_consistency_residuals(
            instance, state.routing_decisions
        )[0]
        selection = result_for_pair(
            compute_configuration_selection_residuals(
                instance, state.parallelism_decisions
            )
        )
        gpu_link = result_for_pair(
            compute_gpu_count_linking_residuals(
                instance, state.parallelism_decisions
            )
        )
        compute = result_for_pair(
            compute_compute_capacity_diagnostics(
                instance, state.parallelism_decisions
            )
        )
        delay = compute_delay_slo_residuals(instance, state)[0]
        error = compute_error_slo_residuals(
            instance, state.routing_decisions
        )[0]
        budget = compute_budget_residual(instance, state)
        storage = compute_storage_capacity_residuals(instance, state)[0]

        self.assertTrue(memory.is_feasible())
        self.assertFalse(demand.is_feasible())
        self.assertFalse(routing.is_feasible())
        self.assertFalse(selection.is_feasible())
        self.assertFalse(gpu_link.is_feasible())
        self.assertFalse(compute.is_feasible())
        self.assertFalse(delay.is_feasible())
        self.assertFalse(error.is_feasible())
        self.assertFalse(budget.is_feasible())
        self.assertFalse(storage.is_feasible())


if __name__ == "__main__":
    unittest.main()
