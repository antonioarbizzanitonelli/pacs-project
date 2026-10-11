"""Hand-derived and isolation tests for Paper v2 Eq. (9), mechanism M1."""

import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from llm_resource_allocation.algorithms import select_minimum_feasible_parallelism
from llm_resource_allocation.data.tiny_instances import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig
from llm_resource_allocation.math.latency import compute_configuration_delay_seconds


QUERY_ID = "q_text"
MODEL_ID = "m_small"
TIER_ID = "gpu_basic_fp16"


def make_m1_instance(
    *,
    weight_gb: float = 1.0,
    memory_capacity_gb: float = 8.0,
    delay_slo_seconds: float = 1.0,
    tensor_degrees: tuple[int, ...] = (1, 2),
    pipeline_depths: tuple[int, ...] = (1,),
    quantization_latency_scale: float = 1.0,
):
    """Return a valid 1x1x1 M1 fixture with transparent delay coefficients."""

    base = make_minimal_tiny_instance()
    query = replace(base.queries[0], delay_slo_seconds=delay_slo_seconds)
    model = replace(base.models[0], weight_gb=weight_gb)
    tier = replace(
        base.gpu_tiers[0],
        memory_capacity_gb=memory_capacity_gb,
        tensor_parallel_degrees=tensor_degrees,
        pipeline_parallel_depths=pipeline_depths,
        quantization_latency_scale=quantization_latency_scale,
    )
    return replace(base, queries=(query,), models=(model,), gpu_tiers=(tier,))


def select(instance, query_id: str = QUERY_ID):
    """Evaluate the canonical model/tier candidate for concise test arithmetic."""

    return select_minimum_feasible_parallelism(
        instance,
        query_id=query_id,
        model_id=MODEL_ID,
        tier_id=TIER_ID,
    )


class M1HandDerivedSelectionTests(unittest.TestCase):
    """Exercise minimum, memory, delay, joint, and no-feasible cases."""

    def test_one_by_one_is_already_the_minimum_feasible_configuration(self):
        """Derive `B/1=1<=8` and `D_cfg=.1*3+.01*1=.31<=1`."""

        result = select(make_m1_instance())

        self.assertTrue(result.is_feasible)
        self.assertEqual(result.selected_configuration, ParallelismConfig(1, 1))
        self.assertEqual(result.selected_weight_shard_gb, 1.0)
        self.assertAlmostEqual(
            result.selected_fixed_configuration_delay_seconds, 0.31
        )
        self.assertTrue(result.selected_diagnostic.is_feasible)

    def test_producer_retains_the_selected_configuration_diagnostic(self):
        for delay_slo, expected in (
            (1.0, ParallelismConfig(1, 1)),
            (0.2, ParallelismConfig(2, 1)),
            (0.001, None),
        ):
            with self.subTest(delay_slo=delay_slo):
                result = select(make_m1_instance(delay_slo_seconds=delay_slo))
                self.assertEqual(result.selected_configuration, expected)
                matching = tuple(
                    diagnostic
                    for diagnostic in result.configuration_diagnostics
                    if diagnostic.configuration == expected
                )
                self.assertEqual(len(matching), int(expected is not None))
                if matching:
                    self.assertIs(result.selected_diagnostic, matching[0])
                    self.assertTrue(matching[0].is_feasible)
                else:
                    self.assertIsNone(result.selected_diagnostic)

    def test_caller_cannot_remove_selected_diagnostic_after_construction(self):
        result = select(make_m1_instance())
        caller_diagnostics = list(result.configuration_diagnostics)
        copied = replace(result, configuration_diagnostics=caller_diagnostics)
        caller_diagnostics.clear()
        self.assertEqual(
            copied.selected_diagnostic.configuration, ParallelismConfig(1, 1)
        )

    def test_memory_forces_the_smallest_larger_gpu_count(self):
        """Derive `8/1>5`, `8/2=4<=5`, hence select `(2,1)` and two GPUs."""

        result = select(
            make_m1_instance(
                weight_gb=8.0,
                memory_capacity_gb=5.0,
                tensor_degrees=(1, 2, 4),
            )
        )

        first, second, _ = result.configuration_diagnostics
        self.assertFalse(first.weight_condition_passes)
        self.assertTrue(first.delay_condition_passes)
        self.assertTrue(second.is_feasible)
        self.assertEqual(result.selected_configuration, ParallelismConfig(2, 1))
        self.assertEqual(result.selected_weight_shard_gb, 4.0)

    def test_delay_forces_the_smallest_larger_tensor_degree(self):
        """Derive `D(1,1)=.31>.2` but `D(2,1)=.16<=.2`."""

        result = select(make_m1_instance(delay_slo_seconds=0.2))

        first, second = result.configuration_diagnostics
        self.assertTrue(first.weight_condition_passes)
        self.assertFalse(first.delay_condition_passes)
        self.assertAlmostEqual(first.fixed_configuration_delay_seconds, 0.31)
        self.assertTrue(second.is_feasible)
        self.assertAlmostEqual(second.fixed_configuration_delay_seconds, 0.16)
        self.assertEqual(result.selected_configuration, ParallelismConfig(2, 1))

    def test_both_conditions_must_hold_on_the_same_configuration(self):
        """Reject separate witnesses: `(1,1)` passes delay, `(1,2)` passes weight."""

        result = select(
            make_m1_instance(
                weight_gb=6.0,
                memory_capacity_gb=4.0,
                delay_slo_seconds=0.315,
                pipeline_depths=(1, 2),
            )
        )
        diagnostics = result.configuration_diagnostics

        # D(1,1)=.31 passes delay but its 6 GB shard fails memory.
        self.assertFalse(diagnostics[0].weight_condition_passes)
        self.assertTrue(diagnostics[0].delay_condition_passes)
        # (1,2) has a 3 GB shard but D=.32, so it fails the delay condition.
        self.assertTrue(diagnostics[1].weight_condition_passes)
        self.assertFalse(diagnostics[1].delay_condition_passes)
        # The later (2,1) has a 3 GB shard and D=.16, satisfying both.
        self.assertTrue(diagnostics[2].is_feasible)
        self.assertEqual(result.selected_configuration, ParallelismConfig(2, 1))

    def test_no_feasible_configuration_returns_none_with_all_diagnostics(self):
        """Never fall back to the largest tier-valid configuration."""

        result = select(
            make_m1_instance(
                weight_gb=100.0,
                memory_capacity_gb=1.0,
                delay_slo_seconds=0.01,
                tensor_degrees=(1, 2),
                pipeline_depths=(1, 2),
            )
        )

        self.assertFalse(result.is_feasible)
        self.assertIsNone(result.selected_configuration)
        self.assertIsNone(result.selected_weight_shard_gb)
        self.assertIsNone(result.selected_fixed_configuration_delay_seconds)
        self.assertIsNone(result.selected_diagnostic)
        self.assertEqual(len(result.configuration_diagnostics), 4)
        self.assertFalse(any(item.is_feasible for item in result.configuration_diagnostics))


class M1ConfigurationAuthorityAndOrderingTests(unittest.TestCase):
    """Protect tier validity, minimum GPU count, and stable semantic ties."""

    def test_unsupported_intermediate_tensor_degree_is_never_considered(self):
        """Select supported TP=4 rather than manufacturing unsupported TP=2."""

        result = select(
            make_m1_instance(
                weight_gb=6.0,
                memory_capacity_gb=4.0,
                tensor_degrees=(1, 4),
            )
        )

        considered = tuple(
            item.configuration for item in result.configuration_diagnostics
        )
        self.assertEqual(
            considered,
            (ParallelismConfig(1, 1), ParallelismConfig(4, 1)),
        )
        self.assertNotIn(ParallelismConfig(2, 1), considered)
        self.assertEqual(result.selected_configuration, ParallelismConfig(4, 1))

    def test_equal_gpu_count_tie_uses_first_tier_semantic_configuration(self):
        """Keep `(1,2)` before equally small `(2,1)` without a TP preference."""

        result = select(
            make_m1_instance(
                weight_gb=6.0,
                memory_capacity_gb=4.0,
                tensor_degrees=(1, 2),
                pipeline_depths=(2, 1),
            )
        )

        self.assertTrue(
            next(
                item
                for item in result.configuration_diagnostics
                if item.configuration == ParallelismConfig(2, 1)
            ).is_feasible
        )
        self.assertEqual(result.selected_configuration, ParallelismConfig(1, 2))

    def test_reversing_semantic_degree_order_changes_only_tied_representative(self):
        """Prove ties follow stored order rather than hashes or lexical IDs."""

        pp_first = select(
            make_m1_instance(
                weight_gb=6.0,
                memory_capacity_gb=4.0,
                tensor_degrees=(1, 2),
                pipeline_depths=(2, 1),
            )
        )
        tp_first = select(
            make_m1_instance(
                weight_gb=6.0,
                memory_capacity_gb=4.0,
                tensor_degrees=(2, 1),
                pipeline_depths=(1, 2),
            )
        )

        self.assertEqual(pp_first.selected_configuration, ParallelismConfig(1, 2))
        self.assertEqual(tp_first.selected_configuration, ParallelismConfig(2, 1))
        self.assertEqual(
            pp_first.selected_configuration.gpu_count,
            tp_first.selected_configuration.gpu_count,
        )

    def test_repeated_calls_are_exactly_deterministic(self):
        """Return identical immutable evidence on repeated validated input."""

        instance = make_m1_instance(
            weight_gb=6.0,
            memory_capacity_gb=4.0,
            pipeline_depths=(2, 1),
        )

        first = select(instance)
        second = select(instance)

        self.assertEqual(first, second)
        with self.assertRaises(FrozenInstanceError):
            first.selected_configuration = ParallelismConfig(2, 1)


class M1ScientificBoundaryTests(unittest.TestCase):
    """Verify query dependence and exclusions from the literal M1 boundary."""

    def make_two_query_instance(self):
        """Return equal workloads whose delay SLOs require TP=1 and TP=2."""

        base = make_m1_instance()
        relaxed = replace(
            base.queries[0], id="q_relaxed", delay_slo_seconds=1.0
        )
        tight = replace(base.queries[0], id="q_tight", delay_slo_seconds=0.2)
        queries = (relaxed, tight)
        pair_values = {(query.id, MODEL_ID): 0.05 for query in queries}
        keys = tuple((query.id, MODEL_ID, TIER_ID) for query in queries)
        return replace(
            base,
            queries=queries,
            base_error_rate=pair_values,
            compute_delay_seconds_per_token={key: 0.10 for key in keys},
            communication_delay_seconds_per_token={key: 0.01 for key in keys},
            compute_gflop_per_token={key: 1.0 for key in keys},
            kv_cache_residency_seconds={key: 1.0 for key in keys},
        )

    def test_same_model_tier_selects_different_configs_for_two_queries(self):
        """Use query-specific `D_cfg`/`Delta_i` while holding model/tier fixed."""

        instance = self.make_two_query_instance()
        relaxed = select(instance, "q_relaxed")
        tight = select(instance, "q_tight")

        self.assertEqual(relaxed.selected_configuration, ParallelismConfig(1, 1))
        self.assertEqual(tight.selected_configuration, ParallelismConfig(2, 1))

    def test_weight_shard_evidence_is_query_independent(self):
        """Keep `B_j/(n*m)` identical across query-specific evaluations."""

        instance = self.make_two_query_instance()
        relaxed = select(instance, "q_relaxed")
        tight = select(instance, "q_tight")

        relaxed_shards = {
            item.configuration: item.weight_shard_gb
            for item in relaxed.configuration_diagnostics
        }
        tight_shards = {
            item.configuration: item.weight_shard_gb
            for item in tight.configuration_diagnostics
        }
        self.assertEqual(relaxed_shards, tight_shards)

    def test_changing_nu_does_not_change_literal_m1_selection(self):
        """Preserve printed Eq. (9), which contains `B_j`, not `nu_k B_j`."""

        fp16 = select(
            make_m1_instance(weight_gb=8.0, memory_capacity_gb=5.0)
        )
        int4_scale = select(
            make_m1_instance(
                weight_gb=8.0,
                memory_capacity_gb=5.0,
                quantization_latency_scale=0.25,
            )
        )

        self.assertEqual(fp16, int4_scale)
        self.assertEqual(fp16.selected_configuration, ParallelismConfig(2, 1))

    def test_full_eq_8f_kv_memory_is_not_used_by_m1(self):
        """Huge KV inputs do not alter the weight-shard-only M1 prefilter."""

        base = make_m1_instance()
        model = replace(base.models[0], kv_cache_gb_per_token=1_000.0)
        instance = replace(
            base,
            models=(model,),
            kv_cache_residency_seconds={(QUERY_ID, MODEL_ID, TIER_ID): 1_000.0},
        )

        result = select(instance)

        self.assertEqual(result.selected_configuration, ParallelismConfig(1, 1))
        self.assertEqual(result.selected_weight_shard_gb, 1.0)

    def test_existing_fixed_configuration_delay_primitive_is_reused(self):
        """Call the verified `D_cfg` helper once per tier-valid configuration."""

        instance = make_m1_instance(pipeline_depths=(1, 2))
        with patch(
            "llm_resource_allocation.algorithms.m1.compute_configuration_delay_seconds",
            wraps=compute_configuration_delay_seconds,
        ) as delay_primitive:
            result = select(instance)

        self.assertEqual(
            delay_primitive.call_count, len(result.configuration_diagnostics)
        )


    def test_unknown_semantic_ids_fail_clearly(self):
        """Reject lookup mistakes rather than silently evaluating another entity."""

        instance = make_m1_instance()
        cases = (
            ("q_unknown", MODEL_ID, TIER_ID),
            (QUERY_ID, "m_unknown", TIER_ID),
            (QUERY_ID, MODEL_ID, "tier_unknown"),
        )
        for query_id, model_id, tier_id in cases:
            with self.subTest(ids=(query_id, model_id, tier_id)):
                with self.assertRaises(ValueError):
                    select_minimum_feasible_parallelism(
                        instance, query_id, model_id, tier_id
                    )


if __name__ == "__main__":
    unittest.main()
