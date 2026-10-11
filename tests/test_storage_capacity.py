"""Hand-calculated tests for the selected Paper v2 Eq. (8h) closure."""

from dataclasses import replace
import unittest

from llm_resource_allocation.data import (
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
)
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
from llm_resource_allocation.math.budget import compute_budget_residual
from llm_resource_allocation.math.compute_capacity import compute_compute_capacity_diagnostics
from llm_resource_allocation.math.storage_capacity import (
    compute_data_storage_rate_by_query,
    compute_model_storage_usage_by_query,
    compute_storage_capacity_residuals,
)
from llm_resource_allocation.math.processing_delay import compute_delay_slo_residuals
from llm_resource_allocation.math.error_slo import compute_error_slo_residuals

if __package__:
    from .placement_support import make_sparse_placement_state as make_state
else:
    from placement_support import make_sparse_placement_state as make_state


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ROUTE[1:]
AUXILIARY = (*ROUTE, 1, 1)


def make_storage_instance(
    *,
    storage_capacity: float = 25.0,
    horizon_hours: float = 1.0,
    storage_price: float = 1.0,
    cost_budget: float = 100.0,
    throughput_tflops: float = 1.0,
    compute_gflop_per_token: float = 1.0,
    compute_delay_seconds_per_token: float = 0.001,
    delay_limit: float = 10.0,
    base_error: float = 0.01,
    error_limit: float = 1.0,
):
    """Build a valid one-query instance with transparent Eq. (8h) terms.

    The model weighs 20 GB. The query has ``theta=0.001 GB/token``, ``r=100
    token/query``, and ``lambda=10 query/hour``. At ``x=0.5`` its printed data
    term is therefore ``0.001*100*10*0.5 = 0.5 GB/hour``.
    """

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        arrival_rate_per_hour=10.0,
        input_tokens=60.0,
        output_tokens=40.0,
        storage_gb_per_token=0.001,
        delay_slo_seconds=delay_limit,
        error_slo=error_limit,
        max_unmet_fraction=1.0,
    )
    model = replace(base.models[0], weight_gb=20.0)
    tier = replace(
        base.gpu_tiers[0],
        compute_throughput_tflops=throughput_tflops,
    )
    return replace(
        base,
        queries=(query,),
        models=(model,),
        gpu_tiers=(tier,),
        base_error_rate={(query.id, model.id): base_error},
        compute_delay_seconds_per_token={
            ROUTE: compute_delay_seconds_per_token
        },
        communication_delay_seconds_per_token={ROUTE: 0.0},
        compute_gflop_per_token={ROUTE: compute_gflop_per_token},
        kv_cache_residency_seconds={ROUTE: 1.0},
        storage_price_usd_per_gb_hour=storage_price,
        cost_budget_usd=cost_budget,
        storage_capacity_gb=storage_capacity,
        horizon_hours=horizon_hours,
        compute_utilization=1.0,
    )


def make_printed_state():
    """Return ``z=1`` and ``x=0.5`` for the transparent hand calculation."""

    return make_state(x={ROUTE: 0.5}, z={ROUTE: 1})


def result_for_query(results, query_id="q_text"):
    """Select one query diagnostic from deterministic output."""

    return next(result for result in results if result.query_id == query_id)


def result_for_pair(results, pair=PAIR):
    """Select one model/tier diagnostic from deterministic output."""

    return next(
        result
        for result in results
        if (result.model_id, result.gpu_tier_id) == pair
    )


class StorageTermHandCalculationTests(unittest.TestCase):
    """Verify the two printed terms and feasible/violated/boundary cases."""

    def test_separate_model_and_data_terms(self):
        """Check model ``20 GB`` and data ``0.001*100*10*0.5=0.5 GB/h``."""

        instance = make_storage_instance()
        state = make_printed_state()

        model = compute_model_storage_usage_by_query(instance, state)
        data = compute_data_storage_rate_by_query(instance, state)

        self.assertEqual(model["q_text"], 20.0)
        self.assertEqual(data["q_text"], 0.5)

    def test_feasible_case(self):
        """Check printed LHS ``20 + 0.5 = 20.5 < 25``, slack ``4.5``."""

        result = result_for_query(
            compute_storage_capacity_residuals(
                make_storage_instance(storage_capacity=25.0),
                make_printed_state(),
            )
        )

        self.assertEqual(result.model_storage_usage_gb, 20.0)
        self.assertEqual(result.data_storage_rate_gb_per_hour, 0.5)
        self.assertEqual(result.printed_storage_lhs, 20.5)
        self.assertEqual(result.storage_limit_gb, 25.0)
        self.assertEqual(result.slack, 4.5)
        self.assertEqual(result.violation, 0.0)
        self.assertTrue(result.is_feasible())

    def test_violated_case(self):
        """Check printed LHS ``20.5 > 20``, slack ``-0.5``, violation ``0.5``."""

        result = result_for_query(
            compute_storage_capacity_residuals(
                make_storage_instance(storage_capacity=20.0),
                make_printed_state(),
            )
        )

        self.assertEqual(result.printed_storage_lhs, 20.5)
        self.assertEqual(result.slack, -0.5)
        self.assertEqual(result.violation, 0.5)
        self.assertFalse(result.is_feasible())

    def test_exact_boundary(self):
        """Check printed LHS equals ``C_s=20.5`` exactly."""

        result = result_for_query(
            compute_storage_capacity_residuals(
                make_storage_instance(storage_capacity=20.5),
                make_printed_state(),
            )
        )

        self.assertEqual(result.printed_storage_lhs, result.storage_limit_gb)
        self.assertEqual(result.slack, 0.0)
        self.assertTrue(result.is_feasible())


class StorageIndexingAuditTests(unittest.TestCase):
    """Protect the selected per-query closure against silent global repair."""

    def test_two_queries_sharing_one_pair_remain_separate_constraints(self):
        """Produce two ``20.5`` residuals, not one global ``41`` residual."""

        base = make_heterogeneous_tiny_instance()
        queries = tuple(
            replace(
                query,
                arrival_rate_per_hour=10.0,
                input_tokens=60.0,
                output_tokens=40.0,
                storage_gb_per_token=0.001,
            )
            for query in base.queries
        )
        models = tuple(
            replace(model, weight_gb=20.0)
            if model.id == "m_compact"
            else model
            for model in base.models
        )
        instance = replace(
            base,
            queries=queries,
            models=models,
            storage_capacity_gb=25.0,
        )
        pair = ("m_compact", "gpu_budget_int8")
        routes = {
            ("q_short", *pair): 0.5,
            ("q_long", *pair): 0.5,
        }
        admissions = {
            ("q_short", *pair): 1,
            ("q_long", *pair): 1,
        }
        results = compute_storage_capacity_residuals(
            instance, make_state(x=routes, z=admissions)
        )

        self.assertEqual(tuple(item.query_id for item in results), ("q_short", "q_long"))
        self.assertEqual(tuple(item.printed_storage_lhs for item in results), (20.5, 20.5))
        self.assertTrue(all(item.is_feasible() for item in results))
        # A global-i-summed alternative would yield 41 and violate C_s=25.
        # It is deliberately documented, not implemented as the primary model.
        self.assertEqual(sum(item.printed_storage_lhs for item in results), 41.0)
        self.assertGreater(41.0, instance.storage_capacity_gb)

    def test_sum_covers_each_model_tier_term_without_deduplication(self):
        """Count the same 20-GB model twice across two tiers for one query."""

        base = make_heterogeneous_tiny_instance()
        query = replace(
            base.queries[0],
            arrival_rate_per_hour=10.0,
            input_tokens=60.0,
            output_tokens=40.0,
            storage_gb_per_token=0.001,
        )
        queries = (query, base.queries[1])
        models = tuple(
            replace(model, weight_gb=20.0)
            if model.id == "m_compact"
            else model
            for model in base.models
        )
        pair_a = ("m_compact", "gpu_budget_int8")
        pair_b = ("m_compact", "gpu_fast_fp16")
        state = make_state(
            x={(query.id, *pair_a): 0.25, (query.id, *pair_b): 0.25},
            z={(query.id, *pair_a): 1, (query.id, *pair_b): 1},
        )
        result = result_for_query(
            compute_storage_capacity_residuals(
                replace(base, queries=queries, models=models), state
            ),
            query.id,
        )

        self.assertEqual(result.model_storage_usage_gb, 40.0)
        self.assertEqual(result.data_storage_rate_gb_per_hour, 0.5)
        self.assertEqual(result.printed_storage_lhs, 40.5)


class StorageCapacitySeparationTests(unittest.TestCase):
    """Keep Eq. (8h) separate from horizon, prices, and monetary Eq. (8c)."""

    def test_horizon_prices_and_cost_budget_do_not_change_storage(self):
        """Exclude ``DeltaT``, ``p_s``, ``p_gpu``, and ``delta`` from Eq. (8h)."""

        base = make_storage_instance()
        expensive_tier = replace(base.gpu_tiers[0], hourly_rental_usd=999.0)
        changed = replace(
            base,
            gpu_tiers=(expensive_tier,),
            horizon_hours=24.0,
            storage_price_usd_per_gb_hour=999.0,
            cost_budget_usd=0.0,
        )
        first = result_for_query(
            compute_storage_capacity_residuals(base, make_printed_state())
        )
        second = result_for_query(
            compute_storage_capacity_residuals(changed, make_printed_state())
        )

        self.assertEqual(first, second)

    def test_budget_and_storage_can_fail_independently(self):
        """Construct budget-pass/storage-fail and budget-fail/storage-pass cases."""

        state = make_printed_state()
        storage_tight = make_storage_instance(
            storage_capacity=20.25,
            storage_price=0.001,
            cost_budget=100.0,
        )
        budget_tight = make_storage_instance(
            storage_capacity=100.0,
            storage_price=1.0,
            cost_budget=10.0,
        )

        first_storage = result_for_query(
            compute_storage_capacity_residuals(storage_tight, state)
        )
        first_budget = compute_budget_residual(storage_tight, state)
        second_storage = result_for_query(
            compute_storage_capacity_residuals(budget_tight, state)
        )
        second_budget = compute_budget_residual(budget_tight, state)

        self.assertFalse(first_storage.is_feasible())
        self.assertTrue(first_budget.is_feasible())
        self.assertTrue(second_storage.is_feasible())
        self.assertFalse(second_budget.is_feasible())


class StorageCapacityBoundaryTests(unittest.TestCase):
    """Verify sparse/domain behavior, tolerance, and constraint independence."""

    def test_sparse_zero_and_invalid_inputs(self):
        """Keep omissions zero while rejecting unknown IDs and incomplete data."""

        instance = make_storage_instance()
        sparse = result_for_query(
            compute_storage_capacity_residuals(instance, make_state())
        )
        self.assertEqual(sparse.printed_storage_lhs, 0.0)
        self.assertEqual(sparse.slack, 25.0)

        invalid_states = (
            make_state(x={("q_unknown", *PAIR): 0.0}),
            make_state(z={("q_text", "m_unknown", PAIR[1]): 0}),
            make_state(x={ROUTE: 1.000000001}),
            make_state(z={ROUTE: 0.5}),
        )
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    compute_storage_capacity_residuals(instance, state)

        incomplete = dict(instance.compute_gflop_per_token)
        incomplete.pop(ROUTE)
        with self.assertRaises(InstanceValidationError):
            compute_storage_capacity_residuals(
                replace(instance, compute_gflop_per_token=incomplete),
                make_state(),
            )

    def test_tolerance_changes_only_classification(self):
        """Preserve all raw fields while tolerating a tiny printed excess."""

        result = result_for_query(
            compute_storage_capacity_residuals(
                make_storage_instance(storage_capacity=20.4999999995),
                make_printed_state(),
            )
        )
        raw = (
            result.model_storage_usage_gb,
            result.data_storage_rate_gb_per_hour,
            result.printed_storage_lhs,
            result.storage_limit_gb,
            result.slack,
            result.violation,
        )

        self.assertFalse(result.is_feasible())
        self.assertTrue(result.is_feasible(tolerance=1e-9))
        self.assertEqual(
            (
                result.model_storage_usage_gb,
                result.data_storage_rate_gb_per_hour,
                result.printed_storage_lhs,
                result.storage_limit_gb,
                result.slack,
                result.violation,
            ),
            raw,
        )

    def test_invalid_tolerance_is_rejected(self):
        """Require a finite non-negative tolerance for the printed comparison."""

        result = result_for_query(
            compute_storage_capacity_residuals(
                make_storage_instance(), make_printed_state()
            )
        )
        for invalid in (-1e-9, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    result.is_feasible(invalid)

    def test_unrelated_constraint_failures_do_not_block_storage(self):
        """Compute Eq. (8h) while Eqs. (8b),(8d)-(8g),(8i)-(8k) fail."""

        instance = make_storage_instance(
            throughput_tflops=0.001,
            compute_gflop_per_token=100.0,
            compute_delay_seconds_per_token=1.0,
            delay_limit=0.1,
            base_error=0.5,
            error_limit=0.1,
        )
        state = make_state(
            x={ROUTE: 0.5},
            z={ROUTE: 0},
            q={PAIR: 1},
            u={"q_text": 0.0},
            y={PAIR: 1},
            v={AUXILIARY: 0.5},
        )

        demand = compute_demand_balance_residuals(
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
        routing = next(
            item
            for item in compute_routing_consistency_residuals(
                instance, state.routing_decisions
            )
            if (item.query_id, item.model_id, item.gpu_tier_id) == ROUTE
        )
        storage = result_for_query(
            compute_storage_capacity_residuals(instance, state)
        )

        for diagnostic in (
            demand,
            selection,
            gpu_link,
            compute,
            delay,
            error,
            routing,
        ):
            with self.subTest(diagnostic=type(diagnostic).__name__):
                self.assertFalse(diagnostic.is_feasible())
        self.assertEqual(storage.model_storage_usage_gb, 0.0)
        self.assertEqual(storage.data_storage_rate_gb_per_hour, 0.5)
        self.assertTrue(storage.is_feasible())


if __name__ == "__main__":
    unittest.main()
