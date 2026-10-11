"""Hand-calculated tests for independent Paper v2 Eq. (8c) diagnostics."""

from dataclasses import replace
from unittest.mock import patch
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
from llm_resource_allocation.math.objective import (
    ObjectiveState,
    PaperVersion,
    compute_data_storage_cost,
    compute_gpu_rental_cost,
    compute_model_storage_cost,
    compute_objective_breakdown,
)
from llm_resource_allocation.math.parallelism import (
    compute_configuration_selection_residuals,
    compute_gpu_count_linking_residuals,
)
from llm_resource_allocation.math.budget import compute_budget_residual
from llm_resource_allocation.math.compute_capacity import compute_compute_capacity_diagnostics
from llm_resource_allocation.math.processing_delay import compute_delay_slo_residuals
from llm_resource_allocation.math.error_slo import compute_error_slo_residuals

if __package__:
    from .placement_support import make_sparse_placement_state as make_state
else:
    from placement_support import make_sparse_placement_state as make_state


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ROUTE[1:]
SELECTOR = (*PAIR, 1, 1)
AUXILIARY = (*ROUTE, 1, 1)


def make_budget_instance(
    *,
    data_storage_cost: float = 10.0,
    budget_limit: float = 100.0,
    horizon_hours: float = 1.0,
    throughput_tflops: float = 1.0,
    compute_gflop_per_token: float = 1.0,
    compute_delay_seconds_per_token: float = 0.001,
    delay_limit: float = 10.0,
    base_error: float = 0.01,
    error_limit: float = 1.0,
    delay_penalty: float = 1.0,
    unmet_penalty: float = 1.0,
):
    """Build a valid one-pair instance with transparent C1-C3 values.

    At the default one-hour horizon and active state, C1=40, C2=20, and
    C3=`data_storage_cost`. Other coefficients are configurable only to create
    independent-constraint failure cases; they do not enter budget arithmetic.
    """

    base = make_minimal_tiny_instance()
    query = replace(
        base.queries[0],
        arrival_rate_per_hour=10.0,
        input_tokens=60.0,
        output_tokens=40.0,
        # With p_s=1, lambda=10, r=100, x=1, this yields the requested C3.
        storage_gb_per_token=(data_storage_cost / 1000.0),
        delay_slo_seconds=delay_limit,
        error_slo=error_limit,
        delay_penalty_usd_per_millisecond_query=delay_penalty,
        unmet_penalty_usd_per_hour=unmet_penalty,
        max_unmet_fraction=1.0,
    )
    model = replace(base.models[0], weight_gb=20.0)
    tier = replace(
        base.gpu_tiers[0],
        hourly_rental_usd=40.0,
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
        storage_price_usd_per_gb_hour=1.0,
        cost_budget_usd=budget_limit,
        horizon_hours=horizon_hours,
        compute_utilization=1.0,
    )


def make_active_state(*, unmet_fraction=0.0):
    """Return a domain-valid active state using every C1-C3 decision family."""

    return make_state(
        x={ROUTE: 1.0},
        z={ROUTE: 1},
        q={PAIR: 1},
        u={"q_text": unmet_fraction},
        w={SELECTOR: 1},
        y={PAIR: 1},
        v={AUXILIARY: 1.0},
    )


def result_for_pair(results, pair=PAIR):
    """Select one model/tier diagnostic from deterministic output."""

    return next(
        result
        for result in results
        if (result.model_id, result.gpu_tier_id) == pair
    )


class BudgetHandCalculationTests(unittest.TestCase):
    """Verify feasible, violated, and exact-boundary C1+C2+C3 cases."""

    def test_feasible_forty_twenty_ten_case(self):
        """Check `40 + 20 + 10 = 70 <= 100`, with slack `30`."""

        result = compute_budget_residual(
            make_budget_instance(data_storage_cost=10.0), make_active_state()
        )

        self.assertEqual(result.gpu_rental_cost, 40.0)
        self.assertEqual(result.model_storage_cost, 20.0)
        self.assertEqual(result.data_storage_cost, 10.0)
        self.assertEqual(result.total_budget_use, 70.0)
        self.assertEqual(result.budget_limit, 100.0)
        self.assertEqual(result.slack, 30.0)
        self.assertEqual(result.violation, 0.0)
        self.assertTrue(result.is_feasible())

    def test_violated_forty_twenty_fifty_case(self):
        """Check `40 + 20 + 50 = 110`, slack `-10`, violation `10`."""

        result = compute_budget_residual(
            make_budget_instance(data_storage_cost=50.0), make_active_state()
        )

        self.assertEqual(result.total_budget_use, 110.0)
        self.assertEqual(result.slack, -10.0)
        self.assertEqual(result.violation, 10.0)
        self.assertFalse(result.is_feasible())

    def test_exact_budget_boundary_is_feasible(self):
        """Check `40 + 20 + 40 = delta = 100` exactly."""

        result = compute_budget_residual(
            make_budget_instance(data_storage_cost=40.0), make_active_state()
        )

        self.assertEqual(result.total_budget_use, result.budget_limit)
        self.assertEqual(result.slack, 0.0)
        self.assertTrue(result.is_feasible())


class BudgetCompositionTests(unittest.TestCase):
    """Protect objective-component reuse and literal published semantics."""

    def test_component_fields_are_exact_direct_c1_c2_c3_calls(self):
        """Prove the diagnostic composes objective functions, not copied formulas."""

        instance = make_budget_instance()
        state = make_active_state()
        with (
            patch(
                "llm_resource_allocation.math.budget.compute_gpu_rental_cost",
                wraps=compute_gpu_rental_cost,
            ) as c1,
            patch(
                "llm_resource_allocation.math.budget.compute_model_storage_cost",
                wraps=compute_model_storage_cost,
            ) as c2,
            patch(
                "llm_resource_allocation.math.budget.compute_data_storage_cost",
                wraps=compute_data_storage_cost,
            ) as c3,
        ):
            result = compute_budget_residual(instance, state)

        c1.assert_called_once_with(instance, state.gpu_counts)
        c2.assert_called_once_with(instance, state.admission_indicators)
        c3.assert_called_once_with(instance, state.routing_fractions)
        self.assertEqual(
            result.gpu_rental_cost,
            compute_gpu_rental_cost(instance, state.gpu_counts),
        )
        self.assertEqual(
            result.model_storage_cost,
            compute_model_storage_cost(instance, state.admission_indicators),
        )
        self.assertEqual(
            result.data_storage_cost,
            compute_data_storage_cost(instance, state.routing_fractions),
        )

    def test_literal_c2_query_indexing_is_not_deduplicated(self):
        """Charge one shared model weight twice for two query-indexed `z` values."""

        base = make_heterogeneous_tiny_instance()
        models = tuple(
            replace(model, weight_gb=10.0)
            if model.id == "m_compact"
            else model
            for model in base.models
        )
        instance = replace(
            base,
            models=models,
            horizon_hours=1.0,
            storage_price_usd_per_gb_hour=1.0,
            cost_budget_usd=100.0,
        )
        pair = ("m_compact", "gpu_budget_int8")
        admissions = {
            ("q_short", *pair): 1,
            ("q_long", *pair): 1,
        }
        result = compute_budget_residual(
            instance, make_state(z=admissions)
        )

        # Paper v2 Eqs. (8a)/(8c): 1 h * $1/GB-h * 10 GB * 2 = $20.
        self.assertEqual(result.model_storage_cost, 20.0)
        self.assertNotEqual(result.model_storage_cost, 10.0)

    def test_horizon_scales_all_budget_use_with_fixed_delta(self):
        """Check `DeltaT -> 3*DeltaT` makes C1+C2+C3 three times larger."""

        short = make_budget_instance(horizon_hours=1.0)
        long = replace(short, horizon_hours=3.0)
        state = make_active_state()
        short_result = compute_budget_residual(short, state)
        long_result = compute_budget_residual(long, state)

        self.assertEqual(short_result.total_budget_use, 70.0)
        self.assertEqual(long_result.total_budget_use, 210.0)
        self.assertEqual(
            long_result.total_budget_use, 3.0 * short_result.total_budget_use
        )
        self.assertEqual(long_result.budget_limit, short_result.budget_limit)

    def test_large_c4_c5_do_not_enter_budget_use(self):
        """Keep Eq. (8c) feasible although the full objective exceeds `delta`."""

        instance = make_budget_instance(
            delay_penalty=1_000_000.0,
            unmet_penalty=1_000_000.0,
        )
        placement = make_active_state(unmet_fraction=1.0)
        budget = compute_budget_residual(instance, placement)
        objective_state = ObjectiveState(
            gpu_counts=placement.gpu_counts,
            admission_indicators=placement.admission_indicators,
            routing_fractions=placement.routing_fractions,
            unmet_fractions=placement.unmet_fractions,
            processing_delays_seconds={"q_text": 1.0},
        )
        objective = compute_objective_breakdown(
            instance, objective_state, PaperVersion.V2
        )

        self.assertEqual(budget.total_budget_use, 70.0)
        self.assertTrue(budget.is_feasible())
        self.assertGreater(objective.delay_penalty, instance.cost_budget_usd)
        self.assertGreater(objective.unmet_demand_penalty, instance.cost_budget_usd)
        self.assertGreater(objective.total, instance.cost_budget_usd)


class BudgetBoundaryAndIntegrationTests(unittest.TestCase):
    """Verify sparse domains, tolerance, and other-constraint independence."""

    def test_sparse_zero_state_and_invalid_inputs(self):
        """Keep decision omissions at zero but reject unknown IDs and bad data."""

        instance = make_budget_instance()
        sparse = compute_budget_residual(instance, make_state())
        self.assertEqual(sparse.total_budget_use, 0.0)
        self.assertEqual(sparse.slack, 100.0)
        self.assertTrue(sparse.is_feasible())

        invalid_states = (
            make_state(x={("q_unknown", *PAIR): 0.0}),
            make_state(z={("q_text", "m_unknown", PAIR[1]): 0}),
            make_state(y={(PAIR[0], "gpu_unknown"): 0}),
            make_state(x={ROUTE: 1.000000001}),
            make_state(z={ROUTE: 0.5}),
            make_state(y={PAIR: 0.5}),
        )
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(DecisionDomainError):
                    compute_budget_residual(instance, state)

        incomplete = dict(instance.compute_gflop_per_token)
        incomplete.pop(ROUTE)
        with self.assertRaises(InstanceValidationError):
            compute_budget_residual(
                replace(instance, compute_gflop_per_token=incomplete),
                make_state(),
            )

    def test_tolerance_changes_only_budget_classification(self):
        """Preserve raw C1-C3, total, delta, slack, and violation under tolerance."""

        instance = make_budget_instance(
            data_storage_cost=40.0000000005,
        )
        result = compute_budget_residual(instance, make_active_state())
        raw = (
            result.gpu_rental_cost,
            result.model_storage_cost,
            result.data_storage_cost,
            result.total_budget_use,
            result.budget_limit,
            result.slack,
            result.violation,
        )

        self.assertFalse(result.is_feasible())
        self.assertTrue(result.is_feasible(tolerance=1e-9))
        self.assertEqual(
            (
                result.gpu_rental_cost,
                result.model_storage_cost,
                result.data_storage_cost,
                result.total_budget_use,
                result.budget_limit,
                result.slack,
                result.violation,
            ),
            raw,
        )

    def test_invalid_tolerance_is_rejected(self):
        """Require a finite non-negative monetary tolerance."""

        result = compute_budget_residual(make_budget_instance(), make_active_state())
        for invalid in (-1e-9, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    result.is_feasible(invalid)

    def test_budget_computes_while_all_other_listed_constraints_fail(self):
        """Expose Eq. (8c) independently of Eqs. (8b),(8d)-(8k)."""

        instance = make_budget_instance(
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
            # Omitted w makes both Eqs. (8d) and (8e) fail with q=y=1.
            y={PAIR: 1},
            # Domain-valid v makes delay independently evaluable despite w=0.
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
        budget = compute_budget_residual(instance, state)

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
        self.assertEqual(budget.total_budget_use, 45.0)
        self.assertEqual(budget.slack, 55.0)
        self.assertTrue(budget.is_feasible())


if __name__ == "__main__":
    unittest.main()
