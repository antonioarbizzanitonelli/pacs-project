"""Hand-calculated and invariant tests for Milestone-3A primitives."""

from dataclasses import replace
import unittest

from llm_resource_allocation.data import make_minimal_tiny_instance
from llm_resource_allocation.domain import ParallelismConfig, QueryType
from llm_resource_allocation.math.latency import (
    compute_configuration_delay_seconds,
    compute_generation_delay_seconds,
    compute_ttft_seconds,
)
from llm_resource_allocation.math.error import compute_effective_error


def make_latency_test_query(*, input_tokens: float, output_tokens: float) -> QueryType:
    """Build a synthetic query whose non-latency fields do not affect formulas."""

    return QueryType(
        id="q_latency_test",
        arrival_rate_per_hour=1.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        storage_gb_per_token=0.0,
        delay_slo_seconds=10.0,
        error_slo=1.0,
        delay_penalty_usd_per_millisecond_query=0.0,
        unmet_penalty_usd_per_hour=0.0,
        max_unmet_fraction=1.0,
    )


class EffectiveErrorTests(unittest.TestCase):
    """Verify Paper v2, Eq. (1), without reusing production calculations."""

    def test_hand_calculated_effective_error(self) -> None:
        """Reproduce the required synthetic ``0.04 * 1.15 = 0.046`` case."""

        self.assertAlmostEqual(0.046, compute_effective_error(0.04, 1.15))

    def test_identity_and_inflation_invariants(self) -> None:
        """Check identity at ``mu=1`` and proportional growth for ``mu>1``."""

        base_error = 0.04
        self.assertAlmostEqual(base_error, compute_effective_error(base_error, 1.0))
        self.assertGreater(compute_effective_error(base_error, 1.15), base_error)
        self.assertAlmostEqual(0.054, compute_effective_error(base_error, 1.35))
        self.assertAlmostEqual(
            1.215,
            compute_effective_error(0.9, 1.35),
            msg="Paper v2, Eq. (1), does not clip effective error at one",
        )


class FixedConfigurationLatencyTests(unittest.TestCase):
    """Verify Paper v2, Eqs. (3)-(4), and its fixed ``D_cfg`` expression."""

    def setUp(self) -> None:
        """Create the transparent numerical case required by the macroprompt."""

        self.query = make_latency_test_query(input_tokens=100.0, output_tokens=50.0)
        self.config = ParallelismConfig(2, 2)
        self.compute_delay = 0.002
        self.communication_delay = 0.0001

    def test_hand_calculated_ttft_generation_and_total(self) -> None:
        """Compare all latency primitives with independently calculated values."""

        ttft = compute_ttft_seconds(self.compute_delay, self.query, self.config)
        generation = compute_generation_delay_seconds(
            self.compute_delay,
            self.communication_delay,
            self.query,
            self.config,
        )
        total = compute_configuration_delay_seconds(
            self.compute_delay,
            self.communication_delay,
            self.query,
            self.config,
        )

        self.assertAlmostEqual(0.1, ttft)
        self.assertAlmostEqual(0.06, generation)
        self.assertAlmostEqual(0.16, total)
        # Independent expansion of the printed D_cfg equation, not a call to
        # either component helper: 0.002*150/2 + 2*0.0001*50 = 0.16.
        direct_paper_formula = 0.002 * (100.0 + 50.0) / 2 + 2 * 0.0001 * 50.0
        self.assertAlmostEqual(direct_paper_formula, total)
        self.assertAlmostEqual(0.16, ttft + generation)

    def test_more_tensor_parallelism_reduces_compute_delay(self) -> None:
        """Check TP monotonicity in both prefill and generation compute terms."""

        low_tp = ParallelismConfig(1, 2)
        high_tp = ParallelismConfig(4, 2)
        self.assertGreater(
            compute_ttft_seconds(self.compute_delay, self.query, low_tp),
            compute_ttft_seconds(self.compute_delay, self.query, high_tp),
        )
        self.assertGreater(
            compute_generation_delay_seconds(
                self.compute_delay, self.communication_delay, self.query, low_tp
            ),
            compute_generation_delay_seconds(
                self.compute_delay, self.communication_delay, self.query, high_tp
            ),
        )

    def test_pipeline_depth_affects_only_positive_communication(self) -> None:
        """Check PP monotonicity and its disappearance when ``d_comm=0``."""

        shallow = ParallelismConfig(2, 1)
        deep = ParallelismConfig(2, 4)
        shallow_delay = compute_generation_delay_seconds(
            self.compute_delay, self.communication_delay, self.query, shallow
        )
        deep_delay = compute_generation_delay_seconds(
            self.compute_delay, self.communication_delay, self.query, deep
        )
        self.assertGreater(deep_delay, shallow_delay)
        self.assertAlmostEqual(
            compute_generation_delay_seconds(
                self.compute_delay, 0.0, self.query, shallow
            ),
            compute_generation_delay_seconds(
                self.compute_delay, 0.0, self.query, deep
            ),
        )

    def test_zero_output_has_no_generation_delay(self) -> None:
        """Check that ``f_i=0`` makes generation zero and total equal TTFT."""

        no_output = replace(self.query, output_tokens=0.0)
        generation = compute_generation_delay_seconds(
            self.compute_delay,
            self.communication_delay,
            no_output,
            self.config,
        )
        ttft = compute_ttft_seconds(self.compute_delay, no_output, self.config)
        total = compute_configuration_delay_seconds(
            self.compute_delay,
            self.communication_delay,
            no_output,
            self.config,
        )
        self.assertAlmostEqual(0.0, generation)
        self.assertAlmostEqual(ttft, total)

    def test_configuration_delay_equals_ttft_plus_generation_across_cases(self) -> None:
        """Regress Paper v2, Eqs. (3)-(4), over several fixed configurations."""

        cases = (
            (0.002, 0.0001, 100.0, 50.0, ParallelismConfig(1, 1)),
            (0.002, 0.0001, 100.0, 50.0, ParallelismConfig(8, 4)),
            (0.015, 0.0, 7.0, 13.0, ParallelismConfig(2, 4)),
            (0.003, 0.004, 20.0, 0.0, ParallelismConfig(4, 2)),
        )
        for d_comp, d_comm, h_i, f_i, config in cases:
            with self.subTest(
                d_comp=d_comp,
                d_comm=d_comm,
                h_i=h_i,
                f_i=f_i,
                config=config,
            ):
                query = make_latency_test_query(
                    input_tokens=h_i,
                    output_tokens=f_i,
                )
                total = compute_configuration_delay_seconds(
                    d_comp, d_comm, query, config
                )
                component_sum = compute_ttft_seconds(
                    d_comp, query, config
                ) + compute_generation_delay_seconds(
                    d_comp, d_comm, query, config
                )
                self.assertAlmostEqual(component_sum, total)

    def test_minimal_instance_coefficients_are_resolved_by_semantic_ids(self) -> None:
        """Integrate the primitives with the existing deterministic 1x1x1 fixture."""

        instance = make_minimal_tiny_instance()
        query = next(query for query in instance.queries if query.id == "q_text")
        tier = next(tier for tier in instance.gpu_tiers if tier.id == "gpu_basic_fp16")
        pair_key = (query.id, "m_small")
        triple_key = (query.id, "m_small", tier.id)
        config = ParallelismConfig(2, 1)

        effective_error = compute_effective_error(
            instance.base_error_rate[pair_key],
            tier.quantization_error_multiplier,
        )
        total = compute_configuration_delay_seconds(
            instance.compute_delay_seconds_per_token[triple_key],
            instance.communication_delay_seconds_per_token[triple_key],
            query,
            config,
        )

        # Fixture arithmetic: 0.05*1 = 0.05 and 0.10*3/2 + 1*0.01*1 = 0.16.
        self.assertAlmostEqual(0.05, effective_error)
        self.assertAlmostEqual(0.16, total)


if __name__ == "__main__":
    unittest.main()
