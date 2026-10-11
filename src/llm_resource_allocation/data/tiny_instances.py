"""Hand-checkable synthetic placement inputs; no paper data or replication evidence."""

from dataclasses import replace

from llm_resource_allocation.domain.types import (
    FoundationModel,
    GpuTier,
    ProblemInstance,
    QueryType,
)
from llm_resource_allocation.domain.units import UnitConvention, kilobytes_to_gigabytes
from llm_resource_allocation.domain.validation import validate_problem_instance


def make_minimal_tiny_instance() -> ProblemInstance:
    """Return the historical literal 1-query x 1-model x 1-tier fixture."""

    query = QueryType(
        id="q_text",
        arrival_rate_per_hour=10.0,
        input_tokens=2.0,
        output_tokens=1.0,
        storage_gb_per_token=kilobytes_to_gigabytes(1.0),
        delay_slo_seconds=10.0,
        error_slo=0.10,
        delay_penalty_usd_per_millisecond_query=0.01,
        unmet_penalty_usd_per_hour=5.0,
        max_unmet_fraction=0.20,
    )
    model = FoundationModel(
        id="m_small",
        weight_gb=1.0,
        kv_cache_gb_per_token=kilobytes_to_gigabytes(2.0),
    )
    tier = GpuTier(
        id="gpu_basic_fp16",
        memory_capacity_gb=8.0,
        compute_throughput_tflops=10.0,
        hourly_rental_usd=1.0,
        memory_bandwidth_gb_per_second=100.0,
        quantization_latency_scale=1.0,
        quantization_error_multiplier=1.0,
        tensor_parallel_degrees=(1, 2),
        pipeline_parallel_depths=(1,),
    )
    pair_key = (query.id, model.id)
    triple_key = (query.id, model.id, tier.id)
    instance = ProblemInstance(
        id="tiny_1x1x1",
        queries=(query,),
        models=(model,),
        gpu_tiers=(tier,),
        base_error_rate={pair_key: 0.05},
        compute_delay_seconds_per_token={triple_key: 0.10},
        communication_delay_seconds_per_token={triple_key: 0.01},
        compute_gflop_per_token={triple_key: 1.0},
        # Synthetic T_res: v2 Section 5.1 leaves its units unresolved.
        kv_cache_residency_seconds={triple_key: 1.0},
        storage_price_usd_per_gb_hour=0.001,
        cost_budget_usd=20.0,
        storage_capacity_gb=100.0,
        horizon_hours=1.0,
        compute_utilization=0.9,
        unit_convention=UnitConvention.LITERAL_PAPER,
    )
    validate_problem_instance(instance)
    return instance


def make_coherent_tiny_instance() -> ProblemInstance:
    """Return a validated synthetic 1x1x1 example with coherent physical units.

    Inputs match the historical minimal fixture; capacity and cost semantics
    follow the current coherent policy, with one-hour steady-state data storage.
    This is a verification example, not calibrated paper data.
    """

    instance = replace(
        make_minimal_tiny_instance(),
        id="tiny_1x1x1_coherent",
        unit_convention=UnitConvention.COHERENT,
    )
    validate_problem_instance(instance)
    return instance


def make_parallelism_tradeoff_tiny_instance() -> ProblemInstance:
    """Return the minimal fixture with a 0.2-second delay SLO requiring TP=2.

    ``u <= 0.2`` requires ``x >= 0.8``; TP=1 gives ``0.31*x > 0.2``.
    TP=2 gives ``D_cfg=0.16`` seconds and satisfies the delay limit.
    """

    base = make_minimal_tiny_instance()
    query = replace(base.queries[0], delay_slo_seconds=0.2)
    instance = replace(
        base,
        id="tiny_1x1x1_parallelism_tradeoff",
        queries=(query,),
    )
    validate_problem_instance(instance)
    return instance


def make_heterogeneous_tiny_instance() -> ProblemInstance:
    """Return the historical literal 2-query x 2-model x 2-tier fixture.

    Models differ in weight/error; tiers in memory, compute, price, precision,
    and PP depth. Arrival rates are 12 and 4 queries/hour.
    """

    queries = (
        QueryType(
            id="q_short",
            arrival_rate_per_hour=12.0,
            input_tokens=2.0,
            output_tokens=2.0,
            storage_gb_per_token=kilobytes_to_gigabytes(1.0),
            delay_slo_seconds=8.0,
            error_slo=0.08,
            delay_penalty_usd_per_millisecond_query=0.01,
            unmet_penalty_usd_per_hour=6.0,
            max_unmet_fraction=0.10,
        ),
        QueryType(
            id="q_long",
            arrival_rate_per_hour=4.0,
            input_tokens=6.0,
            output_tokens=4.0,
            storage_gb_per_token=kilobytes_to_gigabytes(3.0),
            delay_slo_seconds=20.0,
            error_slo=0.04,
            delay_penalty_usd_per_millisecond_query=0.02,
            unmet_penalty_usd_per_hour=10.0,
            max_unmet_fraction=0.05,
        ),
    )
    models = (
        FoundationModel(
            id="m_compact",
            weight_gb=2.0,
            kv_cache_gb_per_token=kilobytes_to_gigabytes(1.0),
        ),
        FoundationModel(
            id="m_accurate",
            weight_gb=6.0,
            kv_cache_gb_per_token=kilobytes_to_gigabytes(2.0),
        ),
    )
    gpu_tiers = (
        GpuTier(
            id="gpu_budget_int8",
            memory_capacity_gb=8.0,
            compute_throughput_tflops=10.0,
            hourly_rental_usd=0.5,
            memory_bandwidth_gb_per_second=100.0,
            quantization_latency_scale=0.5,
            quantization_error_multiplier=1.15,
            tensor_parallel_degrees=(1, 2),
            pipeline_parallel_depths=(1,),
        ),
        GpuTier(
            id="gpu_fast_fp16",
            memory_capacity_gb=24.0,
            compute_throughput_tflops=40.0,
            hourly_rental_usd=2.0,
            memory_bandwidth_gb_per_second=300.0,
            quantization_latency_scale=1.0,
            quantization_error_multiplier=1.0,
            tensor_parallel_degrees=(1, 2),
            pipeline_parallel_depths=(1, 2),
        ),
    )
    instance = ProblemInstance(
        id="tiny_2x2x2",
        queries=queries,
        models=models,
        gpu_tiers=gpu_tiers,
        base_error_rate={
            ("q_short", "m_compact"): 0.06,
            ("q_short", "m_accurate"): 0.03,
            ("q_long", "m_compact"): 0.05,
            ("q_long", "m_accurate"): 0.02,
        },
        compute_delay_seconds_per_token={
            ("q_short", "m_compact", "gpu_budget_int8"): 0.10,
            ("q_short", "m_compact", "gpu_fast_fp16"): 0.05,
            ("q_short", "m_accurate", "gpu_budget_int8"): 0.30,
            ("q_short", "m_accurate", "gpu_fast_fp16"): 0.15,
            ("q_long", "m_compact", "gpu_budget_int8"): 0.12,
            ("q_long", "m_compact", "gpu_fast_fp16"): 0.06,
            ("q_long", "m_accurate", "gpu_budget_int8"): 0.36,
            ("q_long", "m_accurate", "gpu_fast_fp16"): 0.18,
        },
        communication_delay_seconds_per_token={
            ("q_short", "m_compact", "gpu_budget_int8"): 0.020,
            ("q_short", "m_compact", "gpu_fast_fp16"): 0.010,
            ("q_short", "m_accurate", "gpu_budget_int8"): 0.030,
            ("q_short", "m_accurate", "gpu_fast_fp16"): 0.015,
            ("q_long", "m_compact", "gpu_budget_int8"): 0.020,
            ("q_long", "m_compact", "gpu_fast_fp16"): 0.010,
            ("q_long", "m_accurate", "gpu_budget_int8"): 0.030,
            ("q_long", "m_accurate", "gpu_fast_fp16"): 0.015,
        },
        compute_gflop_per_token={
            ("q_short", "m_compact", "gpu_budget_int8"): 1.0,
            ("q_short", "m_compact", "gpu_fast_fp16"): 2.0,
            ("q_short", "m_accurate", "gpu_budget_int8"): 3.0,
            ("q_short", "m_accurate", "gpu_fast_fp16"): 6.0,
            ("q_long", "m_compact", "gpu_budget_int8"): 1.0,
            ("q_long", "m_compact", "gpu_fast_fp16"): 2.0,
            ("q_long", "m_accurate", "gpu_budget_int8"): 3.0,
            ("q_long", "m_accurate", "gpu_fast_fp16"): 6.0,
        },
        # Synthetic T_res values preserve the v2 Section 5.1 unit ambiguity.
        kv_cache_residency_seconds={
            ("q_short", "m_compact", "gpu_budget_int8"): 1.0,
            ("q_short", "m_compact", "gpu_fast_fp16"): 0.5,
            ("q_short", "m_accurate", "gpu_budget_int8"): 2.0,
            ("q_short", "m_accurate", "gpu_fast_fp16"): 1.0,
            ("q_long", "m_compact", "gpu_budget_int8"): 2.0,
            ("q_long", "m_compact", "gpu_fast_fp16"): 1.0,
            ("q_long", "m_accurate", "gpu_budget_int8"): 4.0,
            ("q_long", "m_accurate", "gpu_fast_fp16"): 2.0,
        },
        storage_price_usd_per_gb_hour=0.001,
        cost_budget_usd=50.0,
        storage_capacity_gb=200.0,
        horizon_hours=2.0,
        compute_utilization=0.9,
        unit_convention=UnitConvention.LITERAL_PAPER,
    )
    validate_problem_instance(instance)
    return instance


def make_heterogeneous_solve_tiny_instance() -> ProblemInstance:
    """Return the 2x2x2 fixture with arrival rates of 36 and 12 queries/hour.

    D-0031 originally scaled these rates for exact validation of HiGHS output.
    D-0076 now tolerates the observed ``3e-12`` McCormick undershoot by default;
    this separate fixture retains its original inputs.
    """

    base = make_heterogeneous_tiny_instance()
    arrival_rate_by_query = {"q_short": 36.0, "q_long": 12.0}
    instance = replace(
        base,
        id="tiny_2x2x2_solve",
        queries=tuple(
            replace(
                query,
                arrival_rate_per_hour=arrival_rate_by_query[query.id],
            )
            for query in base.queries
        ),
    )
    validate_problem_instance(instance)
    return instance
