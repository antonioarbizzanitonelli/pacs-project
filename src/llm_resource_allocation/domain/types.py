"""Scientific inputs for the paper-v2 problem.

These structures bring together the workload, models, GPU tiers, and scientific
coefficients in a form that can be used independently of an optimization solver.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .units import SECONDS_PER_HOUR, UnitConvention


PairKey = tuple[str, str]
TripleKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class QueryType:
    """Describe one query type and its workload/SLO inputs.

    Field names include units to keep the paper's mixed units visible. Total
    tokens ``r_i`` are derived from ``h_i + f_i`` as specified by Paper v2,
    Table 1, rather than stored as redundant mutable state.
    """

    id: str
    arrival_rate_per_hour: float
    input_tokens: float
    output_tokens: float
    storage_gb_per_token: float
    delay_slo_seconds: float
    error_slo: float
    delay_penalty_usd_per_millisecond_query: float
    unmet_penalty_usd_per_hour: float
    max_unmet_fraction: float

    @property
    def total_tokens(self) -> float:
        """Return ``r_i = h_i + f_i`` from Paper v2, Table 1."""

        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class FoundationModel:
    """Describe model-weight and per-token KV-cache footprints.

    Both footprints use canonical GB-based units. The instance unit convention
    selects the treatment of KV-cache occupancy.
    """

    id: str
    weight_gb: float
    kv_cache_gb_per_token: float


@dataclass(frozen=True, slots=True)
class ParallelismConfig:
    """Represent one joint tensor/pipeline configuration from Paper v2, Eq. (2)."""

    tensor_parallel_degree: int
    pipeline_parallel_depth: int

    def __post_init__(self) -> None:
        """Enforce positive integer degrees as an intrinsic configuration invariant."""

        for name, value in (
            ("tensor_parallel_degree", self.tensor_parallel_degree),
            ("pipeline_parallel_depth", self.pipeline_parallel_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def gpu_count(self) -> int:
        """Return the selected ``TP * PP`` GPU count from Paper v2, Eq. (2)."""

        return self.tensor_parallel_degree * self.pipeline_parallel_depth


@dataclass(frozen=True, slots=True)
class GpuTier:
    """Describe one hardware/precision tier and its feasible TP/PP sets."""

    id: str
    memory_capacity_gb: float
    compute_throughput_tflops: float
    hourly_rental_usd: float
    memory_bandwidth_gb_per_second: float
    quantization_latency_scale: float
    quantization_error_multiplier: float
    tensor_parallel_degrees: tuple[int, ...]
    pipeline_parallel_depths: tuple[int, ...]

    def __post_init__(self) -> None:
        """Freeze feasible-degree sequences even when callers pass mutable lists."""

        object.__setattr__(
            self, "tensor_parallel_degrees", tuple(self.tensor_parallel_degrees)
        )
        object.__setattr__(
            self, "pipeline_parallel_depths", tuple(self.pipeline_parallel_depths)
        )

    @property
    def parallelism_configs(self) -> tuple[ParallelismConfig, ...]:
        """Return the Cartesian product ``N_k x M_k`` from Paper v2, Eq. (2)."""

        return tuple(
            ParallelismConfig(tensor_degree, pipeline_depth)
            for tensor_degree in self.tensor_parallel_degrees
            for pipeline_depth in self.pipeline_parallel_depths
        )


@dataclass(frozen=True, slots=True)
class ProblemInstance:
    """Store scientific inputs and the explicit physical-unit convention.

    Coefficient mappings use stable string-ID tuples rather than positional
    arrays. The representation makes missing `(i,j)` and `(i,j,k)` entries
    detectable and remains independent of NumPy/Pandas.

    ``kv_cache_residency_seconds`` stores each request's KV lifetime in coherent
    mode; ``data_residency_hours`` is the steady-state data lifetime. These are
    supplied inputs, rather than values reconstructed inside the algorithms.
    ``LITERAL_PAPER`` retains the historical printed formulas and time factors.
    """

    id: str
    queries: tuple[QueryType, ...]
    models: tuple[FoundationModel, ...]
    gpu_tiers: tuple[GpuTier, ...]
    base_error_rate: Mapping[PairKey, float]
    compute_delay_seconds_per_token: Mapping[TripleKey, float]
    communication_delay_seconds_per_token: Mapping[TripleKey, float]
    compute_gflop_per_token: Mapping[TripleKey, float]
    kv_cache_residency_seconds: Mapping[TripleKey, float]
    storage_price_usd_per_gb_hour: float
    cost_budget_usd: float
    storage_capacity_gb: float
    horizon_hours: float
    compute_utilization: float
    seconds_per_hour: float = SECONDS_PER_HOUR
    unit_convention: UnitConvention = UnitConvention.COHERENT
    data_residency_hours: float = 1.0

    def __post_init__(self) -> None:
        """Copy mutable inputs into stable tuples and read-only mappings."""

        object.__setattr__(self, "queries", tuple(self.queries))
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "gpu_tiers", tuple(self.gpu_tiers))
        for field_name in (
            "base_error_rate",
            "compute_delay_seconds_per_token",
            "communication_delay_seconds_per_token",
            "compute_gflop_per_token",
            "kv_cache_residency_seconds",
        ):
            # Copy before wrapping so mutations of a caller-owned dictionary
            # cannot silently change a validated scientific instance.
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )

    @property
    def uses_coherent_units(self) -> bool:
        """Return whether capacity and costs use the coherent unit policy."""

        return self.unit_convention is UnitConvention.COHERENT

    @property
    def data_storage_duration_hours(self) -> float:
        """Return data residency, or the historical numeric factor of one.

        The literal factor preserves the printed storage rate arithmetic; it
        does not assert that the historical equations specified one-hour storage.
        """

        return self.data_residency_hours if self.uses_coherent_units else 1.0

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Return `(I, J, K)` without storing redundant dimension fields."""

        return (len(self.queries), len(self.models), len(self.gpu_tiers))

    @property
    def query_ids(self) -> tuple[str, ...]:
        """Return query identifiers in deterministic instance order."""

        return tuple(query.id for query in self.queries)

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Return model identifiers in deterministic instance order."""

        return tuple(model.id for model in self.models)

    @property
    def gpu_tier_ids(self) -> tuple[str, ...]:
        """Return GPU-tier identifiers in deterministic instance order."""

        return tuple(tier.id for tier in self.gpu_tiers)
