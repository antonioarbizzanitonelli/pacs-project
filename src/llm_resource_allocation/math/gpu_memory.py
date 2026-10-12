"""Per-GPU model and KV-memory diagnostics under the instance unit convention.

Compare model-shard and KV terms with ``C_gpu[k]*q[j,k]``. Linearize
the printed product using supplied auxiliaries v for x*w. Coherent units add
arrival rate times request residency to account for mean KV-cache occupancy;
literal units retain the historical printed term.
"""

from dataclasses import dataclass
from math import fsum, isfinite

from llm_resource_allocation.domain import ProblemInstance, UnitConvention

from .placement import PlacementDecisionState, validate_placement_decision_domains


@dataclass(frozen=True, slots=True)
class GpuMemoryDiagnostic:
    """Record the memory comparison for one model/tier pair.

    ``model_memory_usage_gb`` is the model shard; ``kv_memory_printed_term`` is
    the linearized KV term. ``available_memory_rhs_gb = C_gpu[k]*q[j,k]``.
    ``slack = available_memory_rhs_gb - printed_memory_lhs``; the KV term's
    ``unit_convention`` identifies whether the KV term is coherent GB or the
    historical printed value. The old ``printed_*`` names remain compatible.
    """

    model_id: str
    gpu_tier_id: str
    model_memory_usage_gb: float
    kv_memory_printed_term: float
    available_memory_rhs_gb: float
    unit_convention: UnitConvention = UnitConvention.LITERAL_PAPER

    @property
    def kv_memory_usage_gb(self) -> float:
        """Return coherent KV occupancy in GB; reject literal mixed units."""

        if self.unit_convention is UnitConvention.LITERAL_PAPER:
            raise ValueError("literal KV memory is not a physical GB quantity")
        return self.kv_memory_printed_term

    @property
    def memory_usage_gb(self) -> float:
        """Return model plus KV occupancy under the recorded convention."""

        return self.model_memory_usage_gb + self.kv_memory_usage_gb

    @property
    def printed_memory_lhs(self) -> float:
        """Return the selected memory LHS; historical attribute retained."""

        return self.model_memory_usage_gb + self.kv_memory_printed_term

    @property
    def slack(self) -> float:
        """Return ``C_gpu[k]*q[j,k] - printed_memory_lhs`` (v2 Eq. 8f)."""

        return self.available_memory_rhs_gb - self.printed_memory_lhs

    @property
    def violation(self) -> float:
        """Return ``max(printed_memory_lhs - available_memory_rhs_gb, 0)``."""

        return max(-self.slack, 0.0)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``slack >= -tolerance`` for the printed Eq. (8f).

        The tolerance uses GB in coherent mode, or the legacy comparison scale.
        """

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("tolerance must be a finite non-negative real number")
        return self.slack >= -tolerance


def compute_model_memory_usage_gb_by_pair(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[tuple[str, str], float]:
    """Return ``(j,k) -> sum_(n,m) B_j/(n*m)*w[j,k,n,m]`` (v2 Eq. 8f).

    ``model.weight_gb`` is ``B_j``; ``config.gpu_count`` is ``n*m``.
    The result is per-GPU model memory in GB, using every supplied selector.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_model_memory_usage_gb_by_pair(instance, state)


def compute_kv_memory_printed_term_by_pair(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[tuple[str, str], float]:
    """Return the linearized KV term per pair under the selected convention.

    ``sum_(i,n,m) beta_j/(n*m)*r_i*T_res[i,j,k]*v[i,j,k,n,m]`` uses
    ``model.kv_cache_gb_per_token`` (beta), ``query.total_tokens`` (r),
    ``instance.kv_cache_residency_seconds`` (T_res), and ``auxiliary_products`` (v).
    Supplied v values are used directly.
    Coherent units multiply this term by ``lambda_i/seconds_per_hour`` so
    request arrivals times residency produce mean concurrent KV requests.
    Literal units preserve the historical GB*seconds term. Supplied auxiliaries
    remain independent evidence and are never repaired from x*w.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_kv_memory_printed_term_by_pair(instance, state)


def compute_gpu_memory_residuals(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> tuple[GpuMemoryDiagnostic, ...]:
    """Compare model and KV terms with ``C_gpu[k]*q[j,k]`` (v2 Eq. 8f).

    Return one diagnostic per model/tier pair in instance order, with
    ``slack = C_gpu[k]*q[j,k] - printed_memory_lhs``. Capacity is per GPU.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_gpu_memory_residuals(instance, state)


def _compute_model_memory_usage_gb_by_pair(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[tuple[str, str], float]:
    """Calculate from instance and decision domains validated by the caller."""
    weight_by_model = {model.id: model.weight_gb for model in instance.models}
    return {
        (model_id, tier.id): fsum(
            weight_by_model[model_id]
            / config.gpu_count
            * state.configuration_selectors.get(
                (
                    model_id,
                    tier.id,
                    config.tensor_parallel_degree,
                    config.pipeline_parallel_depth,
                ),
                0.0,
            )
            for config in tier.parallelism_configs
        )
        for model_id in instance.model_ids
        for tier in instance.gpu_tiers
    }


def _compute_kv_memory_printed_term_by_pair(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[tuple[str, str], float]:
    """Calculate from instance and decision domains validated by the caller."""
    kv_by_model = {
        model.id: model.kv_cache_gb_per_token for model in instance.models
    }
    return {
        (model_id, tier.id): fsum(
            kv_by_model[model_id]
            / config.gpu_count
            * query.total_tokens
            * instance.kv_cache_residency_seconds[
                (query.id, model_id, tier.id)
            ]
            * (
                query.arrival_rate_per_hour / instance.seconds_per_hour
                if instance.uses_coherent_units
                else 1.0
            )
            * state.auxiliary_products.get(
                (
                    query.id,
                    model_id,
                    tier.id,
                    config.tensor_parallel_degree,
                    config.pipeline_parallel_depth,
                ),
                0.0,
            )
            for query in instance.queries
            for config in tier.parallelism_configs
        )
        for model_id in instance.model_ids
        for tier in instance.gpu_tiers
    }


def _compute_gpu_memory_residuals(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> tuple[GpuMemoryDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""

    model_usage = _compute_model_memory_usage_gb_by_pair(instance, state)
    kv_usage = _compute_kv_memory_printed_term_by_pair(instance, state)
    residuals: list[GpuMemoryDiagnostic] = []
    for model_id in instance.model_ids:
        for tier in instance.gpu_tiers:
            pair = (model_id, tier.id)
            # Paper v2 Eq. (8f) gates one GPU's capacity with q; using y here
            # would silently turn the constraint into a fleet-memory bound.
            available_rhs = tier.memory_capacity_gb * state.deployment_indicators.get(
                pair, 0.0
            )
            residuals.append(
                GpuMemoryDiagnostic(
                    model_id=model_id,
                    gpu_tier_id=tier.id,
                    model_memory_usage_gb=model_usage[pair],
                    kv_memory_printed_term=kv_usage[pair],
                    available_memory_rhs_gb=available_rhs,
                    unit_convention=instance.unit_convention,
                )
            )
    return tuple(residuals)
