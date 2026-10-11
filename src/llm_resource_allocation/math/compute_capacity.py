"""Compute demand and available throughput for Paper v2 Eq. (8g), in TFLOP/hour."""

from dataclasses import dataclass
from math import fsum, isfinite

from llm_resource_allocation.domain import ProblemInstance
from llm_resource_allocation.domain.units import GFLOPS_PER_TFLOP

from .parallelism import (
    ParallelismDecisionState,
    validate_parallelism_decision_domains,
    _validate_parallelism_configuration_domains,
)
from .routing import RoutingDecisionState, validate_routing_decision_domains


@dataclass(frozen=True, slots=True)
class ComputeCapacityDiagnostic:
    """Record required and available compute for one pair (v2 Eq. 8g).

    ``required_compute`` is the routed load; ``available_compute`` is GPU capacity.
    ``slack = available_compute - required_compute``; all values are TFLOP/hour.
    """

    model_id: str
    gpu_tier_id: str
    required_compute: float
    available_compute: float

    @property
    def slack(self) -> float:
        """Return ``available_compute - required_compute`` (v2 Eq. 8g)."""

        return self.available_compute - self.required_compute

    @property
    def violation(self) -> float:
        """Return ``max(required_compute - available_compute, 0)`` in TFLOP/hour."""

        return max(-self.slack, 0.0)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``slack >= -tolerance`` for Eq. (8g), in TFLOP/hour."""

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("tolerance must be a finite non-negative real number")
        return self.slack >= -tolerance


def compute_required_compute_tflops_per_hour(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> dict[tuple[str, str], float]:
    """Return ``(j,k) -> sum_i alpha[i,j,k]*r_i*lambda_i*x[i,j,k]/10^3``.

    This is routed compute demand in TFLOP/hour (v2 Eq. 8g). ``alpha`` is
    ``instance.compute_gflop_per_token``; ``r_i`` is ``query.total_tokens``,
    ``lambda_i`` is ``query.arrival_rate_per_hour``, and ``x`` is
    ``state.routing_fractions``. Division by 1000 converts GFLOP to TFLOP.
    """

    validate_routing_decision_domains(instance, state)
    return _compute_required_compute_tflops_per_hour(instance, state)


def compute_available_compute_tflops_per_hour(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> dict[tuple[str, str], float]:
    """Return ``(j,k) -> eta*T_conv*P_gpu[k]*y[j,k]`` in TFLOP/hour (v2 Eq. 8g).

    ``eta`` is ``instance.compute_utilization``; ``T_conv`` is
    ``instance.seconds_per_hour``; ``P_gpu`` is ``tier.compute_throughput_tflops``.
    ``state.gpu_counts`` supplies ``y`` directly.
    """

    validate_parallelism_decision_domains(instance, state)
    return _compute_available_compute_tflops_per_hour(instance, state)


def compute_compute_capacity_diagnostics(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> tuple[ComputeCapacityDiagnostic, ...]:
    """Return compute-capacity diagnostics for each model/tier pair (v2 Eq. 8g).

    ``required_compute`` sums the load routed through x[i,j,k];
    ``available_compute`` uses the allocated GPU count y[j,k].
    Both are in TFLOP/hour; each diagnostic derives
    ``slack = available_compute - required_compute``.
    """

    validate_routing_decision_domains(instance, state.routing_decisions)
    _validate_parallelism_configuration_domains(instance, state)
    return _compute_compute_capacity_diagnostics(instance, state)


def _compute_required_compute_tflops_per_hour(
    instance: ProblemInstance,
    state: RoutingDecisionState,
) -> dict[tuple[str, str], float]:
    """Calculate from instance and decision domains validated by the caller."""
    required_by_pair: dict[tuple[str, str], float] = {}
    for model_id in instance.model_ids:
        for tier_id in instance.gpu_tier_ids:
            # Stable summation preserves the printed sum over query types while
            # reducing addition-order artifacts in capacity residuals.
            required_by_pair[(model_id, tier_id)] = fsum(
                instance.compute_gflop_per_token[(query.id, model_id, tier_id)]
                * query.total_tokens
                * query.arrival_rate_per_hour
                * state.routing_fractions.get(
                    (query.id, model_id, tier_id), 0.0
                )
                / GFLOPS_PER_TFLOP
                for query in instance.queries
            )
    return required_by_pair


def _compute_available_compute_tflops_per_hour(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> dict[tuple[str, str], float]:
    """Calculate from instance and decision domains validated by the caller."""
    capacity_by_pair: dict[tuple[str, str], float] = {}
    for model_id in instance.model_ids:
        for tier in instance.gpu_tiers:
            gpu_count = state.gpu_counts.get((model_id, tier.id), 0.0)
            capacity_by_pair[(model_id, tier.id)] = (
                instance.compute_utilization
                * instance.seconds_per_hour
                * tier.compute_throughput_tflops
                * gpu_count
            )
    return capacity_by_pair


def _compute_compute_capacity_diagnostics(
    instance: ProblemInstance,
    state: ParallelismDecisionState,
) -> tuple[ComputeCapacityDiagnostic, ...]:
    """Calculate from instance and decision domains validated by the caller."""

    required_by_pair = _compute_required_compute_tflops_per_hour(
        instance, state.routing_decisions
    )
    available_by_pair = _compute_available_compute_tflops_per_hour(instance, state)
    return tuple(
        ComputeCapacityDiagnostic(
            model_id=model_id,
            gpu_tier_id=tier_id,
            required_compute=required_by_pair[(model_id, tier_id)],
            available_compute=available_by_pair[(model_id, tier_id)],
        )
        for model_id in instance.model_ids
        for tier_id in instance.gpu_tier_ids
    )
