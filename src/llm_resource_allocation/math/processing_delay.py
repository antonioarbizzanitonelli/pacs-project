"""Processing delay from v and D_cfg, with delay-SLO checks (Paper v2 Eq. 8i)."""

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Mapping

from llm_resource_allocation.domain import ProblemInstance

from .latency import compute_configuration_delay_seconds
from .placement import PlacementDecisionState, validate_placement_decision_domains


@dataclass(frozen=True, slots=True)
class DelaySloDiagnostic:
    """Record processing delay and its SLO for one query (v2 Eq. 8i).

    ``processing_delay = D_proc[i]``; ``delay_limit = Delta_i``.
    ``slack = Delta_i - D_proc[i]``; all values are in seconds.
    """

    query_id: str
    processing_delay: float
    delay_limit: float

    @property
    def slack(self) -> float:
        """Return ``Delta_i - D_proc[i]`` in seconds (v2 Eq. 8i)."""

        return self.delay_limit - self.processing_delay

    @property
    def violation(self) -> float:
        """Return ``max(processing_delay - delay_limit, 0)`` in seconds."""

        return max(-self.slack, 0.0)

    def is_feasible(self, tolerance: float = 0.0) -> bool:
        """Return whether ``slack >= -tolerance`` for Eq. (8i), in seconds."""

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("tolerance must be a finite non-negative real number")
        return self.slack >= -tolerance


def compute_processing_delays_seconds(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[str, float]:
    """Return ``i -> D_proc[i] = sum_(j,k,n,m) v[i,j,k,n,m]*D_cfg[i,j,k,n,m]``.

    Use the linearized expression after v2 Eqs. (7a)-(7b).
    ``compute_configuration_delay_seconds`` computes ``D_cfg = D_TTFT + D_Gen``
    in seconds, reusing the v2 Eqs. (3)-(4) functions. Use supplied
    ``state.auxiliary_products`` (v) directly; preserve query order.
    """

    validate_placement_decision_domains(instance, state)
    return _compute_processing_delays_seconds(instance, state)


def compute_delay_slo_residuals(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> tuple[DelaySloDiagnostic, ...]:
    """Return delay-SLO diagnostics for each query (v2 Eq. 8i).

    Compute ``D_proc[i]`` and pair it with ``query.delay_slo_seconds`` (Delta_i).
    Each diagnostic exposes ``slack = Delta_i - D_proc[i]`` in seconds.
    """

    processing_delays = compute_processing_delays_seconds(instance, state)
    return _delay_slo_residuals_from_processing_delays(instance, processing_delays)


def _delay_slo_residuals_from_processing_delays(
    instance: ProblemInstance,
    processing_delays: Mapping[str, float],
) -> tuple[DelaySloDiagnostic, ...]:
    """Pair already computed delays with query SLOs without repeating Eq. (6)."""

    return tuple(
        DelaySloDiagnostic(
            query_id=query.id,
            processing_delay=processing_delays[query.id],
            delay_limit=query.delay_slo_seconds,
        )
        for query in instance.queries
    )


def _compute_processing_delays_seconds(
    instance: ProblemInstance,
    state: PlacementDecisionState,
) -> dict[str, float]:
    """Calculate from instance and decision domains validated by the caller."""
    delays: dict[str, float] = {}
    for query in instance.queries:
        terms: list[float] = []
        for model_id in instance.model_ids:
            for tier in instance.gpu_tiers:
                triple_key = (query.id, model_id, tier.id)
                for config in tier.parallelism_configs:
                    auxiliary_product_key = (
                        *triple_key,
                        config.tensor_parallel_degree,
                        config.pipeline_parallel_depth,
                    )
                    # Sparse extracted v values mean exact zero. We still use
                    # the full lattice so coefficient omissions remain errors.
                    auxiliary = state.auxiliary_products.get(auxiliary_product_key, 0.0)
                    configuration_delay = compute_configuration_delay_seconds(
                        instance.compute_delay_seconds_per_token[triple_key],
                        instance.communication_delay_seconds_per_token[triple_key],
                        query,
                        config,
                    )
                    terms.append(auxiliary * configuration_delay)
        delays[query.id] = fsum(terms)
    return delays
