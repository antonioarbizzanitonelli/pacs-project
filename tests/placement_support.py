"""Mechanical placement builders shared by tests; expected arithmetic stays local."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.math.routing import RoutingDecisionState


ROUTE = ("q_text", "m_small", "gpu_basic_fp16")
PAIR = ("m_small", "gpu_basic_fp16")
W_11 = (*PAIR, 1, 1)
V_11 = (*ROUTE, 1, 1)


def snapshot_value(value):
    """Copy nested record values so later input mutations remain observable."""
    if is_dataclass(value):
        return tuple((field.name, snapshot_value(getattr(value, field.name)))
                     for field in fields(value))
    if isinstance(value, Mapping):
        return {key: snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(snapshot_value(item) for item in value)
    return value


def make_sparse_placement_state(*, x=None, z=None, q=None, u=None, w=None, y=None, v=None):
    """Build one sparse placement state without imposing constraint feasibility."""

    routing = RoutingDecisionState(
        routing_fractions={} if x is None else x,
        admission_indicators={} if z is None else z,
        deployment_indicators={} if q is None else q,
        unmet_fractions={} if u is None else u,
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors={} if w is None else w,
        gpu_counts={} if y is None else y,
    )
    return PlacementDecisionState(
        parallelism_decisions=parallelism,
        auxiliary_products={} if v is None else v,
    )


def make_minimal_placement_state(
    *,
    x: float = 1.0,
    z: float = 1.0,
    q: float = 1.0,
    u: float = 0.0,
    w: dict[tuple[str, str, int, int], float] | None = None,
    y: float = 1.0,
    v: dict[tuple[str, str, str, int, int], float] | None = None,
) -> PlacementDecisionState:
    """Build the established synthetic state without relationship repair."""

    routing = RoutingDecisionState(
        routing_fractions={ROUTE: x},
        admission_indicators={ROUTE: z},
        deployment_indicators={PAIR: q},
        unmet_fractions={"q_text": u},
    )
    parallelism = ParallelismDecisionState(
        routing_decisions=routing,
        configuration_selectors={W_11: 1.0} if w is None else w,
        gpu_counts={PAIR: y},
    )
    return PlacementDecisionState(
        parallelism_decisions=parallelism,
        auxiliary_products={V_11: x} if v is None else v,
    )
