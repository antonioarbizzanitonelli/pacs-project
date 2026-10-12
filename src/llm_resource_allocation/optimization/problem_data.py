"""Ordered indices and structural counts for deterministic ``P_DM``.

Paper v2 Table 1: ``i in I``, ``j in J``, ``k in K``, ``n in N_k``,
``m in M_k``. Model construction and extraction share these keys;
``ProblemInstance`` retains the coefficients. Counts precede solver presolve.
"""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.domain import ProblemInstance, validate_problem_instance


ModelTierKey = tuple[str, str]
RouteKey = tuple[str, str, str]
ConfigurationKey = tuple[str, str, int, int]
AuxiliaryProductKey = tuple[str, str, str, int, int]


class PdmVariableFamily(str, Enum):
    """Name the ``x,z,q,u,w,y,v`` decision families (v2 Table 1)."""

    X = "x"
    Z = "z"
    Q = "q"
    U = "u"
    W = "w"
    Y = "y"
    V = "v"


@dataclass(frozen=True, slots=True)
class PdmIndexSets:
    """Store ``I,J,K,N_k,M_k`` in validated instance order (v2 Table 1).

    ``query_ids``, ``model_ids``, ``tier_ids`` map to ``I,J,K``;
    tier mappings hold TP degrees ``N_k`` and PP depths ``M_k``.
    """

    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    tier_ids: tuple[str, ...]
    tensor_parallel_degrees_by_tier: Mapping[str, tuple[int, ...]]
    pipeline_parallel_depths_by_tier: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        """Copy each tier's TP/PP sequences to tuples and freeze both mappings."""

        object.__setattr__(
            self,
            "tensor_parallel_degrees_by_tier",
            MappingProxyType(
                {
                    tier_id: tuple(degrees)
                    for tier_id, degrees in self.tensor_parallel_degrees_by_tier.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "pipeline_parallel_depths_by_tier",
            MappingProxyType(
                {
                    tier_id: tuple(depths)
                    for tier_id, depths in self.pipeline_parallel_depths_by_tier.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PdmVariableLattices:
    """Store ordered keys for ``x,z,q,u,w,y,v`` (v2 Table 1).

    ``unmet_keys``: i for u; ``model_tier_keys``: (j,k) for q,y;
    ``route_keys``: (i,j,k) for x,z; ``configuration_keys``: (j,k,n,m) for w,
    using each tier's ``N_k x M_k``. ``auxiliary_product_keys`` adds i for v.
    """

    unmet_keys: tuple[str, ...]
    model_tier_keys: tuple[ModelTierKey, ...]
    route_keys: tuple[RouteKey, ...]
    configuration_keys: tuple[ConfigurationKey, ...]
    auxiliary_product_keys: tuple[AuxiliaryProductKey, ...]


@dataclass(frozen=True, slots=True)
class PdmVariableCounts:
    """Count decision variables before solver presolve.

    ``n_x = n_z = |I|*|J|*|K|``, ``n_q = n_y = |J|*|K|``, ``n_u = |I|``;
    ``n_w = sum_(j,k) |N_k|*|M_k|`` and ``n_v = |I|*n_w``.
    """

    n_x: int
    n_z: int
    n_q: int
    n_u: int
    n_w: int
    n_y: int
    n_v: int

    @property
    def total_decision_variable_count(self) -> int:
        """Return ``n_x + n_z + n_q + n_u + n_w + n_y + n_v``."""

        return (
            self.n_x
            + self.n_z
            + self.n_q
            + self.n_u
            + self.n_w
            + self.n_y
            + self.n_v
        )


@dataclass(frozen=True, slots=True)
class PdmConstraintCounts:
    """Count reference constraints before solver presolve (v2 Eqs. 7-8).

    Eq. (8k), ``0 <= x <= z <= q``: two rows plus one x lower bound.
    ``eq_8h_d_0016`` counts per-query storage constraints;
    ``mccormick_inequalities`` counts four rows per v key, including ``v >= 0``.
    """

    eq_8b: int
    eq_8c: int
    eq_8d: int
    eq_8e: int
    eq_8f: int
    eq_8g: int
    eq_8h_d_0016: int
    eq_8i: int
    eq_8j: int
    eq_8k_paper_semantic_inequalities: int
    eq_8k_explicit_linking_rows: int
    eq_8k_x_lower_bounds: int
    mccormick_inequalities: int

    @property
    def total_explicit_reference_rows(self) -> int:
        """Sum explicit Eq. (8b)-(8k) and McCormick rows.

        Include Eq. (8k)'s two linking rows; exclude x bounds and its separate
        three-inequality count to avoid double-counting.
        """

        return (
            self.eq_8b
            + self.eq_8c
            + self.eq_8d
            + self.eq_8e
            + self.eq_8f
            + self.eq_8g
            + self.eq_8h_d_0016
            + self.eq_8i
            + self.eq_8j
            + self.eq_8k_explicit_linking_rows
            + self.mccormick_inequalities
        )


@dataclass(frozen=True, slots=True)
class PdmProblemData:
    """Derive immutable ``P_DM`` indices, variable keys, and structural counts.

    Retain the validated ``instance`` by identity as coefficient authority;
    preserve its order and tier-specific TP/PP domains.
    """

    instance: ProblemInstance
    index_sets: PdmIndexSets = field(init=False)
    variable_lattices: PdmVariableLattices = field(init=False)
    variable_counts: PdmVariableCounts = field(init=False)
    constraint_counts: PdmConstraintCounts = field(init=False)

    def __post_init__(self) -> None:
        """Validate the instance, then derive ordered indices, keys, and counts."""

        validate_problem_instance(self.instance)

        index_sets = _build_index_sets(self.instance)
        lattices = _build_variable_lattices(index_sets)
        variable_counts = _count_variables(lattices)
        constraint_counts = _count_constraints(lattices)

        object.__setattr__(self, "index_sets", index_sets)
        object.__setattr__(self, "variable_lattices", lattices)
        object.__setattr__(self, "variable_counts", variable_counts)
        object.__setattr__(self, "constraint_counts", constraint_counts)


def _build_index_sets(instance: ProblemInstance) -> PdmIndexSets:
    """Collect validated instance IDs and tier-specific TP/PP domains in order."""

    return PdmIndexSets(
        query_ids=instance.query_ids,
        model_ids=instance.model_ids,
        tier_ids=instance.gpu_tier_ids,
        tensor_parallel_degrees_by_tier={
            tier.id: tier.tensor_parallel_degrees
            for tier in instance.gpu_tiers
        },
        pipeline_parallel_depths_by_tier={
            tier.id: tier.pipeline_parallel_depths
            for tier in instance.gpu_tiers
        },
    )


def _build_variable_lattices(index_sets: PdmIndexSets) -> PdmVariableLattices:
    """Form ordered variable keys from ``I,J,K`` and each tier's ``N_k x M_k``."""

    model_tier_keys = tuple(
        (model_id, tier_id)
        for model_id in index_sets.model_ids
        for tier_id in index_sets.tier_ids
    )
    route_keys = tuple(
        (query_id, model_id, tier_id)
        for query_id in index_sets.query_ids
        for model_id in index_sets.model_ids
        for tier_id in index_sets.tier_ids
    )
    # Use each tier's N_k x M_k; exclude unsupported TP/PP combinations.
    configuration_keys = tuple(
        (model_id, tier_id, tensor_degree, pipeline_depth)
        for model_id in index_sets.model_ids
        for tier_id in index_sets.tier_ids
        for tensor_degree in index_sets.tensor_parallel_degrees_by_tier[tier_id]
        for pipeline_depth in index_sets.pipeline_parallel_depths_by_tier[tier_id]
    )
    auxiliary_product_keys = tuple(
        (query_id, *configuration_key)
        for query_id in index_sets.query_ids
        for configuration_key in configuration_keys
    )
    return PdmVariableLattices(
        unmet_keys=index_sets.query_ids,
        model_tier_keys=model_tier_keys,
        route_keys=route_keys,
        configuration_keys=configuration_keys,
        auxiliary_product_keys=auxiliary_product_keys,
    )


def _count_variables(lattices: PdmVariableLattices) -> PdmVariableCounts:
    """Count the seven decision families from their variable-key sequences."""

    return PdmVariableCounts(
        n_x=len(lattices.route_keys),
        n_z=len(lattices.route_keys),
        n_q=len(lattices.model_tier_keys),
        n_u=len(lattices.unmet_keys),
        n_w=len(lattices.configuration_keys),
        n_y=len(lattices.model_tier_keys),
        n_v=len(lattices.auxiliary_product_keys),
    )


def _count_constraints(lattices: PdmVariableLattices) -> PdmConstraintCounts:
    """Count Eq. (8b)-(8k) and McCormick constraints from variable-key sequences."""

    n_queries = len(lattices.unmet_keys)
    n_model_tiers = len(lattices.model_tier_keys)
    n_routes = len(lattices.route_keys)
    return PdmConstraintCounts(
        eq_8b=n_queries,
        eq_8c=1,
        eq_8d=n_model_tiers,
        eq_8e=n_model_tiers,
        eq_8f=n_model_tiers,
        eq_8g=n_model_tiers,
        eq_8h_d_0016=n_queries,
        eq_8i=n_queries,
        eq_8j=n_queries,
        eq_8k_paper_semantic_inequalities=3 * n_routes,
        eq_8k_explicit_linking_rows=2 * n_routes,
        eq_8k_x_lower_bounds=n_routes,
        # Keep all four McCormick rows, including explicit v >= 0.
        mccormick_inequalities=4 * len(lattices.auxiliary_product_keys),
    )


def format_semantic_variable_name(
    family: PdmVariableFamily,
    key: str | ModelTierKey | RouteKey | ConfigurationKey | AuxiliaryProductKey,
) -> str:
    """Return a display name such as ``x[i,j,k]`` or ``w[j,k,n,m]``.

    Require a ``PdmVariableFamily`` and matching key arity or raise ``ValueError``;
    u accepts a query ID or a one-element tuple. Extraction uses semantic keys,
    not display names; validated IDs exclude commas and brackets.
    """

    if not isinstance(family, PdmVariableFamily):
        raise ValueError("family must be a PdmVariableFamily")

    expected_arity = {
        PdmVariableFamily.X: 3,
        PdmVariableFamily.Z: 3,
        PdmVariableFamily.Q: 2,
        PdmVariableFamily.U: 1,
        PdmVariableFamily.W: 4,
        PdmVariableFamily.Y: 2,
        PdmVariableFamily.V: 5,
    }[family]
    components = (key,) if isinstance(key, str) else key
    if not isinstance(components, tuple) or len(components) != expected_arity:
        raise ValueError(
            f"{family.value} keys must contain {expected_arity} semantic components"
        )
    return f"{family.value}[{','.join(str(component) for component in components)}]"
