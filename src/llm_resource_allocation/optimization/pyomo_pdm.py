"""Build the ``P_DM`` MILP from Paper v2 Eqs. (7) and (8a)-(8k).

Symbolic arithmetic is independent of the numerical validator; solving is separate.
"""

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeIntegers,
    Objective,
    Set,
    UnitInterval,
    Var,
    minimize,
)

from llm_resource_allocation.domain import FoundationModel, GpuTier, QueryType
from llm_resource_allocation.domain.units import GFLOPS_PER_TFLOP, MILLISECONDS_PER_SECOND
from llm_resource_allocation.math.objective import PaperVersion

from .problem_data import (
    AuxiliaryProductKey,
    ConfigurationKey,
    ModelTierKey,
    PdmProblemData,
    RouteKey,
)


@dataclass(frozen=True, slots=True)
class _PdmModelLookups:
    """Share ordered lookup tables among symbolic construction helpers."""

    query_by_id: Mapping[str, QueryType]
    model_by_id: Mapping[str, FoundationModel]
    tier_by_id: Mapping[str, GpuTier]
    routes_by_query: Mapping[str, tuple[RouteKey, ...]]
    routes_by_pair: Mapping[ModelTierKey, tuple[RouteKey, ...]]
    configurations_by_pair: Mapping[ModelTierKey, tuple[ConfigurationKey, ...]]
    auxiliaries_by_query: Mapping[str, tuple[AuxiliaryProductKey, ...]]
    auxiliaries_by_pair: Mapping[ModelTierKey, tuple[AuxiliaryProductKey, ...]]


def _prepare_model_lookups(problem_data: PdmProblemData) -> _PdmModelLookups:
    """Partition authoritative keys without sorting or generating new keys."""

    instance = problem_data.instance
    lattices = problem_data.variable_lattices

    query_by_id = {query.id: query for query in instance.queries}
    model_by_id = {
        foundation_model.id: foundation_model
        for foundation_model in instance.models
    }
    tier_by_id = {tier.id: tier for tier in instance.gpu_tiers}

    routes_by_query: dict[str, list[RouteKey]] = defaultdict(list)
    routes_by_pair: dict[ModelTierKey, list[RouteKey]] = defaultdict(list)
    for route_key in lattices.route_keys:
        query_id, model_id, tier_id = route_key
        routes_by_query[query_id].append(route_key)
        routes_by_pair[(model_id, tier_id)].append(route_key)

    configurations_by_pair: dict[ModelTierKey, list[ConfigurationKey]] = defaultdict(list)
    for configuration_key in lattices.configuration_keys:
        model_id, tier_id, _, _ = configuration_key
        configurations_by_pair[(model_id, tier_id)].append(configuration_key)

    auxiliaries_by_query: dict[str, list[AuxiliaryProductKey]] = defaultdict(list)
    auxiliaries_by_pair: dict[ModelTierKey, list[AuxiliaryProductKey]] = defaultdict(list)
    for auxiliary_product_key in lattices.auxiliary_product_keys:
        query_id, model_id, tier_id, _, _ = auxiliary_product_key
        auxiliaries_by_query[query_id].append(auxiliary_product_key)
        auxiliaries_by_pair[(model_id, tier_id)].append(auxiliary_product_key)

    return _PdmModelLookups(
        query_by_id=MappingProxyType(query_by_id),
        model_by_id=MappingProxyType(model_by_id),
        tier_by_id=MappingProxyType(tier_by_id),
        routes_by_query=MappingProxyType({
            key: tuple(keys) for key, keys in routes_by_query.items()
        }),
        routes_by_pair=MappingProxyType({
            key: tuple(keys) for key, keys in routes_by_pair.items()
        }),
        configurations_by_pair=MappingProxyType({
            key: tuple(keys) for key, keys in configurations_by_pair.items()
        }),
        auxiliaries_by_query=MappingProxyType({
            key: tuple(keys) for key, keys in auxiliaries_by_query.items()
        }),
        auxiliaries_by_pair=MappingProxyType({
            key: tuple(keys) for key, keys in auxiliaries_by_pair.items()
        }),
    )


def _add_sets_and_variables(
    model: ConcreteModel,
    problem_data: PdmProblemData,
    lookups: _PdmModelLookups,
) -> None:
    """Attach ordered sets and the seven variable families with their domains."""

    indices = problem_data.index_sets
    lattices = problem_data.variable_lattices
    query_by_id = lookups.query_by_id

    # Preserve problem-data keys and order for model inspection and extraction.
    model.query_ids = Set(initialize=indices.query_ids, ordered=True)
    model.model_tier_keys = Set(
        dimen=2, initialize=lattices.model_tier_keys, ordered=True
    )
    model.route_keys = Set(dimen=3, initialize=lattices.route_keys, ordered=True)
    model.configuration_keys = Set(
        dimen=4, initialize=lattices.configuration_keys, ordered=True
    )
    model.auxiliary_product_keys = Set(
        dimen=5, initialize=lattices.auxiliary_product_keys, ordered=True
    )

    model.x = Var(model.route_keys, domain=UnitInterval)
    model.z = Var(model.route_keys, domain=Binary)
    model.q = Var(model.model_tier_keys, domain=Binary)
    model.u = Var(
        model.query_ids,
        domain=UnitInterval,
        bounds=lambda _model, query_id: (
            0.0,
            query_by_id[query_id].max_unmet_fraction,
        ),
    )
    model.w = Var(model.configuration_keys, domain=Binary)
    model.y = Var(model.model_tier_keys, domain=NonNegativeIntegers)
    model.v = Var(model.auxiliary_product_keys, domain=UnitInterval)


def _add_processing_delay(
    model: ConcreteModel,
    problem_data: PdmProblemData,
    lookups: _PdmModelLookups,
) -> None:
    """Build Eq. (6) from supplied v, preserving the printed coefficient arithmetic."""

    instance = problem_data.instance
    query_by_id = lookups.query_by_id
    auxiliaries_by_query = lookups.auxiliaries_by_query

    # v2 Eq. (6): D_proc = sum(D_cfg*v), in seconds; use supplied v.
    def processing_delay_rule(_model: ConcreteModel, query_id: str):
        query = query_by_id[query_id]
        return sum(
            (
                instance.compute_delay_seconds_per_token[
                    (query_id, model_id, tier_id)
                ]
                * query.total_tokens
                / tensor_degree
                + pipeline_depth
                * instance.communication_delay_seconds_per_token[
                    (query_id, model_id, tier_id)
                ]
                * query.output_tokens
            )
            * _model.v[query_id, model_id, tier_id, tensor_degree, pipeline_depth]
            for (
                _,
                model_id,
                tier_id,
                tensor_degree,
                pipeline_depth,
            ) in auxiliaries_by_query[query_id]
        )

    model.processing_delay = Expression(model.query_ids, rule=processing_delay_rule)


def _add_objective(
    model: ConcreteModel,
    problem_data: PdmProblemData,
    lookups: _PdmModelLookups,
    paper_version: PaperVersion,
) -> None:
    """Attach C1-C5, the versioned objective, and the budget expression."""

    instance = problem_data.instance
    indices = problem_data.index_sets
    lattices = problem_data.variable_lattices
    query_by_id = lookups.query_by_id
    model_by_id = lookups.model_by_id
    tier_by_id = lookups.tier_by_id

    # v2 Eq. (8a): C1-C5, independent of numerical objective recomputation.
    model.cost_gpu = Expression(
        expr=instance.horizon_hours
        * sum(
            tier_by_id[tier_id].hourly_rental_usd * model.y[model_id, tier_id]
            for model_id, tier_id in lattices.model_tier_keys
        )
    )
    model.cost_model_storage = Expression(
        expr=instance.horizon_hours
        * instance.storage_price_usd_per_gb_hour
        * sum(
            model_by_id[model_id].weight_gb * model.z[query_id, model_id, tier_id]
            for query_id, model_id, tier_id in lattices.route_keys
        )
    )
    data_storage_cost = (
        instance.horizon_hours
        * instance.storage_price_usd_per_gb_hour
        * sum(
            query_by_id[query_id].storage_gb_per_token
            * query_by_id[query_id].total_tokens
            * query_by_id[query_id].arrival_rate_per_hour
            * model.x[query_id, model_id, tier_id]
            for query_id, model_id, tier_id in lattices.route_keys
        )
    )
    if instance.uses_coherent_units:
        data_storage_cost *= instance.data_storage_duration_hours
    model.cost_data_storage = Expression(expr=data_storage_cost)

    def delay_penalty_term(query_id: str):
        query = query_by_id[query_id]
        term = (
            query.delay_penalty_usd_per_millisecond_query
            * model.processing_delay[query_id]
        )
        if instance.uses_coherent_units:
            term = (
                term
                * MILLISECONDS_PER_SECOND
                * query.arrival_rate_per_hour
                * instance.horizon_hours
            )
        return term

    model.delay_penalty = Expression(
        expr=sum(
            delay_penalty_term(query_id)
            for query_id in indices.query_ids
        )
    )
    unmet_horizon_factor = (
        1.0 if paper_version is PaperVersion.V1 else instance.horizon_hours
    )
    model.unmet_penalty = Expression(
        expr=unmet_horizon_factor
        * sum(
            query_by_id[query_id].unmet_penalty_usd_per_hour * model.u[query_id]
            for query_id in indices.query_ids
        )
    )
    # Literal mode preserves printed unit tensions; coherent mode charges USD.
    model.total_objective = Expression(
        expr=model.cost_gpu
        + model.cost_model_storage
        + model.cost_data_storage
        + model.delay_penalty
        + model.unmet_penalty
    )
    model.objective = Objective(expr=model.total_objective, sense=minimize)

    # v2 Eq. (8c): budget uses C1+C2+C3 under the selected unit convention.
    model.budget_use = Expression(
        expr=model.cost_gpu + model.cost_model_storage + model.cost_data_storage
    )


def _add_compute_expressions(
    model: ConcreteModel,
    problem_data: PdmProblemData,
    lookups: _PdmModelLookups,
) -> None:
    """Build the required and available compute expressions for Eq. (8g)."""

    instance = problem_data.instance
    query_by_id = lookups.query_by_id
    tier_by_id = lookups.tier_by_id
    routes_by_pair = lookups.routes_by_pair

    # v2 Eq. (8g): convert GFLOP demand to TFLOP; capacity uses seconds/hour.
    def required_compute_rule(
        _model: ConcreteModel, model_id: str, tier_id: str
    ):
        return sum(
            instance.compute_gflop_per_token[(query_id, model_id, tier_id)]
            * query_by_id[query_id].total_tokens
            * query_by_id[query_id].arrival_rate_per_hour
            * _model.x[query_id, model_id, tier_id]
            / GFLOPS_PER_TFLOP
            for query_id, _, _ in routes_by_pair[(model_id, tier_id)]
        )

    model.required_compute = Expression(
        model.model_tier_keys, rule=required_compute_rule
    )
    model.available_compute = Expression(
        model.model_tier_keys,
        rule=lambda _model, model_id, tier_id: (
            instance.compute_utilization
            * instance.seconds_per_hour
            * tier_by_id[tier_id].compute_throughput_tflops
            * _model.y[model_id, tier_id]
        ),
    )


def _add_memory_expressions(
    model: ConcreteModel,
    problem_data: PdmProblemData,
    lookups: _PdmModelLookups,
) -> None:
    """Build model and KV memory under the selected units, using supplied v."""

    instance = problem_data.instance
    query_by_id = lookups.query_by_id
    model_by_id = lookups.model_by_id
    tier_by_id = lookups.tier_by_id
    configurations_by_pair = lookups.configurations_by_pair
    auxiliaries_by_pair = lookups.auxiliaries_by_pair

    def model_memory_usage_rule(
        _model: ConcreteModel, model_id: str, tier_id: str
    ):
        return sum(
            model_by_id[model_id].weight_gb
            / (tensor_degree * pipeline_depth)
            * _model.w[model_id, tier_id, tensor_degree, pipeline_depth]
            for _, _, tensor_degree, pipeline_depth in configurations_by_pair[
                (model_id, tier_id)
            ]
        )

    model.model_memory_usage = Expression(
        model.model_tier_keys, rule=model_memory_usage_rule
    )

    def kv_memory_coefficient(
        query_id: str, model_id: str, tier_id: str,
        tensor_degree: int, pipeline_depth: int,
    ) -> float:
        coefficient = (
            model_by_id[model_id].kv_cache_gb_per_token
            / (tensor_degree * pipeline_depth)
            * query_by_id[query_id].total_tokens
            * instance.kv_cache_residency_seconds[(query_id, model_id, tier_id)]
        )
        if instance.uses_coherent_units:
            coefficient *= (
                query_by_id[query_id].arrival_rate_per_hour / instance.seconds_per_hour
            )
        return coefficient

    # Coherent KV occupancy = footprint * residency * arrivals/second.
    # The expression names remain available for historical model inspection.
    def kv_memory_printed_term_rule(
        _model: ConcreteModel, model_id: str, tier_id: str
    ):
        return sum(
            kv_memory_coefficient(
                query_id, model_id, tier_id, tensor_degree, pipeline_depth
            )
            * _model.v[
                query_id, model_id, tier_id, tensor_degree, pipeline_depth
            ]
            for (
                query_id,
                _,
                _,
                tensor_degree,
                pipeline_depth,
            ) in auxiliaries_by_pair[(model_id, tier_id)]
        )

    model.kv_memory_printed_term = Expression(
        model.model_tier_keys, rule=kv_memory_printed_term_rule
    )
    model.printed_memory_lhs = Expression(
        model.model_tier_keys,
        rule=lambda _model, model_id, tier_id: (
            _model.model_memory_usage[model_id, tier_id]
            + _model.kv_memory_printed_term[model_id, tier_id]
        ),
    )
    model.available_memory_rhs = Expression(
        model.model_tier_keys,
        rule=lambda _model, model_id, tier_id: (
            tier_by_id[tier_id].memory_capacity_gb * _model.q[model_id, tier_id]
        ),
    )


def _add_storage_and_error_expressions(
    model: ConcreteModel,
    problem_data: PdmProblemData,
    lookups: _PdmModelLookups,
) -> None:
    """Build one storage and one error expression per query."""

    query_by_id = lookups.query_by_id
    model_by_id = lookups.model_by_id
    tier_by_id = lookups.tier_by_id
    instance = problem_data.instance
    routes_by_query = lookups.routes_by_query

    model.model_storage_usage = Expression(
        model.query_ids,
        rule=lambda _model, query_id: sum(
            model_by_id[model_id].weight_gb
            * _model.z[query_id, model_id, tier_id]
            for _, model_id, tier_id in routes_by_query[query_id]
        ),
    )
    # Preserve the per-query closure and retain the raw data rate in GB/hour.
    model.data_storage_rate = Expression(
        model.query_ids,
        rule=lambda _model, query_id: sum(
            query_by_id[query_id].storage_gb_per_token
            * query_by_id[query_id].total_tokens
            * query_by_id[query_id].arrival_rate_per_hour
            * _model.x[query_id, model_id, tier_id]
            for _, model_id, tier_id in routes_by_query[query_id]
        ),
    )
    model.printed_storage_lhs = Expression(
        model.query_ids,
        rule=lambda _model, query_id: (
            _model.model_storage_usage[query_id]
            + (
                _model.data_storage_rate[query_id] * instance.data_storage_duration_hours
                if instance.uses_coherent_units
                else _model.data_storage_rate[query_id]
            )
        ),
    )
    model.weighted_error = Expression(
        model.query_ids,
        rule=lambda _model, query_id: sum(
            tier_by_id[tier_id].quantization_error_multiplier
            * instance.base_error_rate[(query_id, model_id)]
            * _model.x[query_id, model_id, tier_id]
            for _, model_id, tier_id in routes_by_query[query_id]
        ),
    )


def _add_mccormick_constraints(
    model: ConcreteModel,
) -> None:
    """Attach all four McCormick rows per v, including explicit non-negativity."""

    # v2 Eqs. (7a)-(7b): retain all four rows, including v >= 0.
    model.mccormick_v_le_x = Constraint(
        model.auxiliary_product_keys,
        rule=lambda _model, query_id, model_id, tier_id, tensor_degree, pipeline_depth: (
            _model.v[
                query_id, model_id, tier_id, tensor_degree, pipeline_depth
            ]
            <= _model.x[query_id, model_id, tier_id]
        ),
    )
    model.mccormick_v_le_w = Constraint(
        model.auxiliary_product_keys,
        rule=lambda _model, query_id, model_id, tier_id, tensor_degree, pipeline_depth: (
            _model.v[
                query_id, model_id, tier_id, tensor_degree, pipeline_depth
            ]
            <= _model.w[model_id, tier_id, tensor_degree, pipeline_depth]
        ),
    )
    model.mccormick_v_ge_x_plus_w_minus_1 = Constraint(
        model.auxiliary_product_keys,
        rule=lambda _model, query_id, model_id, tier_id, tensor_degree, pipeline_depth: (
            _model.v[
                query_id, model_id, tier_id, tensor_degree, pipeline_depth
            ]
            >= _model.x[query_id, model_id, tier_id]
            + _model.w[model_id, tier_id, tensor_degree, pipeline_depth]
            - 1.0
        ),
    )
    model.mccormick_v_nonnegative = Constraint(
        model.auxiliary_product_keys,
        rule=lambda _model, query_id, model_id, tier_id, tensor_degree, pipeline_depth: (
            _model.v[
                query_id, model_id, tier_id, tensor_degree, pipeline_depth
            ]
            >= 0.0
        ),
    )


def _add_placement_constraints(
    model: ConcreteModel,
    problem_data: PdmProblemData,
    lookups: _PdmModelLookups,
) -> None:
    """Attach Eqs. (8b)-(8k) using the previously constructed expressions."""

    instance = problem_data.instance
    query_by_id = lookups.query_by_id
    routes_by_query = lookups.routes_by_query
    configurations_by_pair = lookups.configurations_by_pair

    model.demand_balance = Constraint(
        model.query_ids,
        rule=lambda _model, query_id: (
            sum(
                _model.x[route_key]
                for route_key in routes_by_query[query_id]
            )
            + _model.u[query_id]
            == 1.0
        ),
    )
    model.budget = Constraint(expr=model.budget_use <= instance.cost_budget_usd)
    model.configuration_selection = Constraint(
        model.model_tier_keys,
        rule=lambda _model, model_id, tier_id: (
            sum(
                _model.w[configuration_key]
                for configuration_key in configurations_by_pair[(model_id, tier_id)]
            )
            == _model.q[model_id, tier_id]
        ),
    )
    model.gpu_count_linking = Constraint(
        model.model_tier_keys,
        rule=lambda _model, model_id, tier_id: (
            _model.y[model_id, tier_id]
            == sum(
                tensor_degree
                * pipeline_depth
                * _model.w[model_id, tier_id, tensor_degree, pipeline_depth]
                for _, _, tensor_degree, pipeline_depth in configurations_by_pair[
                    (model_id, tier_id)
                ]
            )
        ),
    )
    model.gpu_memory = Constraint(
        model.model_tier_keys,
        rule=lambda _model, model_id, tier_id: (
            _model.printed_memory_lhs[model_id, tier_id]
            <= _model.available_memory_rhs[model_id, tier_id]
        ),
    )
    model.compute_capacity = Constraint(
        model.model_tier_keys,
        rule=lambda _model, model_id, tier_id: (
            _model.required_compute[model_id, tier_id]
            <= _model.available_compute[model_id, tier_id]
        ),
    )
    model.storage_capacity = Constraint(
        model.query_ids,
        rule=lambda _model, query_id: (
            _model.printed_storage_lhs[query_id] <= instance.storage_capacity_gb
        ),
    )
    model.delay_slo = Constraint(
        model.query_ids,
        rule=lambda _model, query_id: (
            _model.processing_delay[query_id]
            <= query_by_id[query_id].delay_slo_seconds
        ),
    )
    model.error_slo = Constraint(
        model.query_ids,
        rule=lambda _model, query_id: (
            _model.weighted_error[query_id] <= query_by_id[query_id].error_slo
        ),
    )
    # x >= 0 is represented only by UnitInterval; Eq. (8k) adds two
    # explicit linking rows per route key.
    model.routing_x_le_z = Constraint(
        model.route_keys,
        rule=lambda _model, query_id, model_id, tier_id: (
            _model.x[query_id, model_id, tier_id]
            <= _model.z[query_id, model_id, tier_id]
        ),
    )
    model.routing_z_le_q = Constraint(
        model.route_keys,
        rule=lambda _model, query_id, model_id, tier_id: (
            _model.z[query_id, model_id, tier_id]
            <= _model.q[model_id, tier_id]
        ),
    )


def build_pdm_pyomo_model(
    problem_data: PdmProblemData,
    paper_version: PaperVersion = PaperVersion.V2,
) -> ConcreteModel:
    """Build symbolic variables, objective C1-C5, and all placement constraints.

    Index sets come directly from ``problem_data``. Paper v1 changes only C5's
    horizon factor in literal mode. Coherent mode supports primary Paper v2,
    converts KV occupancy and storage to GB, and charges delay in USD.
    """

    if not isinstance(problem_data, PdmProblemData):
        raise TypeError("problem_data must be a PdmProblemData")
    if not isinstance(paper_version, PaperVersion):
        raise ValueError("paper_version must be PaperVersion.V1 or PaperVersion.V2")
    if problem_data.instance.uses_coherent_units and paper_version is PaperVersion.V1:
        raise ValueError("coherent units require PaperVersion.V2; use literal_paper for v1")

    lookups = _prepare_model_lookups(problem_data)
    model = ConcreteModel(
        name=f"P_DM[{problem_data.instance.id},{paper_version.value}]"
    )
    _add_sets_and_variables(model, problem_data, lookups)
    _add_processing_delay(model, problem_data, lookups)
    _add_objective(model, problem_data, lookups, paper_version)
    _add_compute_expressions(model, problem_data, lookups)
    _add_memory_expressions(model, problem_data, lookups)
    _add_storage_and_error_expressions(model, problem_data, lookups)
    _add_mccormick_constraints(model)
    _add_placement_constraints(model, problem_data, lookups)
    return model
