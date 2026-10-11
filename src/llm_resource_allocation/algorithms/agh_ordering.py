"""Build deterministic and seeded random query orderings for AGH."""

from dataclasses import dataclass
from enum import Enum
from random import Random

from llm_resource_allocation.domain import ProblemInstance, QueryType, validate_problem_instance


class AGHOrderingCriterion(str, Enum):
    """Select the query value used for deterministic sorting."""

    ARRIVAL_RATE = "arrival_rate"
    UNMET_PENALTY = "unmet_penalty"
    STORAGE_PER_TOKEN = "storage_per_token"
    ERROR_SLO = "error_slo"


class AGHOrderingDirection(str, Enum):
    """Select ascending or descending query sorting."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class AGHOrderingSource(str, Enum):
    """Identify how a query ordering was generated."""

    DETERMINISTIC = "deterministic"
    SEEDED_RANDOM = "seeded_random"


@dataclass(frozen=True, slots=True)
class AGHDeterministicOrderingSpec:
    """Store the criterion, direction, and label for one deterministic ordering."""

    criterion: AGHOrderingCriterion
    direction: AGHOrderingDirection
    label: str

    def __post_init__(self) -> None:
        """Reject malformed criteria, directions, or labels."""

        if not isinstance(self.criterion, AGHOrderingCriterion):
            raise ValueError("criterion must be an AGHOrderingCriterion")
        if not isinstance(self.direction, AGHOrderingDirection):
            raise ValueError("direction must be an AGHOrderingDirection")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")


DETERMINISTIC_AGH_ORDERING_SPECS = (
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.ARRIVAL_RATE,
        AGHOrderingDirection.ASCENDING,
        "lambda_ascending",
    ),
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.ARRIVAL_RATE,
        AGHOrderingDirection.DESCENDING,
        "lambda_descending",
    ),
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.UNMET_PENALTY,
        AGHOrderingDirection.ASCENDING,
        "phi_ascending",
    ),
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.UNMET_PENALTY,
        AGHOrderingDirection.DESCENDING,
        "phi_descending",
    ),
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.STORAGE_PER_TOKEN,
        AGHOrderingDirection.ASCENDING,
        "theta_ascending",
    ),
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.STORAGE_PER_TOKEN,
        AGHOrderingDirection.DESCENDING,
        "theta_descending",
    ),
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.ERROR_SLO,
        AGHOrderingDirection.ASCENDING,
        "epsilon_ascending",
    ),
    AGHDeterministicOrderingSpec(
        AGHOrderingCriterion.ERROR_SLO,
        AGHOrderingDirection.DESCENDING,
        "epsilon_descending",
    ),
)
"""Eight lambda/phi/theta/epsilon specifications, ascending then descending."""


@dataclass(frozen=True, slots=True)
class AGHQueryOrdering:
    """Store one query permutation and the inputs used to generate it."""

    source: AGHOrderingSource
    query_ids: tuple[str, ...]
    deterministic_spec: AGHDeterministicOrderingSpec | None = None
    seed: int | None = None
    random_start_index: int | None = None

    def __post_init__(self) -> None:
        """Freeze query IDs and check deterministic or random generation inputs."""

        object.__setattr__(self, "query_ids", tuple(self.query_ids))
        if not isinstance(self.source, AGHOrderingSource):
            raise ValueError("source must be an AGHOrderingSource")
        _validate_query_ids(self.query_ids)
        if self.source is AGHOrderingSource.DETERMINISTIC:
            if not isinstance(self.deterministic_spec, AGHDeterministicOrderingSpec):
                raise ValueError("deterministic ordering requires its specification")
            if self.seed is not None or self.random_start_index is not None:
                raise ValueError("deterministic ordering cannot contain random evidence")
        else:
            if self.deterministic_spec is not None:
                raise ValueError("random ordering cannot contain a deterministic spec")
            _validate_seed_and_index(self.seed, self.random_start_index)


def build_deterministic_agh_ordering(
    instance: ProblemInstance,
    specification: AGHDeterministicOrderingSpec,
) -> AGHQueryOrdering:
    """Sort queries by one criterion, preserving instance order on exact ties.

    Equal values retain the same relative order in both sorting directions.
    """

    validate_problem_instance(instance)
    if not isinstance(specification, AGHDeterministicOrderingSpec):
        raise ValueError("specification must be an AGHDeterministicOrderingSpec")
    return _build_deterministic_agh_ordering(instance, specification)


def build_all_deterministic_agh_orderings(
    instance: ProblemInstance,
) -> tuple[AGHQueryOrdering, ...]:
    """Return all eight deterministic orderings in specification order.

    Keep each ordering even when its query permutation duplicates another.
    """

    validate_problem_instance(instance)
    return _build_all_deterministic_agh_orderings(instance)


def build_seeded_random_agh_ordering(
    query_ids: tuple[str, ...],
    *,
    seed: int,
    random_start_index: int,
) -> AGHQueryOrdering:
    """Return the shuffle at a zero-based index in a local seeded stream.

    Each shuffle starts from the supplied query order. Repeated permutations
    are retained, and the result records the seed and index for replay.
    """

    input_query_ids = tuple(query_ids)
    _validate_query_ids(input_query_ids)
    _validate_seed_and_index(seed, random_start_index)
    rng = Random(seed)
    for _ in range(random_start_index + 1):
        permutation = list(input_query_ids)
        rng.shuffle(permutation)
    return AGHQueryOrdering(
        source=AGHOrderingSource.SEEDED_RANDOM,
        query_ids=tuple(permutation),
        seed=seed,
        random_start_index=random_start_index,
    )


def compute_adaptive_agh_random_start_count(
    query_type_count: int,
    model_count: int,
    gpu_tier_count: int,
) -> int:
    """Return the random-start limit for the product of the three entity counts.

    ``R=3`` above 5000, ``5`` above 2000, ``10`` above 500, and ``20`` otherwise.
    The eight deterministic starts are additional.
    """

    for name, value in (
        ("query_type_count", query_type_count),
        ("model_count", model_count),
        ("gpu_tier_count", gpu_tier_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    problem_scale = query_type_count * model_count * gpu_tier_count
    if problem_scale > 5000:
        return 3
    if problem_scale > 2000:
        return 5
    if problem_scale > 500:
        return 10
    return 20


def _build_deterministic_agh_ordering(
    instance: ProblemInstance,
    specification: AGHDeterministicOrderingSpec,
) -> AGHQueryOrdering:
    """Build one deterministic ordering from validated inputs."""

    reverse = specification.direction is AGHOrderingDirection.DESCENDING
    ordered_queries = sorted(
        instance.queries,
        key=lambda query: _criterion_value(query, specification.criterion),
        reverse=reverse,
    )
    return AGHQueryOrdering(
        source=AGHOrderingSource.DETERMINISTIC,
        query_ids=tuple(query.id for query in ordered_queries),
        deterministic_spec=specification,
    )


def _build_all_deterministic_agh_orderings(
    instance: ProblemInstance,
) -> tuple[AGHQueryOrdering, ...]:
    """Build all deterministic orderings from a validated instance."""

    return tuple(
        _build_deterministic_agh_ordering(instance, specification)
        for specification in DETERMINISTIC_AGH_ORDERING_SPECS
    )


def _criterion_value(query: QueryType, criterion: AGHOrderingCriterion) -> float:
    """Read the query rate, unmet penalty, storage per token, or error SLO."""

    if criterion is AGHOrderingCriterion.ARRIVAL_RATE:
        return query.arrival_rate_per_hour
    if criterion is AGHOrderingCriterion.UNMET_PENALTY:
        return query.unmet_penalty_usd_per_hour
    if criterion is AGHOrderingCriterion.STORAGE_PER_TOKEN:
        return query.storage_gb_per_token
    if criterion is AGHOrderingCriterion.ERROR_SLO:
        return query.error_slo
    raise ValueError("unsupported AGH ordering criterion")


def _validate_query_ids(query_ids: tuple[str, ...]) -> None:
    """Require a non-empty sequence of unique, non-empty string query IDs."""

    if not query_ids or any(not isinstance(item, str) or not item for item in query_ids):
        raise ValueError("query_ids must contain non-empty strings")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique")


def _validate_seed_and_index(seed: int | None, random_start_index: int | None) -> None:
    """Require an integer seed and a nonnegative integer draw index."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(random_start_index, bool)
        or not isinstance(random_start_index, int)
        or random_start_index < 0
    ):
        raise ValueError("random_start_index must be a nonnegative integer")
