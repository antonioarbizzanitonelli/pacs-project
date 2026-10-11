"""Extract ``P_DM`` values and normalize variable domains.

Independent validation checks scientific constraints after normalization.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping, TypeVar

from llm_resource_allocation.math.parallelism import ParallelismDecisionState
from llm_resource_allocation.math.placement import (
    PlacementDecisionState,
    validate_placement_decision_domains,
)
from llm_resource_allocation.math.routing import RoutingDecisionState

from .problem_data import (
    AuxiliaryProductKey,
    ConfigurationKey,
    ModelTierKey,
    PdmProblemData,
    PdmVariableFamily,
    RouteKey,
)


SemanticKey = str | ModelTierKey | RouteKey | ConfigurationKey | AuxiliaryProductKey
_KeyT = TypeVar("_KeyT", bound=SemanticKey)


class PdmSolutionExtractionError(ValueError):
    """Report unusable model values or mismatched variable keys."""


class PdmModelLatticeMismatchError(PdmSolutionExtractionError):
    """Report variable keys that differ from the problem-data lattice."""

    def __init__(
        self,
        family: PdmVariableFamily,
        expected_keys: tuple[SemanticKey, ...],
        actual_keys: tuple[SemanticKey, ...],
    ) -> None:
        """Retain missing keys, unexpected keys, and ordering mismatches."""

        actual_key_set = set(actual_keys)
        expected_key_set = set(expected_keys)
        self.family = family
        self.missing_keys = tuple(
            key for key in expected_keys if key not in actual_key_set
        )
        self.unexpected_keys = tuple(
            key for key in actual_keys if key not in expected_key_set
        )
        self.order_mismatch = (
            not self.missing_keys
            and not self.unexpected_keys
            and actual_keys != expected_keys
        )
        details = (
            f"missing={self.missing_keys!r}, "
            f"unexpected={self.unexpected_keys!r}, "
            f"order_mismatch={self.order_mismatch}"
        )
        super().__init__(
            f"Pyomo variable family {family.value!r} does not match its exact "
            f"PdmProblemData lattice: {details}"
        )


class PdmVariableValueExtractionError(PdmSolutionExtractionError):
    """Report one unusable raw Pyomo value with its family and semantic key."""

    def __init__(
        self,
        family: PdmVariableFamily,
        semantic_key: SemanticKey,
        raw_value: object,
        reason: str,
    ) -> None:
        self.family = family
        self.semantic_key = semantic_key
        self.raw_value = raw_value
        self.reason = reason
        super().__init__(
            f"Cannot extract {family.value}[{semantic_key!r}]: {reason}; "
            f"raw value={raw_value!r}"
        )


class PdmSolutionNormalizationError(ValueError):
    """Report a raw value or lattice that cannot be normalized safely."""

    def __init__(
        self,
        family: PdmVariableFamily,
        semantic_key: SemanticKey | None,
        raw_value: object,
        requirement: str,
        tolerance: float | None,
    ) -> None:
        self.family = family
        self.semantic_key = semantic_key
        self.raw_value = raw_value
        self.requirement = requirement
        self.tolerance = tolerance
        tolerance_text = "not applicable" if tolerance is None else repr(tolerance)
        super().__init__(
            f"Cannot normalize {family.value}[{semantic_key!r}] from "
            f"{raw_value!r}: required {requirement}; tolerance={tolerance_text}"
        )


@dataclass(frozen=True, slots=True)
class RawPdmSolution:
    """Snapshot dense ``x,z,q,u,w,y,v`` mappings as immutable floats.

    Binary and integer values remain unrounded; supplied ``v`` is preserved.
    """

    x: Mapping[RouteKey, float]
    z: Mapping[RouteKey, float]
    q: Mapping[ModelTierKey, float]
    u: Mapping[str, float]
    w: Mapping[ConfigurationKey, float]
    y: Mapping[ModelTierKey, float]
    v: Mapping[AuxiliaryProductKey, float]

    def __post_init__(self) -> None:
        for field_name in ("x", "z", "q", "u", "w", "y", "v"):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )


@dataclass(frozen=True, slots=True)
class SolutionNormalizationTolerances:
    """Set domain-normalization tolerances, separate from feasibility (D-0022).

    Defaults: ``1e-7`` binary/integer distance; ``1e-9`` bound clipping.
    """

    binary_integrality: float = 1e-7
    integer_integrality: float = 1e-7
    bound_clipping: float = 1e-9

    def __post_init__(self) -> None:
        for field_name in (
            "binary_integrality",
            "integer_integrality",
            "bound_clipping",
        ):
            tolerance = getattr(self, field_name)
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not isfinite(tolerance)
                or tolerance < 0.0
            ):
                raise ValueError(
                    f"{field_name} must be a finite non-negative real number"
                )
        if self.binary_integrality >= 0.5:
            raise ValueError("binary_integrality must be less than 0.5")
        if self.integer_integrality >= 0.5:
            raise ValueError("integer_integrality must be less than 0.5")


def _family_lattices(
    problem_data: PdmProblemData,
) -> tuple[tuple[PdmVariableFamily, tuple[SemanticKey, ...]], ...]:
    """Return variable families and their ordered problem-data keys."""

    lattices = problem_data.variable_lattices
    return (
        (PdmVariableFamily.X, lattices.route_keys),
        (PdmVariableFamily.Z, lattices.route_keys),
        (PdmVariableFamily.Q, lattices.model_tier_keys),
        (PdmVariableFamily.U, problem_data.index_sets.query_ids),
        (PdmVariableFamily.W, lattices.configuration_keys),
        (PdmVariableFamily.Y, lattices.model_tier_keys),
        (PdmVariableFamily.V, lattices.auxiliary_product_keys),
    )


def _extract_family(
    model: object,
    family: PdmVariableFamily,
    expected_keys: tuple[SemanticKey, ...],
) -> dict[SemanticKey, float]:
    """Extract one dense family after exact component-lattice verification."""

    component = getattr(model, family.value, None)
    if component is None or not hasattr(component, "keys"):
        raise PdmModelLatticeMismatchError(family, expected_keys, ())

    actual_keys = tuple(component.keys())
    if actual_keys != expected_keys:
        raise PdmModelLatticeMismatchError(family, expected_keys, actual_keys)

    extracted: dict[SemanticKey, float] = {}
    for semantic_key in expected_keys:
        raw_value = component[semantic_key].value
        if raw_value is None:
            raise PdmVariableValueExtractionError(
                family, semantic_key, raw_value, "value is uninitialized"
            )
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise PdmVariableValueExtractionError(
                family, semantic_key, raw_value, "value is not a real number"
            )
        plain_value = float(raw_value)
        if not isfinite(plain_value):
            raise PdmVariableValueExtractionError(
                family, semantic_key, raw_value, "value is not finite"
            )
        extracted[semantic_key] = plain_value
    return extracted


def extract_raw_pdm_solution_from_pyomo(
    model: object,
    problem_data: PdmProblemData,
) -> RawPdmSolution:
    """Copy finite model values in the exact problem-data key order.

    Reject mismatched keys and unusable values; preserve ``v`` rather than
    recomputing ``x*w``.
    """

    if not isinstance(problem_data, PdmProblemData):
        raise TypeError("problem_data must be a PdmProblemData")

    extracted = {
        family.value: _extract_family(model, family, expected_keys)
        for family, expected_keys in _family_lattices(problem_data)
    }
    return RawPdmSolution(**extracted)


def _assert_raw_lattice(
    raw_solution: RawPdmSolution,
    problem_data: PdmProblemData,
) -> None:
    """Require each raw mapping to have the exact problem-data key set."""

    for family, expected_keys in _family_lattices(problem_data):
        mapping = getattr(raw_solution, family.value)
        actual_keys = tuple(mapping)
        if set(actual_keys) != set(expected_keys) or len(actual_keys) != len(
            expected_keys
        ):
            raise PdmSolutionNormalizationError(
                family,
                None,
                actual_keys,
                f"exact key set {expected_keys!r}",
                None,
            )


def _require_finite_raw(
    family: PdmVariableFamily,
    semantic_key: SemanticKey,
    raw_value: object,
) -> float:
    """Return one finite float or raise a structured normalization error."""

    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        raise PdmSolutionNormalizationError(
            family, semantic_key, raw_value, "a finite real number", None
        )
    value = float(raw_value)
    if not isfinite(value):
        raise PdmSolutionNormalizationError(
            family, semantic_key, raw_value, "a finite real number", None
        )
    return value


def _normalize_binary(
    family: PdmVariableFamily,
    semantic_key: SemanticKey,
    raw_value: object,
    tolerance: float,
) -> int:
    """Map only tolerance-close values to an exact binary integer."""

    value = _require_finite_raw(family, semantic_key, raw_value)
    if abs(value) <= tolerance:
        return 0
    if abs(value - 1.0) <= tolerance:
        return 1
    raise PdmSolutionNormalizationError(
        family,
        semantic_key,
        raw_value,
        "distance to binary domain {0, 1} within binary_integrality",
        tolerance,
    )


def _normalize_nonnegative_integer(
    semantic_key: ModelTierKey,
    raw_value: object,
    tolerance: float,
) -> int:
    """Map only tolerance-close values to a non-negative integer."""

    family = PdmVariableFamily.Y
    value = _require_finite_raw(family, semantic_key, raw_value)
    nearest_integer = round(value)
    if nearest_integer >= 0 and abs(value - nearest_integer) <= tolerance:
        return nearest_integer
    raise PdmSolutionNormalizationError(
        family,
        semantic_key,
        raw_value,
        "distance to a non-negative integer within integer_integrality",
        tolerance,
    )


def _normalize_bounded_continuous(
    family: PdmVariableFamily,
    semantic_key: SemanticKey,
    raw_value: object,
    upper_bound: float,
    tolerance: float,
) -> float:
    """Preserve an in-domain value and clip only tolerance-scale excursions."""

    value = _require_finite_raw(family, semantic_key, raw_value)
    if 0.0 <= value <= upper_bound:
        return value
    if value < 0.0 and -value <= tolerance:
        return 0.0
    if value > upper_bound and value - upper_bound <= tolerance:
        return upper_bound
    raise PdmSolutionNormalizationError(
        family,
        semantic_key,
        raw_value,
        f"domain [0, {upper_bound!r}] with only bound-scale clipping",
        tolerance,
    )


def _normalize_binary_family(
    family: PdmVariableFamily,
    raw_values: Mapping[_KeyT, float],
    keys: tuple[_KeyT, ...],
    tolerance: float,
) -> dict[_KeyT, int]:
    """Normalize a binary family in its authoritative key order."""

    return {
        key: _normalize_binary(family, key, raw_values[key], tolerance)
        for key in keys
    }


def _normalize_continuous_family(
    family: PdmVariableFamily,
    raw_values: Mapping[_KeyT, float],
    keys: tuple[_KeyT, ...],
    upper_bound: float | Mapping[_KeyT, float],
    tolerance: float,
) -> dict[_KeyT, float]:
    """Normalize continuous values with a shared or per-key upper bound."""

    return {
        key: _normalize_bounded_continuous(
            family,
            key,
            raw_values[key],
            upper_bound[key] if isinstance(upper_bound, Mapping) else upper_bound,
            tolerance,
        )
        for key in keys
    }


def normalize_raw_pdm_solution(
    raw_solution: RawPdmSolution,
    problem_data: PdmProblemData,
    tolerances: SolutionNormalizationTolerances | None = None,
) -> PlacementDecisionState:
    """Normalize domain noise and validate the resulting decision domains.

    Snap nearby binaries/GPU counts and clip continuous bound excursions within
    their tolerances. Scientific relationships, including ``v=x*w``, remain for
    independent validation (D-0022).
    """

    if not isinstance(raw_solution, RawPdmSolution):
        raise TypeError("raw_solution must be a RawPdmSolution")
    if not isinstance(problem_data, PdmProblemData):
        raise TypeError("problem_data must be a PdmProblemData")
    selected_tolerances = tolerances or SolutionNormalizationTolerances()
    if not isinstance(selected_tolerances, SolutionNormalizationTolerances):
        raise TypeError("tolerances must be a SolutionNormalizationTolerances")

    _assert_raw_lattice(raw_solution, problem_data)
    lattices = problem_data.variable_lattices
    max_unmet_by_query = {
        query.id: query.max_unmet_fraction
        for query in problem_data.instance.queries
    }

    normalized_x = _normalize_continuous_family(
        PdmVariableFamily.X, raw_solution.x, lattices.route_keys,
        1.0, selected_tolerances.bound_clipping,
    )
    normalized_z = _normalize_binary_family(
        PdmVariableFamily.Z, raw_solution.z, lattices.route_keys,
        selected_tolerances.binary_integrality,
    )
    normalized_q = _normalize_binary_family(
        PdmVariableFamily.Q, raw_solution.q, lattices.model_tier_keys,
        selected_tolerances.binary_integrality,
    )
    normalized_u = _normalize_continuous_family(
        PdmVariableFamily.U, raw_solution.u, problem_data.index_sets.query_ids,
        max_unmet_by_query, selected_tolerances.bound_clipping,
    )
    normalized_w = _normalize_binary_family(
        PdmVariableFamily.W, raw_solution.w, lattices.configuration_keys,
        selected_tolerances.binary_integrality,
    )
    normalized_y = {
        key: _normalize_nonnegative_integer(
            key,
            raw_solution.y[key],
            selected_tolerances.integer_integrality,
        )
        for key in lattices.model_tier_keys
    }
    normalized_v = _normalize_continuous_family(
        PdmVariableFamily.V, raw_solution.v, lattices.auxiliary_product_keys,
        1.0, selected_tolerances.bound_clipping,
    )

    routing_decisions = RoutingDecisionState(
        routing_fractions=normalized_x,
        admission_indicators=normalized_z,
        deployment_indicators=normalized_q,
        unmet_fractions=normalized_u,
    )
    parallelism_decisions = ParallelismDecisionState(
        routing_decisions=routing_decisions,
        configuration_selectors=normalized_w,
        gpu_counts=normalized_y,
    )
    state = PlacementDecisionState(
        parallelism_decisions=parallelism_decisions,
        auxiliary_products=normalized_v,
    )
    validate_placement_decision_domains(problem_data.instance, state)
    return state
