"""Independent residual-classification defaults (D-0097), not solver options.

Absolute floors use the existing row's numerical unit. Memory/storage units
remain unresolved; these constants do not supply a physical interpretation.
"""

from math import isfinite
from enum import Enum
from typing import TypeVar


_FamilyT = TypeVar("_FamilyT", bound=Enum)


DEFAULT_FRACTION_ABSOLUTE_TOLERANCE = 1e-10
DEFAULT_BUDGET_ABSOLUTE_TOLERANCE = 1e-8
DEFAULT_COMPUTE_ABSOLUTE_TOLERANCE = 1e-8
DEFAULT_DELAY_ABSOLUTE_TOLERANCE = 1e-10
DEFAULT_ERROR_ABSOLUTE_TOLERANCE = 1e-12
DEFAULT_PRINTED_CAPACITY_ABSOLUTE_TOLERANCE = 1e-9
DEFAULT_SCALED_RELATIVE_TOLERANCE = 1e-12


def _record_applied_tolerance(
    thresholds: dict[tuple[_FamilyT, tuple[str | int, ...]], float],
    family: _FamilyT,
    location: tuple[str | int, ...],
    absolute: float,
    relative: float = 0.0,
    lhs: float = 0.0,
    rhs: float = 0.0,
) -> float:
    """Retain the inclusive threshold for a row without changing its residual."""

    threshold = absolute + relative * max(abs(lhs), abs(rhs))
    if not isfinite(threshold):
        raise ValueError("applied feasibility tolerance must be finite")
    thresholds[family, location] = threshold
    return threshold
