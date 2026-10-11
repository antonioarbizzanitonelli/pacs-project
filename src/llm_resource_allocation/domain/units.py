"""Canonical unit conversions at the problem-data input boundary.

The internal unit policy is an implementation choice documented by Decision
D-0008. Converting published KB-valued inputs to the chosen GB representation
does not alter or repair any paper equation.
"""

from math import isfinite
from enum import Enum


KILOBYTES_PER_GIGABYTE = 1_000_000.0
GFLOPS_PER_TFLOP = 1_000.0
MILLISECONDS_PER_SECOND = 1_000.0
SECONDS_PER_HOUR = 3_600.0


class UnitConvention(str, Enum):
    """Select coherent physical units or the historical printed arithmetic.

    Coherent units account for mean in-flight KV requests, explicit data
    residency, and delay penalties on every arriving request in the horizon.
    Literal paper units retain the earlier reference for historical comparisons.
    """

    COHERENT = "coherent"
    LITERAL_PAPER = "literal_paper"


def kilobytes_to_gigabytes(kilobytes: float) -> float:
    """Convert non-negative decimal KB to GB for canonical storage fields.

    The papers label capacities as GB and per-token footprints as KB/token but
    do not print conversion factors in Eqs. (8a), (8c), (8f), or (8h). This
    helper performs only the explicit SI input conversion selected in Decision
    D-0008; it does not change the scientific meaning of those equations.

    Args:
        kilobytes: A finite, non-negative value expressed in decimal KB.

    Returns:
        The same physical quantity expressed in decimal GB.

    Raises:
        ValueError: If ``kilobytes`` is negative, non-finite, or boolean.
    """

    if isinstance(kilobytes, bool) or not isinstance(kilobytes, (int, float)):
        raise ValueError("kilobytes must be a real number")
    if not isfinite(kilobytes) or kilobytes < 0:
        raise ValueError("kilobytes must be finite and non-negative")
    return float(kilobytes) / KILOBYTES_PER_GIGABYTE
