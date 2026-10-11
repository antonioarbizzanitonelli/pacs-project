"""Product ``v = x*w`` and McCormick slacks (Paper v2 Eqs. 6, 7a-7b)."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class McCormickResiduals:
    """Record the four McCormick slacks (v2 Eqs. 7a-7b).

    ``x_minus_v = x-v`` and ``w_minus_v = w-v`` enforce the upper bounds;
    ``v_minus_x_minus_w_plus_one = v-x-w+1`` enforces ``v >= x+w-1``;
    ``v_nonnegative = v`` enforces ``v >= 0``. Each satisfied bound has
    non-negative slack.
    """

    x_minus_v: float
    w_minus_v: float
    v_minus_x_minus_w_plus_one: float
    v_nonnegative: float

    @property
    def minimum_residual(self) -> float:
        """Return ``min(x-v, w-v, v-x-w+1, v)`` (v2 Eqs. 7a-7b)."""

        return min(
            self.x_minus_v,
            self.w_minus_v,
            self.v_minus_x_minus_w_plus_one,
            self.v_nonnegative,
        )


def _validate_product_domain(x: float, w: int | float) -> None:
    """Require finite ``x in [0,1]`` and exactly binary ``w`` (v2 Eq. 6)."""

    if isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x):
        raise ValueError("x must be a finite real number in [0, 1]")
    if x < 0.0 or x > 1.0:
        raise ValueError("x must lie in [0, 1]")
    # Exact numeric 0.0/1.0 values remain useful for solver-independent
    # validation, but near-binary values are never rounded into the domain.
    if (
        isinstance(w, bool)
        or not isinstance(w, (int, float))
        or not isfinite(w)
        or w not in (0, 1)
    ):
        raise ValueError("w must be exactly 0 or 1")


def _validate_finite_value(value: float, name: str) -> None:
    """Require a finite real scalar, excluding booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")


def compute_binary_continuous_product(x: float, w: int | float) -> float:
    """Return the intended auxiliary ``v = x*w`` (v2 Eq. 6).

    Require ``x in [0,1]`` and ``w in {0,1}``; return their product directly.
    """

    _validate_product_domain(x, w)
    return float(x * w)


def compute_mccormick_residuals(
    x: float,
    w: int | float,
    v: float,
) -> McCormickResiduals:
    """Compute ``x-v``, ``w-v``, ``v-x-w+1``, and ``v`` (v2 Eqs. 7a-7b).

    Require ``x in [0,1]``, binary ``w``, and finite real ``v``. Supplied ``v``
    may violate its bounds; non-negative slacks satisfy the envelope.
    """

    _validate_product_domain(x, w)
    _validate_finite_value(v, "v")
    return McCormickResiduals(
        x_minus_v=float(x - v),
        w_minus_v=float(w - v),
        v_minus_x_minus_w_plus_one=float(v - x - w + 1),
        v_nonnegative=float(v),
    )


def satisfies_mccormick_envelope(
    x: float,
    w: int | float,
    v: float,
    tolerance: float = 0.0,
) -> bool:
    """Return whether all McCormick slacks are ``>= -tolerance`` (v2 Eqs. 7a-7b).

    The exact domains ``x in [0,1]`` and ``w in {0,1}`` are checked separately.
    """

    _validate_finite_value(tolerance, "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    residuals = compute_mccormick_residuals(x, w, v)
    return residuals.minimum_residual >= -tolerance
