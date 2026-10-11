"""Effective error ``e_bar[i,j,k] = mu[k]*e_base[i,j]`` (Paper v2 Eq. 1)."""


def compute_effective_error(
    base_error_rate: float,
    quantization_error_multiplier: float,
) -> float:
    """Return ``e_bar[i,j,k] = mu[k]*e_base[i,j]`` (v2 Eq. 1).

    ``base_error_rate`` is the FP16 baseline ``e_base[i,j]``;
    ``quantization_error_multiplier`` is ``mu[k]``. Input and output values
    are dimensionless; the product is returned without clipping.
    """

    return quantization_error_multiplier * base_error_rate
