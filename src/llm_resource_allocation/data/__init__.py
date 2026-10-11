"""Synthetic input fixtures for deterministic placement and GH/AGH verification."""

from .tiny_instances import (
    make_coherent_tiny_instance,
    make_heterogeneous_solve_tiny_instance,
    make_heterogeneous_tiny_instance,
    make_minimal_tiny_instance,
    make_parallelism_tradeoff_tiny_instance,
)


__all__ = [
    "make_coherent_tiny_instance",
    "make_heterogeneous_solve_tiny_instance",
    "make_heterogeneous_tiny_instance",
    "make_minimal_tiny_instance",
    "make_parallelism_tradeoff_tiny_instance",
]
