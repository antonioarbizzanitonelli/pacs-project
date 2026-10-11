"""Typed scientific inputs and validation for placement instances."""

from .types import (
    FoundationModel,
    GpuTier,
    ParallelismConfig,
    ProblemInstance,
    QueryType,
)
from .validation import InstanceValidationError, validate_problem_instance
from .units import UnitConvention

__all__ = [
    "FoundationModel",
    "GpuTier",
    "InstanceValidationError",
    "ParallelismConfig",
    "ProblemInstance",
    "QueryType",
    "UnitConvention",
    "validate_problem_instance",
]
