"""Scientific reference structures for heterogeneous LLM allocation."""

from .domain import (
    FoundationModel,
    GpuTier,
    InstanceValidationError,
    ParallelismConfig,
    ProblemInstance,
    QueryType,
    validate_problem_instance,
)

__all__ = [
    "FoundationModel",
    "GpuTier",
    "InstanceValidationError",
    "ParallelismConfig",
    "ProblemInstance",
    "QueryType",
    "validate_problem_instance",
]
