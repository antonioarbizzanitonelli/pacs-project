"""Resolve semantic entity identifiers shared by M1-M3."""

from typing import TypeVar

from llm_resource_allocation.domain import FoundationModel, GpuTier, QueryType


EntityT = TypeVar("EntityT", QueryType, FoundationModel, GpuTier)


def _entity_by_id(
    entities: tuple[EntityT, ...], entity_id: str, label: str
) -> EntityT:
    """Return one validated semantic entity or reject an unknown identifier."""

    for entity in entities:
        if entity.id == entity_id:
            return entity
    raise ValueError(f"unknown {label} ID: {entity_id!r}")
