"""SchemaInferer: schema inference via schema-free extraction + recursive merge.

Each text is first reduced to a schema-free entity tree via
``NestedEntityExtractor`` -- the same step that's useful on its own for
seeing what information a corpus is actually made of, before any schema
exists. ``EntityTypeNormalizer`` then reconciles entity type names across
the whole corpus, one level at a time. Finally, a coverage-based merge
turns the normalized entity trees into a JSON Schema: a canonical type is
kept as a field only if it appears in at least ``min_coverage`` of the
instances at its level, repetition within a single instance signals an
array, and a canonical type is treated as an object (recursing into its
own coverage-based merge) if most of its occurrences across the corpus
carried ``children`` rather than a scalar ``value``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .backends.base import Backend
from .entities import (
    DEFAULT_MAX_DEPTH,
    Entity,
    EntityExtractionError,
    EntityTypeNormalizer,
    NestedEntityExtractor,
)
from .schema import Schema

DEFAULT_MIN_COVERAGE = 0.6


class InferError(Exception):
    """Raised when schema inference cannot produce a usable schema."""


class SchemaInferer:
    """Infers a ``Schema`` shared by a collection of texts, via a backend."""

    def __init__(
        self,
        backend: Backend,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> None:
        self._backend = backend
        self._max_depth = max_depth
        self._min_coverage = min_coverage

    def infer(self, texts: list[str]) -> Schema:
        """Infer a JSON Schema shared by ``texts``."""
        if not texts:
            raise InferError("cannot infer a schema from an empty text collection")

        extractor = NestedEntityExtractor(self._backend, max_depth=self._max_depth)
        entity_lists: list[list[Entity]] = []
        for text in texts:
            try:
                entity_lists.append(extractor.extract(text))
            except EntityExtractionError:
                continue
        if not entity_lists:
            raise InferError("backend did not return a usable entity list for any text")

        normalizer = EntityTypeNormalizer(self._backend)
        normalizer.fit(entity_lists)
        normalized = normalizer.transform(entity_lists)

        properties, required = _merge_entity_lists(normalized, self._min_coverage)
        if not properties:
            raise InferError(
                f"no fields met the min_coverage={self._min_coverage} threshold across "
                f"{len(entity_lists)} text(s)"
            )
        return Schema({"type": "object", "properties": properties, "required": required})


def _merge_entity_lists(
    entity_lists: list[list[Entity]], min_coverage: float
) -> tuple[dict[str, Any], list[str]]:
    n = len(entity_lists)
    occurrences: dict[str, list[Entity]] = {}
    presence: Counter[str] = Counter()
    repeated: set[str] = set()

    for entities in entity_lists:
        counts: Counter[str] = Counter()
        for entity in entities:
            occurrences.setdefault(entity.type, []).append(entity)
            counts[entity.type] += 1
        for entity_type, count in counts.items():
            presence[entity_type] += 1
            if count > 1:
                repeated.add(entity_type)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for entity_type, count in presence.items():
        coverage = count / n
        if coverage < min_coverage:
            continue

        entity_occurrences = occurrences[entity_type]
        group_count = sum(1 for entity in entity_occurrences if entity.children is not None)
        # Majority vote: a canonical type observed as both scalar and group
        # across the corpus is resolved here, at merge time, rather than by
        # the normalizer -- this is the one place with corpus-wide coverage.
        is_group = group_count / len(entity_occurrences) >= 0.5

        if is_group:
            children_lists = [
                entity.children for entity in entity_occurrences if entity.children is not None
            ]
            child_properties, child_required = _merge_entity_lists(children_lists, min_coverage)
            field_schema: dict[str, Any] = {
                "type": "object",
                "properties": child_properties,
                "required": child_required,
            }
        else:
            field_schema = {"type": "string"}

        if entity_type in repeated:
            field_schema = {"type": "array", "items": field_schema}

        properties[entity_type] = field_schema
        if coverage >= 1.0:
            required.append(entity_type)

    return properties, sorted(required)
