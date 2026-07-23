"""Schema-free structured extraction: a building block for schema inference.

Rather than asking a backend to invent a JSON Schema directly, texts are
first reduced to a flat list of ``(type, value)`` entities. The same type
may repeat within one text (that repetition is itself a signal, e.g. of an
array field). A later normalization pass reconciles entity types across a
corpus before a schema is built from them.

``NestedEntityExtractor`` extends this: an entity may carry ``children``
(its own entities) instead of a scalar ``value``, for repeating structured
records (e.g. a receipt's line items) that a flat ``(type, value)`` pair
cannot express -- repetition of the same group ``type`` at a given level
then reads as "array of that group," the same repetition signal used for
scalar arrays. ``children`` may themselves carry ``children``, down to
``max_depth`` levels below the top level (defaulting to one, matching the
depth ``EntityTypeNormalizer``/``SchemaInferer`` are exercised at today);
any nesting beyond that is dropped, both by the (bounded, non-infinitely-
recursive) output JSON Schema handed to the backend and defensively when
parsing its response.

``EntityTypeNormalizer`` reconciles the free-form type labels a corpus of
such extractions ends up with into a shared vocabulary, one level at a
time: it normalizes a level's own types first, then pools every occurrence
of a canonical group type's ``children`` -- across every entity list and
every occurrence within it -- and recurses into that pool to normalize the
next level down. Deciding whether a canonical type is ultimately a scalar
or an object/array field is left to schema construction (``SchemaInferer``),
which has the corpus-wide coverage numbers to make that call; this module
only unifies names.

``EntityRenderer`` provides the reverse direction (entities -> text), useful
for round-trip checks on extraction/normalization quality. It only renders
one level of ``children`` and is not a general-purpose D2T replacement --
see ``StructuredEntityRenderer`` for that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ._util import extract_json
from .backends.base import Backend
from .prompts import (
    build_entity_extraction_prompt,
    build_entity_render_prompt,
    build_entity_type_merge_prompt,
    build_nested_entity_extraction_prompt,
)

DEFAULT_MAX_EXAMPLES_PER_TYPE = 3
DEFAULT_MAX_DEPTH = 1

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _entity_item_schema(remaining_depth: int) -> dict[str, Any]:
    """JSON Schema for one entity item, allowing ``children`` ``remaining_depth`` levels deeper."""
    properties: dict[str, Any] = {"type": {"type": "string"}, "value": {"type": "string"}}
    if remaining_depth > 0:
        properties = dict(properties)
        properties["children"] = {
            "type": "array",
            "items": _entity_item_schema(remaining_depth - 1),
        }
    return {"type": "object", "properties": properties, "required": ["type"]}


def _nested_entity_output_schema(max_depth: int) -> dict[str, Any]:
    return {"type": "array", "items": _entity_item_schema(max_depth)}


def _normalize_type_string(raw: str) -> str:
    """Cheap, LLM-free normalization: lowercase + snake_case."""
    return _NON_ALNUM.sub("_", raw.strip().lower()).strip("_")


class Entity(BaseModel):
    """A single entity extracted from text.

    Either a leaf -- ``value`` set, ``children`` ``None`` -- or a group
    representing one instance of a repeating structured record -- ``value``
    ``None``, ``children`` set to that instance's own entities (which may
    themselves be leaves or further groups, depending on the extractor's
    ``max_depth``).
    """

    type: str
    value: str | None = None
    children: list[Entity] | None = None


class EntityExtractionError(Exception):
    """Raised when a backend fails to produce a usable entity list."""


class EntityNormalizationError(Exception):
    """Raised when a backend fails to produce a usable type mapping."""


class EntityRenderError(Exception):
    """Raised when a backend fails to produce text from an entity list."""


class FlatEntityExtractor:
    """Extracts a flat list of entities from a single text via a backend.

    By default, the backend is free to invent its own ``type`` labels (the
    right mode for first-pass corpus discovery ahead of
    :class:`EntityTypeNormalizer`). Passing ``entity_schema`` -- typically
    :meth:`EntityTypeNormalizer.entity_schema`'s output -- constrains
    extraction to a known type vocabulary instead: the prompt lists the
    allowed types, the backend is asked to generate against a schema
    reflecting them (letting a ``constrained_decoding`` backend enforce it
    at the grammar level), and any entity whose type slips through outside
    the vocabulary is dropped as a defensive net.
    """

    def __init__(self, backend: Backend, *, entity_schema: dict[str, Any] | None = None) -> None:
        self._backend = backend
        self._entity_schema = entity_schema

    def extract(self, text: str) -> list[Entity]:
        """Extract entities mentioned in ``text``."""
        prompt = build_entity_extraction_prompt(text, entity_schema=self._entity_schema)
        output_schema = (
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"type": self._entity_schema, "value": {"type": "string"}},
                    "required": ["type", "value"],
                },
            }
            if self._entity_schema is not None
            else None
        )
        raw = self._backend.generate(prompt, schema=output_schema)
        try:
            data = extract_json(raw)
        except json.JSONDecodeError as exc:
            raise EntityExtractionError(f"backend did not return valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise EntityExtractionError("backend did not return a JSON array of entities")

        allowed_types = (
            set(self._entity_schema["enum"]) if self._entity_schema is not None else None
        )
        entities = []
        for item in data:
            if not isinstance(item, dict) or "type" not in item or "value" not in item:
                continue
            entity_type = str(item["type"])
            if allowed_types is not None and entity_type not in allowed_types:
                continue
            entities.append(Entity(type=entity_type, value=str(item["value"])))
        return entities


class NestedEntityExtractor:
    """Extracts entities from a single text via a backend, schema-free and depth-bounded.

    Like :class:`FlatEntityExtractor`, but an entity may represent one
    instance of a repeating structured record (e.g. a receipt's line
    items) by carrying ``children`` -- that instance's own entities --
    instead of a scalar ``value``. Repeated entities of the same group
    ``type`` at a given level then read as "array of that group," the same
    repetition-as-array-signal :class:`FlatEntityExtractor` relies on for
    scalar arrays.

    ``max_depth`` bounds how many levels of ``children`` are allowed below
    the top level (default ``1``): both the output JSON Schema handed to
    the backend and the defensive parsing of its response cap nesting
    there, dropping anything deeper. Raising it lets the schema-free
    extraction step see deeper structures (e.g. a group within a group),
    at the cost of a larger/more recursive grammar for
    ``constrained_decoding`` backends.

    Unconstrained only: unlike :class:`FlatEntityExtractor`, there is no
    ``entity_schema`` vocabulary constraint here. Constraining extraction to
    a known vocabulary for the nested case is schema inference's job (see
    ``SchemaInferer``), not this extractor's.
    """

    def __init__(self, backend: Backend, *, max_depth: int = DEFAULT_MAX_DEPTH) -> None:
        self._backend = backend
        self._max_depth = max_depth

    def extract(self, text: str) -> list[Entity]:
        """Extract entities mentioned in ``text``, including any groups."""
        prompt = build_nested_entity_extraction_prompt(text)
        schema = _nested_entity_output_schema(self._max_depth)
        raw = self._backend.generate(prompt, schema=schema)
        try:
            data = extract_json(raw)
        except json.JSONDecodeError as exc:
            raise EntityExtractionError(f"backend did not return valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise EntityExtractionError("backend did not return a JSON array of entities")

        entities = []
        for item in data:
            entity = _parse_entity(item, self._max_depth)
            if entity is not None:
                entities.append(entity)
        return entities


def _parse_entity(item: Any, remaining_depth: int) -> Entity | None:
    if not isinstance(item, dict) or "type" not in item:
        return None
    entity_type = str(item["type"])
    children_raw = item.get("children")
    if remaining_depth > 0 and isinstance(children_raw, list):
        children = [
            child
            for child in (
                _parse_entity(raw_child, remaining_depth - 1) for raw_child in children_raw
            )
            if child is not None
        ]
        return Entity(type=entity_type, children=children)
    if "value" in item:
        return Entity(type=entity_type, value=str(item["value"]))
    return None


class EntityRenderer:
    """Renders a list of entities back into text via a backend.

    The reverse of :class:`FlatEntityExtractor` and :class:`NestedEntityExtractor`.
    Not schema-aware -- unlike ``StructuredEntityRenderer``, there is no
    ``Schema`` here to supply field descriptions/examples/style, just the
    raw entities (including a group's ``children``, one level of them).
    This makes it useful mainly as a round-trip check on entity extraction
    and normalization quality (extract -> normalize -> render ->
    re-extract), not as a general-purpose D2T replacement.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def render(self, entities: list[Entity]) -> str:
        """Render ``entities`` into a single piece of natural-language text."""
        prompt = build_entity_render_prompt([_render_item(entity) for entity in entities])
        text = self._backend.generate(prompt)
        if not text.strip():
            raise EntityRenderError("backend returned empty text")
        return text


def _render_item(entity: Entity) -> tuple[str, str | list[tuple[str, str]]]:
    if entity.children is not None:
        return (entity.type, [(child.type, child.value or "") for child in entity.children])
    return (entity.type, entity.value or "")


class EntityTypeNormalizer:
    """Reconciles entity types observed across a corpus into canonical names, level by level.

    ``fit`` calls the backend once per level to merge synonymous types
    (e.g. "name" and "full_name") into a single canonical, snake_case type
    name for that level, keeping the result on ``self.mapping``. It then
    pools every occurrence of each canonical group type's ``children`` --
    across every entity list and every occurrence within it -- and
    recurses into a child :class:`EntityTypeNormalizer` (kept on
    ``self.children``, keyed by canonical group type) to normalize the
    next level down the same way. A group's ``children`` are themselves
    entity lists, so this recursion bottoms out naturally once entities
    stop carrying ``children`` (i.e. at whatever ``max_depth``
    :class:`NestedEntityExtractor` was run with).

    A canonical type observed as both a scalar leaf and a group across the
    corpus is not disambiguated here -- both occurrence kinds are pooled
    and passed through as-is. Deciding the field's ultimate shape (scalar
    vs. object/array) is schema construction's job, which has the
    corpus-wide coverage numbers to make that call.

    ``transform`` applies the fitted mapping (recursively) without
    touching the backend, so a fitted mapping can be persisted with
    ``save`` and reused later via ``load``.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self.mapping: dict[str, str] = {}
        self.children: dict[str, EntityTypeNormalizer] = {}

    def fit(self, entity_lists: list[list[Entity]]) -> None:
        """Compute ``self.mapping`` (and recursively, ``self.children``) from ``entity_lists``."""
        examples: dict[str, list[str]] = {}
        for entities in entity_lists:
            for entity in entities:
                normalized = _normalize_type_string(entity.type)
                values = examples.setdefault(normalized, [])
                if (
                    entity.value is not None
                    and len(values) < DEFAULT_MAX_EXAMPLES_PER_TYPE
                    and entity.value not in values
                ):
                    values.append(entity.value)

        if not examples:
            self.mapping = {}
            self.children = {}
            return

        prompt = build_entity_type_merge_prompt(examples)
        raw = self._backend.generate(prompt)
        try:
            data = extract_json(raw)
        except json.JSONDecodeError as exc:
            raise EntityNormalizationError(f"backend did not return valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise EntityNormalizationError("backend did not return a JSON object mapping types")

        self.mapping = {
            raw_type: canonical
            for raw_type, canonical in data.items()
            if isinstance(raw_type, str) and isinstance(canonical, str)
        }

        pooled: dict[str, list[list[Entity]]] = {}
        for entities in entity_lists:
            for entity in entities:
                if entity.children is None:
                    continue
                normalized = _normalize_type_string(entity.type)
                canonical = self.mapping.get(normalized, normalized)
                pooled.setdefault(canonical, []).append(entity.children)

        self.children = {}
        for canonical, children_lists in pooled.items():
            child_normalizer = EntityTypeNormalizer(self._backend)
            child_normalizer.fit(children_lists)
            self.children[canonical] = child_normalizer

    def entity_schema(self) -> dict[str, Any] | None:
        """A JSON Schema fragment constraining an entity's ``type`` to this
        level's canonical vocabulary learned by ``fit``, for reuse with
        :class:`FlatEntityExtractor`'s ``entity_schema`` param. ``None`` if
        ``fit`` hasn't run yet (or ran on empty input).
        """
        if not self.mapping:
            return None
        return {"type": "string", "enum": sorted(set(self.mapping.values()))}

    def transform(self, entity_lists: list[list[Entity]]) -> list[list[Entity]]:
        """Apply the fitted mapping (recursively), without calling the backend."""
        return [
            [self._transform_entity(entity) for entity in entities] for entities in entity_lists
        ]

    def _transform_entity(self, entity: Entity) -> Entity:
        normalized = _normalize_type_string(entity.type)
        canonical = self.mapping.get(normalized, normalized)
        if entity.children is None:
            return Entity(type=canonical, value=entity.value)
        child_normalizer = self.children.get(canonical)
        if child_normalizer is not None:
            children = [child_normalizer._transform_entity(child) for child in entity.children]
        else:
            # Never seen during fit (e.g. a group type absent from the fitted corpus):
            # fall back to rule-normalization, recursively, rather than dropping children.
            children = [_rule_normalize_entity(child) for child in entity.children]
        return Entity(type=canonical, children=children)

    def to_dict(self) -> dict[str, Any]:
        """This normalizer's fitted state as a plain, JSON-serializable nested dict."""
        return {
            "mapping": dict(self.mapping),
            "children": {canonical: child.to_dict() for canonical, child in self.children.items()},
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any], backend: Backend) -> EntityTypeNormalizer:
        normalizer = cls(backend)
        mapping = data.get("mapping", {})
        if isinstance(mapping, dict):
            normalizer.mapping = {str(k): str(v) for k, v in mapping.items()}
        children = data.get("children", {})
        if isinstance(children, dict):
            normalizer.children = {
                str(canonical): cls._from_dict(child, backend)
                for canonical, child in children.items()
                if isinstance(child, dict)
            }
        return normalizer

    def save(self, path: str | Path) -> None:
        """Persist this normalizer's fitted state (recursively) as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path, backend: Backend) -> EntityTypeNormalizer:
        """Load a previously ``save``d state into a new normalizer.

        ``backend`` is still required (e.g. to ``fit`` further later) even
        though ``transform`` alone would not need it.
        """
        loaded = json.loads(Path(path).read_text())
        if not isinstance(loaded, dict):
            raise EntityNormalizationError(f"{path} does not contain a JSON object")
        return cls._from_dict(loaded, backend)


def _rule_normalize_entity(entity: Entity) -> Entity:
    normalized = _normalize_type_string(entity.type)
    if entity.children is None:
        return Entity(type=normalized, value=entity.value)
    return Entity(
        type=normalized, children=[_rule_normalize_entity(child) for child in entity.children]
    )
