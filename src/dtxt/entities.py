"""Flat entity extraction: a building block for schema inference.

Rather than asking a backend to invent a JSON Schema directly, texts are
first reduced to a flat list of ``(type, value)`` entities. The same type
may repeat within one text (that repetition is itself a signal, e.g. of an
array field). A later normalization pass reconciles entity types across a
corpus before a schema is built from them.

``EntityRenderer`` provides the reverse direction (entities -> text), useful
for round-trip checks on extraction/normalization quality. It is not a
general-purpose D2T replacement -- see ``dtxt.d2t.render`` for that.
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
)

DEFAULT_MAX_EXAMPLES_PER_TYPE = 3

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_type_string(raw: str) -> str:
    """Cheap, LLM-free normalization: lowercase + snake_case."""
    return _NON_ALNUM.sub("_", raw.strip().lower()).strip("_")


class Entity(BaseModel):
    """A single ``(type, value)`` pair extracted from text."""

    type: str
    value: str


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


class EntityRenderer:
    """Renders a flat list of entities back into text via a backend.

    The reverse of :class:`FlatEntityExtractor`. Not schema-aware -- unlike
    :func:`dtxt.d2t.render`, there is no ``Schema`` here to supply field
    descriptions/examples/style, just the raw ``(type, value)`` pairs. This
    makes it useful mainly as a round-trip check on entity extraction and
    normalization quality (extract -> normalize -> render -> re-extract),
    not as a general-purpose D2T replacement.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def render(self, entities: list[Entity]) -> str:
        """Render ``entities`` into a single piece of natural-language text."""
        prompt = build_entity_render_prompt([(entity.type, entity.value) for entity in entities])
        text = self._backend.generate(prompt)
        if not text.strip():
            raise EntityRenderError("backend returned empty text")
        return text


class EntityTypeNormalizer:
    """Reconciles entity types observed across a corpus into canonical names.

    ``fit`` calls the backend once to merge synonymous types (e.g. "name"
    and "full_name") into a single canonical, snake_case type name, and
    keeps the resulting mapping on ``self.mapping``. ``transform`` applies
    that mapping without touching the backend, so a fitted mapping can be
    persisted with ``save`` and reused later via ``load``.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self.mapping: dict[str, str] = {}

    def fit(self, entity_lists: list[list[Entity]]) -> None:
        """Compute ``self.mapping`` from the entity types observed in ``entity_lists``."""
        examples: dict[str, list[str]] = {}
        for entities in entity_lists:
            for entity in entities:
                normalized = _normalize_type_string(entity.type)
                values = examples.setdefault(normalized, [])
                if len(values) < DEFAULT_MAX_EXAMPLES_PER_TYPE and entity.value not in values:
                    values.append(entity.value)

        if not examples:
            self.mapping = {}
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

    def entity_schema(self) -> dict[str, Any] | None:
        """A JSON Schema fragment constraining an entity's ``type`` to the
        canonical vocabulary learned by ``fit``, for reuse with
        :class:`FlatEntityExtractor`'s ``entity_schema`` param. ``None`` if
        ``fit`` hasn't run yet (or ran on empty input).
        """
        if not self.mapping:
            return None
        return {"type": "string", "enum": sorted(set(self.mapping.values()))}

    def transform(self, entity_lists: list[list[Entity]]) -> list[list[Entity]]:
        """Apply ``self.mapping`` to ``entity_lists``, without calling the backend.

        A type not covered by ``self.mapping`` (e.g. never seen during
        ``fit``) falls back to its rule-normalized form.
        """
        result = []
        for entities in entity_lists:
            mapped = []
            for entity in entities:
                normalized = _normalize_type_string(entity.type)
                canonical = self.mapping.get(normalized, normalized)
                mapped.append(Entity(type=canonical, value=entity.value))
            result.append(mapped)
        return result

    def save(self, path: str | Path) -> None:
        """Persist ``self.mapping`` as JSON."""
        Path(path).write_text(json.dumps(self.mapping, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path, backend: Backend) -> EntityTypeNormalizer:
        """Load a previously ``save``d mapping into a new normalizer.

        ``backend`` is still required (e.g. to ``fit`` further later) even
        though ``transform`` alone would not need it.
        """
        normalizer = cls(backend)
        loaded = json.loads(Path(path).read_text())
        if not isinstance(loaded, dict):
            raise EntityNormalizationError(f"{path} does not contain a JSON object mapping")
        normalizer.mapping = {
            str(raw_type): str(canonical) for raw_type, canonical in loaded.items()
        }
        return normalizer
