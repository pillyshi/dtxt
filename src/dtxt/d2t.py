"""D2T: object -> text."""

from __future__ import annotations

from typing import Any

from .backends.base import Backend
from .prompts import build_d2t_prompt
from .schema import Schema


def _field_guidance(schema: Schema) -> str:
    lines: list[str] = []
    for name in schema.properties:
        parts: list[str] = []
        description = schema.field_description(name)
        if description:
            parts.append(description)
        examples = schema.field_examples(name)
        if examples:
            parts.append(f"examples: {examples}")
        style = schema.field_style(name)
        if style:
            parts.append(f"style: {style}")
        if parts:
            lines.append(f"- {name}: {'; '.join(parts)}")
    return "\n".join(lines) if lines else "(none)"


class StructuredEntityRenderer:
    """Converts an object conforming to a fixed ``Schema`` into text, via a backend.

    The schema-aware counterpart to :class:`dtxt.entities.EntityRenderer`,
    which renders schema-free entities.
    """

    def __init__(self, backend: Backend, schema: Schema) -> None:
        self._backend = backend
        self._schema = schema

    @property
    def schema(self) -> Schema:
        return self._schema

    def render(self, obj: dict[str, Any], *, style: str | None = None) -> str:
        """Convert ``obj`` into text, guided by the schema's ``x-dtxt-*`` metadata.

        ``style`` overrides the schema's own ``x-dtxt-style`` (its root-level
        style hint) for this call; per-field style hints still apply on top of
        either one.
        """
        json_schema = self._schema.to_json_schema()
        effective_style = style if style is not None else self._schema.style
        prompt = build_d2t_prompt(
            obj,
            json_schema,
            style=effective_style or "(none)",
            field_guidance=_field_guidance(self._schema),
        )
        return self._backend.generate(prompt)
