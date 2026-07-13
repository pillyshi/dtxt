"""D2T: object -> text."""

from __future__ import annotations

from typing import Any

from ._config import resolve_backend
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


def render(
    obj: dict[str, Any],
    schema: Schema,
    *,
    style: str | None = None,
    backend: Backend | None = None,
) -> str:
    """Convert ``obj`` into text, guided by ``schema``'s ``x-dtxt-*`` metadata.

    ``style`` overrides the schema's own ``x-dtxt-style`` (its root-level
    style hint) for this call; per-field style hints still apply on top of
    either one.
    """
    resolved = resolve_backend("render", backend)
    json_schema = schema.to_json_schema()
    effective_style = style if style is not None else schema.style
    prompt = build_d2t_prompt(
        obj,
        json_schema,
        style=effective_style or "(none)",
        field_guidance=_field_guidance(schema),
    )
    return resolved.generate(prompt)
