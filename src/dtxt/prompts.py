"""Prompt templates for T2D, D2T, and schema inference.

Kept as plain module constants (rather than inline f-strings in the
pipeline modules) so a caller or a backend can override them -- e.g. a
small local model with different prompt sensitivities -- without touching
``t2d.py`` / ``d2t.py`` / ``infer.py``.
"""

from __future__ import annotations

import json
from typing import Any

T2D_TEMPLATE = """\
You convert unstructured text into a single JSON object that strictly follows the given JSON Schema.

# JSON Schema
{schema}

# Text
{text}

Respond with ONLY the JSON object. Do not include markdown fences or commentary.
If a field's value cannot be determined from the text, set it to null."""

T2D_RETRY_TEMPLATE = """\
{base_prompt}

# Previous attempt
{previous_output}

# Validation errors
{errors}

Fix the JSON object so it satisfies the schema and passes validation.
Respond with ONLY the corrected JSON object."""

D2T_TEMPLATE = """\
You convert a JSON object into natural, fluent text that expresses every non-null field.

# JSON Schema
{schema}

# Overall style
{style}

# Field guidance
{field_guidance}

# Object
{obj}

Write the text now. Do not include the JSON, labels, or commentary -- only the resulting text."""

INFER_TEMPLATE = """\
You infer a JSON Schema that describes the common structure shared by the following texts.

# Texts
{texts}

Respond with ONLY a JSON Schema object (type "object", with "properties" and "required").
Do not include markdown fences or commentary."""

ENTITY_EXTRACTION_TEMPLATE = """\
You extract every named entity or notable attribute mentioned in the text as a flat list of \
(type, value) pairs. Choose short, lowercase, snake_case labels for "type" based on what the \
value represents (e.g. "person_name", "date", "email"). If the text mentions multiple values of \
the same kind, emit one entity per value -- do not merge them.

# Text
{text}

Respond with ONLY a JSON array of objects, each with a "type" key and a "value" key.
Do not include markdown fences or commentary."""

ENTITY_EXTRACTION_CONSTRAINED_TEMPLATE = """\
You extract every named entity or notable attribute mentioned in the text as a flat list of \
(type, value) pairs. Only use one of the following allowed types for "type" -- do not invent \
new ones, and skip any entity that does not fit one of them. If the text mentions multiple \
values of the same kind, emit one entity per value -- do not merge them.

# Allowed types
{allowed_types}

# Text
{text}

Respond with ONLY a JSON array of objects, each with a "type" key and a "value" key.
Do not include markdown fences or commentary."""


def build_t2d_prompt(text: str, schema: dict[str, Any], *, template: str = T2D_TEMPLATE) -> str:
    return template.format(schema=json.dumps(schema, ensure_ascii=False, indent=2), text=text)


def build_t2d_retry_prompt(
    base_prompt: str,
    previous_output: str,
    errors: list[str],
    *,
    template: str = T2D_RETRY_TEMPLATE,
) -> str:
    return template.format(
        base_prompt=base_prompt,
        previous_output=previous_output,
        errors="\n".join(f"- {error}" for error in errors),
    )


def build_d2t_prompt(
    obj: dict[str, Any],
    schema: dict[str, Any],
    *,
    style: str = "(none)",
    field_guidance: str = "(none)",
    template: str = D2T_TEMPLATE,
) -> str:
    return template.format(
        schema=json.dumps(schema, ensure_ascii=False, indent=2),
        obj=json.dumps(obj, ensure_ascii=False, indent=2),
        style=style,
        field_guidance=field_guidance,
    )


def build_infer_prompt(texts: list[str], *, template: str = INFER_TEMPLATE) -> str:
    joined = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(texts))
    return template.format(texts=joined)


def build_entity_extraction_prompt(
    text: str,
    *,
    entity_schema: dict[str, Any] | None = None,
    template: str | None = None,
) -> str:
    if entity_schema is not None:
        allowed_types = "\n".join(f"- {t}" for t in entity_schema.get("enum", []))
        return (template or ENTITY_EXTRACTION_CONSTRAINED_TEMPLATE).format(
            text=text, allowed_types=allowed_types
        )
    return (template or ENTITY_EXTRACTION_TEMPLATE).format(text=text)


ENTITY_TYPE_MERGE_TEMPLATE = """\
You are given entity types observed across a text corpus, each with a few example values. Some \
types are synonyms or near-duplicates of each other (e.g. "name" and "full_name" both referring \
to a person's name). Group synonymous types together and assign each group a single canonical, \
lowercase, snake_case type name.

# Observed types
{types}

Respond with ONLY a JSON object mapping every observed type listed above (as its key, unchanged) \
to its canonical type name (as its value). Do not include markdown fences or commentary."""


def build_entity_type_merge_prompt(
    type_examples: dict[str, list[str]],
    *,
    template: str = ENTITY_TYPE_MERGE_TEMPLATE,
) -> str:
    listing = "\n".join(
        f"- {type_}: {', '.join(json.dumps(value, ensure_ascii=False) for value in values)}"
        for type_, values in type_examples.items()
    )
    return template.format(types=listing)


ENTITY_RENDER_TEMPLATE = """\
You write natural, fluent text that expresses every one of the following (type, value) entities. \
Do not add entities that are not listed, and do not drop any of the listed ones.

# Entities
{entities}

Write the text now. Do not include the entity list, labels, or commentary -- only the \
resulting text."""


def build_entity_render_prompt(
    entities: list[tuple[str, str]],
    *,
    template: str = ENTITY_RENDER_TEMPLATE,
) -> str:
    listing = "\n".join(f"- {type_}: {value}" for type_, value in entities)
    return template.format(entities=listing)
