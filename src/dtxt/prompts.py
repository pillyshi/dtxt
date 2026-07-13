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
    field_guidance: str = "(none)",
    template: str = D2T_TEMPLATE,
) -> str:
    return template.format(
        schema=json.dumps(schema, ensure_ascii=False, indent=2),
        obj=json.dumps(obj, ensure_ascii=False, indent=2),
        field_guidance=field_guidance,
    )


def build_infer_prompt(texts: list[str], *, template: str = INFER_TEMPLATE) -> str:
    joined = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(texts))
    return template.format(texts=joined)
