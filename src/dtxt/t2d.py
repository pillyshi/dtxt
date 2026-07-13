"""T2D: text -> schema-conformant object."""

from __future__ import annotations

import json
from typing import Any

from ._config import resolve_backend
from ._util import extract_json
from .backends.base import CONSTRAINED_DECODING, Backend
from .prompts import build_t2d_prompt, build_t2d_retry_prompt
from .schema import Schema

DEFAULT_MAX_RETRIES = 3


class ParseError(Exception):
    """Raised when a backend fails to produce a schema-conformant object."""


def parse(
    text: str,
    schema: Schema,
    *,
    backend: Backend | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Convert ``text`` into an object conforming to ``schema``.

    Missing or unextractable fields are represented as ``None``.
    """
    resolved = resolve_backend("parse", backend)
    json_schema = schema.to_json_schema()
    prompt = build_t2d_prompt(text, json_schema)

    # A constrained-decoding backend guarantees schema-conformant JSON at the
    # grammar level, so there is nothing for a retry loop to fix.
    attempts = 1 if CONSTRAINED_DECODING in resolved.capabilities else max_retries

    last_error: Exception = ParseError("backend produced no output")
    raw = resolved.generate(prompt, schema=json_schema)
    for attempt in range(attempts):
        is_last_attempt = attempt == attempts - 1
        try:
            obj = extract_json(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            if is_last_attempt:
                break
            raw = resolved.generate(
                build_t2d_retry_prompt(prompt, raw, [f"invalid JSON: {exc}"]),
                schema=json_schema,
            )
            continue

        errors = schema.iter_errors(obj)
        if not errors:
            return dict(obj)
        last_error = ValueError("; ".join(errors))
        if is_last_attempt:
            break
        raw = resolved.generate(build_t2d_retry_prompt(prompt, raw, errors), schema=json_schema)

    raise ParseError(
        f"failed to parse a schema-conformant object after {attempts} attempt(s): {last_error}"
    ) from last_error


def parse_many(
    texts: list[str],
    schema: Schema,
    *,
    backend: Backend | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict[str, Any]]:
    """Batch version of ``parse``.

    Sequential by default; async API backends optimize this internally by
    overriding this function's backend resolution with their own executor.
    """
    resolved = resolve_backend("parse", backend)
    return [parse(text, schema, backend=resolved, max_retries=max_retries) for text in texts]
