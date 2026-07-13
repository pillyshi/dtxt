"""T2D: text -> schema-conformant object."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ._config import resolve_backend
from ._util import extract_json
from .backends.base import Backend
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

    # A constrained-decoding backend only guarantees the grammar-facing part
    # of the schema (see backends/llamacpp.py's two-stage split); keywords
    # like `format`/`pattern` still need this retry + validation loop, so
    # `max_retries` applies uniformly regardless of backend capabilities.
    attempts = max_retries

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

    When the resolved backend exposes an ``agenerate`` coroutine (API
    backends such as Anthropic/OpenAI), texts are parsed concurrently via
    asyncio. Otherwise falls back to sequential ``parse`` calls.
    """
    resolved = resolve_backend("parse", backend)
    if not texts:
        return []
    if getattr(resolved, "agenerate", None) is not None:
        return asyncio.run(_parse_many_async(texts, schema, resolved, max_retries))
    return [parse(text, schema, backend=resolved, max_retries=max_retries) for text in texts]


async def _parse_many_async(
    texts: list[str],
    schema: Schema,
    backend: Any,
    max_retries: int,
) -> list[dict[str, Any]]:
    results = await asyncio.gather(
        *(_parse_async(text, schema, backend, max_retries) for text in texts)
    )
    return list(results)


async def _parse_async(
    text: str,
    schema: Schema,
    backend: Any,
    max_retries: int,
) -> dict[str, Any]:
    json_schema = schema.to_json_schema()
    prompt = build_t2d_prompt(text, json_schema)
    attempts = max_retries

    last_error: Exception = ParseError("backend produced no output")
    raw = await backend.agenerate(prompt, schema=json_schema)
    for attempt in range(attempts):
        is_last_attempt = attempt == attempts - 1
        try:
            obj = extract_json(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            if is_last_attempt:
                break
            raw = await backend.agenerate(
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
        raw = await backend.agenerate(
            build_t2d_retry_prompt(prompt, raw, errors), schema=json_schema
        )

    raise ParseError(
        f"failed to parse a schema-conformant object after {attempts} attempt(s): {last_error}"
    ) from last_error
