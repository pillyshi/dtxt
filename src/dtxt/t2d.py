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
DEFAULT_MAX_CONCURRENCY = 8


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
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Batch version of ``parse``.

    When the resolved backend exposes an ``agenerate`` coroutine (API
    backends such as Anthropic/OpenAI), texts are parsed concurrently via
    asyncio, bounded by ``max_concurrency`` to avoid tripping rate limits.
    Otherwise falls back to sequential ``parse`` calls (e.g. llama.cpp,
    which assumes a single in-process stream).
    """
    resolved = resolve_backend("parse", backend)
    if not texts:
        return []
    if getattr(resolved, "agenerate", None) is not None:
        return asyncio.run(_parse_many_async(texts, schema, resolved, max_retries, max_concurrency))
    return [parse(text, schema, backend=resolved, max_retries=max_retries) for text in texts]


async def _parse_many_async(
    texts: list[str],
    schema: Schema,
    backend: Any,
    max_retries: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(text: str) -> dict[str, Any]:
        async with semaphore:
            return await _parse_async(text, schema, backend, max_retries)

    # `return_exceptions=True` lets every text finish (success or failure)
    # instead of raising on the first failure and leaving other in-flight
    # requests to be torn down as dangling tasks.
    results: list[dict[str, Any] | BaseException] = await asyncio.gather(
        *(_bounded(text) for text in texts), return_exceptions=True
    )

    failures = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
    if failures:
        first_index, first_error = failures[0]
        raise ParseError(
            f"parse_many failed for {len(failures)}/{len(texts)} text(s); "
            f"first failure at index {first_index}: {first_error}"
        ) from first_error

    return [result for result in results if isinstance(result, dict)]


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
