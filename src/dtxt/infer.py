"""infer_schema: schema inference via sampling + merge.

Texts are sampled in small batches (kind to small context windows); each
batch yields a candidate schema, and a field is kept in the merged schema
only if it appears in at least ``min_coverage`` of the candidates.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ._config import resolve_backend
from ._util import extract_json
from .backends.base import Backend
from .prompts import build_infer_prompt
from .schema import Schema

DEFAULT_BATCH_SIZE = 5
DEFAULT_MIN_COVERAGE = 0.6


class InferError(Exception):
    """Raised when schema inference cannot produce a usable schema."""


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _candidate_schema(batch: list[str], backend: Backend) -> dict[str, Any] | None:
    prompt = build_infer_prompt(batch)
    raw = backend.generate(prompt)
    try:
        candidate = extract_json(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(candidate, dict) or not isinstance(candidate.get("properties"), dict):
        return None
    return candidate


def _merge_candidates(candidates: list[dict[str, Any]], min_coverage: float) -> dict[str, Any]:
    n = len(candidates)
    field_types: dict[str, Counter[str]] = {}
    field_count: Counter[str] = Counter()

    for candidate in candidates:
        for name, field_schema in candidate["properties"].items():
            field_count[name] += 1
            field_types.setdefault(name, Counter())[field_schema.get("type", "string")] += 1

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, count in field_count.items():
        coverage = count / n
        if coverage < min_coverage:
            continue
        most_common_type, _ = field_types[name].most_common(1)[0]
        properties[name] = {"type": most_common_type}
        if coverage >= 1.0:
            required.append(name)

    return {"type": "object", "properties": properties, "required": sorted(required)}


def infer_schema(
    texts: list[str],
    *,
    backend: Backend | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> Schema:
    """Infer a JSON Schema shared by ``texts``."""
    if not texts:
        raise InferError("cannot infer a schema from an empty text collection")

    resolved = resolve_backend("infer", backend)
    batches = _batched(texts, batch_size)

    candidates = [
        candidate
        for candidate in (_candidate_schema(batch, resolved) for batch in batches)
        if candidate is not None
    ]
    if not candidates:
        raise InferError("backend did not return any usable candidate schema")

    merged = _merge_candidates(candidates, min_coverage)
    if not merged["properties"]:
        raise InferError(
            f"no fields met the min_coverage={min_coverage} threshold across "
            f"{len(candidates)} candidate schema(s)"
        )
    return Schema(merged)
