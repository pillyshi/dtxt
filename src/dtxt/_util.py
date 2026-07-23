"""Small helpers shared across the T2D / D2T / infer pipelines."""

from __future__ import annotations

import json
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, without a numpy dependency.

    Kept dependency-free (rather than using numpy) since this is on the
    core T2D path; embedding backends themselves may depend on numpy
    transitively, but core does not.
    """
    dot: float = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a: float = sum(x * x for x in a) ** 0.5
    norm_b: float = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_json(raw: str) -> Any:
    """Parse a JSON value out of a model response, tolerating code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().isalpha():
                text = rest
    return json.loads(text)
