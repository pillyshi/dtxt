"""Small helpers shared across the T2D / D2T / infer pipelines."""

from __future__ import annotations

import json
from typing import Any


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
