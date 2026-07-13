"""A deterministic, in-process backend used for unit tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

_TYPE_PLACEHOLDERS: dict[str, Any] = {
    "string": "",
    "integer": 0,
    "number": 0.0,
    "boolean": False,
    "array": [],
    "object": {},
}


def _placeholder_for(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    return {
        name: _TYPE_PLACEHOLDERS.get(prop.get("type"), None) for name, prop in properties.items()
    }


class MockBackend:
    """No network, no randomness: responses are canned or computed locally.

    Every call is recorded in ``calls`` so tests can assert on the prompts
    a caller produced.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        generate_fn: Callable[[str, dict[str, Any] | None], str] | None = None,
        capabilities: set[str] | None = None,
    ) -> None:
        self._responses: Iterator[str] | None = iter(responses) if responses is not None else None
        self._generate_fn = generate_fn
        self._capabilities = capabilities if capabilities is not None else set()
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def generate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> str:
        self.calls.append((prompt, schema))
        if self._generate_fn is not None:
            return self._generate_fn(prompt, schema)
        if self._responses is not None:
            try:
                return next(self._responses)
            except StopIteration as exc:
                raise RuntimeError("MockBackend ran out of canned responses") from exc
        return json.dumps(_placeholder_for(schema)) if schema is not None else ""

    @property
    def capabilities(self) -> set[str]:
        return self._capabilities
