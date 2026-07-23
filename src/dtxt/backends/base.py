"""Backend abstraction: the interface every dtxt backend implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

CONSTRAINED_DECODING = "constrained_decoding"
JSON_MODE = "json_mode"
TOOL_CALLING = "tool_calling"


@runtime_checkable
class Backend(Protocol):
    """Anything that can turn a prompt into text, optionally schema-guided.

    ``capabilities`` tells the caller which execution path to take:

    - ``constrained_decoding``: the backend enforces the JSON schema at the
      grammar level, so callers may skip syntactic validation and only run
      semantic checks.
    - otherwise: callers must retry and validate the output themselves.
    """

    def generate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> str: ...

    @property
    def capabilities(self) -> set[str]: ...


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn texts into embedding vectors.

    Used for few-shot example retrieval (see
    :class:`dtxt.t2d.StructuredEntityExtractor`'s ``fewshots`` argument):
    the extractor embeds its fixed pool of few-shot texts once, then embeds
    each input text at extraction time to rank the pool by cosine
    similarity.
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...
