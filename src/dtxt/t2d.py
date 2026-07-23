"""T2D: text -> schema-conformant object."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ._util import cosine_similarity, extract_json
from .backends.base import Backend, Embedder
from .prompts import build_t2d_prompt, build_t2d_retry_prompt
from .schema import Schema

DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_FEWSHOT_K = 3

Fewshot = tuple[str, dict[str, Any]]


class ParseError(Exception):
    """Raised when a backend fails to produce a schema-conformant object."""


class StructuredEntityExtractor:
    """Converts text into an object conforming to a fixed ``Schema``, via a backend.

    The schema-specified counterpart to :class:`dtxt.entities.FlatEntityExtractor`
    / :class:`dtxt.entities.NestedEntityExtractor`, which extract schema-free.

    Optionally accepts a fixed pool of few-shot examples (``fewshots``, a
    list of ``(text, obj)`` pairs). When given, each call embeds the input
    text via ``embedder`` and picks the ``fewshot_k`` most similar examples
    (cosine similarity) to include in the prompt. ``embedder`` defaults to
    :class:`dtxt.backends.SentenceTransformersEmbedder` (a small
    multilingual model) if not given. Leaving ``fewshots`` empty/``None``
    reproduces the exact prompt used before few-shot support existed.
    """

    def __init__(
        self,
        backend: Backend,
        schema: Schema,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        fewshots: list[Fewshot] | None = None,
        embedder: Embedder | None = None,
        fewshot_k: int = DEFAULT_FEWSHOT_K,
    ) -> None:
        self._backend = backend
        self._schema = schema
        self._max_retries = max_retries
        self._max_concurrency = max_concurrency
        self._fewshots = list(fewshots) if fewshots else []
        self._fewshot_k = fewshot_k
        self._embedder: Embedder | None = None
        self._fewshot_embeddings: list[list[float]] = []
        if self._fewshots:
            if embedder is None:
                from .backends.sentence_transformers import SentenceTransformersEmbedder

                embedder = SentenceTransformersEmbedder()
            self._embedder = embedder
            self._fewshot_embeddings = embedder.embed([text for text, _ in self._fewshots])

    @property
    def schema(self) -> Schema:
        return self._schema

    def _rank_fewshots(self, query_embedding: list[float]) -> list[Fewshot]:
        ranked = sorted(
            zip(self._fewshots, self._fewshot_embeddings, strict=True),
            key=lambda pair: cosine_similarity(query_embedding, pair[1]),
            reverse=True,
        )
        return [fewshot for fewshot, _ in ranked[: self._fewshot_k]]

    def _select_fewshots(self, text: str) -> list[Fewshot]:
        if not self._fewshots:
            return []
        assert self._embedder is not None
        query_embedding = self._embedder.embed([text])[0]
        return self._rank_fewshots(query_embedding)

    def _select_fewshots_many(self, texts: list[str]) -> list[list[Fewshot]]:
        if not self._fewshots:
            return [[] for _ in texts]
        assert self._embedder is not None
        # Embedding all query texts in one call lets embedder implementations
        # batch (e.g. sentence-transformers' encode()), instead of paying a
        # per-text model call for every text in the batch.
        query_embeddings = self._embedder.embed(texts)
        return [self._rank_fewshots(embedding) for embedding in query_embeddings]

    def extract(self, text: str) -> dict[str, Any]:
        """Convert ``text`` into an object conforming to this extractor's schema.

        Missing or unextractable fields are represented as ``None``.
        """
        return self._extract_with_fewshots(text, self._select_fewshots(text))

    def _extract_with_fewshots(self, text: str, fewshots: list[Fewshot]) -> dict[str, Any]:
        json_schema = self._schema.to_json_schema()
        prompt = build_t2d_prompt(text, json_schema, fewshots=fewshots)

        # A constrained-decoding backend only guarantees the grammar-facing part
        # of the schema (see backends/llamacpp.py's two-stage split); keywords
        # like `format`/`pattern` still need this retry + validation loop, so
        # `max_retries` applies uniformly regardless of backend capabilities.
        attempts = self._max_retries

        last_error: Exception = ParseError("backend produced no output")
        raw = self._backend.generate(prompt, schema=json_schema)
        for attempt in range(attempts):
            is_last_attempt = attempt == attempts - 1
            try:
                obj = extract_json(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                if is_last_attempt:
                    break
                raw = self._backend.generate(
                    build_t2d_retry_prompt(prompt, raw, [f"invalid JSON: {exc}"]),
                    schema=json_schema,
                )
                continue

            errors = self._schema.iter_errors(obj)
            if not errors:
                return dict(obj)
            last_error = ValueError("; ".join(errors))
            if is_last_attempt:
                break
            raw = self._backend.generate(
                build_t2d_retry_prompt(prompt, raw, errors), schema=json_schema
            )

        raise ParseError(
            f"failed to parse a schema-conformant object after {attempts} attempt(s): {last_error}"
        ) from last_error

    def extract_many(self, texts: list[str]) -> list[dict[str, Any]]:
        """Batch version of ``extract``.

        When the backend exposes an ``agenerate`` coroutine (API backends
        such as Anthropic/OpenAI), texts are extracted concurrently via
        asyncio, bounded by ``max_concurrency`` to avoid tripping rate
        limits. Otherwise falls back to sequential ``extract`` calls (e.g.
        llama.cpp, which assumes a single in-process stream).
        """
        if not texts:
            return []
        if getattr(self._backend, "agenerate", None) is not None:
            return asyncio.run(self._extract_many_async(texts))
        fewshots_per_text = self._select_fewshots_many(texts)
        return [
            self._extract_with_fewshots(text, fewshots)
            for text, fewshots in zip(texts, fewshots_per_text, strict=True)
        ]

    async def _extract_many_async(self, texts: list[str]) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self._max_concurrency)
        fewshots_per_text = self._select_fewshots_many(texts)

        async def _bounded(text: str, fewshots: list[Fewshot]) -> dict[str, Any]:
            async with semaphore:
                return await self._extract_async(text, fewshots)

        # `return_exceptions=True` lets every text finish (success or failure)
        # instead of raising on the first failure and leaving other in-flight
        # requests to be torn down as dangling tasks.
        results: list[dict[str, Any] | BaseException] = await asyncio.gather(
            *(
                _bounded(text, fewshots)
                for text, fewshots in zip(texts, fewshots_per_text, strict=True)
            ),
            return_exceptions=True,
        )

        failures = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
        if failures:
            first_index, first_error = failures[0]
            raise ParseError(
                f"extract_many failed for {len(failures)}/{len(texts)} text(s); "
                f"first failure at index {first_index}: {first_error}"
            ) from first_error

        return [result for result in results if isinstance(result, dict)]

    async def _extract_async(
        self, text: str, fewshots: list[Fewshot] | None = None
    ) -> dict[str, Any]:
        if fewshots is None:
            fewshots = self._select_fewshots(text)
        backend: Any = self._backend
        json_schema = self._schema.to_json_schema()
        prompt = build_t2d_prompt(text, json_schema, fewshots=fewshots)
        attempts = self._max_retries

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

            errors = self._schema.iter_errors(obj)
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
