import asyncio
import json

import pytest

from dtxt import Schema
from dtxt.backends import MockBackend
from dtxt.t2d import ParseError, StructuredEntityExtractor

SCHEMA = Schema(
    {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
)


def test_extract_succeeds_on_valid_json() -> None:
    backend = MockBackend(responses=[json.dumps({"name": "Alice"})])
    extractor = StructuredEntityExtractor(backend, SCHEMA)
    assert extractor.extract("Alice said hi.") == {"name": "Alice"}


def test_extract_strips_markdown_fences() -> None:
    backend = MockBackend(responses=['```json\n{"name": "Alice"}\n```'])
    extractor = StructuredEntityExtractor(backend, SCHEMA)
    assert extractor.extract("Alice said hi.") == {"name": "Alice"}


def test_extract_retries_on_invalid_json_then_succeeds() -> None:
    backend = MockBackend(responses=["not json", json.dumps({"name": "Bob"})])
    extractor = StructuredEntityExtractor(backend, SCHEMA, max_retries=2)
    assert extractor.extract("Bob said hi.") == {"name": "Bob"}


def test_extract_retries_on_schema_violation_then_succeeds() -> None:
    backend = MockBackend(responses=[json.dumps({}), json.dumps({"name": "Carol"})])
    extractor = StructuredEntityExtractor(backend, SCHEMA, max_retries=2)
    assert extractor.extract("Carol said hi.") == {"name": "Carol"}


def test_extract_raises_after_max_retries_exhausted() -> None:
    backend = MockBackend(responses=["bad", "bad"])
    extractor = StructuredEntityExtractor(backend, SCHEMA, max_retries=2)
    with pytest.raises(ParseError):
        extractor.extract("text")


def test_constrained_decoding_backend_still_honors_max_retries() -> None:
    # A constrained-decoding backend (e.g. llama.cpp + GBNF) only
    # guarantees the grammar-facing part of the schema; keywords it can't
    # enforce (format/pattern) still rely on this retry loop, so
    # `max_retries` applies the same as for any other backend.
    backend = MockBackend(responses=["bad", "bad"], capabilities={"constrained_decoding"})
    extractor = StructuredEntityExtractor(backend, SCHEMA, max_retries=2)
    with pytest.raises(ParseError):
        extractor.extract("text")
    assert len(backend.calls) == 2


def test_constrained_decoding_backend_retries_on_semantic_validation_failure() -> None:
    # Simulates the two-stage split: the grammar can't enforce `pattern`,
    # so a first attempt violating it should still be retried and can
    # succeed on a later attempt via post-hoc validation.
    schema_with_pattern = Schema(
        {
            "type": "object",
            "properties": {"code": {"type": "string", "pattern": "^[A-Z]{3}$"}},
            "required": ["code"],
        }
    )
    backend = MockBackend(
        responses=[json.dumps({"code": "abc"}), json.dumps({"code": "ABC"})],
        capabilities={"constrained_decoding"},
    )
    extractor = StructuredEntityExtractor(backend, schema_with_pattern, max_retries=2)
    result = extractor.extract("some text")
    assert result == {"code": "ABC"}
    assert len(backend.calls) == 2


def test_extract_many_processes_each_text() -> None:
    backend = MockBackend(responses=[json.dumps({"name": "Alice"}), json.dumps({"name": "Bob"})])
    extractor = StructuredEntityExtractor(backend, SCHEMA)
    result = extractor.extract_many(["Alice said hi.", "Bob said hi."])
    assert result == [{"name": "Alice"}, {"name": "Bob"}]


def test_extract_many_returns_empty_list_for_no_texts() -> None:
    extractor = StructuredEntityExtractor(MockBackend(), SCHEMA)
    assert extractor.extract_many([]) == []


class _FakeAsyncBackend:
    """Duck-typed backend exposing agenerate, like Anthropic/OpenAI do."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.async_calls: list[str] = []

    @property
    def capabilities(self) -> set[str]:
        return set()

    def generate(self, prompt: str, *, schema: dict | None = None) -> str:
        raise AssertionError("sync generate should not be used when agenerate is available")

    async def agenerate(self, prompt: str, *, schema: dict | None = None) -> str:
        self.async_calls.append(prompt)
        for key, value in self._responses.items():
            if key in prompt:
                return value
        raise AssertionError(f"unexpected prompt: {prompt}")


def test_extract_many_uses_asyncio_when_backend_supports_agenerate() -> None:
    backend = _FakeAsyncBackend(
        {
            "Alice": json.dumps({"name": "Alice"}),
            "Bob": json.dumps({"name": "Bob"}),
        }
    )
    extractor = StructuredEntityExtractor(backend, SCHEMA)

    result = extractor.extract_many(["Alice said hi.", "Bob said hi."])

    assert result == [{"name": "Alice"}, {"name": "Bob"}]
    assert len(backend.async_calls) == 2


class _ConcurrencyTrackingBackend:
    """Tracks how many agenerate() calls are in flight at once."""

    def __init__(self) -> None:
        self._in_flight = 0
        self.max_concurrent_seen = 0

    @property
    def capabilities(self) -> set[str]:
        return set()

    def generate(self, prompt: str, *, schema: dict | None = None) -> str:
        raise AssertionError("sync generate should not be used when agenerate is available")

    async def agenerate(self, prompt: str, *, schema: dict | None = None) -> str:
        self._in_flight += 1
        self.max_concurrent_seen = max(self.max_concurrent_seen, self._in_flight)
        await asyncio.sleep(0.01)
        self._in_flight -= 1
        return json.dumps({"name": "X"})


def test_extract_many_respects_max_concurrency() -> None:
    backend = _ConcurrencyTrackingBackend()
    texts = [f"text {i}" for i in range(10)]
    extractor = StructuredEntityExtractor(backend, SCHEMA, max_concurrency=2)

    result = extractor.extract_many(texts)

    assert len(result) == 10
    assert backend.max_concurrent_seen <= 2


class _PartiallyFailingAsyncBackend:
    @property
    def capabilities(self) -> set[str]:
        return set()

    def generate(self, prompt: str, *, schema: dict | None = None) -> str:
        raise AssertionError("sync generate should not be used when agenerate is available")

    async def agenerate(self, prompt: str, *, schema: dict | None = None) -> str:
        if "fail" in prompt:
            return "not json"
        return json.dumps({"name": "OK"})


def test_extract_many_raises_aggregated_error_on_partial_failure() -> None:
    backend = _PartiallyFailingAsyncBackend()
    texts = ["ok text 1", "please fail here", "ok text 2"]
    extractor = StructuredEntityExtractor(backend, SCHEMA, max_retries=1)

    with pytest.raises(ParseError, match=r"failed for 1/3"):
        extractor.extract_many(texts)
