import json

import pytest

from dtxt import Schema
from dtxt.backends import MockBackend
from dtxt.t2d import ParseError, parse, parse_many

SCHEMA = Schema(
    {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
)


def test_parse_succeeds_on_valid_json() -> None:
    backend = MockBackend(responses=[json.dumps({"name": "Alice"})])
    assert parse("Alice said hi.", SCHEMA, backend=backend) == {"name": "Alice"}


def test_parse_strips_markdown_fences() -> None:
    backend = MockBackend(responses=['```json\n{"name": "Alice"}\n```'])
    assert parse("Alice said hi.", SCHEMA, backend=backend) == {"name": "Alice"}


def test_parse_retries_on_invalid_json_then_succeeds() -> None:
    backend = MockBackend(responses=["not json", json.dumps({"name": "Bob"})])
    assert parse("Bob said hi.", SCHEMA, backend=backend, max_retries=2) == {"name": "Bob"}


def test_parse_retries_on_schema_violation_then_succeeds() -> None:
    backend = MockBackend(responses=[json.dumps({}), json.dumps({"name": "Carol"})])
    assert parse("Carol said hi.", SCHEMA, backend=backend, max_retries=2) == {"name": "Carol"}


def test_parse_raises_after_max_retries_exhausted() -> None:
    backend = MockBackend(responses=["bad", "bad"])
    with pytest.raises(ParseError):
        parse("text", SCHEMA, backend=backend, max_retries=2)


def test_constrained_decoding_backend_skips_retry_loop() -> None:
    backend = MockBackend(responses=["bad"], capabilities={"constrained_decoding"})
    with pytest.raises(ParseError):
        parse("text", SCHEMA, backend=backend, max_retries=5)
    assert len(backend.calls) == 1


def test_parse_many_processes_each_text() -> None:
    backend = MockBackend(responses=[json.dumps({"name": "Alice"}), json.dumps({"name": "Bob"})])
    result = parse_many(["Alice said hi.", "Bob said hi."], SCHEMA, backend=backend)
    assert result == [{"name": "Alice"}, {"name": "Bob"}]


def test_parse_many_returns_empty_list_for_no_texts() -> None:
    assert parse_many([], SCHEMA, backend=MockBackend()) == []


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


def test_parse_many_uses_asyncio_when_backend_supports_agenerate() -> None:
    backend = _FakeAsyncBackend(
        {
            "Alice": json.dumps({"name": "Alice"}),
            "Bob": json.dumps({"name": "Bob"}),
        }
    )

    result = parse_many(["Alice said hi.", "Bob said hi."], SCHEMA, backend=backend)

    assert result == [{"name": "Alice"}, {"name": "Bob"}]
    assert len(backend.async_calls) == 2
