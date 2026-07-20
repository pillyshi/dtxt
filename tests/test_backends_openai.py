import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from dtxt.backends.openai import OpenAI


class _FakeCompletions:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeAsyncCompletions:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeChat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.chat = _FakeChat(_FakeCompletions(response))


class _FakeAsyncClient:
    def __init__(self, response: Any) -> None:
        self.chat = _FakeChat(_FakeAsyncCompletions(response))


def _response(content: str | None) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_generate_returns_message_content() -> None:
    client = _FakeClient(_response("hello"))
    backend = OpenAI("gpt-x", client=client)

    assert backend.generate("prompt") == "hello"
    assert "response_format" not in client.chat.completions.calls[0]


def test_generate_with_schema_sets_response_format() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client = _FakeClient(_response('{"name": "Alice"}'))
    backend = OpenAI("gpt-x", client=client)

    result = backend.generate("prompt", schema=schema)

    assert result == '{"name": "Alice"}'
    response_format = client.chat.completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema


def test_temperature_defaults_to_omitted() -> None:
    client = _FakeClient(_response("hello"))
    backend = OpenAI("gpt-x", client=client)

    backend.generate("prompt")

    assert "temperature" not in client.chat.completions.calls[0]


def test_temperature_is_forwarded_to_chat_completions_create() -> None:
    client = _FakeClient(_response("hello"))
    backend = OpenAI("gpt-x", client=client, temperature=0.7)

    backend.generate("prompt")

    assert client.chat.completions.calls[0]["temperature"] == 0.7


def test_missing_content_raises() -> None:
    client = _FakeClient(_response(None))
    backend = OpenAI("gpt-x", client=client)
    with pytest.raises(RuntimeError):
        backend.generate("prompt")


def test_capabilities_is_json_mode() -> None:
    backend = OpenAI("gpt-x", client=_FakeClient(_response("x")))
    assert backend.capabilities == {"json_mode"}


def test_agenerate_uses_async_client() -> None:
    async_client = _FakeAsyncClient(_response("async result"))
    backend = OpenAI("gpt-x", client=_FakeClient(_response("unused")), async_client=async_client)

    result = asyncio.run(backend.agenerate("prompt"))

    assert result == "async result"


def test_missing_package_raises_helpful_error() -> None:
    backend = OpenAI("gpt-x")
    with pytest.raises(ImportError, match=r"pip install dtxt\[openai\]"):
        backend.generate("prompt")
