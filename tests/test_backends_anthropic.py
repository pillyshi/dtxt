import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from dtxt.backends.anthropic import Anthropic


class _FakeMessages:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeAsyncMessages:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = _FakeMessages(response)


class _FakeAsyncClient:
    def __init__(self, response: Any) -> None:
        self.messages = _FakeAsyncMessages(response)


def _text_message(text: str) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _tool_use_message(input_obj: dict[str, Any]) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="emit_result", input=input_obj)]
    )


def test_generate_without_schema_returns_text_blocks() -> None:
    client = _FakeClient(_text_message("hello there"))
    backend = Anthropic("claude-x", client=client)

    assert backend.generate("prompt") == "hello there"
    call = client.messages.calls[0]
    assert call["messages"] == [{"role": "user", "content": "prompt"}]
    assert "tools" not in call


def test_generate_with_schema_forces_tool_use() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client = _FakeClient(_tool_use_message({"name": "Alice"}))
    backend = Anthropic("claude-x", client=client)

    result = backend.generate("prompt", schema=schema)

    assert json.loads(result) == {"name": "Alice"}
    call = client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "emit_result"}
    assert call["tools"][0]["input_schema"] == schema


def test_temperature_defaults_to_omitted() -> None:
    client = _FakeClient(_text_message("hello"))
    backend = Anthropic("claude-x", client=client)

    backend.generate("prompt")

    assert "temperature" not in client.messages.calls[0]


def test_temperature_is_forwarded_to_messages_create() -> None:
    client = _FakeClient(_text_message("hello"))
    backend = Anthropic("claude-x", client=client, temperature=0.7)

    backend.generate("prompt")

    assert client.messages.calls[0]["temperature"] == 0.7


def test_missing_tool_use_block_raises() -> None:
    client = _FakeClient(_text_message("oops, no tool call"))
    backend = Anthropic("claude-x", client=client)

    with pytest.raises(RuntimeError):
        backend.generate("prompt", schema={"type": "object", "properties": {}})


def test_capabilities_is_tool_calling() -> None:
    backend = Anthropic("claude-x", client=_FakeClient(_text_message("x")))
    assert backend.capabilities == {"tool_calling"}


def test_agenerate_uses_async_client() -> None:
    schema = {"type": "object", "properties": {}}
    async_client = _FakeAsyncClient(_tool_use_message({}))
    backend = Anthropic(
        "claude-x", client=_FakeClient(_text_message("unused")), async_client=async_client
    )

    result = asyncio.run(backend.agenerate("prompt", schema=schema))

    assert json.loads(result) == {}
    assert async_client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "emit_result"}


def test_missing_package_raises_helpful_error() -> None:
    backend = Anthropic("claude-x")
    with pytest.raises(ImportError, match=r"pip install dtxt\[anthropic\]"):
        backend.generate("prompt")
