from typing import Any

import pytest

from dtxt.backends.llamacpp import LlamaCpp, grammar_safe_schema


class _FakeLlama:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self._content}}]}


def test_generate_without_schema_omits_response_format() -> None:
    llama = _FakeLlama("hello there")
    backend = LlamaCpp("model.gguf", llama=llama)

    assert backend.generate("prompt") == "hello there"
    call = llama.calls[0]
    assert call["messages"] == [{"role": "user", "content": "prompt"}]
    assert "response_format" not in call


def test_generate_with_schema_sets_grammar_response_format() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string", "pattern": "^[A-Z]"}}}
    llama = _FakeLlama('{"name": "Alice"}')
    backend = LlamaCpp("model.gguf", llama=llama)

    result = backend.generate("prompt", schema=schema)

    assert result == '{"name": "Alice"}'
    response_format = llama.calls[0]["response_format"]
    assert response_format["type"] == "json_object"
    # `pattern` cannot be reliably grammar-constrained, so it is stripped
    # from the grammar-facing schema (still present in the original for
    # post-hoc jsonschema validation).
    assert "pattern" not in response_format["schema"]["properties"]["name"]


def test_prompt_template_override_wraps_prompt() -> None:
    llama = _FakeLlama("ok")
    backend = LlamaCpp("model.gguf", llama=llama, prompt_template="<s>[INST] {prompt} [/INST]")

    backend.generate("hello")

    assert llama.calls[0]["messages"] == [{"role": "user", "content": "<s>[INST] hello [/INST]"}]


def test_capabilities_is_constrained_decoding() -> None:
    backend = LlamaCpp("model.gguf", llama=_FakeLlama("x"))
    assert backend.capabilities == {"constrained_decoding"}


def test_missing_package_raises_helpful_error() -> None:
    backend = LlamaCpp("model.gguf")
    with pytest.raises(ImportError, match=r"pip install dtxt\[llamacpp\]"):
        backend.generate("prompt")


def test_grammar_safe_schema_strips_format_and_pattern() -> None:
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "format": "uuid"},
            "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
        },
    }
    safe = grammar_safe_schema(schema)
    assert "format" not in safe["properties"]["id"]
    assert "pattern" not in safe["properties"]["code"]
    # the original schema is untouched
    assert schema["properties"]["id"]["format"] == "uuid"


def test_grammar_safe_schema_collapses_past_max_depth() -> None:
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    node = schema
    for _ in range(10):
        child: dict[str, Any] = {"type": "object", "properties": {}}
        node["properties"]["child"] = child
        node = child

    safe = grammar_safe_schema(schema, max_depth=2)

    node = safe
    for _ in range(2):
        node = node["properties"]["child"]
    # past max_depth, nested object schemas collapse to an unconstrained
    # placeholder instead of being fully expanded into the grammar
    assert node["properties"]["child"] == {"type": "object"}
