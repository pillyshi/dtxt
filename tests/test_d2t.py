from dtxt import Schema
from dtxt.backends import MockBackend
from dtxt.d2t import StructuredEntityRenderer

SCHEMA = Schema(
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "x-dtxt-description": "the person's full name",
                "x-dtxt-style": "friendly",
            }
        },
        "required": ["name"],
    }
)


def test_render_returns_backend_output() -> None:
    backend = MockBackend(responses=["Alice says hello."])
    renderer = StructuredEntityRenderer(backend, SCHEMA)
    assert renderer.render({"name": "Alice"}) == "Alice says hello."


def test_render_prompt_includes_object_and_field_guidance() -> None:
    backend = MockBackend(responses=["ignored"])
    renderer = StructuredEntityRenderer(backend, SCHEMA)
    renderer.render({"name": "Alice"})

    assert len(backend.calls) == 1
    prompt, schema_arg = backend.calls[0]
    assert schema_arg is None
    assert "Alice" in prompt
    assert "the person's full name" in prompt
    assert "friendly" in prompt


def test_render_without_style_shows_none_placeholder() -> None:
    backend = MockBackend(responses=["ignored"])
    renderer = StructuredEntityRenderer(backend, SCHEMA)
    renderer.render({"name": "Alice"})
    prompt, _ = backend.calls[0]
    assert "# Overall style\n(none)" in prompt


def test_render_uses_schema_root_style_by_default() -> None:
    schema = Schema(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "x-dtxt-style": "formal, third person",
        }
    )
    backend = MockBackend(responses=["ignored"])
    renderer = StructuredEntityRenderer(backend, schema)
    renderer.render({"name": "Alice"})
    prompt, _ = backend.calls[0]
    assert "# Overall style\nformal, third person" in prompt


def test_render_style_argument_overrides_schema_style() -> None:
    schema = Schema(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "x-dtxt-style": "formal",
        }
    )
    backend = MockBackend(responses=["ignored"])
    renderer = StructuredEntityRenderer(backend, schema)
    renderer.render({"name": "Alice"}, style="casual, upbeat")
    prompt, _ = backend.calls[0]
    assert "# Overall style\ncasual, upbeat" in prompt
