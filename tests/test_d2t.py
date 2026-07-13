from dtxt import Schema, render
from dtxt.backends import MockBackend

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
    assert render({"name": "Alice"}, SCHEMA, backend=backend) == "Alice says hello."


def test_render_prompt_includes_object_and_field_guidance() -> None:
    backend = MockBackend(responses=["ignored"])
    render({"name": "Alice"}, SCHEMA, backend=backend)

    assert len(backend.calls) == 1
    prompt, schema_arg = backend.calls[0]
    assert schema_arg is None
    assert "Alice" in prompt
    assert "the person's full name" in prompt
    assert "friendly" in prompt
