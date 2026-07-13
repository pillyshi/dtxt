# dtxt

Schema-centric bidirectional conversion between text and structured data.

`dtxt` is not related to [`dtx`](https://pypi.org/project/dtx/) (an AI red-teaming tool).

## Core features

1. **Schema inference** (`infer_schema`): derive a schema from a collection of texts
2. **T2D** (`parse`): convert text into a schema-conformant object
3. **D2T** (`render`): convert an object into text
4. **Round-trip verification** (`check_roundtrip`): check that
   `parse(render(obj)) ≈ obj` for a given schema and backend

## Install

```bash
pip install dtxt              # core only
pip install dtxt[anthropic]   # + Anthropic backend
pip install dtxt[openai]      # + OpenAI backend
pip install dtxt[llamacpp]    # + local GGUF models via llama.cpp
pip install dtxt[all]         # everything
```

The core package depends only on `pydantic` and `jsonschema`. Backends are
optional extras, imported lazily.

## Usage

```python
import dtxt
from dtxt import Schema
from dtxt.backends import MockBackend

schema = Schema({
    "type": "object",
    "properties": {
        "name": {"type": "string", "x-dtxt-description": "the person's full name"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
})

# Backends can be set globally per function...
dtxt.configure(parse=MockBackend(), render=MockBackend())

# ...or overridden per call via backend=.
obj = dtxt.parse("Alice is 30 years old.", schema, backend=MockBackend())
text = dtxt.render({"name": "Alice", "age": 30}, schema)

result = dtxt.check_roundtrip({"name": "Alice", "age": 30}, schema)
result.ok  # True if parse(render(obj)) == obj on every schema field
```

Swap `MockBackend` for a real one once available:

```python
dtxt.configure(
    infer=dtxt.backends.Anthropic("claude-sonnet-4-6"),
    parse=dtxt.backends.LlamaCpp("model.gguf", n_ctx=8192),
    render=dtxt.backends.Anthropic("claude-sonnet-4-6"),
)
```

## Status

Early development (`0.0.x`). Implemented so far: `Schema`, `parse` /
`parse_many`, `render`, `infer_schema`, `check_roundtrip`, `configure`, and
a mock backend for testing. The Anthropic / OpenAI / llama.cpp backends are
not implemented yet -- see `CLAUDE.md` for the milestone plan.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```
