# dtxt

Schema-centric bidirectional conversion between text and structured data.

`dtxt` is not related to [`dtx`](https://pypi.org/project/dtx/) (an AI red-teaming tool).

## Core features

1. **Schema inference** (`SchemaInferer`): derive a schema from a collection of texts
2. **T2D** (`StructuredEntityExtractor`): convert text into a schema-conformant object
3. **D2T** (`StructuredEntityRenderer`): convert an object into text
4. **Round-trip verification** (`check_roundtrip`): check that
   `extract(render(obj)) ≈ obj` for a given schema and backend

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

# Each class takes its backend at construction -- there is no global config.
extractor = dtxt.StructuredEntityExtractor(MockBackend(), schema)
renderer = dtxt.StructuredEntityRenderer(MockBackend(), schema)

obj = extractor.extract("Alice is 30 years old.")
text = renderer.render({"name": "Alice", "age": 30})

result = dtxt.check_roundtrip(
    {"name": "Alice", "age": 30}, schema, renderer=renderer, extractor=extractor
)
result.ok  # True if extract(render(obj)) == obj on every schema field
```

Swap `MockBackend` for a real one:

```python
extractor = dtxt.StructuredEntityExtractor(dtxt.backends.LlamaCpp("model.gguf", n_ctx=8192), schema)
renderer = dtxt.StructuredEntityRenderer(dtxt.backends.Anthropic("claude-sonnet-4-6"), schema)
inferer = dtxt.SchemaInferer(dtxt.backends.Anthropic("claude-sonnet-4-6"))
```

`LlamaCpp` locates a model either by local path (`model_path`) or by
pulling from the Hugging Face Hub (`repo_id` + `filename`, forwarded to
`Llama.from_pretrained`); `n_gpu_layers` and `flash_attn`, among other
`llama-cpp-python` constructor options, are also exposed:

```python
dtxt.backends.LlamaCpp(
    repo_id="TheBloke/some-model-GGUF",
    filename="some-model.Q4_K_M.gguf",
    n_ctx=8192,
    n_gpu_layers=32,
    flash_attn=True,
)
```

`Anthropic` uses forced tool use to get structured output; `OpenAI` uses
`response_format={"type": "json_schema", ...}`; `LlamaCpp` constrains
decoding at the grammar level via GBNF. None of them guarantee full schema
conformance on their own:

- Anthropic/OpenAI guarantee valid JSON syntax, not every schema keyword.
- `LlamaCpp` strips constructs GBNF can't reliably express (`format`,
  `pattern`, deeply nested objects/arrays) from the grammar-facing schema;
  the original schema is still checked afterwards.

So all three go through dtxt's retry + validation loop the same way.
`StructuredEntityExtractor.extract_many` runs concurrently via asyncio for
Anthropic/OpenAI, bounded by `max_concurrency` (default 8) to avoid
tripping rate limits; `LlamaCpp` processes it sequentially in-process so
its prompt cache stays warm. A partial batch failure raises one
`ParseError` naming how many texts failed and the first failing index,
rather than aborting on the first error.

Style is controllable at both the schema and call level:

```python
schema = Schema({
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "x-dtxt-style": "formal, third person",  # schema-wide default
})
renderer = dtxt.StructuredEntityRenderer(backend, schema)
renderer.render(obj)                       # uses "formal, third person"
renderer.render(obj, style="casual, upbeat")  # overrides it for this call
```

Schema inference (`SchemaInferer`) works schema-free first: each text is
reduced to a tree of `(type, value)` entities (optionally with `children`
for repeating structured records, e.g. a receipt's line items) via
`dtxt.entities`, type names are reconciled across the corpus, and a field
is kept only if it meets a `min_coverage` threshold -- applied recursively,
so nested object fields go through the same coverage test as top-level
ones:

```python
inferer = dtxt.SchemaInferer(backend, min_coverage=0.6)
schema = inferer.infer(texts)
```

## Status

Released as [`dtxt` `0.7.0`](https://pypi.org/project/dtxt/0.7.0/) on PyPI;
`0.8.0` (unreleased) reworks the public API to be class-based -- see
`CHANGELOG.md`. `Schema`, `StructuredEntityExtractor` (T2D, with
`extract_many` asyncio batching), `StructuredEntityRenderer` (D2T, with
schema-level and per-call style control), `SchemaInferer` (schema-free
extraction + recursive coverage-based merge, `min_coverage`),
`check_roundtrip`, a mock backend for testing, and the Anthropic / OpenAI /
llama.cpp backends are implemented. `dtxt.entities`
(`FlatEntityExtractor`, `NestedEntityExtractor`, `EntityTypeNormalizer`,
`EntityRenderer`) is `SchemaInferer`'s internal implementation. See
`CHANGELOG.md` for release notes and `CLAUDE.md` for what's next.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```
