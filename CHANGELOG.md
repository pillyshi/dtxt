# Changelog

All notable changes to this project are documented in this file.

## [0.9.0] - 2026-07-23

### Added

- `StructuredEntityExtractor(backend, schema, *, fewshots=None, embedder=None,
  fewshot_k=3)`: optional few-shot example retrieval for T2D. `fewshots` is a
  fixed pool of `(text, obj)` pairs; each `extract()`/`extract_many()` call
  embeds the input text via `embedder` and includes the `fewshot_k` most
  similar examples (cosine similarity) in the prompt. `embedder` defaults to
  `dtxt.backends.SentenceTransformersEmbedder` (a small multilingual model)
  when `fewshots` is given but no embedder is passed. Leaving `fewshots`
  unset (default) reproduces the exact prompt used before this change --
  fully backward compatible. `extract_many()` and the asyncio batch path
  embed all query texts in a single batched call rather than one call per
  text.
- `dtxt.backends.Embedder` protocol (`embed(texts) -> list[list[float]]`)
  and `dtxt.backends.SentenceTransformersEmbedder`, a lazily-imported
  wrapper around `sentence-transformers` (new `dtxt[sentence-transformers]`
  extra, included in `dtxt[all]`). `dtxt.backends.MockEmbedder` added for
  testing.

## [0.8.0] - 2026-07-23

### Changed (breaking)

- Public API is now class-based; `dtxt.configure()` is removed along with
  the global-default-backend mechanism (`_config.py`). Every class takes
  its `backend` as a constructor argument instead.
  - `dtxt.parse` / `dtxt.parse_many` -> `dtxt.StructuredEntityExtractor`
    (constructed with `backend, schema`; methods `.extract(text)` /
    `.extract_many(texts)`).
  - `dtxt.render` -> `dtxt.StructuredEntityRenderer` (constructed with
    `backend, schema`; method `.render(obj, *, style=None)`).
  - `dtxt.infer_schema` -> `dtxt.SchemaInferer` (constructed with `backend,
    max_depth=1, min_coverage=0.6`; method `.infer(texts)`). Its
    implementation changed along with the rename (see below), so this is
    not a pure rename for `infer_schema` -- inferred schemas may differ
    from `0.7.0`'s.
  - `check_roundtrip(obj, schema, *, renderer=..., extractor=...)` now
    takes a `StructuredEntityRenderer`/`StructuredEntityExtractor` pair
    instead of raw `render_backend`/`parse_backend`.
- `SchemaInferer` no longer asks a backend to invent a candidate JSON
  Schema directly per batch. It now reuses `dtxt.entities`'s building
  blocks: `NestedEntityExtractor` extracts each text schema-free (one
  backend call per text, skipping texts that fail extraction rather than
  aborting), `EntityTypeNormalizer` reconciles type names across the
  corpus, and a new recursive coverage-based merge (replacing the old
  flat-only one) turns the normalized entity trees into a `Schema`.
  `min_coverage` is now applied at every nesting level, not just the top
  one; `batch_size` is gone (there's no longer a "batch" to size --
  extraction is one call per text). A field's type is now inferred as
  `string`, `array`, or a recursively-merged `object`/array-of-`object`
  based on entity shape, rather than by asking the backend to guess a
  JSON Schema type.
- `NestedEntityExtractor(backend, *, max_depth=1)`: the one-level nesting
  cap is now a configurable `max_depth` (still defaulting to one level, so
  default behavior is unchanged). Raising it lets `children` themselves
  carry `children`, recursively, at the cost of a larger/more recursive
  output JSON Schema.
- `EntityTypeNormalizer` now normalizes recursively, one level at a time:
  after fitting a level's own type-name mapping, it pools every occurrence
  of each canonical group type's `children` across the whole corpus and
  recurses into a child `EntityTypeNormalizer` (`self.children`, keyed by
  canonical group type) to normalize the next level down. `transform()` no
  longer drops a group entity's `children` (`0.7.0`'s `transform` silently
  discarded them -- a data-loss bug, not an intentional behavior being
  changed here). `save`/`load`'s on-disk format changed to a nested
  `{"mapping": ..., "children": {...}}` structure to hold this; `0.7.0`'s
  flat mapping files are not compatible with `0.8.0`'s `load()`.

## [0.7.0] - 2026-07-22

### Added

- `dtxt.entities.NestedEntityExtractor`: like `FlatEntityExtractor`, but an
  entity may represent one instance of a repeating structured record (e.g.
  a receipt's line items) by carrying `children` -- that instance's own
  flat entities -- instead of a scalar `value`. Repeated top-level entities
  of the same group `type` read as "array of that group," the same
  repetition-as-array signal `FlatEntityExtractor` already relies on for
  scalar arrays. Nesting is capped at one level: children never carry their
  own `children`, both by the (non-recursive) output JSON Schema handed to
  the backend and defensively when parsing its response. Unconstrained
  only for now -- no `entity_schema` vocabulary constraint yet; that's
  planned as a follow-up once `EntityTypeNormalizer` can learn a two-tier
  group/child vocabulary.
- `Entity.children`: optional field (`list[Entity] | None`, default
  `None`) added to the existing `Entity` model to support the above.
  `Entity.value` is now `str | None` (default `None`) to allow a group
  entity to omit it. Fully backward compatible -- `FlatEntityExtractor`
  and `EntityTypeNormalizer` are unchanged.
- `EntityRenderer.render()` now also handles group entities (rendering a
  group's `children` alongside it), so it still serves as a round-trip
  check (extract -> render -> re-extract) on `NestedEntityExtractor`'s
  output, not just `FlatEntityExtractor`'s.

## [0.6.0] - 2026-07-21

### Added

- `EntityTypeNormalizer.entity_schema()`: emits a JSON Schema fragment
  (`{"type": "string", "enum": [...]}`) constraining an entity's `type` to
  the canonical vocabulary learned by `fit`. Returns `None` before `fit`
  has run (or ran on empty input).
- `FlatEntityExtractor(backend, entity_schema=...)`: an optional
  constructor param, typically fed `EntityTypeNormalizer.entity_schema()`'s
  output, that constrains `extract()` to a known type vocabulary instead of
  letting the backend invent labels freely. The prompt lists the allowed
  types, the backend is called with a schema reflecting them (so a
  `constrained_decoding` backend can enforce it at the grammar level), and
  any entity whose type slips through outside the vocabulary is dropped.
  Left `None` (default), `extract()`'s behavior is unchanged.

## [0.5.0] - 2026-07-21

### Added

- `dtxt.entities.EntityRenderer`: renders a flat entity list back into
  text via a backend, the reverse of `FlatEntityExtractor`. Not
  schema-aware (no `x-dtxt-*` description/examples/style guidance like
  `dtxt.d2t.render` gets), so it's meant for round-trip checks on entity
  extraction/normalization quality rather than as a general D2T
  replacement.

## [0.4.0] - 2026-07-21

### Added

- `temperature` constructor parameter on `Anthropic`, `OpenAI`, and
  `LlamaCpp` backends: a fixed, per-instance sampling temperature
  forwarded to the underlying API/`create_chat_completion` call on every
  `generate`/`agenerate`. Left `None` (default), the provider's own
  default is used and the parameter is omitted from the request.

## [0.3.0] - 2026-07-17

### Added

- `dtxt.entities`: a flat-entity building block for schema inference,
  usable ahead of a full `infer_schema` integration.
  - `Entity`: a `(type, value)` pair.
  - `FlatEntityExtractor`: extracts a flat entity list from a single text
    via a backend (repeated types are kept, not deduplicated -- that
    repetition is itself an array-field signal).
  - `EntityTypeNormalizer`: reconciles entity types observed across a
    corpus into canonical names. `fit()` merges synonymous types (e.g.
    "name"/"full_name") into `self.mapping` via one backend call, seeded
    with a few example values per type; `transform()` applies the fitted
    mapping without touching the backend, falling back to a rule-based
    (lowercase/snake_case) normalization for types not seen during `fit`.
    `save()`/`load()` persist the mapping as JSON for reuse without
    re-fitting.

## [0.2.0] - 2026-07-16

### Added

- `LlamaCpp`: model can now be located via `repo_id`+`filename` (forwarded
  to `Llama.from_pretrained`) as an alternative to `model_path`. Added
  explicit `n_gpu_layers` and `flash_attn` constructor parameters.

## [0.1.0] - 2026-07-14

Initial public release.

### Added

- `Schema`: JSON-Schema-compatible schema with Pydantic conversion
  (`from_pydantic`/`to_pydantic`) and `x-dtxt-*` extension keywords for
  D2T metadata (`description`, `examples`, per-field and schema-wide
  `style`).
- `parse` / `parse_many` (T2D): text to schema-conformant object, with a
  retry + validation loop. `parse_many` runs concurrently via asyncio for
  backends that support it (bounded by `max_concurrency`), and reports
  partial batch failures as a single aggregated error.
- `render` (D2T): object to text, guided by schema metadata and an
  optional per-call `style` override.
- `check_roundtrip`: verifies `parse(render(obj)) ≈ obj`.
- `infer_schema`: sampling + merge schema inference with a `min_coverage`
  threshold.
- `configure()` for setting a default backend per function, with
  per-call `backend=` overrides.
- Backends: `MockBackend` (testing), `Anthropic` (forced tool use),
  `OpenAI` (`response_format` structured outputs), `LlamaCpp`
  (GBNF-constrained decoding with a two-stage grammar/post-hoc-validation
  split for `format`/`pattern`/deep nesting). All non-mock backends are
  optional extras, imported lazily.
