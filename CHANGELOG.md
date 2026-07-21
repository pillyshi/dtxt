# Changelog

All notable changes to this project are documented in this file.

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
