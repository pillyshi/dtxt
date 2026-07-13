# Changelog

All notable changes to this project are documented in this file.

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
