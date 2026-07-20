"""llama.cpp backend: local GGUF models with GBNF-constrained decoding.

``llama-cpp-python`` is a heavy, optional extra (``dtxt[llamacpp]``) and is
imported lazily. Generation constrains JSON output at the grammar level
via ``response_format``. Schema constructs GBNF can't reliably express
(``format``, ``pattern``, deep nesting) are stripped from the
grammar-facing copy of the schema; the original schema is left untouched
for dtxt's normal post-hoc jsonschema validation. That is the two-stage
split: what the grammar guarantees during decoding, and what gets checked
afterwards.
"""

from __future__ import annotations

from typing import Any, cast

from .base import CONSTRAINED_DECODING

_INSTALL_HINT = "pip install dtxt[llamacpp]"

DEFAULT_MAX_GRAMMAR_DEPTH = 6
DEFAULT_PROMPT_TEMPLATE = "{prompt}"

_UNGRAMMARABLE_KEYWORDS = ("format", "pattern")


def _import_llama_cpp() -> Any:
    try:
        import llama_cpp
    except ImportError as exc:
        raise ImportError(
            "the 'llama-cpp-python' package is required to use dtxt.backends.LlamaCpp. "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc
    return llama_cpp


def grammar_safe_schema(
    schema: dict[str, Any], *, max_depth: int = DEFAULT_MAX_GRAMMAR_DEPTH
) -> dict[str, Any]:
    """Return a copy of ``schema`` with GBNF-unfriendly constructs removed.

    Drops ``format``/``pattern`` everywhere, and replaces ``object``/
    ``array`` schemas past ``max_depth`` nested levels with an
    unconstrained placeholder of the same type. The caller is expected to
    still validate the original schema post-hoc.
    """
    return cast(dict[str, Any], _strip(schema, depth=0, max_depth=max_depth))


def _strip(node: Any, *, depth: int, max_depth: int) -> Any:
    if not isinstance(node, dict):
        return node

    if depth > max_depth:
        node_type = node.get("type")
        if node_type in ("object", "array"):
            return {"type": node_type}
        return dict(node)

    stripped = {key: value for key, value in node.items() if key not in _UNGRAMMARABLE_KEYWORDS}
    if "properties" in stripped:
        stripped["properties"] = {
            name: _strip(prop, depth=depth + 1, max_depth=max_depth)
            for name, prop in stripped["properties"].items()
        }
    if "items" in stripped:
        stripped["items"] = _strip(stripped["items"], depth=depth + 1, max_depth=max_depth)
    return stripped


class LlamaCpp:
    """Backend for local GGUF models via ``llama-cpp-python``.

    A model is located either by local path (``model_path``) or by pulling
    from the Hugging Face Hub (``repo_id`` + ``filename``, forwarded to
    ``Llama.from_pretrained``). Exactly one of the two must be given.

    Small local models are prompt-sensitive, so the prompt can be wrapped
    in a model-specific chat template by passing ``prompt_template`` (a
    ``str.format`` template with a single ``{prompt}`` placeholder).

    ``temperature`` is a fixed, per-instance sampling parameter forwarded
    to ``create_chat_completion`` on every call; leave it ``None`` to use
    llama.cpp's own default.

    In-process inference assumes a single stream: batches are processed
    sequentially (no ``agenerate``), so llama.cpp's prompt cache is reused
    across calls sharing a prefix instead of contending for one model
    instance in parallel.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        flash_attn: bool = False,
        max_tokens: int = 1024,
        temperature: float | None = None,
        max_grammar_depth: int = DEFAULT_MAX_GRAMMAR_DEPTH,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        llama: Any | None = None,
        **llama_kwargs: Any,
    ) -> None:
        if llama is None:
            has_path = model_path is not None
            has_repo = repo_id is not None or filename is not None
            if has_path == has_repo:
                raise ValueError(
                    "LlamaCpp requires exactly one of `model_path` or "
                    "`repo_id`+`filename` to locate a model."
                )
            if has_repo and (repo_id is None or filename is None):
                raise ValueError("`repo_id` and `filename` must be given together.")

        self._model_path = model_path
        self._repo_id = repo_id
        self._filename = filename
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._flash_attn = flash_attn
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_grammar_depth = max_grammar_depth
        self._prompt_template = prompt_template
        self._llama_kwargs = llama_kwargs
        self._llama = llama

    def _get_llama(self) -> Any:
        if self._llama is None:
            llama_cpp = _import_llama_cpp()
            kwargs: dict[str, Any] = {
                "n_ctx": self._n_ctx,
                "n_gpu_layers": self._n_gpu_layers,
                "flash_attn": self._flash_attn,
                **self._llama_kwargs,
            }
            if self._model_path is not None:
                self._llama = llama_cpp.Llama(model_path=self._model_path, **kwargs)
            else:
                self._llama = llama_cpp.Llama.from_pretrained(
                    repo_id=self._repo_id, filename=self._filename, **kwargs
                )
        return self._llama

    @property
    def capabilities(self) -> set[str]:
        return {CONSTRAINED_DECODING}

    def generate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> str:
        llama = self._get_llama()
        kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": self._prompt_template.format(prompt=prompt)}],
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if schema is not None:
            safe_schema = grammar_safe_schema(schema, max_depth=self._max_grammar_depth)
            kwargs["response_format"] = {"type": "json_object", "schema": safe_schema}
        response = llama.create_chat_completion(**kwargs)
        return str(response["choices"][0]["message"]["content"])
