"""Anthropic API backend: structured output via forced tool use.

The ``anthropic`` package is an optional extra (``dtxt[anthropic]``) and is
imported lazily, only when a client actually needs to be built.
"""

from __future__ import annotations

import json
from typing import Any

from .base import TOOL_CALLING

_INSTALL_HINT = "pip install dtxt[anthropic]"
_TOOL_NAME = "emit_result"


def _import_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "the 'anthropic' package is required to use dtxt.backends.Anthropic. "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc
    return anthropic


def _build_request_kwargs(
    model: str, prompt: str, schema: dict[str, Any] | None, max_tokens: int
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if schema is not None:
        kwargs["tools"] = [
            {
                "name": _TOOL_NAME,
                "description": "Emit the result as structured data matching the schema.",
                "input_schema": schema,
            }
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}
    return kwargs


def _extract_result(message: Any, schema: dict[str, Any] | None) -> str:
    if schema is not None:
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == (
                _TOOL_NAME
            ):
                return json.dumps(block.input)
        raise RuntimeError("Anthropic response did not include the expected tool_use block")
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )


class Anthropic:
    """Backend for the Anthropic Messages API.

    Structured output is obtained by wrapping the JSON Schema as a single
    tool and forcing ``tool_choice`` onto it, so ``tool_use.input`` is the
    extracted object. This is not treated as ``constrained_decoding``: the
    API does not guarantee full schema conformance the way grammar-based
    decoding does, so dtxt still runs it through the retry + validation
    loop.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = client
        self._async_client = async_client

    def _get_client(self) -> Any:
        if self._client is None:
            anthropic = _import_anthropic()
            kwargs: dict[str, Any] = {"api_key": self._api_key} if self._api_key else {}
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            anthropic = _import_anthropic()
            kwargs: dict[str, Any] = {"api_key": self._api_key} if self._api_key else {}
            self._async_client = anthropic.AsyncAnthropic(**kwargs)
        return self._async_client

    @property
    def capabilities(self) -> set[str]:
        return {TOOL_CALLING}

    def generate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> str:
        kwargs = _build_request_kwargs(self._model, prompt, schema, self._max_tokens)
        message = self._get_client().messages.create(**kwargs)
        return _extract_result(message, schema)

    async def agenerate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> str:
        kwargs = _build_request_kwargs(self._model, prompt, schema, self._max_tokens)
        message = await self._get_async_client().messages.create(**kwargs)
        return _extract_result(message, schema)
