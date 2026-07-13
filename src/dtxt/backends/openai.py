"""OpenAI API backend: structured outputs via ``response_format``.

The ``openai`` package is an optional extra (``dtxt[openai]``) and is
imported lazily, only when a client actually needs to be built.
"""

from __future__ import annotations

from typing import Any

from .base import JSON_MODE

_INSTALL_HINT = "pip install dtxt[openai]"


def _import_openai() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "the 'openai' package is required to use dtxt.backends.OpenAI. "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc
    return openai


def _build_request_kwargs(model: str, prompt: str, schema: dict[str, Any] | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "dtxt_result", "schema": schema},
        }
    return kwargs


def _extract_result(response: Any) -> str:
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI response contained no content")
    return str(content)


class OpenAI:
    """Backend for the OpenAI Chat Completions API.

    Uses ``response_format={"type": "json_schema", ...}`` (structured
    outputs) when a schema is given. This is not treated as
    ``constrained_decoding``: OpenAI's guarantee is on JSON syntax, not on
    every schema keyword dtxt supports, so dtxt still runs it through the
    retry + validation loop.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client = client
        self._async_client = async_client

    def _get_client(self) -> Any:
        if self._client is None:
            openai = _import_openai()
            kwargs: dict[str, Any] = {"api_key": self._api_key} if self._api_key else {}
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            openai = _import_openai()
            kwargs: dict[str, Any] = {"api_key": self._api_key} if self._api_key else {}
            self._async_client = openai.AsyncOpenAI(**kwargs)
        return self._async_client

    @property
    def capabilities(self) -> set[str]:
        return {JSON_MODE}

    def generate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> str:
        kwargs = _build_request_kwargs(self._model, prompt, schema)
        response = self._get_client().chat.completions.create(**kwargs)
        return _extract_result(response)

    async def agenerate(self, prompt: str, *, schema: dict[str, Any] | None = None) -> str:
        kwargs = _build_request_kwargs(self._model, prompt, schema)
        response = await self._get_async_client().chat.completions.create(**kwargs)
        return _extract_result(response)
