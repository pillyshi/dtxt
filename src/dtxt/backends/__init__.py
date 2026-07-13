"""Backend implementations for dtxt.

Only the mock backend is imported eagerly. API/local backends
(``Anthropic``, ``OpenAI``, ``LlamaCpp``) live in optional extras; their
modules -- and the underlying SDK packages -- are imported lazily on first
attribute access, so importing ``dtxt`` never requires them to be
installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import Backend
from .mock import MockBackend

__all__ = ["Backend", "MockBackend", "Anthropic", "OpenAI"]

_LAZY: dict[str, tuple[str, str]] = {
    "Anthropic": (".anthropic", "Anthropic"),
    "OpenAI": (".openai", "OpenAI"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attr_name)
