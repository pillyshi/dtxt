"""Backend implementations for dtxt.

Only the mock backend is imported eagerly. API/local backends
(``anthropic``, ``openai``, ``llamacpp``) live in optional extras and are
imported lazily by their own modules to keep the core dependency-free.
"""

from .base import Backend
from .mock import MockBackend

__all__ = ["Backend", "MockBackend"]
