"""Sentence-Transformers-based embedder, for few-shot example retrieval.

``sentence-transformers`` is a heavy, optional extra (``dtxt[sentence-transformers]``)
and is imported lazily, matching the llama.cpp backend's approach. The
default model is a small multilingual one (see ``DEFAULT_MODEL``) so
few-shot retrieval works out of the box across languages without the
caller having to pick a model.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = "pip install dtxt[sentence-transformers]"

# A widely-used, lightweight multilingual sentence-embedding model
# (50+ languages, including Japanese) -- a reasonable default for few-shot
# retrieval without requiring the caller to pick a model up front.
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _import_sentence_transformers() -> Any:
    try:
        import sentence_transformers
    except ImportError as exc:
        raise ImportError(
            "the 'sentence-transformers' package is required to use "
            f"dtxt.backends.SentenceTransformersEmbedder. Install it with: {_INSTALL_HINT}"
        ) from exc
    return sentence_transformers


class SentenceTransformersEmbedder:
    """Embeds texts with a local ``sentence-transformers`` model.

    The model is loaded lazily on first ``embed()`` call, and reused
    across calls.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, client: Any | None = None) -> None:
        self._model_name = model
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            sentence_transformers = _import_sentence_transformers()
            self._client = sentence_transformers.SentenceTransformer(self._model_name)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        embeddings = client.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [[float(x) for x in row] for row in embeddings]
