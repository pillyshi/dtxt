from typing import Any

import pytest

from dtxt.backends.sentence_transformers import DEFAULT_MODEL, SentenceTransformersEmbedder


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]


def test_embed_returns_a_vector_per_text() -> None:
    client = _FakeSentenceTransformer()
    embedder = SentenceTransformersEmbedder(client=client)

    result = embedder.embed(["ab", "abcd"])

    assert result == [[2.0, 0.0], [4.0, 0.0]]
    assert client.encode_calls == [["ab", "abcd"]]


def test_client_is_reused_across_calls() -> None:
    client = _FakeSentenceTransformer()
    embedder = SentenceTransformersEmbedder(client=client)

    embedder.embed(["a"])
    embedder.embed(["b"])

    assert len(client.encode_calls) == 2


def test_default_model_is_multilingual() -> None:
    assert "multilingual" in DEFAULT_MODEL


def test_missing_package_raises_helpful_error() -> None:
    embedder = SentenceTransformersEmbedder()
    with pytest.raises(ImportError, match=r"pip install dtxt\[sentence-transformers\]"):
        embedder.embed(["text"])
