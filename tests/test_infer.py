import json

import pytest

from dtxt.backends import MockBackend
from dtxt.infer import InferError, infer_schema

CANDIDATES = [
    {"name": {"type": "string"}, "age": {"type": "integer"}},
    {"name": {"type": "string"}},
    {"name": {"type": "string"}},
    {"name": {"type": "string"}},
    {"age": {"type": "integer"}},
]


def test_infer_schema_keeps_fields_meeting_min_coverage() -> None:
    responses = [
        json.dumps({"type": "object", "properties": properties}) for properties in CANDIDATES
    ]
    backend = MockBackend(responses=responses)
    texts = [f"text {i}" for i in range(len(CANDIDATES))]

    schema = infer_schema(texts, backend=backend, batch_size=1, min_coverage=0.6)

    assert set(schema.properties) == {"name"}
    assert schema.required == []


def test_infer_schema_raises_on_empty_input() -> None:
    with pytest.raises(InferError):
        infer_schema([], backend=MockBackend())


def test_infer_schema_raises_when_no_candidate_is_usable() -> None:
    backend = MockBackend(responses=["not json", "also not json"])
    with pytest.raises(InferError):
        infer_schema(["a", "b"], backend=backend, batch_size=1)
