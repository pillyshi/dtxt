import json

import pytest

from dtxt.backends import MockBackend
from dtxt.infer import InferError, SchemaInferer

# 5 texts' worth of NestedEntityExtractor.extract() responses, followed by one
# EntityTypeNormalizer.fit() response (identity mapping, since the raw types
# are already canonical here). "name" appears in 4/5 texts (coverage 0.8),
# "age" in 2/5 (coverage 0.4).
EXTRACTIONS = [
    [{"type": "name", "value": "Alice"}, {"type": "age", "value": "30"}],
    [{"type": "name", "value": "Bob"}],
    [{"type": "name", "value": "Carol"}],
    [{"type": "name", "value": "Dave"}],
    [{"type": "age", "value": "40"}],
]


def _responses(extractions: list[list[dict[str, str]]], mapping: dict[str, str]) -> list[str]:
    return [json.dumps(entities) for entities in extractions] + [json.dumps(mapping)]


def test_infer_keeps_fields_meeting_min_coverage() -> None:
    backend = MockBackend(responses=_responses(EXTRACTIONS, {"name": "name", "age": "age"}))
    texts = [f"text {i}" for i in range(len(EXTRACTIONS))]
    inferer = SchemaInferer(backend, min_coverage=0.6)

    schema = inferer.infer(texts)

    assert set(schema.properties) == {"name"}
    assert schema.properties["name"] == {"type": "string"}
    assert schema.required == []


def test_infer_raises_on_empty_input() -> None:
    with pytest.raises(InferError):
        SchemaInferer(MockBackend()).infer([])


def test_infer_raises_when_no_text_yields_a_usable_entity_list() -> None:
    backend = MockBackend(responses=["not json", "also not json"])
    with pytest.raises(InferError):
        SchemaInferer(backend).infer(["a", "b"])


def test_infer_marks_field_required_only_when_present_in_every_text() -> None:
    extractions = [
        [{"type": "name", "value": "Alice"}],
        [{"type": "name", "value": "Bob"}],
    ]
    backend = MockBackend(responses=_responses(extractions, {"name": "name"}))
    inferer = SchemaInferer(backend)

    schema = inferer.infer(["text 0", "text 1"])

    assert schema.required == ["name"]


def test_infer_detects_array_field_from_repetition_within_a_text() -> None:
    extractions = [
        [{"type": "tag", "value": "a"}, {"type": "tag", "value": "b"}],
        [{"type": "tag", "value": "c"}],
    ]
    backend = MockBackend(responses=_responses(extractions, {"tag": "tag"}))
    inferer = SchemaInferer(backend)

    schema = inferer.infer(["text 0", "text 1"])

    assert schema.properties["tag"] == {"type": "array", "items": {"type": "string"}}


def test_infer_builds_nested_object_field_from_majority_group_occurrences() -> None:
    extractions = [
        [
            {
                "type": "line_item",
                "children": [
                    {"type": "item_name", "value": "Widget"},
                    {"type": "quantity", "value": "3"},
                ],
            }
        ],
        [
            {
                "type": "line_item",
                "children": [
                    {"type": "item_name", "value": "Gadget"},
                    {"type": "quantity", "value": "1"},
                ],
            }
        ],
    ]
    backend = MockBackend(
        responses=_responses(extractions, {"line_item": "line_item"})
        + [json.dumps({"item_name": "item_name", "quantity": "quantity"})]
    )
    inferer = SchemaInferer(backend, max_depth=1)

    schema = inferer.infer(["text 0", "text 1"])

    assert schema.properties["line_item"] == {
        "type": "object",
        "properties": {
            "item_name": {"type": "string"},
            "quantity": {"type": "string"},
        },
        "required": ["item_name", "quantity"],
    }
