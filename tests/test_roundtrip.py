import json

from dtxt import Schema
from dtxt.backends import MockBackend
from dtxt.roundtrip import check_roundtrip

SCHEMA = Schema(
    {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
)


def test_check_roundtrip_reports_ok_when_fields_are_preserved() -> None:
    obj = {"name": "Alice", "age": 30}
    render_backend = MockBackend(responses=["Alice is 30 years old."])
    parse_backend = MockBackend(responses=[json.dumps(obj)])

    result = check_roundtrip(
        obj, SCHEMA, render_backend=render_backend, parse_backend=parse_backend
    )

    assert result.ok is True
    assert result.mismatches == {}
    assert result.rendered_text == "Alice is 30 years old."
    assert result.reparsed == obj


def test_check_roundtrip_reports_mismatches() -> None:
    obj = {"name": "Alice", "age": 30}
    render_backend = MockBackend(responses=["Alice is 30 years old."])
    parse_backend = MockBackend(responses=[json.dumps({"name": "Alice", "age": 31})])

    result = check_roundtrip(
        obj, SCHEMA, render_backend=render_backend, parse_backend=parse_backend
    )

    assert result.ok is False
    assert result.mismatches == {"age": (30, 31)}
