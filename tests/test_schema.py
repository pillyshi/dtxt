from pydantic import BaseModel

from dtxt import Schema


def _person_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "x-dtxt-description": "the person's full name",
                "x-dtxt-examples": ["Alice", "Bob"],
                "x-dtxt-style": "formal",
            },
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }


def test_to_json_schema_round_trips() -> None:
    raw = _person_json_schema()
    schema = Schema(raw)
    assert schema.to_json_schema() == raw


def test_to_json_schema_returns_a_copy() -> None:
    raw = _person_json_schema()
    schema = Schema(raw)
    mutated = schema.to_json_schema()
    mutated["properties"]["name"]["type"] = "integer"
    assert schema.to_json_schema()["properties"]["name"]["type"] == "string"


def test_from_pydantic() -> None:
    class Person(BaseModel):
        name: str
        age: int

    schema = Schema.from_pydantic(Person)
    assert schema.properties["name"]["type"] == "string"
    assert schema.properties["age"]["type"] == "integer"
    assert set(schema.required) == {"name", "age"}


def test_to_pydantic_builds_a_working_model() -> None:
    schema = Schema(_person_json_schema())
    model_cls = schema.to_pydantic("Person")
    instance = model_cls(name="Alice", age=30)
    assert instance.name == "Alice"  # type: ignore[attr-defined]
    assert instance.age == 30  # type: ignore[attr-defined]


def test_field_metadata_accessors() -> None:
    schema = Schema(_person_json_schema())
    assert schema.field_description("name") == "the person's full name"
    assert schema.field_examples("name") == ["Alice", "Bob"]
    assert schema.field_style("name") == "formal"
    assert schema.field_description("age") is None
    assert schema.field_examples("age") == []
    assert schema.field_style("age") is None


def test_validate_accepts_conformant_object() -> None:
    schema = Schema(_person_json_schema())
    schema.validate({"name": "Alice", "age": 30})


def test_is_valid_rejects_missing_required_field() -> None:
    schema = Schema(_person_json_schema())
    assert schema.is_valid({"age": 30}) is False
    assert schema.is_valid({"name": "Alice"}) is True


def test_iter_errors_reports_problems() -> None:
    schema = Schema(_person_json_schema())
    errors = schema.iter_errors({"age": "not-a-number"})
    assert len(errors) >= 1
