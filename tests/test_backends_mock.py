import json

from dtxt.backends import MockBackend


def test_default_response_matches_placeholder_for_schema() -> None:
    backend = MockBackend()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    result = json.loads(backend.generate("prompt", schema=schema))
    assert result == {"name": "", "age": 0}


def test_canned_responses_are_returned_in_order() -> None:
    backend = MockBackend(responses=["first", "second"])
    assert backend.generate("p1") == "first"
    assert backend.generate("p2") == "second"


def test_generate_fn_overrides_canned_responses() -> None:
    backend = MockBackend(generate_fn=lambda prompt, schema: f"echo:{prompt}")
    assert backend.generate("hello") == "echo:hello"


def test_calls_are_recorded() -> None:
    backend = MockBackend(responses=["ok"])
    backend.generate("prompt", schema={"type": "object"})
    assert backend.calls == [("prompt", {"type": "object"})]


def test_capabilities_default_to_empty_set() -> None:
    assert MockBackend().capabilities == set()
    assert MockBackend(capabilities={"constrained_decoding"}).capabilities == {
        "constrained_decoding"
    }
