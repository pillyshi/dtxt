import json

import pytest

from dtxt import Schema, configure
from dtxt._config import resolve_backend
from dtxt.backends import MockBackend
from dtxt.t2d import parse


def test_resolve_backend_raises_when_unconfigured() -> None:
    with pytest.raises(RuntimeError):
        resolve_backend("parse", None)


def test_resolve_backend_prefers_explicit_override() -> None:
    configure(parse=MockBackend(responses=["configured"]))
    override = MockBackend(responses=["override"])
    assert resolve_backend("parse", override) is override


def test_configure_sets_global_default_used_by_parse() -> None:
    schema = Schema(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )
    backend = MockBackend(responses=[json.dumps({"name": "Alice"})])
    configure(parse=backend)
    assert parse("Alice said hi.", schema) == {"name": "Alice"}
