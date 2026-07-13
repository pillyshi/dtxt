"""Schema: the JSON-Schema-compatible internal representation used across dtxt."""

from __future__ import annotations

import copy
from typing import Any, Optional

import jsonschema
from pydantic import BaseModel, create_model

_EXT_DESCRIPTION = "x-dtxt-description"
_EXT_EXAMPLES = "x-dtxt-examples"
_EXT_STYLE = "x-dtxt-style"

_JSON_TYPE_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class Schema:
    """A JSON-Schema-compatible schema.

    D2T description metadata (description, examples, style hints) is carried
    as ``x-dtxt-*`` extension keywords on field schemas so it round-trips
    through plain JSON Schema tooling.
    """

    def __init__(self, json_schema: dict[str, Any]) -> None:
        self._json_schema = copy.deepcopy(json_schema)

    def __repr__(self) -> str:
        return f"Schema({self._json_schema!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        return self._json_schema == other._json_schema

    @classmethod
    def from_pydantic(cls, model: type[BaseModel]) -> Schema:
        return cls(model.model_json_schema())

    @classmethod
    def from_dict(cls, json_schema: dict[str, Any]) -> Schema:
        return cls(json_schema)

    def to_json_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._json_schema)

    def to_pydantic(self, name: str = "DtxtModel") -> type[BaseModel]:
        """Build a best-effort Pydantic model from the top-level properties.

        This covers the common flat-record case; deeply nested or
        format-constrained schemas keep their full fidelity in
        ``to_json_schema()`` instead.
        """
        fields: dict[str, Any] = {}
        required = set(self.required)
        for field_name, field_schema in self.properties.items():
            py_type = _JSON_TYPE_TO_PY.get(field_schema.get("type"), Any)
            if field_name in required:
                fields[field_name] = (py_type, ...)
            else:
                fields[field_name] = (Optional[py_type], None)  # noqa: UP045
        return create_model(name, **fields)

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self._json_schema.get("properties", {}))

    @property
    def required(self) -> list[str]:
        return list(self._json_schema.get("required", []))

    def field_description(self, field_name: str) -> str | None:
        result = self.properties.get(field_name, {}).get(_EXT_DESCRIPTION)
        return result if isinstance(result, str) else None

    def field_examples(self, field_name: str) -> list[Any]:
        result = self.properties.get(field_name, {}).get(_EXT_EXAMPLES, [])
        return list(result) if isinstance(result, list) else []

    def field_style(self, field_name: str) -> str | None:
        result = self.properties.get(field_name, {}).get(_EXT_STYLE)
        return result if isinstance(result, str) else None

    def validate(self, obj: dict[str, Any]) -> None:
        jsonschema.validate(instance=obj, schema=self._json_schema)

    def is_valid(self, obj: dict[str, Any]) -> bool:
        try:
            self.validate(obj)
        except jsonschema.ValidationError:
            return False
        return True

    def iter_errors(self, obj: dict[str, Any]) -> list[str]:
        validator_cls = jsonschema.validators.validator_for(self._json_schema)
        validator_cls.check_schema(self._json_schema)
        validator = validator_cls(self._json_schema)
        return [error.message for error in validator.iter_errors(obj)]
