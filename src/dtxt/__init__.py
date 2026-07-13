"""dtxt: schema-centric bidirectional conversion between text and structured data."""

from . import backends
from ._config import configure
from .d2t import render
from .infer import InferError, infer_schema
from .roundtrip import RoundtripResult, check_roundtrip
from .schema import Schema
from .t2d import ParseError, parse, parse_many

__all__ = [
    "InferError",
    "ParseError",
    "RoundtripResult",
    "Schema",
    "backends",
    "check_roundtrip",
    "configure",
    "infer_schema",
    "parse",
    "parse_many",
    "render",
]
