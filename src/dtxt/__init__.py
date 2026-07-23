"""dtxt: schema-centric bidirectional conversion between text and structured data."""

from . import backends
from .d2t import StructuredEntityRenderer
from .infer import InferError, SchemaInferer
from .roundtrip import RoundtripResult, check_roundtrip
from .schema import Schema
from .t2d import ParseError, StructuredEntityExtractor

__all__ = [
    "InferError",
    "ParseError",
    "RoundtripResult",
    "Schema",
    "SchemaInferer",
    "StructuredEntityExtractor",
    "StructuredEntityRenderer",
    "backends",
    "check_roundtrip",
]
