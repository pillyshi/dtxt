"""round-trip verification: ``extract(render(obj)) ≈ obj``.

Treated as a first-class feature: this is the main way to tell whether a
schema + backend pair actually preserves information through D2T then T2D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .d2t import StructuredEntityRenderer
from .schema import Schema
from .t2d import StructuredEntityExtractor


@dataclass
class RoundtripResult:
    original: dict[str, Any]
    rendered_text: str
    reparsed: dict[str, Any]
    mismatches: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def check_roundtrip(
    obj: dict[str, Any],
    schema: Schema,
    *,
    renderer: StructuredEntityRenderer,
    extractor: StructuredEntityExtractor,
) -> RoundtripResult:
    """Render ``obj`` to text, extract it back, and diff against ``obj``.

    Comparison is restricted to ``schema``'s declared fields, since
    ``extractor`` only ever populates those.
    """
    text = renderer.render(obj)
    reparsed = extractor.extract(text)

    mismatches: dict[str, tuple[Any, Any]] = {}
    for key in schema.properties:
        original_value = obj.get(key)
        reparsed_value = reparsed.get(key)
        if original_value != reparsed_value:
            mismatches[key] = (original_value, reparsed_value)

    return RoundtripResult(
        original=obj, rendered_text=text, reparsed=reparsed, mismatches=mismatches
    )
