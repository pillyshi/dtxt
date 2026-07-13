"""Global backend configuration, one slot per public function.

``configure()`` sets the default; each function's own ``backend=``
argument overrides it for that single call.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backends.base import Backend


@dataclass
class _Config:
    infer: Backend | None = None
    parse: Backend | None = None
    render: Backend | None = None


_config = _Config()


def configure(
    *,
    infer: Backend | None = None,
    parse: Backend | None = None,
    render: Backend | None = None,
) -> None:
    """Set the default backend used by ``infer_schema`` / ``parse`` / ``render``."""
    if infer is not None:
        _config.infer = infer
    if parse is not None:
        _config.parse = parse
    if render is not None:
        _config.render = render


def resolve_backend(function: str, override: Backend | None) -> Backend:
    if override is not None:
        return override
    configured: Backend | None = getattr(_config, function)
    if configured is None:
        raise RuntimeError(
            f"No backend configured for '{function}'. "
            f"Call dtxt.configure({function}=...) or pass backend=... explicitly."
        )
    return configured


def reset() -> None:
    """Clear all configured backends. Mainly useful between tests."""
    _config.infer = None
    _config.parse = None
    _config.render = None
