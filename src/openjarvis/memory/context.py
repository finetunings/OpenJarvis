"""Backward-compatibility shim — canonical location is openjarvis.tools.storage.context."""
from openjarvis.tools.storage.context import (  # noqa: F401
    ContextConfig,
    build_context_message,
    format_context,
    inject_context,
)

__all__ = [
    "ContextConfig",
    "build_context_message",
    "format_context",
    "inject_context",
]
