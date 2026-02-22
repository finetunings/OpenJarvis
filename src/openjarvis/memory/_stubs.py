"""Backward-compatibility shim — canonical location is openjarvis.tools.storage._stubs."""
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult  # noqa: F401

__all__ = ["MemoryBackend", "RetrievalResult"]
