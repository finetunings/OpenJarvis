"""Backward-compatibility shim — canonical location is openjarvis.tools.storage.colbert_backend."""
from openjarvis.tools.storage.colbert_backend import ColBERTMemory  # noqa: F401

__all__ = ["ColBERTMemory"]
