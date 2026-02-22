"""Backward-compatibility shim — canonical location is openjarvis.tools.storage.faiss_backend."""
from openjarvis.tools.storage.faiss_backend import FAISSMemory  # noqa: F401

__all__ = ["FAISSMemory"]
