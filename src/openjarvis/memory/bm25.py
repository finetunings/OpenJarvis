"""Backward-compatibility shim — canonical location is openjarvis.tools.storage.bm25."""
from openjarvis.tools.storage.bm25 import BM25Memory  # noqa: F401

__all__ = ["BM25Memory"]
