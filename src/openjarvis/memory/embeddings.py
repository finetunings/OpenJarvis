"""Backward-compatibility shim — canonical location is openjarvis.tools.storage.embeddings."""
from openjarvis.tools.storage.embeddings import (  # noqa: F401
    Embedder,
    SentenceTransformerEmbedder,
)

__all__ = ["Embedder", "SentenceTransformerEmbedder"]
