"""Backward-compatibility shim — canonical location is openjarvis.tools.storage.chunking."""
from openjarvis.tools.storage.chunking import (  # noqa: F401
    Chunk,
    ChunkConfig,
    chunk_text,
)

__all__ = ["Chunk", "ChunkConfig", "chunk_text"]
