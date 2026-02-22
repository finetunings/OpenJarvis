"""Backward-compatibility shim — canonical location is openjarvis.tools.storage.ingest."""
from openjarvis.tools.storage.ingest import (  # noqa: F401
    DocumentMeta,
    detect_file_type,
    ingest_path,
    read_document,
)

__all__ = ["DocumentMeta", "detect_file_type", "ingest_path", "read_document"]
