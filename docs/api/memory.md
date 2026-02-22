# Memory Module

The memory module implements persistent searchable storage for document
retrieval. All backends implement the `MemoryBackend` ABC with `store()`,
`retrieve()`, `delete()`, and `clear()` methods. The module also includes
a document ingestion pipeline (chunking, file reading) and context injection
for augmenting prompts with retrieved knowledge.

!!! note "Canonical import location"
    Memory backends have moved to `openjarvis.tools.storage.*`. The old
    `openjarvis.memory.*` imports still work via backward-compatibility shims.

## Abstract Base Class

### MemoryBackend

::: openjarvis.tools.storage._stubs.MemoryBackend
    options:
      show_source: true
      members_order: source

### RetrievalResult

::: openjarvis.tools.storage._stubs.RetrievalResult
    options:
      show_source: true
      members_order: source

---

## Backend Implementations

### SQLiteMemory

::: openjarvis.tools.storage.sqlite.SQLiteMemory
    options:
      show_source: true
      members_order: source

### FAISSMemory

::: openjarvis.tools.storage.faiss_backend.FAISSMemory
    options:
      show_source: true
      members_order: source

### ColBERTMemory

::: openjarvis.tools.storage.colbert_backend.ColBERTMemory
    options:
      show_source: true
      members_order: source

### BM25Memory

::: openjarvis.tools.storage.bm25.BM25Memory
    options:
      show_source: true
      members_order: source

### HybridMemory

::: openjarvis.tools.storage.hybrid.HybridMemory
    options:
      show_source: true
      members_order: source

### reciprocal_rank_fusion

::: openjarvis.tools.storage.hybrid.reciprocal_rank_fusion
    options:
      show_source: true

---

## Document Chunking

Splits text into fixed-size chunks with configurable overlap, respecting
paragraph boundaries.

### ChunkConfig

::: openjarvis.tools.storage.chunking.ChunkConfig
    options:
      show_source: true
      members_order: source

### Chunk

::: openjarvis.tools.storage.chunking.Chunk
    options:
      show_source: true
      members_order: source

### chunk_text

::: openjarvis.tools.storage.chunking.chunk_text
    options:
      show_source: true

---

## Document Ingestion

File reading, type detection, and directory walking for the ingestion
pipeline.

### DocumentMeta

::: openjarvis.tools.storage.ingest.DocumentMeta
    options:
      show_source: true
      members_order: source

### detect_file_type

::: openjarvis.tools.storage.ingest.detect_file_type
    options:
      show_source: true

### read_document

::: openjarvis.tools.storage.ingest.read_document
    options:
      show_source: true

### ingest_path

::: openjarvis.tools.storage.ingest.ingest_path
    options:
      show_source: true

---

## Context Injection

Retrieves relevant memory and injects it into prompts as system messages
with source attribution.

### ContextConfig

::: openjarvis.tools.storage.context.ContextConfig
    options:
      show_source: true
      members_order: source

### inject_context

::: openjarvis.tools.storage.context.inject_context
    options:
      show_source: true

### format_context

::: openjarvis.tools.storage.context.format_context
    options:
      show_source: true

### build_context_message

::: openjarvis.tools.storage.context.build_context_message
    options:
      show_source: true

---

## Embeddings

Abstraction layer for text embedding models used by dense retrieval backends.

### Embedder

::: openjarvis.tools.storage.embeddings.Embedder
    options:
      show_source: true
      members_order: source

### SentenceTransformerEmbedder

::: openjarvis.tools.storage.embeddings.SentenceTransformerEmbedder
    options:
      show_source: true
      members_order: source
