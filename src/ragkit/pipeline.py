"""Ingestion glue: load -> chunk -> embed -> store, with the settings recorded in the store
so later queries and re-ingests use the same configuration."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.chunking import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    ChunkStrategy,
    chunk_document,
)
from ragkit.embed import Embedder
from ragkit.loaders import Document, load_path
from ragkit.store import Store


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int
    seconds: float


def ingest_documents(
    store: Store,
    docs: Sequence[Document],
    embedder: Embedder,
    *,
    strategy: ChunkStrategy = "recursive",
    size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> IngestResult:
    """Chunk, embed, and store every document. Existing documents with the same id are replaced."""
    started = time.perf_counter()
    total_chunks = 0
    for doc in docs:
        chunks = chunk_document(doc, strategy=strategy, size=size, overlap=overlap)
        vectors = embedder.embed([c.text for c in chunks])
        store.add_document(doc, chunks, vectors, embedder_name=embedder.name)
        total_chunks += len(chunks)
    store.set_meta("chunk_strategy", strategy)
    store.set_meta("chunk_size", str(size))
    store.set_meta("chunk_overlap", str(overlap))
    return IngestResult(len(docs), total_chunks, time.perf_counter() - started)


def ingest_path(
    store: Store,
    path: str | Path,
    embedder: Embedder,
    *,
    strategy: ChunkStrategy = "recursive",
    size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> IngestResult:
    """Load a file or directory and ingest it."""
    docs = load_path(path)
    return ingest_documents(store, docs, embedder, strategy=strategy, size=size, overlap=overlap)
