"""ragkit: hybrid retrieval-augmented generation with a built-in evaluation loop.

The public surface is small on purpose: load documents, chunk them, index them in a
single-file SQLite store, retrieve with BM25 / dense / hybrid (RRF) search, and answer
with citations. Every piece works offline; hosted models are optional extras.
"""

from ragkit.chunking import Chunk, chunk_document, fixed_size_chunks, recursive_chunks
from ragkit.embed import Embedder, HashEmbedder, get_embedder
from ragkit.errors import (
    ConfigurationError,
    GenerationError,
    LoadError,
    OptionalDependencyError,
    RagkitError,
    StoreError,
)
from ragkit.generate import Answer, Citation, ExtractiveGenerator, Generator, get_generator
from ragkit.loaders import Document, load_file, load_path
from ragkit.pipeline import ingest_path
from ragkit.retrieve import Hit, LexicalOverlapReranker, Reranker, Retriever, reciprocal_rank_fusion
from ragkit.store import Store

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "ConfigurationError",
    "Document",
    "Embedder",
    "ExtractiveGenerator",
    "GenerationError",
    "Generator",
    "HashEmbedder",
    "Hit",
    "LexicalOverlapReranker",
    "LoadError",
    "OptionalDependencyError",
    "RagkitError",
    "Reranker",
    "Retriever",
    "Store",
    "StoreError",
    "__version__",
    "chunk_document",
    "fixed_size_chunks",
    "get_embedder",
    "get_generator",
    "ingest_path",
    "load_file",
    "load_path",
    "reciprocal_rank_fusion",
    "recursive_chunks",
]
