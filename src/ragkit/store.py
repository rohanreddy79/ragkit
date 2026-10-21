"""Single-file SQLite store holding documents, chunks, embeddings, and BM25 postings.

Design: SQLite is the only persistence layer. Dense search loads the embedding matrix into
memory once (lazily) and uses a numpy dot product; BM25 search reads only the postings of
the query's terms. Both caches are invalidated on write, so a store can be read and
extended through the same object.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import TracebackType

import numpy as np

from ragkit.bm25 import DEFAULT_B, DEFAULT_K1, idf, term_score
from ragkit.chunking import Chunk
from ragkit.errors import StoreError
from ragkit.loaders import Document
from ragkit.text import normalize, tokenize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS postings (
    term TEXT NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tf INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS postings_by_chunk ON postings(chunk_id);
CREATE TABLE IF NOT EXISTS chunk_stats (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    length INTEGER NOT NULL
);
"""


class Store:
    """Documents + chunks + embeddings + BM25 postings in one SQLite file (or `:memory:`)."""

    def __init__(
        self, path: str | Path = ":memory:", *, k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> None:
        self.path = str(path)
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open store at {self.path}: {exc}") from exc
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._matrix: np.ndarray | None = None
        self._matrix_ids: np.ndarray | None = None
        self._lengths: dict[int, int] | None = None

    # -- lifecycle -----------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- metadata ------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
            )

    @property
    def embedder_name(self) -> str | None:
        return self.get_meta("embedder")

    @property
    def dim(self) -> int | None:
        value = self.get_meta("dim")
        return int(value) if value else None

    # -- documents and chunks ------------------------------------------------------------

    def add_document(
        self, doc: Document, chunks: Sequence[Chunk], vectors: np.ndarray, *, embedder_name: str
    ) -> list[int]:
        """Insert a document with its chunks, embeddings, and postings in one transaction.

        A document with the same id is replaced. The embedder name and dimension are
        pinned on first write; later writes must match so one index never mixes spaces.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise StoreError(f"expected {len(chunks)} vectors, got array of shape {vectors.shape}")
        with self._lock, self._conn:
            self._check_space(embedder_name, int(vectors.shape[1]) if len(chunks) else None)
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc.id,))
            self._conn.execute(
                "INSERT INTO documents (id, text, source, metadata) VALUES (?, ?, ?, ?)",
                (doc.id, doc.text, doc.source, json.dumps(doc.metadata, sort_keys=True)),
            )
            ids: list[int] = []
            for chunk, vec in zip(chunks, vectors, strict=True):
                cur = self._conn.execute(
                    "INSERT INTO chunks (doc_id, idx, text, start, end) VALUES (?, ?, ?, ?, ?)",
                    (doc.id, chunk.index, chunk.text, chunk.start, chunk.end),
                )
                chunk_id = int(cur.lastrowid or 0)
                ids.append(chunk_id)
                self._conn.execute(
                    "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                    (chunk_id, vec.tobytes()),
                )
                tokens = tokenize(chunk.text)
                tf: dict[str, int] = {}
                for tok in tokens:
                    tf[tok] = tf.get(tok, 0) + 1
                self._conn.executemany(
                    "INSERT INTO postings (term, chunk_id, tf) VALUES (?, ?, ?)",
                    [(term, chunk_id, count) for term, count in tf.items()],
                )
                self._conn.execute(
                    "INSERT INTO chunk_stats (chunk_id, length) VALUES (?, ?)",
                    (chunk_id, len(tokens)),
                )
            self._invalidate()
        return ids

    def delete_document(self, doc_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._invalidate()
        return cur.rowcount > 0

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, source, metadata FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def sources(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """Map document ids to their source paths (missing ids are skipped)."""
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, source FROM documents WHERE id IN ({placeholders})", ids
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def document_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM documents ORDER BY id").fetchall()
        return [r[0] for r in rows]

    def count_documents(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def count_chunks(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get_chunk(self, chunk_id: int) -> Chunk:
        chunks = self.get_chunks([chunk_id])
        if chunk_id not in chunks:
            raise StoreError(f"no chunk with id {chunk_id}")
        return chunks[chunk_id]

    def get_chunks(self, chunk_ids: Iterable[int]) -> dict[int, Chunk]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, doc_id, idx, text, start, end FROM chunks "
                f"WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return {int(r[0]): Chunk(r[1], int(r[2]), r[3], int(r[4]), int(r[5])) for r in rows}

    def chunk_ids_for_document(self, doc_id: str) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)
            ).fetchall()
        return [int(r[0]) for r in rows]

    # -- search --------------------------------------------------------------------------

    def bm25_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (chunk_id, score) by Okapi BM25, reading only the query terms' postings."""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        with self._lock:
            lengths = self._load_lengths()
            n_docs = len(lengths)
            if n_docs == 0:
                return []
            avgdl = sum(lengths.values()) / n_docs
            scores: dict[int, float] = {}
            for term in terms:
                rows = self._conn.execute(
                    "SELECT chunk_id, tf FROM postings WHERE term = ?", (term,)
                ).fetchall()
                if not rows:
                    continue
                idf_value = idf(n_docs, len(rows))
                for chunk_id, tf in rows:
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score(
                        int(tf), lengths[int(chunk_id)], avgdl, idf_value, self.k1, self.b
                    )
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(int(cid), float(s)) for cid, s in ranked[:k]]

    def idf(self, term: str) -> float:
        """Inverse document frequency of a term over the indexed chunks. The term is
        normalised the same way the index was built, so raw words work too."""
        term = normalize(term.lower())
        with self._lock:
            n_docs = len(self._load_lengths())
            df = self._conn.execute(
                "SELECT COUNT(*) FROM postings WHERE term = ?", (term,)
            ).fetchone()
        return idf(n_docs, int(df[0])) if n_docs else 0.0

    def dense_search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (chunk_id, cosine) over the lazily loaded, unit-norm embedding matrix."""
        with self._lock:
            matrix, ids = self._load_matrix()
        if matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != matrix.shape[1]:
            raise StoreError(f"query vector has dim {q.shape[0]}, store has dim {matrix.shape[1]}")
        sims = matrix @ q
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        order = sorted(top, key=lambda i: (-float(sims[i]), int(ids[i])))
        return [(int(ids[i]), float(sims[i])) for i in order]

    def vectors_for(self, chunk_ids: Sequence[int]) -> np.ndarray:
        """Embedding rows for the given chunk ids, in the given order."""
        with self._lock:
            matrix, ids = self._load_matrix()
        position = {int(cid): i for i, cid in enumerate(ids)}
        try:
            rows = [position[int(c)] for c in chunk_ids]
        except KeyError as exc:
            raise StoreError(f"no embedding for chunk {exc.args[0]}") from exc
        return matrix[rows] if rows else np.zeros((0, matrix.shape[1]), dtype=np.float32)

    # -- internals -----------------------------------------------------------------------

    def _check_space(self, embedder_name: str, dim: int | None) -> None:
        existing = self.get_meta("embedder")
        if existing is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedder', ?)", (embedder_name,)
            )
            if dim is not None:
                self._conn.execute("INSERT INTO meta (key, value) VALUES ('dim', ?)", (str(dim),))
            return
        if existing != embedder_name:
            raise StoreError(
                f"store was built with embedder {existing!r}; "
                f"refusing to add vectors from {embedder_name!r}"
            )
        stored_dim = self.get_meta("dim")
        if stored_dim is None and dim is not None:
            self._conn.execute("INSERT INTO meta (key, value) VALUES ('dim', ?)", (str(dim),))
        elif dim is not None and int(stored_dim or 0) != dim:
            raise StoreError(f"store has dim {stored_dim}, got vectors of dim {dim}")

    def _invalidate(self) -> None:
        self._matrix = None
        self._matrix_ids = None
        self._lengths = None

    def _load_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None or self._matrix_ids is None:
            rows = self._conn.execute(
                "SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id"
            ).fetchall()
            dim = self.dim or 0
            if rows:
                self._matrix = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
            else:
                self._matrix = np.zeros((0, dim), dtype=np.float32)
            self._matrix_ids = np.array([int(cid) for cid, _ in rows], dtype=np.int64)
        return self._matrix, self._matrix_ids

    def _load_lengths(self) -> dict[int, int]:
        if self._lengths is None:
            rows = self._conn.execute("SELECT chunk_id, length FROM chunk_stats").fetchall()
            self._lengths = {int(cid): int(length) for cid, length in rows}
        return self._lengths


def _row_to_document(row: tuple[str, str, str, str]) -> Document:
    return Document(id=row[0], text=row[1], source=row[2], metadata=json.loads(row[3]))
