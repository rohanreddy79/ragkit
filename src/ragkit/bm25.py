"""Okapi BM25 from scratch: scoring functions plus an in-memory inverted index.

The SQLite store reuses `idf()` and `term_score()` so on-disk and in-memory search agree
to the last digit. The in-memory index is what the unit tests check against a hand
computation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def idf(n_docs: int, df: int) -> float:
    """Lucene-style smoothed IDF; never negative, even for terms in most documents."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def term_score(tf: int, doc_len: int, avgdl: float, idf_value: float, k1: float, b: float) -> float:
    """Contribution of one term to a document's BM25 score."""
    norm = k1 * (1.0 - b + b * doc_len / avgdl) if avgdl > 0 else k1
    return idf_value * tf * (k1 + 1.0) / (tf + norm)


class BM25Index:
    """Inverted index keyed by integer document ids, serialisable with to_dict/from_dict."""

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self.postings: dict[str, dict[int, int]] = {}
        self.lengths: dict[int, int] = {}

    @property
    def n_docs(self) -> int:
        return len(self.lengths)

    @property
    def avgdl(self) -> float:
        return sum(self.lengths.values()) / self.n_docs if self.lengths else 0.0

    def add(self, doc_key: int, tokens: Iterable[str]) -> None:
        """Index an already-tokenised document. Re-adding a key replaces it."""
        if doc_key in self.lengths:
            self.remove(doc_key)
        tf: dict[str, int] = {}
        length = 0
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
            length += 1
        for term, count in tf.items():
            self.postings.setdefault(term, {})[doc_key] = count
        self.lengths[doc_key] = length

    def remove(self, doc_key: int) -> None:
        if doc_key not in self.lengths:
            return
        del self.lengths[doc_key]
        for term in [t for t, docs in self.postings.items() if doc_key in docs]:
            del self.postings[term][doc_key]
            if not self.postings[term]:
                del self.postings[term]

    def idf(self, term: str) -> float:
        return idf(self.n_docs, len(self.postings.get(term, {})))

    def score(self, query_tokens: Iterable[str], doc_key: int) -> float:
        """BM25 score of one document for a query; each distinct query term counts once."""
        avgdl = self.avgdl
        total = 0.0
        for term in dict.fromkeys(query_tokens):
            tf = self.postings.get(term, {}).get(doc_key, 0)
            if tf:
                total += term_score(
                    tf, self.lengths[doc_key], avgdl, self.idf(term), self.k1, self.b
                )
        return total

    def search(self, query_tokens: Iterable[str], k: int = 10) -> list[tuple[int, float]]:
        """Top-k (doc_key, score) pairs, best first; ties broken by doc_key for determinism."""
        avgdl = self.avgdl
        scores: dict[int, float] = {}
        for term in dict.fromkeys(query_tokens):
            docs = self.postings.get(term)
            if not docs:
                continue
            idf_value = idf(self.n_docs, len(docs))
            for doc_key, tf in docs.items():
                scores[doc_key] = scores.get(doc_key, 0.0) + term_score(
                    tf, self.lengths[doc_key], avgdl, idf_value, self.k1, self.b
                )
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "lengths": {str(k): v for k, v in self.lengths.items()},
            "postings": {t: {str(k): v for k, v in d.items()} for t, d in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        index = cls(k1=float(data["k1"]), b=float(data["b"]))
        index.lengths = {int(k): int(v) for k, v in data["lengths"].items()}
        index.postings = {
            t: {int(k): int(v) for k, v in d.items()} for t, d in data["postings"].items()
        }
        return index
