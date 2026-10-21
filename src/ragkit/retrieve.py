"""Retrieval: BM25, dense, and hybrid (reciprocal rank fusion) search over a Store, with
optional MMR diversification and pluggable rerankers.

Hybrid search fuses *ranks*, not scores. BM25 scores are unbounded and corpus-dependent;
cosine similarities live in [-1, 1]. RRF needs no calibration between the two and is
robust to either list being noisy, which is why it is the default here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from ragkit.chunking import Chunk
from ragkit.embed import Embedder
from ragkit.errors import OptionalDependencyError
from ragkit.lexical import PHRASE_WEIGHT, coverage, query_weights
from ragkit.store import Store
from ragkit.text import tokenize

Mode = Literal["bm25", "dense", "hybrid"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid")
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `score` is the final ranking score; `scores` keeps every stage's
    raw number (bm25, dense, rrf, rerank) so results stay explainable."""

    chunk_id: int
    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fuse several ranked id lists: score(id) = sum_source weight / (k + rank).

    Ranks are 1-based. Ties are broken by id so the output is deterministic.
    """
    if k <= 0:
        raise ValueError("rrf k must be positive")
    fused: dict[int, float] = {}
    for source, ranking in rankings.items():
        weight = weights.get(source, 1.0) if weights else 1.0
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def mmr_order(
    query_vector: np.ndarray, vectors: np.ndarray, *, k: int, lambda_: float = 0.7
) -> list[int]:
    """Maximal Marginal Relevance: greedily pick rows that are relevant to the query but
    not redundant with rows already picked. Returns row indices in selection order."""
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    n = vectors.shape[0]
    if n == 0 or k <= 0:
        return []
    relevance = vectors @ query_vector
    pairwise = vectors @ vectors.T
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        if selected:
            redundancy = pairwise[np.ix_(remaining, selected)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining), dtype=vectors.dtype)
        objective = lambda_ * relevance[remaining] - (1.0 - lambda_) * redundancy
        best = remaining[int(np.argmax(objective))]
        selected.append(best)
        remaining.remove(best)
    return selected


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate hits for a query; returns new Hit objects with updated scores."""

    name: str

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


class LexicalOverlapReranker:
    """Offline reranker: `ragkit.lexical.coverage` of the query by each chunk (IDF-weighted
    when `idf` is given), blended with the incoming rank via `alpha` so retrieval evidence
    is not thrown away. alpha=1 ranks by coverage alone; alpha=0 is a no-op."""

    name = "lexical-overlap"

    def __init__(self, alpha: float = 0.5, *, idf: Callable[[str], float] | None = None) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.idf = idf

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        weights = query_weights(query, self.idf)
        n = len(hits)
        rescored: list[Hit] = []
        for rank, hit in enumerate(hits):
            overlap = coverage(weights, tokenize(hit.chunk.text)) / (1.0 + PHRASE_WEIGHT)
            prior = 1.0 - rank / n  # 1.0 for the top incoming hit, falling linearly
            score = self.alpha * overlap + (1.0 - self.alpha) * prior
            rescored.append(Hit(hit.chunk_id, hit.chunk, score, {**hit.scores, "rerank": score}))
        return sorted(rescored, key=lambda h: (-h.score, h.chunk_id))


class CrossEncoderReranker:
    """Semantic reranker via a sentence-transformers cross-encoder (extra: `st`)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise OptionalDependencyError(
                "st", "sentence-transformers", "CrossEncoderReranker"
            ) from exc
        self._model = CrossEncoder(model_name)
        self.name = f"cross-encoder:{model_name}"

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict([(query, hit.chunk.text) for hit in hits])
        rescored = [
            Hit(hit.chunk_id, hit.chunk, float(s), {**hit.scores, "rerank": float(s)})
            for hit, s in zip(hits, scores, strict=True)
        ]
        return sorted(rescored, key=lambda h: (-h.score, h.chunk_id))


class Retriever:
    """Runs one of the three retrieval modes over a store, then optional rerank and MMR."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.weights = dict(weights) if weights else {"bm25": 1.0, "dense": 1.0}

    def search(
        self,
        query: str,
        *,
        mode: Mode = "hybrid",
        k: int = 5,
        candidates: int | None = None,
        rerank: bool | None = None,
        mmr_lambda: float | None = None,
    ) -> list[Hit]:
        """Return the top-k hits. `candidates` is the pool size fetched from each source
        before fusion / reranking / MMR (default max(4k, 20)). `rerank=None` means "use the
        reranker if one is configured"."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if k <= 0:
            return []
        pool = candidates or max(4 * k, 20)
        bm25_hits = self.store.bm25_search(query, pool) if mode in ("bm25", "hybrid") else []
        dense_hits: list[tuple[int, float]] = []
        query_vector: np.ndarray | None = None
        if mode in ("dense", "hybrid"):
            query_vector = self.embedder.embed([query])[0]
            dense_hits = self.store.dense_search(query_vector, pool)

        scores: dict[int, dict[str, float]] = {}
        for cid, s in bm25_hits:
            scores.setdefault(cid, {})["bm25"] = s
        for cid, s in dense_hits:
            scores.setdefault(cid, {})["dense"] = s

        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                {"bm25": [c for c, _ in bm25_hits], "dense": [c for c, _ in dense_hits]},
                weights=self.weights,
                k=self.rrf_k,
            )
            ranked = [(cid, s) for cid, s in fused]
            for cid, s in ranked:
                scores[cid]["rrf"] = s
        else:
            ranked = bm25_hits if mode == "bm25" else dense_hits

        ranked = ranked[:pool]
        chunks = self.store.get_chunks([cid for cid, _ in ranked])
        hits = [Hit(cid, chunks[cid], s, scores[cid]) for cid, s in ranked if cid in chunks]

        use_reranker = self.reranker is not None and rerank is not False
        if use_reranker and self.reranker is not None:
            hits = self.reranker.rerank(query, hits)
        if mmr_lambda is not None and hits:
            if query_vector is None:
                query_vector = self.embedder.embed([query])[0]
            vectors = self.store.vectors_for([h.chunk_id for h in hits])
            hits = [hits[i] for i in mmr_order(query_vector, vectors, k=k, lambda_=mmr_lambda)]
        return hits[:k]
