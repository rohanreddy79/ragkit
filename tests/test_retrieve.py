import sys

import numpy as np
import pytest

from ragkit.chunking import Chunk
from ragkit.embed import HashEmbedder
from ragkit.errors import OptionalDependencyError
from ragkit.pipeline import ingest_documents
from ragkit.retrieve import (
    CrossEncoderReranker,
    Hit,
    LexicalOverlapReranker,
    Retriever,
    mmr_order,
    reciprocal_rank_fusion,
)
from ragkit.store import Store


def test_rrf_ordering_and_weights() -> None:
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1, 4]})
    assert [i for i, _ in fused] == [1, 3, 2, 4]
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62) and scores[4] == pytest.approx(1 / 63)
    tied = reciprocal_rank_fusion({"a": [9, 4], "b": [4, 9]})
    assert [i for i, _ in tied] == [4, 9]  # equal scores: tie broken by id
    weighted = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1, 4]}, weights={"b": 2.0})
    assert [i for i, _ in weighted][:2] == [3, 1]
    assert reciprocal_rank_fusion({}) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": [1]}, k=0)


def test_mmr_prefers_diverse_results() -> None:
    def unit(v: list[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        return arr / np.linalg.norm(arr)

    q = unit([1, 0, 0])
    vectors = np.stack([unit([1, 0.2, 0]), unit([1, 0.25, 0]), unit([1, 0, 0.9])])
    assert mmr_order(q, vectors, k=3, lambda_=1.0) == [0, 1, 2]
    assert mmr_order(q, vectors, k=3, lambda_=0.5) == [0, 2, 1]
    assert mmr_order(q, vectors, k=2, lambda_=0.5) == [0, 2]
    assert mmr_order(q, vectors[:0], k=2) == []
    with pytest.raises(ValueError):
        mmr_order(q, vectors, k=2, lambda_=1.5)


def _hit(cid: int, text: str) -> Hit:
    return Hit(cid, Chunk("d", cid, text, 0, len(text)), 1.0 / cid, {"rrf": 1.0 / cid})


def test_lexical_overlap_reranker() -> None:
    hits = [
        _hit(1, "Unrelated text about parking and bicycles."),
        _hit(2, "The primary on-call engineer acknowledges a page within 15 minutes."),
        _hit(3, "On-call engineers rotate weekly."),
    ]
    query = "how quickly must the on-call engineer acknowledge a page"
    by_overlap = LexicalOverlapReranker(alpha=1.0).rerank(query, hits)
    assert [h.chunk_id for h in by_overlap] == [2, 3, 1]
    assert all("rerank" in h.scores and "rrf" in h.scores for h in by_overlap)
    unchanged = LexicalOverlapReranker(alpha=0.0).rerank(query, hits)
    assert [h.chunk_id for h in unchanged] == [1, 2, 3]
    assert LexicalOverlapReranker().rerank("", hits)[0].chunk_id == 1
    with pytest.raises(ValueError):
        LexicalOverlapReranker(alpha=2.0)


def test_cross_encoder_extra_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(OptionalDependencyError):
        CrossEncoderReranker()


@pytest.fixture
def retriever(docs) -> Retriever:
    store = Store(":memory:")
    emb = HashEmbedder()
    ingest_documents(store, docs, emb, size=40, overlap=5)
    return Retriever(store, emb, reranker=LexicalOverlapReranker())


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_mode_finds_the_right_document(retriever: Retriever, mode: str) -> None:
    cases = {
        "how much is the Beacon API rate limit per key": "product/beacon",
        "payload of the Atlas robot": "product/atlas",
        "how do I sign in to the VPN": "it/vpn",
    }
    for query, expected in cases.items():
        hits = retriever.search(query, mode=mode, k=3, rerank=False)  # type: ignore[arg-type]
        assert hits and hits[0].doc_id == expected, (mode, query, [h.doc_id for h in hits])
        assert mode in hits[0].scores or (mode == "hybrid" and "rrf" in hits[0].scores)


def test_hybrid_keeps_both_source_scores(retriever: Retriever) -> None:
    hits = retriever.search("paid time off days per year", mode="hybrid", k=2)
    assert hits[0].doc_id == "hr/pto-policy"
    assert {"bm25", "dense", "rrf", "rerank"} <= set(hits[0].scores)
    assert hits[0].scores["rerank"] == hits[0].score


def test_retriever_options(retriever: Retriever) -> None:
    assert retriever.search("anything", k=0) == []
    assert retriever.search("zzzz qqqq", mode="bm25", k=3) == []
    diverse = retriever.search("on-call page", mode="hybrid", k=3, mmr_lambda=0.5)
    assert len(diverse) == 3 and len({h.chunk_id for h in diverse}) == 3
    with pytest.raises(ValueError):
        retriever.search("x", mode="fuzzy")  # type: ignore[arg-type]
