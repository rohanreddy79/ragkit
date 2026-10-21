from pathlib import Path

import numpy as np
import pytest

from ragkit.chunking import recursive_chunks
from ragkit.embed import HashEmbedder
from ragkit.errors import StoreError
from ragkit.pipeline import ingest_documents
from ragkit.store import Store


def test_roundtrip_close_reopen_same_results(tmp_path: Path, docs) -> None:
    db = tmp_path / "idx.db"
    emb = HashEmbedder()
    with Store(db) as store:
        result = ingest_documents(store, docs, emb, size=40, overlap=5)
        assert result.documents == 6 and result.chunks == store.count_chunks()
        q = emb.embed(["payload of the atlas robot"])[0]
        bm25_before = store.bm25_search("payload of the atlas robot", k=5)
        dense_before = store.dense_search(q, k=5)
        assert store.get_meta("chunk_strategy") == "recursive"
    assert db.exists() and not db.with_suffix(".db-journal").exists()

    with Store(db) as reopened:
        assert reopened.count_documents() == 6
        assert reopened.embedder_name == "hash:2048:0.5:3:0.5" and reopened.dim == 2048
        assert reopened.bm25_search("payload of the atlas robot", k=5) == bm25_before
        assert reopened.dense_search(q, k=5) == dense_before
        assert reopened.get_chunk(bm25_before[0][0]).doc_id == "product/atlas"
        doc = reopened.get_document("product/atlas")
        assert (
            doc is not None and doc.metadata == {"type": "md"} and doc.source == "product/atlas.md"
        )
        assert reopened.sources(["product/atlas", "nope"]) == {"product/atlas": "product/atlas.md"}
        assert reopened.get_document("nope") is None
        assert reopened.idf("atlas") > reopened.idf("day") > 0.0
        assert reopened.idf("zzz") > reopened.idf("atlas")


def test_replace_and_delete_cascade(docs) -> None:
    emb = HashEmbedder()
    with Store() as store:
        ingest_documents(store, docs, emb, size=40, overlap=5)
        before_chunks = store.count_chunks()
        ingest_documents(store, docs[:1], emb, size=40, overlap=5)
        assert store.count_documents() == 6 and store.count_chunks() == before_chunks
        assert store.delete_document("product/beacon") is True
        assert store.delete_document("product/beacon") is False
        assert store.bm25_search("webhook hmac", k=5) == []
        assert all(cid for cid, _ in store.dense_search(emb.embed(["webhook hmac"])[0], k=50))
        assert "product/beacon" not in store.document_ids()
        assert store.chunk_ids_for_document("product/beacon") == []


def test_store_rejects_mixed_embedding_spaces(docs) -> None:
    doc = docs[0]
    chunks = recursive_chunks(doc, max_words=40)
    with Store() as store:
        store.add_document(
            doc, chunks, HashEmbedder(64).embed([c.text for c in chunks]), embedder_name="hash:64"
        )
        with pytest.raises(StoreError, match="refusing"):
            store.add_document(
                doc, chunks, HashEmbedder(64).embed([c.text for c in chunks]), embedder_name="st:x"
            )
        with pytest.raises(StoreError, match="dim"):
            store.add_document(
                doc, chunks, np.zeros((len(chunks), 32), dtype=np.float32), embedder_name="hash:64"
            )
        with pytest.raises(StoreError, match="expected"):
            store.add_document(
                doc,
                chunks,
                np.zeros((len(chunks) + 1, 64), dtype=np.float32),
                embedder_name="hash:64",
            )
        with pytest.raises(StoreError):
            store.dense_search(np.zeros(8, dtype=np.float32), k=3)
        with pytest.raises(StoreError):
            store.vectors_for([9999])
        with pytest.raises(StoreError):
            store.get_chunk(9999)


def test_empty_store_searches_cleanly() -> None:
    with Store() as store:
        assert store.bm25_search("anything", k=3) == []
        assert store.dense_search(np.zeros(4, dtype=np.float32), k=3) == []
        assert store.vectors_for([]).shape == (0, 0)
        assert store.idf("anything") == 0.0
