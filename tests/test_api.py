from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragkit.api import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "api.db"))


def test_health_ingest_query_ask(client: TestClient, corpus_dir: Path) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["documents"] == 0 and health["embedder"] is None

    r = client.post("/ingest", json={"path": str(corpus_dir), "chunk_size": 40, "overlap": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["documents"] == 6 and body["total_documents"] == 6 and body["chunks"] >= 6

    r = client.post(
        "/ingest", json={"documents": [{"id": "inline/faq", "text": "Badges cost $25 to replace."}]}
    )
    assert r.status_code == 200 and r.json()["total_documents"] == 7
    assert client.get("/health").json()["embedder"] == "hash:2048:0.5:3:0.5"

    r = client.post("/query", json={"query": "Beacon API rate limit", "mode": "hybrid", "k": 3})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits[0]["doc_id"] == "product/beacon" and hits[0]["rank"] == 1
    assert {"bm25", "dense", "rrf"} <= set(hits[0]["scores"])

    r = client.post(
        "/query", json={"query": "badge replacement cost", "mode": "bm25", "k": 2, "rerank": True}
    )
    assert r.json()["hits"][0]["doc_id"] == "inline/faq"

    r = client.post("/ask", json={"query": "how much does a replacement badge cost", "k": 3})
    assert r.status_code == 200
    answer = r.json()
    assert "$25" in answer["answer"] and answer["generator"] == "extractive"
    assert (
        answer["citations"][0]["doc_id"] == "inline/faq"
        and answer["citations"][0]["source"] == "api"
    )


def test_api_validation_and_gating(client: TestClient) -> None:
    assert client.post("/ingest", json={}).status_code == 422
    assert client.post("/ingest", json={"path": "/definitely/missing"}).status_code == 400
    assert client.post("/query", json={"query": "x", "mode": "fuzzy"}).status_code == 422
    assert client.post("/query", json={"query": ""}).status_code == 422
    r = client.post("/ask", json={"query": "x", "generator": "anthropic"})
    assert r.status_code == 503
