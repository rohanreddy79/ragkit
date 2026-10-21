"""FastAPI service: POST /ingest, POST /query, POST /ask, GET /health.

The app owns one Store and one Retriever for its lifetime; the store serialises access
with a lock so the sync endpoints are safe under FastAPI's thread pool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragkit import __version__
from ragkit.chunking import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, ChunkStrategy
from ragkit.embed import get_embedder
from ragkit.errors import ConfigurationError, OptionalDependencyError, RagkitError
from ragkit.generate import get_generator
from ragkit.loaders import Document
from ragkit.pipeline import ingest_documents, ingest_path
from ragkit.retrieve import LexicalOverlapReranker, Mode, Retriever
from ragkit.store import Store


class DocumentIn(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = "api"
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    path: str | None = Field(default=None, description="Server-side file or directory to ingest.")
    documents: list[DocumentIn] = Field(default_factory=list, description="Inline documents.")
    chunker: ChunkStrategy = "recursive"
    chunk_size: int = Field(default=DEFAULT_CHUNK_WORDS, ge=1)
    overlap: int = Field(default=DEFAULT_OVERLAP_WORDS, ge=0)


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    seconds: float
    total_documents: int
    total_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Mode = "hybrid"
    k: int = Field(default=5, ge=1, le=100)
    rerank: bool = False
    mmr: float | None = Field(default=None, ge=0.0, le=1.0)


class HitOut(BaseModel):
    rank: int
    chunk_id: int
    doc_id: str
    chunk_index: int
    start: int
    end: int
    score: float
    scores: dict[str, float]
    text: str


class QueryResponse(BaseModel):
    query: str
    mode: Mode
    hits: list[HitOut]


class AskRequest(QueryRequest):
    generator: Literal["extractive", "anthropic"] = "extractive"


class CitationOut(BaseModel):
    n: int
    chunk_id: int
    doc_id: str
    source: str
    chunk_index: int
    start: int
    end: int


class AskResponse(BaseModel):
    question: str
    answer: str
    generator: str
    citations: list[CitationOut]
    hits: list[HitOut]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    db: str
    embedder: str | None
    documents: int
    chunks: int


def create_app(db: str | Path = "idx.db", *, embedder_spec: str = "hash") -> FastAPI:
    """Build the API around one store. The embedder must match what the index was built with."""
    store = Store(db)
    spec = store.embedder_name or embedder_spec
    embedder = get_embedder(spec)
    retriever = Retriever(store, embedder)
    reranker = LexicalOverlapReranker(idf=store.idf)
    app = FastAPI(
        title="ragkit", version=__version__, description="Hybrid RAG over a SQLite index."
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            db=store.path,
            embedder=store.embedder_name,
            documents=store.count_documents(),
            chunks=store.count_chunks(),
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        if not req.path and not req.documents:
            raise HTTPException(status_code=422, detail="provide 'path' or 'documents'")
        docs = 0
        chunks = 0
        seconds = 0.0
        try:
            if req.path:
                r = ingest_path(
                    store,
                    req.path,
                    embedder,
                    strategy=req.chunker,
                    size=req.chunk_size,
                    overlap=req.overlap,
                )
                docs, chunks, seconds = r.documents, r.chunks, r.seconds
            if req.documents:
                inline = [Document(d.id, d.text, d.source, d.metadata) for d in req.documents]
                r = ingest_documents(
                    store,
                    inline,
                    embedder,
                    strategy=req.chunker,
                    size=req.chunk_size,
                    overlap=req.overlap,
                )
                docs, chunks, seconds = docs + r.documents, chunks + r.chunks, seconds + r.seconds
        except RagkitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IngestResponse(
            documents=docs,
            chunks=chunks,
            seconds=seconds,
            total_documents=store.count_documents(),
            total_chunks=store.count_chunks(),
        )

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        hits = _search(req)
        return QueryResponse(query=req.query, mode=req.mode, hits=hits)

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        try:
            generator = get_generator(req.generator, idf=store.idf)
        except (OptionalDependencyError, ConfigurationError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raw_hits = retriever.search(
            req.query, mode=req.mode, k=req.k, rerank=req.rerank, mmr_lambda=req.mmr
        )
        try:
            answer = generator.generate(req.query, raw_hits)
        except RagkitError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        sources = store.sources(h.doc_id for h in raw_hits)
        return AskResponse(
            question=req.query,
            answer=answer.text,
            generator=answer.generator,
            citations=[
                CitationOut(
                    n=c.n,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=sources.get(c.doc_id, ""),
                    chunk_index=c.chunk_index,
                    start=c.start,
                    end=c.end,
                )
                for c in answer.citations
            ],
            hits=_to_hits(raw_hits),
        )

    def _search(req: QueryRequest) -> list[HitOut]:
        retriever.reranker = reranker if req.rerank else None
        try:
            hits = retriever.search(
                req.query, mode=req.mode, k=req.k, rerank=req.rerank, mmr_lambda=req.mmr
            )
        except RagkitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_hits(hits)

    app.state.store = store
    return app


def _to_hits(hits: list) -> list[HitOut]:
    return [
        HitOut(
            rank=i,
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            chunk_index=h.chunk.index,
            start=h.chunk.start,
            end=h.chunk.end,
            score=h.score,
            scores=h.scores,
            text=h.chunk.text,
        )
        for i, h in enumerate(hits, start=1)
    ]
