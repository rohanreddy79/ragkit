"""Retrieval evaluation: recall@k, MRR, and nDCG@k over a bundled question set, run for
bm25 / dense / hybrid / hybrid+rerank so every change to the pipeline is a number.

Gold labels are document ids. A ranked list of chunks is collapsed to document order
(first occurrence wins) before scoring, which keeps labels stable when chunking changes.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ragkit.chunking import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, ChunkStrategy
from ragkit.embed import Embedder, HashEmbedder
from ragkit.errors import LoadError
from ragkit.loaders import load_path
from ragkit.pipeline import ingest_documents
from ragkit.retrieve import Hit, LexicalOverlapReranker, Mode, Retriever
from ragkit.store import Store


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    gold: tuple[str, ...]


@dataclass(frozen=True)
class EvalConfig:
    """One row of the comparison table."""

    name: str
    mode: Mode
    rerank: bool = False


DEFAULT_CONFIGS: tuple[EvalConfig, ...] = (
    EvalConfig("bm25", "bm25"),
    EvalConfig("dense", "dense"),
    EvalConfig("hybrid", "hybrid"),
    EvalConfig("hybrid+rerank", "hybrid", rerank=True),
)


@dataclass(frozen=True)
class MetricRow:
    name: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    mean_latency_ms: float


@dataclass(frozen=True)
class EvalReport:
    rows: list[MetricRow]
    n_questions: int
    n_documents: int
    n_chunks: int
    embedder: str
    chunking: str

    def to_markdown(self) -> str:
        header = "| retrieval | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | ms/query |"
        rule = "|---|---:|---:|---:|---:|---:|---:|"
        lines = [header, rule]
        for r in self.rows:
            lines.append(
                f"| {r.name} | {r.recall_at_1:.3f} | {r.recall_at_3:.3f} | {r.recall_at_5:.3f} "
                f"| {r.mrr:.3f} | {r.ndcg_at_5:.3f} | {r.mean_latency_ms:.1f} |"
            )
        return "\n".join(lines)


# -- metrics -----------------------------------------------------------------------------


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold ids present in the top k."""
    if not gold:
        raise ValueError("gold must not be empty")
    gold_set = set(gold)
    return len(gold_set & set(ranked[:k])) / len(gold_set)


def mrr(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal rank of the first gold id, or 0 if none is ranked."""
    gold_set = set(gold)
    for rank, item in enumerate(ranked, start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k: DCG with 1/log2(rank+1) gains, normalised by the ideal DCG."""
    if not gold:
        raise ValueError("gold must not be empty")
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in gold_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def collapse_to_documents(hits: Sequence[Hit]) -> list[str]:
    """Document ids in order of first appearance in the chunk ranking."""
    return list(dict.fromkeys(hit.doc_id for hit in hits))


# -- dataset -----------------------------------------------------------------------------


def default_dataset_dir() -> Path:
    """The bundled fictional-company wiki shipped inside the package."""
    return Path(str(resources.files("ragkit") / "data" / "eval"))


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Read a JSONL file of {"id", "question", "gold": [doc_id, ...]} records."""
    p = Path(path)
    if not p.is_file():
        raise LoadError(f"questions file not found: {p}")
    questions: list[EvalQuestion] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            gold = tuple(str(g) for g in record["gold"])
            if not gold:
                raise ValueError("gold is empty")
            questions.append(EvalQuestion(str(record["id"]), str(record["question"]), gold))
        except (KeyError, ValueError, TypeError) as exc:
            raise LoadError(f"{p}:{line_no}: bad question record: {exc}") from exc
    if not questions:
        raise LoadError(f"no questions in {p}")
    return questions


# -- runner ------------------------------------------------------------------------------


def run_eval(
    dataset_dir: str | Path | None = None,
    *,
    embedder: Embedder | None = None,
    configs: Sequence[EvalConfig] = DEFAULT_CONFIGS,
    strategy: ChunkStrategy = "recursive",
    size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
    k: int = 5,
    rerank_alpha: float = 0.5,
) -> EvalReport:
    """Index the dataset's docs into an in-memory store and score every config."""
    root = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    docs = load_path(root / "docs")
    questions = load_questions(root / "questions.jsonl")
    known = {d.id for d in docs}
    for q in questions:
        missing = [g for g in q.gold if g not in known]
        if missing:
            raise LoadError(f"question {q.id} references unknown document(s) {missing}")

    embedder = embedder or HashEmbedder()
    with Store(":memory:") as store:
        result = ingest_documents(
            store, docs, embedder, strategy=strategy, size=size, overlap=overlap
        )
        reranker = LexicalOverlapReranker(alpha=rerank_alpha, idf=store.idf)
        retriever = Retriever(store, embedder, reranker=reranker)
        rows = [_score_config(retriever, questions, cfg, k) for cfg in configs]
        return EvalReport(
            rows=rows,
            n_questions=len(questions),
            n_documents=result.documents,
            n_chunks=result.chunks,
            embedder=embedder.name,
            chunking=f"{strategy}, size={size}, overlap={overlap}",
        )


def _score_config(
    retriever: Retriever, questions: Sequence[EvalQuestion], cfg: EvalConfig, k: int
) -> MetricRow:
    r1 = r3 = r5 = rr = nd = 0.0
    elapsed = 0.0
    for q in questions:
        started = time.perf_counter()
        hits = retriever.search(q.question, mode=cfg.mode, k=k, rerank=cfg.rerank)
        elapsed += time.perf_counter() - started
        ranked = collapse_to_documents(hits)
        r1 += recall_at_k(ranked, q.gold, 1)
        r3 += recall_at_k(ranked, q.gold, 3)
        r5 += recall_at_k(ranked, q.gold, 5)
        rr += mrr(ranked, q.gold)
        nd += ndcg_at_k(ranked, q.gold, 5)
    n = len(questions)
    return MetricRow(cfg.name, r1 / n, r3 / n, r5 / n, rr / n, nd / n, 1000.0 * elapsed / n)
