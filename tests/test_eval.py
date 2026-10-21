import json
import math
from pathlib import Path

import pytest

from ragkit.errors import LoadError
from ragkit.eval import (
    DEFAULT_CONFIGS,
    EvalConfig,
    collapse_to_documents,
    default_dataset_dir,
    load_questions,
    mrr,
    ndcg_at_k,
    recall_at_k,
    run_eval,
)
from ragkit.loaders import load_path


def test_metrics_single_gold() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["c"], 1) == 0.0
    assert recall_at_k(ranked, ["c"], 3) == 1.0
    assert mrr(ranked, ["c"]) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, ["c"], 5) == pytest.approx(1 / math.log2(4))
    assert mrr(ranked, ["zzz"]) == 0.0 and ndcg_at_k(ranked, ["zzz"], 5) == 0.0


def test_metrics_multi_gold() -> None:
    ranked = ["a", "b", "c", "d"]
    gold = ["a", "c"]
    assert recall_at_k(ranked, gold, 1) == 0.5
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert mrr(ranked, gold) == 1.0
    ideal = 1.0 + 1 / math.log2(3)
    assert ndcg_at_k(ranked, gold, 5) == pytest.approx((1.0 + 1 / math.log2(4)) / ideal)
    assert ndcg_at_k(["c", "a"], gold, 5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        recall_at_k(ranked, [], 3)


def test_load_questions_validation(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "question": "x", "gold": ["d"]}) + "\n\n", encoding="utf-8"
    )
    [q] = load_questions(path)
    assert q.gold == ("d",)
    path.write_text(json.dumps({"id": "q1", "question": "x", "gold": []}) + "\n", encoding="utf-8")
    with pytest.raises(LoadError, match="gold is empty"):
        load_questions(path)
    path.write_text(json.dumps({"id": "q1"}) + "\n", encoding="utf-8")
    with pytest.raises(LoadError):
        load_questions(path)
    with pytest.raises(LoadError):
        load_questions(tmp_path / "none.jsonl")
    with pytest.raises(LoadError):
        run_eval(tmp_path)


def test_bundled_dataset_is_consistent() -> None:
    root = default_dataset_dir()
    docs = load_path(root / "docs")
    questions = load_questions(root / "questions.jsonl")
    ids = {d.id for d in docs}
    assert len(docs) >= 30 and len(questions) >= 40
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert set(q.gold) <= ids, q.id
    for d in docs:
        n = len(d.text.split())
        assert 150 <= n <= 420, (d.id, n)


def test_run_eval_on_bundled_dataset() -> None:
    report = run_eval(configs=(*DEFAULT_CONFIGS[:1], EvalConfig("hybrid+rerank", "hybrid", True)))
    assert [r.name for r in report.rows] == ["bm25", "hybrid+rerank"]
    assert report.n_documents >= 30 and report.n_chunks >= report.n_documents
    for row in report.rows:
        assert 0.0 <= row.recall_at_1 <= row.recall_at_3 <= row.recall_at_5 <= 1.0
        assert 0.0 <= row.mrr <= 1.0 and 0.0 <= row.ndcg_at_5 <= 1.0
        assert row.mean_latency_ms > 0.0
    table = report.to_markdown()
    assert table.splitlines()[0].startswith("| retrieval |") and "| bm25 |" in table


def test_collapse_to_documents() -> None:
    from ragkit.chunking import Chunk
    from ragkit.retrieve import Hit

    hits = [Hit(i, Chunk(d, 0, "t", 0, 1), 1.0) for i, d in enumerate(["a", "b", "a", "c"])]
    assert collapse_to_documents(hits) == ["a", "b", "c"]
