import math

import pytest

from ragkit.bm25 import BM25Index, idf, term_score


def test_scores_match_hand_computation() -> None:
    index = BM25Index(k1=1.5, b=0.75)
    index.add(0, ["cat", "sat", "mat"])
    index.add(1, ["cat", "dog"])
    index.add(2, ["dog", "dog", "bird", "tree", "sky"])
    assert index.n_docs == 3
    assert index.avgdl == pytest.approx(10 / 3)

    # idf for a term in 2 of 3 docs: ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    assert index.idf("cat") == pytest.approx(math.log(1.6))
    assert index.idf("unseen") == pytest.approx(math.log(1 + 3.5 / 0.5))

    # doc 1: tf=1 for both terms, dl=2 -> norm = 1.5 * (0.25 + 0.75 * 2 / (10/3)) = 1.05
    expected_doc1 = 2 * math.log(1.6) * 1 * 2.5 / (1 + 1.05)
    assert index.score(["cat", "dog"], 1) == pytest.approx(expected_doc1)
    assert index.score(["cat", "dog"], 1) == pytest.approx(1.1464, abs=1e-4)
    # doc 2: dog tf=2, dl=5 -> norm = 1.5 * (0.25 + 0.75 * 1.5) = 2.0625
    assert index.score(["cat", "dog"], 2) == pytest.approx(math.log(1.6) * 2 * 2.5 / (2 + 2.0625))
    assert index.score(["cat", "dog"], 2) == pytest.approx(0.5785, abs=1e-4)
    # doc 0: cat tf=1, dl=3 -> norm = 1.5 * (0.25 + 0.75 * 0.9) = 1.3875
    assert index.score(["cat", "dog"], 0) == pytest.approx(0.4922, abs=1e-4)

    ranked = index.search(["cat", "dog", "cat"], k=10)
    assert [d for d, _ in ranked] == [1, 2, 0]
    assert index.search(["nothing"], k=5) == []
    assert len(index.search(["dog"], k=1)) == 1


def test_term_score_and_idf_edge_cases() -> None:
    assert idf(1, 1) == pytest.approx(math.log(1 + 0.5 / 1.5))
    assert idf(10, 10) > 0.0
    assert term_score(1, 5, 0.0, 1.0, 1.5, 0.75) == pytest.approx(2.5 / 2.5)


def test_remove_and_roundtrip() -> None:
    index = BM25Index()
    index.add(7, ["alpha", "beta"])
    index.add(8, ["beta", "gamma"])
    index.add(7, ["alpha"])  # re-adding replaces
    assert index.lengths[7] == 1 and 7 not in index.postings["beta"]
    clone = BM25Index.from_dict(index.to_dict())
    assert clone.search(["alpha", "beta"], k=5) == index.search(["alpha", "beta"], k=5)
    index.remove(8)
    index.remove(999)
    assert index.n_docs == 1 and "gamma" not in index.postings
