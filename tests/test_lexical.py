from ragkit.lexical import coverage, query_weights, term_covers


def test_term_covers_exact_and_prefix() -> None:
    assert term_covers("limit", ["key", "limited"])
    assert term_covers("reimbursement", ["reimburse"])
    assert not term_covers("rate", ["rated"])  # too short for prefix matching
    assert not term_covers("page", ["pager"])


def test_query_weights_and_coverage() -> None:
    weights = query_weights(
        "Beacon API rate limit", idf=lambda t: 2.0 if t in ("rate", "limit") else 1.0
    )
    assert weights == {"beacon": 1.0, "api": 1.0, "rate": 2.0, "limit": 2.0}
    sentence = ["each", "key", "limited", "600", "request", "per", "minute"]
    assert coverage(weights, sentence) == 2.0 / 6.0
    with_heading = coverage(weights, sentence, heading_tokens=["beacon", "api", "rate", "limit"])
    assert with_heading == (2.0 + 0.5 * 4.0) / 6.0
    full = coverage(weights, ["beacon", "api", "rate", "limit"])
    assert full == 1.0 + 0.5  # every term and every bigram
    assert coverage({}, sentence) == 0.0
