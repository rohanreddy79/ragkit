"""Lexical coverage scoring shared by the extractive generator and the lexical reranker.

Coverage answers "what fraction of the query's (IDF-weighted) terms does this text contain?"
A term also counts as covered through a shared prefix of five or more characters (cheap
stemming: "limit" ~ "limited", "reimburse" ~ "reimbursement"), and at half weight when it
appears only in the heading above the text, so section titles carry their meaning down.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from ragkit.text import bigrams, tokenize

MIN_PREFIX = 5
HEADING_WEIGHT = 0.5
PHRASE_WEIGHT = 0.5


def query_weights(query: str, idf: Callable[[str], float] | None = None) -> dict[str, float]:
    """Distinct query terms mapped to their weight (IDF if given, else 1.0)."""
    return {t: (idf(t) if idf else 1.0) for t in dict.fromkeys(tokenize(query))}


def term_covers(term: str, tokens: Iterable[str]) -> bool:
    """Exact match, or a shared prefix of at least MIN_PREFIX characters."""
    for tok in tokens:
        if tok == term:
            return True
        if (
            len(term) >= MIN_PREFIX
            and len(tok) >= MIN_PREFIX
            and (tok.startswith(term) or term.startswith(tok))
        ):
            return True
    return False


def coverage(
    weights: Mapping[str, float],
    tokens: Sequence[str],
    *,
    heading_tokens: Sequence[str] = (),
) -> float:
    """Weighted fraction of query terms covered by `tokens` (or, at HEADING_WEIGHT, by the
    heading tokens), plus PHRASE_WEIGHT times the fraction of query bigrams present."""
    total = sum(weights.values())
    if total <= 0.0:
        return 0.0
    token_set = set(tokens)
    heading_set = set(heading_tokens)
    covered = 0.0
    for term, weight in weights.items():
        if term_covers(term, token_set):
            covered += weight
        elif heading_set and term_covers(term, heading_set):
            covered += HEADING_WEIGHT * weight
    q_bi = set(bigrams(list(weights)))
    phrase = len(q_bi & set(bigrams(list(tokens)))) / len(q_bi) if q_bi else 0.0
    return covered / total + PHRASE_WEIGHT * phrase
