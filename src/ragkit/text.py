"""Tokenisation and sentence/paragraph splitting shared by BM25, embeddings, and generation.

Everything here is regex-based and dependency-free so the whole pipeline stays deterministic
and offline. The stopword list is deliberately small: it removes function words that would
dominate BM25 document frequencies without removing domain words ("not", "no", "off").
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from itertools import pairwise

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "shall",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z0-9\"'(\[])|[ \t]*\n+[ \t]*")
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n+")
_WORD_RE = re.compile(r"\S+")


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, normalise plurals, and (optionally) drop
    stopwords. The same function runs on queries and documents, so what matters is that
    it is consistent, not that every stem is a real word."""
    tokens = (normalize(t) for t in _TOKEN_RE.findall(text.lower()))
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return list(tokens)


def normalize(token: str) -> str:
    """Strip possessives and regular plurals: engineer's -> engineer, policies -> policy,
    minutes -> minute. Short tokens and -ss/-us/-is endings are left alone (status, access)."""
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def bigrams(tokens: list[str]) -> list[str]:
    """Adjacent token pairs joined with an underscore ("on_call")."""
    return [f"{a}_{b}" for a, b in pairwise(tokens)]


def word_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of every whitespace-delimited word."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of paragraphs, split on blank lines and trimmed of whitespace."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in _PARAGRAPH_BREAK_RE.finditer(text):
        spans.append((cursor, m.start()))
        cursor = m.end()
    spans.append((cursor, len(text)))
    return [s for s in (_trim(text, a, b) for a, b in spans) if s is not None]


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """(start, end) offsets of sentences inside text[start:end], trimmed of whitespace.

    A boundary is either "terminal punctuation, spaces, then an uppercase letter, digit or
    opening quote" (which keeps "e.g. the" together) or any line break, so Markdown
    headings and list items are their own units.
    """
    end = len(text) if end is None else end
    segment = text[start:end]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in _SENTENCE_END_RE.finditer(segment):
        spans.append((start + cursor, start + m.start()))
        cursor = m.end()
    spans.append((start + cursor, end))
    return [s for s in (_trim(text, a, b) for a, b in spans) if s is not None]


def sentences(text: str) -> list[str]:
    """Sentence strings of a text, in order."""
    return [text[a:b] for a, b in sentence_spans(text)]


def iter_windows(n: int, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Yield (lo, hi) index windows of `size` over n items, advancing by size - overlap."""
    if size <= 0:
        raise ValueError("window size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    lo = 0
    while True:
        hi = min(lo + size, n)
        yield lo, hi
        if hi >= n:
            return
        lo += step


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None
