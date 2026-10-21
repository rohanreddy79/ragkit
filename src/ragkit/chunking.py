"""Chunkers: fixed-size word windows with overlap, and a recursive paragraph/sentence/word
splitter that respects a maximum size.

Every Chunk records (start, end) character offsets such that
`document.text[start:end] == chunk.text` holds exactly. That invariant is what makes
citations and highlighting trustworthy, so it is tested directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from ragkit.loaders import Document
from ragkit.text import iter_windows, paragraph_spans, sentence_spans, word_spans

ChunkStrategy = Literal["recursive", "fixed"]
Span = tuple[int, int]

# Measured on the bundled eval set: 120-word chunks beat 200-word ones for every retrieval
# mode (less dilution per chunk), and overlap only matters for word-window fallbacks.
DEFAULT_CHUNK_WORDS = 120
DEFAULT_OVERLAP_WORDS = 20


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of one document."""

    doc_id: str
    index: int
    text: str
    start: int
    end: int

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def fixed_size_chunks(
    doc: Document, *, size: int = DEFAULT_CHUNK_WORDS, overlap: int = DEFAULT_OVERLAP_WORDS
) -> list[Chunk]:
    """Windows of `size` words advancing by `size - overlap` words.

    Offsets come from the words' own positions, so each chunk is an exact substring.
    """
    spans = word_spans(doc.text)
    if not spans:
        return []
    chunks: list[Chunk] = []
    for lo, hi in iter_windows(len(spans), size, overlap):
        start, end = spans[lo][0], spans[hi - 1][1]
        chunks.append(Chunk(doc.id, len(chunks), doc.text[start:end], start, end))
    return chunks


def recursive_chunks(
    doc: Document, *, max_words: int = DEFAULT_CHUNK_WORDS, overlap: int = DEFAULT_OVERLAP_WORDS
) -> list[Chunk]:
    """Pack whole paragraphs up to `max_words`; split oversized paragraphs into sentences,
    and oversized sentences into word windows (with `overlap` words of context).

    Boundaries therefore fall on the most natural unit that fits, which keeps related
    sentences together and avoids cutting a policy mid-clause.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    pieces = _split_level(doc.text, paragraph_spans(doc.text), max_words, overlap, level=0)
    return [
        Chunk(doc.id, i, doc.text[start:end], start, end) for i, (start, end) in enumerate(pieces)
    ]


def chunk_document(
    doc: Document,
    *,
    strategy: ChunkStrategy = "recursive",
    size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Dispatch to a chunker by name. `overlap` applies to word windows in both strategies."""
    if strategy == "fixed":
        return fixed_size_chunks(doc, size=size, overlap=overlap)
    if strategy == "recursive":
        return recursive_chunks(doc, max_words=size, overlap=overlap)
    raise ValueError(f"unknown chunk strategy {strategy!r}; expected 'recursive' or 'fixed'")


_LEVELS: Sequence[Callable[[str, int, int], list[Span]]] = (
    lambda text, a, b: sentence_spans(text, a, b),
    lambda text, a, b: [(a + s, a + e) for s, e in word_spans(text[a:b])],
)


def _split_level(
    text: str, spans: list[Span], max_words: int, overlap: int, level: int
) -> list[Span]:
    """Greedily pack consecutive spans; recurse into any single span that is too big."""
    out: list[Span] = []
    group: list[Span] = []
    group_words = 0
    for span in spans:
        n = _count_words(text, span)
        if n > max_words:
            if group:
                out.append((group[0][0], group[-1][1]))
                group, group_words = [], 0
            out.extend(_split_oversized(text, span, max_words, overlap, level))
            continue
        if group and group_words + n > max_words:
            out.append((group[0][0], group[-1][1]))
            group, group_words = [], 0
        group.append(span)
        group_words += n
    if group:
        out.append((group[0][0], group[-1][1]))
    return out


def _split_oversized(text: str, span: Span, max_words: int, overlap: int, level: int) -> list[Span]:
    start, end = span
    if level < len(_LEVELS) - 1:
        sub = _LEVELS[level](text, start, end)
        if len(sub) > 1:
            return _split_level(text, sub, max_words, overlap, level + 1)
    # Last resort: word windows. Overlap is capped so the window always advances.
    words = _LEVELS[-1](text, start, end)
    step_overlap = min(overlap, max_words - 1)
    return [
        (words[lo][0], words[hi - 1][1])
        for lo, hi in iter_windows(len(words), max_words, step_overlap)
    ]


def _count_words(text: str, span: Span) -> int:
    return len(text[span[0] : span[1]].split())
