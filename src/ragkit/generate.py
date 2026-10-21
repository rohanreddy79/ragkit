"""Answer generation with citations: an offline extractive generator and an optional
Claude-backed generator (extra: `anthropic`, gated on ANTHROPIC_API_KEY).

Both return an Answer whose text cites retrieved chunks as [n], where n is the 1-based
position of the hit in the retrieval results, and whose `citations` list resolves each n
back to a document, source path, and character range.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ragkit.errors import ConfigurationError, GenerationError, OptionalDependencyError
from ragkit.lexical import coverage, query_weights
from ragkit.retrieve import Hit
from ragkit.text import sentences, tokenize

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
NO_ANSWER = "I could not find an answer to that in the indexed documents."

_CITATION_RE = re.compile(r"\[(\d+)\]")
_MARKUP_PREFIX_RE = re.compile(r"^[\s#>*\-]+")


@dataclass(frozen=True)
class Citation:
    """Resolves a [n] marker in an answer back to the chunk it came from."""

    n: int
    chunk_id: int
    doc_id: str
    chunk_index: int
    start: int
    end: int


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    generator: str


@runtime_checkable
class Generator(Protocol):
    """Turns a question plus retrieved hits into a cited answer."""

    name: str

    def generate(self, question: str, hits: Sequence[Hit]) -> Answer: ...


class ExtractiveGenerator:
    """Picks the sentences across the top chunks that best cover the question's terms.

    Scoring is `ragkit.lexical.coverage`: the IDF-weighted fraction of question terms a
    sentence contains (prefix-tolerant, with the section heading above it counting at half
    weight) plus a phrase bonus. Headings themselves and sentences shorter than
    `min_tokens` are never answers. Sentences scoring at least `min_ratio` of the best one
    are kept (up to `max_sentences`) and emitted in retrieval order with a [n] citation
    each. It never invents text, which makes it a safe default and a useful lower bound
    for any hosted generator.
    """

    name = "extractive"

    def __init__(
        self,
        max_sentences: int = 3,
        min_ratio: float = 0.5,
        *,
        min_tokens: int = 3,
        idf: Callable[[str], float] | None = None,
    ) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError("min_ratio must be in [0, 1]")
        self.max_sentences = max_sentences
        self.min_ratio = min_ratio
        self.min_tokens = min_tokens
        self.idf = idf

    def generate(self, question: str, hits: Sequence[Hit]) -> Answer:
        weights = query_weights(question, self.idf)
        if not weights:
            return Answer(NO_ANSWER, [], self.name)
        scored: list[tuple[float, int, int, str]] = []
        for n, hit in enumerate(hits, start=1):
            heading_tokens: list[str] = []
            for pos, raw in enumerate(sentences(hit.chunk.text)):
                if raw.lstrip().startswith("#"):
                    heading_tokens.extend(tokenize(raw))
                    continue
                sent = _MARKUP_PREFIX_RE.sub("", raw).strip()
                toks = tokenize(sent)
                if len(toks) < self.min_tokens:
                    continue
                score = coverage(weights, toks, heading_tokens=heading_tokens)
                if score > 0.0:
                    scored.append((score, n, pos, sent))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        # Weak sentences (relative to the best one) add noise rather than evidence.
        floor = scored[0][0] * self.min_ratio if scored else 0.0
        chosen: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        for score, n, pos, sent in scored:
            if sent in seen or score < floor:
                continue
            seen.add(sent)
            chosen.append((n, pos, sent))
            if len(chosen) == self.max_sentences:
                break
        if not chosen:
            return Answer(NO_ANSWER, [], self.name)
        chosen.sort()
        text = " ".join(f"{sent} [{n}]" for n, _, sent in chosen)
        cited = sorted({n for n, _, _ in chosen})
        return Answer(text, [_citation(n, hits[n - 1]) for n in cited], self.name)


_SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite every claim with the passage number in square brackets, like [2]. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Be concise."
)


class AnthropicGenerator:
    """Grounded generation with Claude (extra: `anthropic`).

    Construction fails fast with ConfigurationError if ANTHROPIC_API_KEY is unset, so the
    CLI and API can report a clear message instead of a stack trace mid-request. A
    pre-built client can be injected for tests.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        max_tokens: int = 1024,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.environ.get("RAGKIT_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.max_tokens = max_tokens
        self._api_errors: tuple[type[Exception], ...] = ()
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise OptionalDependencyError("anthropic", "anthropic", "AnthropicGenerator") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set; export it (see .env.example) "
                "or use --generator extractive"
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._api_errors = (anthropic.APIConnectionError, anthropic.APIStatusError)

    def generate(self, question: str, hits: Sequence[Hit]) -> Answer:
        if not hits:
            return Answer(NO_ANSWER, [], self.name)
        context = "\n\n".join(
            f"[{n}] (document: {hit.chunk.doc_id})\n{hit.chunk.text}"
            for n, hit in enumerate(hits, 1)
        )
        prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._api_errors as exc:
            raise GenerationError(f"Claude request failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise GenerationError("Claude declined to answer this request")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise GenerationError("Claude returned an empty answer")
        valid = {int(m) for m in _CITATION_RE.findall(text)} & set(range(1, len(hits) + 1))
        return Answer(text, [_citation(n, hits[n - 1]) for n in sorted(valid)], self.name)


def get_generator(
    name: str = "extractive", *, idf: Callable[[str], float] | None = None, **kwargs: Any
) -> Generator:
    """Build a generator by name: `extractive` (default, offline) or `anthropic`.

    `idf` is forwarded to the extractive generator only (hosted models do not need it).
    """
    if name == "extractive":
        return ExtractiveGenerator(idf=idf, **kwargs)
    if name == "anthropic":
        return AnthropicGenerator(**kwargs)
    raise ValueError(f"unknown generator {name!r}; expected 'extractive' or 'anthropic'")


def _citation(n: int, hit: Hit) -> Citation:
    c = hit.chunk
    return Citation(n, hit.chunk_id, c.doc_id, c.index, c.start, c.end)
