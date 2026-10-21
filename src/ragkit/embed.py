"""Embedders: the Embedder protocol, an offline feature-hashing baseline, and an optional
sentence-transformers wrapper behind the `st` extra.

All embedders return float32 arrays of shape (n, dim) with unit L2 norm, so cosine
similarity is a plain dot product everywhere downstream.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np

from ragkit.errors import OptionalDependencyError
from ragkit.text import bigrams, tokenize


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps texts to unit-norm float32 vectors."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim); rows are L2-normalised."""
        ...


class HashEmbedder:
    """Deterministic, dependency-free embedder based on signed feature hashing.

    Features are unigrams, bigrams (weight `bigram_weight`), and character n-grams of each
    token padded with `#` (weight `char_weight`). Each feature is hashed into one of `dim`
    buckets with a hash-derived sign (so collisions cancel in expectation rather than
    accumulate), weighted by sublinear term frequency, and L2-normalised. Character n-grams
    give it morphology ("reimbursement" shares most trigrams with "reimburses"), which plain
    BM25 lacks; it still has no notion of synonyms. It is exact, fast, needs no model
    download, and gives dense retrieval a reproducible baseline to measure against.

    The `name` encodes every setting, so an index can rebuild the identical embedder.
    """

    def __init__(
        self,
        dim: int = 2048,
        *,
        bigram_weight: float = 0.5,
        char_ngram: int = 3,
        char_weight: float = 0.5,
        seed: int = 0,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if char_ngram < 0:
            raise ValueError("char_ngram must be 0 (off) or positive")
        if not 0 <= seed < 2**32:
            raise ValueError("seed must fit in 32 bits")
        self.dim = dim
        self.bigram_weight = bigram_weight
        self.char_ngram = char_ngram
        self.char_weight = char_weight
        self.seed = seed
        base = f"hash:{dim}:{bigram_weight:g}:{char_ngram}:{char_weight:g}"
        self.name = f"{base}:{seed}" if seed else base

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            self._embed_into(out[row], text)
        return out

    def features(self, text: str) -> dict[str, float]:
        """Weighted feature counts for one text (exposed for tests and ablations)."""
        tokens = tokenize(text)
        counts: dict[str, float] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        if self.bigram_weight:
            for bg in bigrams(tokens):
                counts[bg] = counts.get(bg, 0.0) + self.bigram_weight
        if self.char_ngram and self.char_weight:
            n = self.char_ngram
            for tok in tokens:
                padded = f"#{tok}#"
                for i in range(len(padded) - n + 1):
                    gram = "c:" + padded[i : i + n]
                    counts[gram] = counts.get(gram, 0.0) + self.char_weight
        return counts

    def _embed_into(self, vec: np.ndarray, text: str) -> None:
        for feature, tf in self.features(text).items():
            index, sign = _slot(feature, self.dim, self.seed)
            vec[index] += sign * (1.0 + math.log(tf)) if tf >= 1.0 else sign * tf
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm


@lru_cache(maxsize=1 << 18)
def _slot(feature: str, dim: int, seed: int) -> tuple[int, float]:
    """Bucket index and sign for a feature. blake2b keeps it stable across processes; the
    seed is a salt, so different seeds give independent collision patterns."""
    digest = hashlib.blake2b(
        feature.encode("utf-8"), digest_size=8, salt=seed.to_bytes(4, "little")
    ).digest()
    h = int.from_bytes(digest, "little")
    return h % dim, 1.0 if (h >> 63) & 1 else -1.0


class SentenceTransformerEmbedder:
    """Semantic embeddings via sentence-transformers (extra: `st`).

    The import is deferred to construction so the core package never pays for torch.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                "st", "sentence-transformers", "SentenceTransformerEmbedder"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self.dim)


def get_embedder(spec: str = "hash") -> Embedder:
    """Build an embedder from a spec string.

    `hash[:dim[:bigram_weight:char_ngram:char_weight[:seed]]]` (offline) or `st[:model-name]`
    (extra `st`). Every embedder's `name` is a valid spec, so an index records exactly how
    to rebuild its embedder.
    """
    kind, _, rest = spec.partition(":")
    if kind == "hash":
        parts = [part for part in rest.split(":") if part] if rest else []
        try:
            if len(parts) == 0:
                return HashEmbedder()
            if len(parts) == 1:
                return HashEmbedder(int(parts[0]))
            if len(parts) in (4, 5):
                return HashEmbedder(
                    int(parts[0]),
                    bigram_weight=float(parts[1]),
                    char_ngram=int(parts[2]),
                    char_weight=float(parts[3]),
                    seed=int(parts[4]) if len(parts) == 5 else 0,
                )
        except ValueError as exc:
            raise ValueError(f"bad hash embedder spec {spec!r}: {exc}") from exc
        raise ValueError(f"bad hash embedder spec {spec!r}; expected hash[:dim[:bw:cn:cw[:seed]]]")
    if kind == "st":
        return SentenceTransformerEmbedder(rest) if rest else SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder spec {spec!r}; expected 'hash[:dim]' or 'st[:model]'")
