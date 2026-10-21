import sys

import numpy as np
import pytest

from ragkit.embed import HashEmbedder, SentenceTransformerEmbedder, get_embedder
from ragkit.errors import OptionalDependencyError


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    texts = ["The on-call engineer acknowledges the page.", "Procurement thresholds for vendors."]
    a = HashEmbedder(dim=512).embed(texts)
    b = HashEmbedder(dim=512).embed(texts)
    assert a.shape == (2, 512) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(
        HashEmbedder().embed(["the and of"]), np.zeros((1, 2048), dtype=np.float32)
    )
    assert HashEmbedder(dim=64).embed(["x"]).shape == (1, 64)


def test_similar_texts_are_closer_than_unrelated() -> None:
    emb = HashEmbedder()
    v = emb.embed(
        [
            "the on-call engineer must acknowledge the page within fifteen minutes",
            "on-call engineers acknowledge pages quickly",
            "quarterly procurement thresholds require three vendor quotes",
        ]
    )
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])
    assert float(v[0] @ v[1]) > 0.1 and float(v[0] @ v[2]) < 0.05


def test_get_embedder_specs_roundtrip() -> None:
    assert get_embedder("hash").dim == 2048 and get_embedder("hash").name == "hash:2048:0.5:3:0.5"
    assert get_embedder("hash:128").name == "hash:128:0.5:3:0.5"
    custom = get_embedder("hash:256:0:0:0")
    assert (custom.dim, custom.bigram_weight, custom.char_ngram) == (256, 0.0, 0)
    assert get_embedder(custom.name).name == custom.name
    seeded = get_embedder("hash:256:0.5:3:0.5:7")
    assert seeded.seed == 7 and seeded.name == "hash:256:0.5:3:0.5:7"
    assert not np.array_equal(
        seeded.embed(["atlas robot"]), HashEmbedder(256).embed(["atlas robot"])
    )
    for bad in ("word2vec", "hash:abc", "hash:1:2", "hash:1:2:3:4:5:6"):
        with pytest.raises(ValueError):
            get_embedder(bad)
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)
    with pytest.raises(ValueError):
        HashEmbedder(char_ngram=-1)
    with pytest.raises(ValueError):
        HashEmbedder(seed=-1)


def test_features_include_bigrams_and_char_grams() -> None:
    feats = HashEmbedder().features("on-call pages")
    assert feats["call"] == 1.0 and feats["page"] == 1.0
    assert feats["call_page"] == 0.5
    assert feats["c:#ca"] == 0.5 and feats["c:ge#"] == 0.5
    plain = HashEmbedder(bigram_weight=0.0, char_ngram=0).features("on-call pages")
    assert plain == {"call": 1.0, "page": 1.0}


def test_sentence_transformer_extra_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(OptionalDependencyError, match="uv sync --extra st"):
        SentenceTransformerEmbedder()
    with pytest.raises(OptionalDependencyError):
        get_embedder("st")
