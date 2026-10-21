"""Ablation: which HashEmbedder features and chunk sizes matter on the bundled eval set.

Run with `uv run python examples/ablation.py`. Everything is offline and takes a few seconds.
The point is the method, not the numbers: every pipeline change gets measured the same way.
"""

from __future__ import annotations

import statistics

from ragkit.embed import HashEmbedder
from ragkit.eval import DEFAULT_CONFIGS, run_eval

VARIANTS = {
    "unigrams only": HashEmbedder(bigram_weight=0.0, char_ngram=0),
    "unigrams + bigrams (1.0)": HashEmbedder(bigram_weight=1.0, char_ngram=0),
    "unigrams + bigrams (0.5)": HashEmbedder(bigram_weight=0.5, char_ngram=0),
    "unigrams + bigrams + char-3grams, 512 dims": HashEmbedder(512),
    "unigrams + bigrams + char-3grams, 2048 dims (default)": HashEmbedder(),
}


def main() -> None:
    configs = tuple(c for c in DEFAULT_CONFIGS if not c.rerank)
    print("## Embedder features (recursive chunks, 120 words)\n")
    print("| embedder | dense R@1 | dense MRR | hybrid R@1 | hybrid MRR |")
    print("|---|---:|---:|---:|---:|")
    for label, embedder in VARIANTS.items():
        rows = {r.name: r for r in run_eval(embedder=embedder, configs=configs).rows}
        d, h = rows["dense"], rows["hybrid"]
        print(
            f"| {label} | {d.recall_at_1:.3f} | {d.mrr:.3f} | {h.recall_at_1:.3f} | {h.mrr:.3f} |"
        )

    print("\n## Chunk size (default embedder)\n")
    print("| chunking | chunks | bm25 R@1 | dense R@1 | hybrid R@1 | hybrid MRR |")
    print("|---|---:|---:|---:|---:|---:|")
    for strategy, size, overlap in (
        ("recursive", 80, 20),
        ("recursive", 120, 20),
        ("recursive", 200, 40),
        ("recursive", 300, 40),
        ("fixed", 120, 20),
    ):
        report = run_eval(configs=configs, strategy=strategy, size=size, overlap=overlap)
        rows = {r.name: r for r in report.rows}
        print(
            f"| {strategy} {size}/{overlap} | {report.n_chunks} | {rows['bm25'].recall_at_1:.3f} "
            f"| {rows['dense'].recall_at_1:.3f} | {rows['hybrid'].recall_at_1:.3f} "
            f"| {rows['hybrid'].mrr:.3f} |"
        )


def seed_variance(n_seeds: int = 5) -> None:
    """How much of a dense/hybrid difference is just hash-collision luck."""
    configs = tuple(c for c in DEFAULT_CONFIGS if not c.rerank)
    print(f"\n## Hash-seed variance (default config, seeds 0-{n_seeds - 1})\n")
    print("| retrieval | recall@1 mean ± std | MRR mean ± std |")
    print("|---|---:|---:|")
    per_mode: dict[str, list[tuple[float, float]]] = {}
    for seed in range(n_seeds):
        report = run_eval(embedder=HashEmbedder(seed=seed), configs=configs)
        for row in report.rows:
            per_mode.setdefault(row.name, []).append((row.recall_at_1, row.mrr))
    for name, values in per_mode.items():
        r1 = [v[0] for v in values]
        rr = [v[1] for v in values]
        print(
            f"| {name} | {statistics.mean(r1):.3f} ± {statistics.pstdev(r1):.3f} "
            f"| {statistics.mean(rr):.3f} ± {statistics.pstdev(rr):.3f} |"
        )


if __name__ == "__main__":
    main()
    seed_variance()
