# ragkit

**Hybrid retrieval-augmented generation (BM25 + dense + reciprocal rank fusion) in one SQLite file, with an evaluation loop so every pipeline change is a number, not a vibe.**

[![ci](https://github.com/rohanreddy79/ragkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanreddy79/ragkit/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why

Most RAG demos are a vector database plus a prompt, and nobody can say whether the last
change to chunking or retrieval made answers better or worse. This repo is the
production-shaped version of that pipeline: loaders, two chunkers that keep exact character
offsets, a from-scratch Okapi BM25, a deterministic offline embedder, a single-file SQLite
index, rank-fused hybrid retrieval, citations, a CLI and an HTTP API. It also ships a
31-document fictional company wiki with 45 labelled questions and a `ragkit eval` command
that reports recall@k, MRR and nDCG for BM25, dense, hybrid and hybrid+rerank. The numbers
below are the point: they show where hybrid search helps, where it does not, and how much
of a difference is just noise.

## Features

- Loaders for `.txt`, `.md`, `.pdf` (pypdf) and recursive directory walks with stable ids.
- Two chunkers: fixed word windows with overlap, and a recursive paragraph -> sentence -> word
  splitter. Every chunk's `(start, end)` satisfies `doc.text[start:end] == chunk.text`.
- Okapi BM25 implemented from scratch (inverted index, Lucene-style smoothed IDF, `k1`, `b`),
  tested against a hand computation, with on-disk postings in SQLite.
- `HashEmbedder`: signed feature hashing of unigrams, bigrams and character trigrams into
  2048 dims. Deterministic, dependency-free, no model download; `sentence-transformers`
  behind the optional `st` extra via one lazy import.
- Retrieval modes `bm25`, `dense`, `hybrid` (reciprocal rank fusion, `k=60`, per-source
  weights), optional MMR diversification, and a `Reranker` protocol with a lexical-overlap
  default and a cross-encoder behind `st`.
- Generators with `[n]` citations: offline extractive (IDF-weighted, section-aware sentence
  selection) by default, Claude (`claude-opus-5-5`) behind the `anthropic` extra and
  `ANTHROPIC_API_KEY`.
- `ragkit eval`: bundled dataset, recall@1/3/5, MRR, nDCG@5, latency, Markdown output.
- Typer CLI (`ingest`, `query`, `ask`, `eval`, `serve`) and a FastAPI service with pydantic
  models (`POST /ingest`, `POST /query`, `POST /ask`, `GET /health`).

## Quickstart

```bash
uv sync
uv run ragkit ingest src/ragkit/data/eval/docs --db idx.db
uv run ragkit query "how many days of paid time off do I get" --db idx.db -k 3
uv run ragkit ask "what is the Beacon API rate limit" --db idx.db
uv run ragkit eval
```

Expected output (trimmed):

```text
indexed 31 documents as 81 chunks into idx.db (31 documents total) in 0.06s

           hybrid results for: how many days of paid time off do I get
┃ # ┃ doc                  ┃  score ┃ chunk                                    ┃
│ 1 │ hr-pto-policy#0      │ 0.0328 │ # Paid Time Off (PTO) Policy Helios ...  │
│ 2 │ eng-oncall-runbook#2 │ 0.0323 │ On-call engineers receive $350 per ...   │
│ 3 │ product-beacon-api#1 │ 0.0305 │ The current version is v2, reachable ... │

Each key is limited to 600 requests per minute, with bursts of up to 100
requests per second. [1] ...
  [1] product-beacon-api (chunk 0, chars 0-677) src/ragkit/data/eval/docs/product-beacon-api.md
```

The `eval` table is reproduced in full under [Results](#results). Everything above runs
offline; no API key or model download is involved. `examples/quickstart.sh` runs the same
sequence, and `examples/ablation.py` reproduces the ablation tables.

## How it works

```text
 .txt/.md/.pdf ──▶ loaders ──▶ chunking ──┬──▶ BM25 postings ─────┐
                   Document     Chunk     │     (SQLite)           │
                                          └──▶ HashEmbedder ──────▶│  Store (one .db file)
                                               float32 blobs       │
                                                                   ▼
 query ──┬──▶ bm25_search  ─────▶ top-N ids ─┐                 reciprocal
         └──▶ embed ──▶ dense_search ─▶ ids ─┴──▶ rank fusion ──▶ [rerank] ──▶ [MMR] ──▶ Hits
                                                                                        │
                                                   ExtractiveGenerator / Claude ◀───────┘
                                                          │
                                                 Answer + [n] citations
```

| Module | Responsibility |
|---|---|
| `loaders.py` | `Document(id, text, source, metadata)`; txt/md/pdf, directory walk with ids relative to the root |
| `text.py` | Tokeniser (lowercase, plural/possessive normalisation, small stopword list), sentence and paragraph spans |
| `chunking.py` | `fixed_size_chunks`, `recursive_chunks`; exact character offsets on every `Chunk` |
| `bm25.py` | `idf`, `term_score`, in-memory `BM25Index` (also serialisable) |
| `embed.py` | `Embedder` protocol, `HashEmbedder`, lazy `SentenceTransformerEmbedder`, `get_embedder(spec)` |
| `store.py` | SQLite schema: documents, chunks, embeddings, postings, chunk_stats, meta; BM25 and cosine search |
| `retrieve.py` | `reciprocal_rank_fusion`, `mmr_order`, `Reranker` protocol, `Retriever` |
| `lexical.py` | IDF-weighted, prefix-tolerant, heading-aware term coverage shared by the generator and reranker |
| `generate.py` | `Generator` protocol, `ExtractiveGenerator`, `AnthropicGenerator`, `Citation` |
| `eval.py` | Metrics, dataset loading, `run_eval` -> `EvalReport.to_markdown()` |
| `cli.py`, `api.py` | Typer commands and the FastAPI app |

### Design decisions

**Rank fusion, not score interpolation.** BM25 scores are unbounded and corpus-dependent
(13.2 for the top PTO hit above); cosine similarities live in [-1, 1]. Any weighted sum of
the two needs per-corpus calibration and silently breaks when one side's scale shifts.
Reciprocal rank fusion (`sum(w / (60 + rank))`) only looks at positions, so it needs no
calibration, tolerates a noisy input list, and makes per-source weights meaningful
("count BM25 twice"). The cost is that it discards score margins; the eval shows that cost
is small here.

**One SQLite file is the whole index.** Documents, chunks, float32 embedding blobs, BM25
postings and per-chunk lengths live in one `.db` with foreign keys and cascading deletes,
so re-ingesting a document atomically replaces its chunks, postings and vectors. BM25
search reads only the postings of the query's terms (`WHERE term = ?`), so it scales with
the query, not the corpus. Dense search loads the embedding matrix lazily into numpy once
and does a single matrix-vector product; the cache is invalidated on write. For the corpus
sizes this is built for (thousands to low hundreds of thousands of chunks) this beats
running a vector database, and the index is a file you can copy, diff and back up.

**Chunking is a measured trade-off, not a constant.** The recursive chunker packs whole
paragraphs up to a word budget and only splits a paragraph into sentences, or a sentence
into word windows, when it has to, so boundaries fall on the most natural unit that fits.
The budget matters more than the strategy: 120-word chunks beat 200-word chunks for every
retrieval mode on the eval set (dilution: a long chunk's vector and length-normalised BM25
score are dragged down by the sentences that are not about the question), and 80-word
chunks lose again because answers get split from their context. Offsets are exact so a
citation can be highlighted in the source file.

**A hashing embedder is a legitimate offline baseline.** `HashEmbedder` hashes unigrams,
half-weighted bigrams and `#`-padded character trigrams into 2048 signed buckets with
sublinear term frequency and L2 normalisation. It has no notion of synonyms, but it is
exact, reproducible across machines, costs nothing to run, and the character trigrams give
it morphology that exact-token BM25 lacks. It exists so that `dense` and `hybrid` have a
measurable meaning without a 90 MB model download, and so the tests are fully offline. The
ablation below shows why bigrams are down-weighted and why 2048 dims: a 120-word chunk
produces roughly 700 features, which collide badly in 512 buckets. Swapping in real
embeddings is one flag (see below).

**Measure the noise before believing a number.** With 45 questions, one question is 0.022
of recall@1, and the hashing embedder's collisions depend on a hash seed. `examples/ablation.py`
reruns the default configuration under five seeds and reports mean ± std; differences
inside that band are not evidence. That table is what stopped me from "tuning" the embedder
on single lucky runs.

**Extractive generation is the safe default.** The offline generator scores sentences by
IDF-weighted coverage of the question (prefix-tolerant, with the section heading above a
sentence counting at half weight so "Rate limits" carries down to "Each key is limited to
600 requests per minute"), keeps the sentences within 50% of the best, and cites each one.
It never invents text, so it is both a usable answerer for policy-style corpora and a lower
bound for any hosted model.

### Swapping in real models

```bash
uv sync --extra st                          # sentence-transformers (pulls torch; ~2 GB of wheels)
uv run ragkit ingest docs/ --db st.db --embedder st:sentence-transformers/all-MiniLM-L6-v2
uv run ragkit query "..." --db st.db        # the index remembers its embedder; no flag needed

uv sync --extra anthropic                   # Claude generation
cp .env.example .env && $EDITOR .env        # set ANTHROPIC_API_KEY
uv run ragkit ask "..." --db idx.db --generator anthropic
```

The index stores the embedder spec (`hash:2048:0.5:3:0.5` or `st:<model>`) and refuses to
mix vector spaces. `CrossEncoderReranker` (`st` extra) plugs into `Retriever(reranker=...)`.
Both optional paths are behind lazy imports and raise `OptionalDependencyError` with the
exact `uv sync --extra ...` command; neither is installed in CI or was run on the machine
below, so they are documented, not benchmarked.

### HTTP API

```bash
uv run ragkit serve --db idx.db --port 8765
curl -s http://127.0.0.1:8765/health
curl -s -X POST http://127.0.0.1:8765/query -H 'content-type: application/json' \
  -d '{"query": "canary rollout percentage", "mode": "hybrid", "k": 2}'
curl -s -X POST http://127.0.0.1:8765/ask -H 'content-type: application/json' \
  -d '{"query": "how long are robot camera images retained", "k": 3}'
```

`POST /ingest` takes either a server-side `path` or inline `documents`; every response is a
pydantic model, and errors map to 400 (bad input), 422 (validation), 503 (optional extra
or key missing) and 502 (generation failure).

## Results

Machine: Apple M3, 8 GB RAM, macOS, CPU only, Python 3.12. Dataset: the bundled wiki of a
fictional warehouse-robotics company (31 documents, 206-264 words each, 45 questions with
document-level gold labels; several questions have two gold documents and several are
paraphrased so they share few exact tokens with the answer). Chunk rankings are collapsed
to document order before scoring. Reproduce with `uv run ragkit eval --markdown`.

| retrieval | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | ms/query |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.867 | 0.989 | 0.989 | 0.956 | 0.955 | 0.1 |
| dense | 0.833 | 0.956 | 0.956 | 0.919 | 0.918 | 0.1 |
| hybrid | 0.856 | 0.978 | 0.978 | 0.941 | 0.940 | 0.1 |
| hybrid+rerank | 0.811 | 0.978 | 0.978 | 0.919 | 0.924 | 1.3 |

45 questions, 31 documents, 81 chunks; embedder `hash:2048:0.5:3:0.5`; recursive chunks,
120 words, overlap 20.

**Why hybrid, then?** On this corpus BM25 alone is very strong (the questions reuse the
wiki's vocabulary) and the hashing embedder is itself lexical, so fusing them cannot add
information that neither has; hybrid lands between its two inputs and within noise of the
better one. What it buys is robustness: across hash seeds, hybrid's mean is the highest of
the three and it never collapses when one input has a bad day (with the earlier 200-word
chunks and full-weight bigrams, dense fell to 0.533 recall@1 while hybrid held 0.733). The
case for hybrid gets stronger, not weaker, when the dense side is a semantic model that
can match "vacation" to "paid time off"; that is the `st` extra, which this machine cannot
run. The lexical reranker is a negative result: it is a protocol demonstration, and
re-scoring by term coverage loses to the fused ranking (-0.045 recall@1), so `--rerank`
is off by default.

Ablations from `uv run python examples/ablation.py` (same machine, same dataset):

| embedder | dense R@1 | dense MRR | hybrid R@1 | hybrid MRR |
|---|---:|---:|---:|---:|
| unigrams only | 0.867 | 0.939 | 0.867 | 0.950 |
| unigrams + bigrams (1.0) | 0.767 | 0.889 | 0.822 | 0.933 |
| unigrams + bigrams (0.5) | 0.822 | 0.919 | 0.844 | 0.944 |
| unigrams + bigrams + char-3grams, 512 dims | 0.822 | 0.911 | 0.878 | 0.950 |
| unigrams + bigrams + char-3grams, 2048 dims (default) | 0.833 | 0.919 | 0.856 | 0.941 |

| chunking | chunks | bm25 R@1 | dense R@1 | hybrid R@1 | hybrid MRR |
|---|---:|---:|---:|---:|---:|
| recursive 80/20 | 131 | 0.789 | 0.811 | 0.811 | 0.915 |
| recursive 120/20 | 81 | 0.867 | 0.833 | 0.856 | 0.941 |
| recursive 200/40 | 62 | 0.800 | 0.856 | 0.811 | 0.911 |
| recursive 300/40 | 31 | 0.822 | 0.844 | 0.833 | 0.933 |
| fixed 120/20 | 87 | 0.844 | 0.811 | 0.833 | 0.933 |

| retrieval (5 hash seeds) | recall@1 mean ± std | MRR mean ± std |
|---|---:|---:|
| bm25 | 0.867 ± 0.000 | 0.956 ± 0.000 |
| dense | 0.858 ± 0.024 | 0.937 ± 0.015 |
| hybrid | 0.880 ± 0.026 | 0.956 ± 0.014 |

Reading these honestly: chunk size is the only lever with an effect clearly larger than the
seed noise; full-weight bigrams hurt; character trigrams and 2048 dims help with longer
chunks but are within noise at 120 words. At 200-word chunks, dense recall@1 goes
0.533 -> 0.744 -> 0.856 for `--embedder hash:512:1:0:0`, `hash:512:1:3:0.5` and the
default `hash:2048:0.5:3:0.5` (`uv run ragkit eval --chunk-size 200 --overlap 40
--embedder <spec>`). The whole eval (4 configurations x 45 queries, including indexing)
runs in about 0.5 s, which is what makes it cheap enough to run on every change. Latency
columns are wall-clock means and vary between runs.

## Testing

```bash
uv run ruff check .
uv run pytest -q        # 55 passed in about 1 s, fully offline
```

Covered: chunk offsets map back to the source text exactly for both chunkers; BM25 scores
against a hand computation; RRF ordering, weights and tie-breaks; MMR picks the diverse
item; `HashEmbedder` determinism, normalisation and seeds; store round-trip (close, reopen,
identical results), replacement and cascading deletes, embedding-space mismatch errors;
end-to-end ingest -> query returns the right document first in every mode; the extractive
generator's sentence choice, citations and no-answer path; the Claude generator against a
fake client (citation parsing, refusal and empty-response errors) and its `ANTHROPIC_API_KEY`
gating; eval metrics with known rankings; the bundled dataset's consistency; the CLI via
`CliRunner`; the API via `TestClient`. The lazy `st` and `anthropic` imports are tested by
blanking the modules in `sys.modules`. The PDF loader is tested on a PDF built by hand in
the test fixture.

## Roadmap / Limitations

- The eval set is small (45 questions) and was also used to choose the defaults (chunk size,
  bigram weight, dimension). Treat the tables as a development signal with a ±0.025 noise
  band, not a benchmark; a held-out split is the next step.
- No semantic matching offline: the hashing embedder cannot map "vacation" to "paid time
  off". The `st` extra provides it but is not exercised here.
- The extractive generator cannot reason over lists or tables ("who can approve a 25%
  discount" needs the discount-authority bullets), only pick sentences.
- The store keeps the full embedding matrix in memory for dense search (8 KB per chunk at
  2048 dims); beyond a few hundred thousand chunks it needs an ANN index.
- Ingest is single-process and re-embeds every chunk of a changed document; incremental
  chunk-level updates are not implemented.
- Streaming answers and conversation history are out of scope; the API answers one question
  at a time.

## License

MIT. See [LICENSE](LICENSE).
