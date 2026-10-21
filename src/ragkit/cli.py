"""Typer CLI: ingest, query, ask, eval, serve."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ragkit.chunking import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, ChunkStrategy
from ragkit.embed import get_embedder
from ragkit.errors import RagkitError
from ragkit.eval import run_eval
from ragkit.generate import get_generator
from ragkit.pipeline import ingest_path
from ragkit.retrieve import LexicalOverlapReranker, Mode, Retriever
from ragkit.store import Store

app = typer.Typer(
    help="Hybrid RAG toolkit: ingest documents, query them, answer with citations, evaluate.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()
err_console = Console(stderr=True)

DbOption = Annotated[Path, typer.Option("--db", help="SQLite index file.")]
EmbedderOption = Annotated[
    str, typer.Option("--embedder", help="hash[:dim] (offline) or st[:model] (extra 'st').")
]


def _fail(exc: Exception) -> None:
    err_console.print(f"[red]error:[/red] {exc}")
    raise typer.Exit(code=1)


def _open_retriever(db: Path, embedder_spec: str, rerank: bool) -> tuple[Store, Retriever]:
    """Open an index and rebuild the embedder it was created with.

    The stored embedder wins; an explicit, different `--embedder` is an error rather than a
    silent mismatch between query and document vectors.
    """
    if not db.exists():
        raise RagkitError(f"index {db} does not exist; run `ragkit ingest <path> --db {db}` first")
    store = Store(db)
    stored = store.embedder_name
    if stored is None or embedder_spec == "hash":
        embedder = get_embedder(stored or embedder_spec)
    else:
        embedder = get_embedder(embedder_spec)
        if embedder.name != stored:
            store.close()
            raise RagkitError(
                f"index was built with embedder {stored!r}, not {embedder.name!r}; "
                "omit --embedder to reuse the stored one"
            )
    reranker = LexicalOverlapReranker(idf=store.idf) if rerank else None
    return store, Retriever(store, embedder, reranker=reranker)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory of .txt/.md/.pdf files.")],
    db: DbOption = Path("idx.db"),
    embedder: EmbedderOption = "hash",
    chunker: Annotated[str, typer.Option(help="recursive or fixed.")] = "recursive",
    chunk_size: Annotated[int, typer.Option(help="Max words per chunk.")] = DEFAULT_CHUNK_WORDS,
    overlap: Annotated[
        int, typer.Option(help="Overlapping words between word windows.")
    ] = DEFAULT_OVERLAP_WORDS,
) -> None:
    """Load, chunk, embed and index documents into a single SQLite file."""
    if chunker not in ("recursive", "fixed"):
        _fail(ValueError("--chunker must be 'recursive' or 'fixed'"))
    strategy: ChunkStrategy = "fixed" if chunker == "fixed" else "recursive"
    try:
        emb = get_embedder(embedder)
        with Store(db) as store:
            result = ingest_path(
                store, path, emb, strategy=strategy, size=chunk_size, overlap=overlap
            )
            console.print(
                f"indexed {result.documents} documents as {result.chunks} chunks into {db} "
                f"({store.count_documents()} documents total) in {result.seconds:.2f}s"
            )
    except RagkitError as exc:
        _fail(exc)


@app.command()
def query(
    text: Annotated[str, typer.Argument(help="Search query.")],
    db: DbOption = Path("idx.db"),
    mode: Annotated[str, typer.Option(help="bm25, dense, or hybrid.")] = "hybrid",
    k: Annotated[int, typer.Option("-k", "--k", help="Number of results.")] = 5,
    rerank: Annotated[bool, typer.Option(help="Apply the lexical-overlap reranker.")] = False,
    mmr: Annotated[
        float | None, typer.Option(help="MMR lambda in [0,1] for diversification.")
    ] = None,
    embedder: EmbedderOption = "hash",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
) -> None:
    """Retrieve the top-k chunks for a query."""
    if mode not in ("bm25", "dense", "hybrid"):
        _fail(ValueError("--mode must be bm25, dense, or hybrid"))
    search_mode: Mode = mode  # type: ignore[assignment]
    try:
        store, retriever = _open_retriever(db, embedder, rerank)
        with store:
            hits = retriever.search(text, mode=search_mode, k=k, mmr_lambda=mmr)
            if as_json:
                payload = [
                    {
                        "rank": i,
                        "chunk_id": h.chunk_id,
                        "doc_id": h.doc_id,
                        "score": h.score,
                        "scores": h.scores,
                        "text": h.chunk.text,
                    }
                    for i, h in enumerate(hits, 1)
                ]
                console.print_json(json.dumps(payload))
                return
            table = Table(title=f"{mode} results for: {text}")
            table.add_column("#", justify="right")
            table.add_column("doc")
            table.add_column("score", justify="right")
            table.add_column("chunk")
            for i, h in enumerate(hits, 1):
                snippet = " ".join(h.chunk.text.split())
                table.add_row(
                    str(i),
                    f"{h.doc_id}#{h.chunk.index}",
                    f"{h.score:.4f}",
                    snippet[:110] + ("…" if len(snippet) > 110 else ""),
                )
            console.print(table)
    except RagkitError as exc:
        _fail(exc)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer.")],
    db: DbOption = Path("idx.db"),
    mode: Annotated[str, typer.Option(help="bm25, dense, or hybrid.")] = "hybrid",
    k: Annotated[int, typer.Option("-k", "--k", help="Chunks given to the generator.")] = 5,
    rerank: Annotated[bool, typer.Option(help="Apply the lexical-overlap reranker.")] = False,
    generator: Annotated[
        str, typer.Option(help="extractive (offline) or anthropic.")
    ] = "extractive",
    embedder: EmbedderOption = "hash",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Answer a question from the index with [n] citations."""
    if mode not in ("bm25", "dense", "hybrid"):
        _fail(ValueError("--mode must be bm25, dense, or hybrid"))
    search_mode: Mode = mode  # type: ignore[assignment]
    try:
        store, retriever = _open_retriever(db, embedder, rerank)
        with store:
            gen = get_generator(generator, idf=store.idf)
            hits = retriever.search(question, mode=search_mode, k=k)
            answer = gen.generate(question, hits)
            sources = store.sources(h.doc_id for h in hits)
            if as_json:
                payload = {
                    "answer": answer.text,
                    "generator": answer.generator,
                    "citations": [
                        {**asdict(c), "source": sources.get(c.doc_id, "")} for c in answer.citations
                    ],
                }
                console.print_json(json.dumps(payload))
                return
            console.print(answer.text)
            if answer.citations:
                console.print()
                for c in answer.citations:
                    where = f"chunk {c.chunk_index}, chars {c.start}-{c.end}"
                    console.print(f"  [{c.n}] {c.doc_id} ({where}) {sources.get(c.doc_id, '')}")
    except RagkitError as exc:
        _fail(exc)


@app.command("eval")
def eval_cmd(
    dataset: Annotated[
        Path | None,
        typer.Option(help="Dataset dir with docs/ and questions.jsonl (default: bundled)."),
    ] = None,
    embedder: EmbedderOption = "hash",
    chunker: Annotated[str, typer.Option(help="recursive or fixed.")] = "recursive",
    chunk_size: Annotated[int, typer.Option(help="Max words per chunk.")] = DEFAULT_CHUNK_WORDS,
    overlap: Annotated[
        int, typer.Option(help="Overlapping words between word windows.")
    ] = DEFAULT_OVERLAP_WORDS,
    rerank: Annotated[bool, typer.Option(help="Include the hybrid+rerank row.")] = True,
    markdown: Annotated[bool, typer.Option(help="Print a Markdown table (for READMEs).")] = False,
) -> None:
    """Compare bm25 / dense / hybrid / hybrid+rerank on the bundled wiki question set."""
    from ragkit.eval import DEFAULT_CONFIGS

    if chunker not in ("recursive", "fixed"):
        _fail(ValueError("--chunker must be 'recursive' or 'fixed'"))
    strategy: ChunkStrategy = "fixed" if chunker == "fixed" else "recursive"
    configs = DEFAULT_CONFIGS if rerank else tuple(c for c in DEFAULT_CONFIGS if not c.rerank)
    try:
        report = run_eval(
            dataset,
            embedder=get_embedder(embedder),
            configs=configs,
            strategy=strategy,
            size=chunk_size,
            overlap=overlap,
        )
    except RagkitError as exc:
        _fail(exc)
        return
    caption = (
        f"{report.n_questions} questions, {report.n_documents} documents, "
        f"{report.n_chunks} chunks; embedder={report.embedder}; chunking={report.chunking}"
    )
    if markdown:
        console.print(report.to_markdown(), markup=False)
        console.print(f"\n{caption}", markup=False)
        return
    table = Table(title="Retrieval quality", caption=caption)
    for col in ("retrieval", "recall@1", "recall@3", "recall@5", "MRR", "nDCG@5", "ms/query"):
        table.add_column(col, justify="left" if col == "retrieval" else "right")
    for r in report.rows:
        table.add_row(
            r.name,
            f"{r.recall_at_1:.3f}",
            f"{r.recall_at_3:.3f}",
            f"{r.recall_at_5:.3f}",
            f"{r.mrr:.3f}",
            f"{r.ndcg_at_5:.3f}",
            f"{r.mean_latency_ms:.1f}",
        )
    console.print(table)


@app.command()
def serve(
    db: DbOption = Path("idx.db"),
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    embedder: EmbedderOption = "hash",
) -> None:
    """Run the HTTP API (FastAPI + uvicorn) over an index. Creates the index if missing."""
    import uvicorn

    from ragkit.api import create_app

    try:
        application = create_app(db, embedder_spec=embedder)
    except RagkitError as exc:
        _fail(exc)
        return
    uvicorn.run(application, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
