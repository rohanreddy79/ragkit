import json
from pathlib import Path

from typer.testing import CliRunner

from ragkit.cli import app

runner = CliRunner()


def test_ingest_query_ask_roundtrip(tmp_path: Path, corpus_dir: Path) -> None:
    db = tmp_path / "idx.db"
    r = runner.invoke(app, ["ingest", str(corpus_dir), "--db", str(db), "--chunk-size", "40"])
    assert r.exit_code == 0, r.output
    assert "indexed 6 documents" in r.output

    r = runner.invoke(app, ["query", "Beacon API rate limit", "--db", str(db), "-k", "2"])
    assert r.exit_code == 0, r.output
    assert "product/beacon" in r.output

    r = runner.invoke(app, ["query", "Atlas payload", "--db", str(db), "--mode", "bm25", "--json"])
    assert r.exit_code == 0, r.output
    hits = json.loads(r.output)
    assert hits[0]["doc_id"] == "product/atlas" and hits[0]["rank"] == 1

    r = runner.invoke(
        app, ["ask", "how many requests per minute does the Beacon API allow", "--db", str(db)]
    )
    assert r.exit_code == 0, r.output
    assert (
        "600 requests per minute" in r.output and "[1]" in r.output and "product/beacon" in r.output
    )

    r = runner.invoke(app, ["ask", "payload of atlas", "--db", str(db), "--json", "--rerank"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload["citations"][0]["doc_id"] == "product/atlas"
    assert payload["citations"][0]["source"].endswith("product/atlas.md")


def test_cli_errors(tmp_path: Path, corpus_dir: Path) -> None:
    r = runner.invoke(app, ["query", "x", "--db", str(tmp_path / "missing.db")])
    assert r.exit_code == 1 and "does not exist" in r.output
    r = runner.invoke(app, ["ingest", str(tmp_path / "nowhere"), "--db", str(tmp_path / "a.db")])
    assert r.exit_code == 1 and "error" in r.output
    r = runner.invoke(app, ["query", "x", "--db", str(tmp_path / "a.db"), "--mode", "fuzzy"])
    assert r.exit_code == 1
    db = tmp_path / "b.db"
    assert runner.invoke(app, ["ingest", str(corpus_dir), "--db", str(db)]).exit_code == 0
    r = runner.invoke(app, ["query", "x", "--db", str(db), "--embedder", "hash:64"])
    assert r.exit_code == 1 and "built with embedder" in r.output
    r = runner.invoke(app, ["ask", "x", "--db", str(db), "--generator", "anthropic"])
    assert r.exit_code == 1 and ("ANTHROPIC_API_KEY" in r.output or "extra anthropic" in r.output)


def test_eval_markdown_table() -> None:
    r = runner.invoke(app, ["eval", "--no-rerank", "--markdown"])
    assert r.exit_code == 0, r.output
    assert "| bm25 |" in r.output and "| hybrid |" in r.output and "hybrid+rerank" not in r.output
    assert "questions" in r.output
