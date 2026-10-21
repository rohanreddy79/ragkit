from pathlib import Path

import pytest

from conftest import minimal_pdf
from ragkit.errors import LoadError
from ragkit.loaders import load_file, load_path


def test_directory_walk_ids_are_relative_and_sorted(corpus_dir: Path) -> None:
    (corpus_dir / "notes.csv").write_text("ignored,file")
    (corpus_dir / ".hidden.md").write_text("# hidden")
    docs = load_path(corpus_dir)
    ids = [d.id for d in docs]
    assert ids == sorted(ids)
    assert "hr/pto-policy" in ids and "eng/oncall" in ids
    assert all(not i.startswith(".") and "csv" not in i for i in ids)
    pto = next(d for d in docs if d.id == "hr/pto-policy")
    assert pto.metadata == {"type": "md", "title": "PTO policy"}
    assert pto.source.endswith("hr/pto-policy.md")


def test_single_file_and_errors(tmp_path: Path) -> None:
    txt = tmp_path / "readme.txt"
    txt.write_text("plain text body", encoding="utf-8")
    [doc] = load_path(txt)
    assert doc.id == "readme" and doc.text == "plain text body" and doc.metadata == {"type": "txt"}
    with pytest.raises(LoadError):
        load_path(tmp_path / "missing")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    with pytest.raises(LoadError):
        load_file(tmp_path / "data.bin")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LoadError):
        load_path(empty)


def test_pdf_extraction(tmp_path: Path) -> None:
    pdf = tmp_path / "spec.pdf"
    pdf.write_bytes(minimal_pdf(["Atlas picks 400 items per hour.", "Payload is 15 kg."]))
    doc = load_file(pdf)
    assert doc.metadata["type"] == "pdf" and doc.metadata["pages"] == "1"
    assert "400 items per hour" in doc.text and "15 kg" in doc.text
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf at all")
    with pytest.raises(LoadError):
        load_file(tmp_path / "broken.pdf")
