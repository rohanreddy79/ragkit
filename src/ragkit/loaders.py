"""Document loaders: plain text, Markdown, PDF (via pypdf), and directory walks.

A Document is immutable and carries its source path so citations can point back to it.
Document ids are derived from file paths (relative to the walked root, suffix stripped),
which keeps them stable across machines and re-ingests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ragkit.errors import LoadError

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".rst"})
PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})
SUPPORTED_SUFFIXES: frozenset[str] = TEXT_SUFFIXES | PDF_SUFFIXES


@dataclass(frozen=True)
class Document:
    """A unit of ingestion: one file (or one API payload)."""

    id: str
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


def load_text(path: Path, *, doc_id: str | None = None) -> Document:
    """Load a plain-text or Markdown file. Markdown titles ("# ...") become metadata."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LoadError(f"cannot read {path}: {exc}") from exc
    metadata = {"type": path.suffix.lstrip(".") or "txt"}
    title = _markdown_title(raw)
    if title:
        metadata["title"] = title
    return Document(id=doc_id or path.stem, text=raw, source=str(path), metadata=metadata)


def load_pdf(path: Path, *, doc_id: str | None = None) -> Document:
    """Extract text from a PDF page by page; pages are joined with blank lines."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except (OSError, PyPdfError, ValueError) as exc:
        raise LoadError(f"cannot read PDF {path}: {exc}") from exc
    text = "\n\n".join(p for p in pages if p)
    if not text:
        raise LoadError(f"PDF {path} contains no extractable text")
    metadata = {"type": "pdf", "pages": str(len(reader.pages))}
    return Document(id=doc_id or path.stem, text=text, source=str(path), metadata=metadata)


def load_file(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Dispatch on file suffix. Raises LoadError for unsupported or unreadable files."""
    p = Path(path)
    if not p.is_file():
        raise LoadError(f"not a file: {p}")
    suffix = p.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return load_text(p, doc_id=doc_id)
    if suffix in PDF_SUFFIXES:
        return load_pdf(p, doc_id=doc_id)
    raise LoadError(
        f"unsupported file type '{suffix}' ({p}); supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def load_path(path: str | Path, *, suffixes: Iterable[str] | None = None) -> list[Document]:
    """Load one file or every supported file under a directory (sorted, recursive).

    Files inside a directory get ids relative to that directory with the suffix removed,
    e.g. `wiki/hr/pto-policy.md` -> `hr/pto-policy`.
    """
    root = Path(path)
    if root.is_file():
        return [load_file(root)]
    if not root.is_dir():
        raise LoadError(f"path does not exist: {root}")
    allowed = frozenset(s.lower() for s in suffixes) if suffixes else SUPPORTED_SUFFIXES
    docs: list[Document] = []
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        if file.suffix.lower() not in allowed or file.name.startswith("."):
            continue
        rel = file.relative_to(root).with_suffix("").as_posix()
        docs.append(load_file(file, doc_id=rel))
    if not docs:
        raise LoadError(f"no supported files ({sorted(allowed)}) found under {root}")
    return docs


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            return None
    return None
