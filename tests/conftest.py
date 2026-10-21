"""Shared fixtures: a tiny six-document wiki, on disk and in memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.loaders import Document

CORPUS: dict[str, str] = {
    "hr/pto-policy": (
        "# PTO policy\n\n"
        "Full-time employees accrue 20 days of paid time off per year. "
        "Requests go through Workday two weeks ahead.\n\n"
        "Up to 5 unused days roll over into the next year."
    ),
    "eng/oncall": (
        "# On-call runbook\n\n"
        "The primary on-call engineer acknowledges a page within 15 minutes. "
        "PagerDuty escalates to the secondary after 30 minutes.\n\n"
        "On-call pays 350 dollars per primary week."
    ),
    "hr/expenses": (
        "# Expense policy\n\n"
        "Submit receipts in Expensify within 30 days. "
        "Meals are capped at 75 dollars per day while travelling domestically."
    ),
    "product/atlas": (
        "# Atlas robot\n\n"
        "Atlas carries a payload of 15 kg and runs for 10 hours on one battery charge. "
        "Top speed is 1.5 m/s in open aisles."
    ),
    "product/beacon": (
        "# Beacon API\n\n"
        "Each API key is limited to 600 requests per minute. "
        "Webhook payloads are signed with HMAC-SHA256."
    ),
    "it/vpn": (
        "# VPN access\n\n"
        "Production systems are reachable only through the Tunnel VPN. "
        "Sign in with Okta and approve the push in Okta Verify."
    ),
}


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(id=k, text=v, source=f"{k}.md", metadata={"type": "md"}) for k, v in CORPUS.items()
    ]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "wiki"
    for doc_id, text in CORPUS.items():
        path = root / f"{doc_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def minimal_pdf(lines: list[str]) -> bytes:
    """A hand-built single-page PDF with Helvetica text that pypdf can extract."""
    content_lines = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"({escaped}) Tj T*")
    content_lines.append("ET")
    content = "\n".join(content_lines).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    out += trailer.encode()
    return bytes(out)
