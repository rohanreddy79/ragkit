import pytest

from ragkit.chunking import chunk_document, fixed_size_chunks, recursive_chunks
from ragkit.loaders import Document


def _doc(text: str) -> Document:
    return Document(id="d", text=text, source="d.txt")


def _assert_exact_offsets(doc: Document, chunks: list) -> None:
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.doc_id == doc.id
        assert doc.text[c.start : c.end] == c.text, f"chunk {i} offsets do not map back"
        assert c.text == c.text.strip()


def test_fixed_size_offsets_overlap_and_coverage() -> None:
    words = [f"w{i}" for i in range(100)]
    text = "  " + "   \n".join(words) + "\n"  # irregular whitespace on purpose
    doc = _doc(text)
    chunks = fixed_size_chunks(doc, size=30, overlap=10)
    _assert_exact_offsets(doc, chunks)
    assert [c.n_words for c in chunks] == [30, 30, 30, 30, 20]  # step 20, tail of 20
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert chunks[-1].end == len(text.rstrip())
    assert chunks[-1].text.split()[-1] == "w99"


def test_fixed_size_single_window_when_short() -> None:
    doc = _doc("one two three")
    chunks = fixed_size_chunks(doc, size=10, overlap=2)
    assert len(chunks) == 1 and chunks[0].text == "one two three"
    assert fixed_size_chunks(_doc("   "), size=10, overlap=2) == []


def test_recursive_respects_max_words_and_offsets() -> None:
    short = "Short paragraph one.\n\nShort paragraph two with a few words."
    many_sentences = " ".join(f"Sentence number {i} has exactly six words." for i in range(12))
    one_long_sentence = " ".join(f"token{i}" for i in range(130))
    doc = _doc(f"{short}\n\n{many_sentences}\n\n{one_long_sentence}\n")
    chunks = recursive_chunks(doc, max_words=50, overlap=5)
    _assert_exact_offsets(doc, chunks)
    assert all(c.n_words <= 50 for c in chunks)
    # The two short paragraphs are packed together, including the blank line between them.
    assert chunks[0].text == short
    # The 12-sentence paragraph is split on sentence boundaries, never mid-sentence.
    sentence_chunks = [c for c in chunks if c.text.startswith("Sentence number")]
    assert len(sentence_chunks) >= 2
    assert all(c.text.endswith("words.") for c in sentence_chunks)
    # The oversized sentence falls back to overlapping word windows.
    word_chunks = [c for c in chunks if c.text.startswith("token")]
    assert len(word_chunks) == 3
    assert word_chunks[0].text.split()[-5:] == word_chunks[1].text.split()[:5]
    assert word_chunks[-1].text.split()[-1] == "token129"


def test_recursive_empty_and_validation() -> None:
    assert recursive_chunks(_doc("\n\n  \n"), max_words=10) == []
    with pytest.raises(ValueError):
        recursive_chunks(_doc("x"), max_words=0)


def test_chunk_document_dispatch() -> None:
    doc = _doc("a b c d e f")
    assert len(chunk_document(doc, strategy="fixed", size=2, overlap=0)) == 3
    assert len(chunk_document(doc, strategy="recursive", size=2, overlap=0)) == 3
    with pytest.raises(ValueError):
        chunk_document(doc, strategy="bogus")  # type: ignore[arg-type]
