from ragkit.text import (
    bigrams,
    iter_windows,
    normalize,
    paragraph_spans,
    sentence_spans,
    sentences,
    tokenize,
    word_spans,
)


def test_tokenize_lowercases_normalises_and_drops_stopwords() -> None:
    assert tokenize("The On-Call Engineer's page, within 15 minutes!") == [
        "call",
        "engineer",
        "page",
        "within",
        "15",
        "minute",
    ]
    assert tokenize("the a and", drop_stopwords=False) == ["the", "a", "and"]
    assert tokenize("the a and") == []


def test_normalize_is_conservative() -> None:
    assert [normalize(t) for t in ["policies", "robots", "acknowledges", "status"]] == [
        "policy",
        "robot",
        "acknowledge",
        "status",
    ]
    assert [normalize(t) for t in ["access", "bus", "gas", "5xx", "sla", "release"]] == [
        "access",
        "bus",
        "gas",
        "5xx",
        "sla",
        "release",
    ]


def test_bigrams() -> None:
    assert bigrams(["a", "b", "c"]) == ["a_b", "b_c"]
    assert bigrams(["solo"]) == []


def test_paragraph_and_sentence_spans_are_trimmed_substrings() -> None:
    text = "  First para. Second sentence!\n\n\n  Second para\nwith a line break.  \n"
    paras = paragraph_spans(text)
    assert [text[a:b] for a, b in paras] == [
        "First para. Second sentence!",
        "Second para\nwith a line break.",
    ]
    sents = sentence_spans(text, *paras[0])
    assert [text[a:b] for a, b in sents] == ["First para.", "Second sentence!"]
    assert sentences("# Title\n- item one\n- item two. Done.") == [
        "# Title",
        "- item one",
        "- item two.",
        "Done.",
    ]


def test_sentence_split_keeps_abbreviations_and_lowercase_continuations() -> None:
    assert sentences("Use e.g. the dashboard. then stop. Next one.") == [
        "Use e.g. the dashboard. then stop.",
        "Next one.",
    ]


def test_word_spans_and_windows() -> None:
    text = "a  bb\tccc\n"
    assert [text[a:b] for a, b in word_spans(text)] == ["a", "bb", "ccc"]
    assert list(iter_windows(5, 2, 0)) == [(0, 2), (2, 4), (4, 5)]
    assert list(iter_windows(5, 3, 1)) == [(0, 3), (2, 5)]
    assert list(iter_windows(0, 3, 1)) == [(0, 0)]
