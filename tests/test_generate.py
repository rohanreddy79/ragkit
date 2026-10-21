import sys
import types

import pytest

from ragkit.chunking import Chunk
from ragkit.errors import ConfigurationError, GenerationError, OptionalDependencyError
from ragkit.generate import NO_ANSWER, AnthropicGenerator, ExtractiveGenerator, get_generator
from ragkit.retrieve import Hit


def _hits() -> list[Hit]:
    texts = [
        "# On-call runbook\n\nThe primary acknowledges a page within 15 minutes. "
        "Swaps are announced in #oncall.",
        "On-call engineers receive $350 per primary week, paid monthly.",
        "Parking is free in the north lot.",
    ]
    return [Hit(i, Chunk(f"doc{i}", 0, t, 0, len(t)), 1.0, {}) for i, t in enumerate(texts, 1)]


def test_extractive_picks_best_sentences_with_citations() -> None:
    answer = ExtractiveGenerator(max_sentences=2).generate(
        "how quickly must the primary acknowledge a page", _hits()
    )
    assert answer.text == "The primary acknowledges a page within 15 minutes. [1]"
    assert [c.n for c in answer.citations] == [1]
    assert answer.citations[0].doc_id == "doc1" and answer.citations[0].chunk_id == 1
    assert answer.generator == "extractive"


def test_extractive_orders_by_retrieval_rank_and_dedupes() -> None:
    hits = _hits()
    hits.append(Hit(4, Chunk("doc4", 0, hits[1].chunk.text, 0, 10), 0.5, {}))
    answer = ExtractiveGenerator(max_sentences=3, min_ratio=0.0).generate("primary page week", hits)
    assert answer.text.count("[") == 2  # duplicate sentence from doc4 is dropped
    assert answer.text.index("[1]") < answer.text.index("[2]")
    assert [c.n for c in answer.citations] == [1, 2]


def test_extractive_skips_headings_and_uses_idf_and_prefix_matching() -> None:
    text = (
        "# Beacon API Overview\n\n## Rate limits\n\n"
        "Beacon is the public REST API for Relay. "
        "Each key is limited to 600 requests per minute."
    )
    hits = [Hit(1, Chunk("beacon", 0, text, 0, len(text)), 1.0, {})]
    rare = {"rate": 3.0, "limit": 3.0, "api": 0.2, "beacon": 0.2}
    answer = ExtractiveGenerator(max_sentences=1, idf=lambda t: rare.get(t, 1.0)).generate(
        "what is the Beacon API rate limit", hits
    )
    assert answer.text == "Each key is limited to 600 requests per minute. [1]"
    assert ExtractiveGenerator(min_tokens=1).generate("rate limits", hits).text != "Rate limits [1]"


def test_extractive_no_answer_and_validation() -> None:
    assert ExtractiveGenerator().generate("quantum teleportation", _hits()).text == NO_ANSWER
    assert ExtractiveGenerator().generate("anything", []).citations == []
    with pytest.raises(ValueError):
        ExtractiveGenerator(max_sentences=0)
    with pytest.raises(ValueError):
        get_generator("gpt")


class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, stop_reason: str, blocks: list[_Block]) -> None:
        self.stop_reason = stop_reason
        self.content = blocks


class _FakeClient:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[dict] = []

    class _Messages:
        def __init__(self, outer: "_FakeClient") -> None:
            self.outer = outer

        def create(self, **kwargs: object) -> _Response:
            self.outer.calls.append(kwargs)
            return self.outer.response

    @property
    def messages(self) -> "_FakeClient._Messages":
        return self._Messages(self)


def test_anthropic_generator_parses_citations_from_fake_client() -> None:
    client = _FakeClient(
        _Response(
            "end_turn",
            [_Block("thinking"), _Block("text", "Within 15 minutes [1]. Pay is $350 [2][9].")],
        )
    )
    gen = AnthropicGenerator(client=client, model="claude-test")
    answer = gen.generate("how fast?", _hits())
    assert answer.text.startswith("Within 15 minutes [1].")
    assert [c.n for c in answer.citations] == [1, 2]  # [9] is out of range and dropped
    call = client.calls[0]
    assert (
        call["model"] == "claude-test" and "[1] (document: doc1)" in call["messages"][0]["content"]
    )
    assert gen.generate("x", []).text == NO_ANSWER and not client.calls[1:]


def test_anthropic_generator_errors() -> None:
    refused = AnthropicGenerator(client=_FakeClient(_Response("refusal", [])))
    with pytest.raises(GenerationError, match="declined"):
        refused.generate("q", _hits())
    empty = AnthropicGenerator(client=_FakeClient(_Response("end_turn", [_Block("text", "  ")])))
    with pytest.raises(GenerationError, match="empty"):
        empty.generate("q", _hits())


def test_anthropic_generator_is_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(OptionalDependencyError, match="uv sync --extra anthropic"):
        get_generator("anthropic")

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda api_key: object()  # type: ignore[attr-defined]
    fake.APIConnectionError = type("APIConnectionError", (Exception,), {})  # type: ignore[attr-defined]
    fake.APIStatusError = type("APIStatusError", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicGenerator()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("RAGKIT_ANTHROPIC_MODEL", "claude-from-env")
    assert AnthropicGenerator().model == "claude-from-env"
