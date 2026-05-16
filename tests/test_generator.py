"""Tests for `rag_kit.generator` — citation enforcement + weak-context refusal.

These tests are hermetic (no Postgres, no network). The `RetrievalResult`
objects are constructed directly so the generator layer can be tested in
isolation from the retriever.
"""

from __future__ import annotations

import pytest

from rag_kit.generator import (
    AnthropicGenerator,
    CitationError,
    GeneratedAnswer,
    Refusal,
    TemplateGenerator,
    enforce_citations,
    split_sentences,
)
from rag_kit.retriever import RetrievalResult


def _result(
    external_id: str, text: str, *, fused: float = 0.5, rerank: float | None = None
) -> RetrievalResult:
    return RetrievalResult(
        external_id=external_id,
        text=text,
        metadata={"source": "test"},
        fused_score=fused,
        ranks={"lexical": 1, "dense": 1},
        rerank_score=rerank,
        rerank_rank=None if rerank is None else 1,
    )


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self) -> None:
        assert split_sentences("First claim. Second claim! Third?") == [
            "First claim.",
            "Second claim!",
            "Third?",
        ]

    def test_drops_whitespace_only_fragments(self) -> None:
        assert split_sentences("   \n\n  ") == []


class TestEnforceCitations:
    def test_accepts_every_sentence_cited(self) -> None:
        retrieved = [_result("A", "alpha"), _result("B", "beta")]
        text = "Alpha is the first letter [cite:A]. Beta is the second [cite:B]."
        citations = enforce_citations(text, retrieved)
        assert [c.external_id for c in citations] == ["A", "B"]

    def test_rejects_missing_cite_marker(self) -> None:
        retrieved = [_result("A", "alpha")]
        text = "Alpha is the first letter."
        with pytest.raises(CitationError) as exc:
            enforce_citations(text, retrieved)
        assert exc.value.reason == "unparseable_output"

    def test_rejects_dangling_citation_id(self) -> None:
        retrieved = [_result("A", "alpha")]
        text = "Alpha is the first letter [cite:Z]."
        with pytest.raises(CitationError) as exc:
            enforce_citations(text, retrieved)
        assert "dangling" in exc.value.detail

    def test_rejects_text_with_no_sentences(self) -> None:
        retrieved = [_result("A", "alpha")]
        with pytest.raises(CitationError):
            enforce_citations("   ", retrieved)

    def test_deduplicates_repeated_citations(self) -> None:
        retrieved = [_result("A", "alpha")]
        text = "Claim one [cite:A]. Claim two [cite:A]."
        citations = enforce_citations(text, retrieved)
        assert len(citations) == 1
        assert citations[0].external_id == "A"


class TestTemplateGenerator:
    def test_happy_path_returns_answer_with_one_citation_per_chunk(self) -> None:
        retrieved = [
            _result("A", "Alpha facts", fused=0.4),
            _result("B", "Beta facts", fused=0.3),
        ]
        gen = TemplateGenerator(max_chunks=2)
        result = gen.generate("what?", retrieved, threshold=0.05)
        assert isinstance(result, GeneratedAnswer)
        assert {c.external_id for c in result.citations} == {"A", "B"}
        assert "[cite:A]" in result.text
        assert "[cite:B]" in result.text
        assert result.top_score == pytest.approx(0.4)

    def test_refuses_when_top_score_below_threshold(self) -> None:
        retrieved = [_result("A", "weak match", fused=0.001)]
        gen = TemplateGenerator()
        result = gen.generate("what?", retrieved, threshold=0.1)
        assert isinstance(result, Refusal)
        assert result.reason == "insufficient_context"
        assert result.top_score == pytest.approx(0.001)

    def test_refuses_on_empty_retrieved(self) -> None:
        gen = TemplateGenerator()
        result = gen.generate("what?", [], threshold=0.1)
        assert isinstance(result, Refusal)
        assert "no chunks retrieved" in result.detail

    def test_uses_rerank_score_when_present(self) -> None:
        retrieved = [_result("A", "fact", fused=0.05, rerank=0.9)]
        gen = TemplateGenerator()
        result = gen.generate("?", retrieved, threshold=0.5)
        assert isinstance(result, GeneratedAnswer)
        assert result.top_score == pytest.approx(0.9)

    def test_max_chunks_limits_citations(self) -> None:
        retrieved = [
            _result("A", "one", fused=0.4),
            _result("B", "two", fused=0.3),
            _result("C", "three", fused=0.2),
            _result("D", "four", fused=0.1),
        ]
        gen = TemplateGenerator(max_chunks=2)
        result = gen.generate("?", retrieved, threshold=0.05)
        assert isinstance(result, GeneratedAnswer)
        assert len(result.citations) == 2

    def test_invalid_max_chunks_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_chunks must be positive"):
            TemplateGenerator(max_chunks=0)


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **_: object) -> _FakeMessage:
        return _FakeMessage(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


class TestAnthropicGenerator:
    def test_validates_well_cited_response(self) -> None:
        retrieved = [_result("A", "alpha", fused=0.4)]
        client = _FakeClient("Alpha is the first letter [cite:A].")
        gen = AnthropicGenerator(client=client)
        result = gen.generate("?", retrieved, threshold=0.05)
        assert isinstance(result, GeneratedAnswer)
        assert result.citations[0].external_id == "A"

    def test_refuses_below_threshold_without_calling_client(self) -> None:
        retrieved = [_result("A", "weak", fused=0.001)]
        sentinel = _FakeClient("Should never be returned [cite:A].")
        gen = AnthropicGenerator(client=sentinel)
        result = gen.generate("?", retrieved, threshold=0.5)
        assert isinstance(result, Refusal)
        assert result.reason == "insufficient_context"

    def test_converts_citation_error_to_refusal(self) -> None:
        retrieved = [_result("A", "alpha", fused=0.4)]
        client = _FakeClient("Uncited claim with no marker.")
        gen = AnthropicGenerator(client=client)
        result = gen.generate("?", retrieved, threshold=0.05)
        assert isinstance(result, Refusal)
        assert result.reason == "unparseable_output"

    def test_recognises_explicit_refuse_marker(self) -> None:
        retrieved = [_result("A", "alpha", fused=0.4)]
        client = _FakeClient("REFUSE: context is off-topic")
        gen = AnthropicGenerator(client=client)
        result = gen.generate("?", retrieved, threshold=0.05)
        assert isinstance(result, Refusal)
        assert "off-topic" in result.detail

    def test_handles_dict_shaped_content_blocks(self) -> None:
        retrieved = [_result("A", "alpha", fused=0.4)]

        class _DictBlockMessage:
            def __init__(self) -> None:
                self.content = [{"text": "Claim [cite:A]."}]

        class _DictMessages:
            def create(self, **_: object) -> _DictBlockMessage:
                return _DictBlockMessage()

        class _DictClient:
            def __init__(self) -> None:
                self.messages = _DictMessages()

        gen = AnthropicGenerator(client=_DictClient())
        result = gen.generate("?", retrieved, threshold=0.05)
        assert isinstance(result, GeneratedAnswer)
