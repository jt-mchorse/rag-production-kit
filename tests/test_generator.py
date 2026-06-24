"""Tests for `rag_kit.generator` — citation enforcement + weak-context refusal.

These tests are hermetic (no Postgres, no network). The `RetrievalResult`
objects are constructed directly so the generator layer can be tested in
isolation from the retriever.
"""

from __future__ import annotations

import math

import pytest

from rag_kit.generator import (
    AnthropicGenerator,
    CitationError,
    GeneratedAnswer,
    Refusal,
    TemplateGenerator,
    _top_score,
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

    def test_drops_claimless_punctuation_only_fragments(self) -> None:
        # A stray terminator splits off a fragment that asserts nothing (no
        # alphanumeric content). It must not survive as a "sentence" that then
        # demands a citation (#67). The real claim sentences are kept intact.
        assert split_sentences("First claim. . Second claim.") == [
            "First claim.",
            "Second claim.",
        ]

    def test_drops_fragment_with_only_punctuation_but_keeps_numbers(self) -> None:
        # Digits are alphanumeric: a number-bearing fragment is a potential
        # claim and stays under enforcement; a pure-punctuation one is dropped.
        assert split_sentences("Pi is 3.14 [cite:A]. !? Done [cite:A].") == [
            "Pi is 3.14 [cite:A].",
            "Done [cite:A].",
        ]


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

    def test_accepts_fully_cited_answer_with_stray_terminator(self) -> None:
        # #67: a stray "." between two cited claims used to split off a claim-less
        # fragment and falsely refuse an otherwise fully-cited answer.
        retrieved = [_result("A", "alpha")]
        text = "Alpha is the first letter [cite:A]. . It comes before beta [cite:A]."
        citations = enforce_citations(text, retrieved)
        assert [c.external_id for c in citations] == ["A"]

    def test_rejects_text_with_only_claimless_fragments(self) -> None:
        # The "no sentences" guard must still fire when nothing in the text is a
        # claim — claim-less fragments are dropped, leaving zero sentences.
        retrieved = [_result("A", "alpha")]
        with pytest.raises(CitationError):
            enforce_citations(". . !?", retrieved)

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

    def test_handles_multi_sentence_chunk_text(self) -> None:
        # The indexer stores whole-document prose, so a retrieved chunk is
        # routinely multi-sentence. The template must cite every sentence it
        # emits, not just the chunk's last one — otherwise enforce_citations
        # fragments the output and falsely refuses a fully-citable answer.
        retrieved = [
            _result(
                "c1",
                "Paris is the capital of France. It has about two million residents",
                fused=0.9,
            )
        ]
        gen = TemplateGenerator(max_chunks=3)
        result = gen.generate("about paris", retrieved, threshold=0.1)
        assert isinstance(result, GeneratedAnswer)
        assert {c.external_id for c in result.citations} == {"c1"}
        # Every sentence in the rendered answer carries the citation marker.
        for sentence in result.text.split("[cite:c1].")[:-1]:
            assert sentence.strip()
        assert result.text.count("[cite:c1]") == 2

    def test_top_score_returns_true_negative_max(self) -> None:
        # `LexicalOverlapReranker` can score a long, low-overlap chunk negative.
        # `_top_score` must report the true (negative) maximum, not clamp to 0.0.
        retrieved = [
            _result("A", "x", fused=0.05, rerank=-0.20),
            _result("B", "y", fused=0.04, rerank=-0.30),
        ]
        assert _top_score(retrieved) == pytest.approx(-0.20)

    def test_refusal_reports_real_negative_top_score(self) -> None:
        # Under the default-style positive threshold an all-negative set still
        # refuses, but the reported top_score must be the real value, not 0.0 —
        # otherwise the "top_score=… below threshold=…" detail lies to operators.
        retrieved = [_result("A", "x", fused=0.05, rerank=-0.20)]
        gen = TemplateGenerator()
        result = gen.generate("?", retrieved, threshold=0.02)
        assert isinstance(result, Refusal)
        assert result.top_score == pytest.approx(-0.20)

    def test_refuses_when_all_scores_negative_at_zero_threshold(self) -> None:
        # The decision-flip: with all-negative scores the old 0.0-clamp made
        # `0.0 < 0.0` False, so the kit answered from chunks it should reject.
        # The true max (-0.20) is below a 0.0 threshold and must refuse.
        retrieved = [
            _result("A", "x", fused=0.05, rerank=-0.20),
            _result("B", "y", fused=0.04, rerank=-0.30),
        ]
        gen = TemplateGenerator()
        result = gen.generate("?", retrieved, threshold=0.0)
        assert isinstance(result, Refusal)
        assert result.reason == "insufficient_context"
        assert result.top_score == pytest.approx(-0.20)

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
        with pytest.raises(ValueError, match="max_chunks must be a positive integer"):
            TemplateGenerator(max_chunks=0)

    # Issue #82: a non-finite threshold silently breaks the refusal gate.
    # `top < NaN` and `top < -inf` are both False so the kit answers from
    # chunks it should refuse; `+inf` forces an unconditional refusal. Reject
    # at the boundary, matching the fusion.k (#38) / length_penalty (#75)
    # finiteness guards. A *finite* negative threshold stays valid (#69).
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_threshold(self, bad: float) -> None:
        retrieved = [_result("A", "Alpha facts", fused=0.8)]
        gen = TemplateGenerator()
        with pytest.raises(ValueError, match="threshold must be a finite number"):
            gen.generate("?", retrieved, threshold=bad)

    def test_finite_negative_threshold_still_accepted(self) -> None:
        # _top_score can be genuinely negative (#69); a negative floor that
        # accepts low-overlap chunks is a legitimate config, not corruption.
        retrieved = [_result("A", "x", fused=0.05, rerank=-0.20)]
        gen = TemplateGenerator()
        result = gen.generate("?", retrieved, threshold=-0.5)
        assert isinstance(result, GeneratedAnswer)
        assert result.top_score == pytest.approx(-0.20)


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

    def test_rejects_non_finite_threshold_before_calling_client(self) -> None:
        # The guard runs before any client call, so a misconfigured threshold
        # fails loud rather than spending an API request on a broken gate.
        retrieved = [_result("A", "alpha", fused=0.8)]
        sentinel = _FakeClient("Should never be returned [cite:A].")
        gen = AnthropicGenerator(client=sentinel)
        with pytest.raises(ValueError, match="threshold must be a finite number"):
            gen.generate("?", retrieved, threshold=math.nan)

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
