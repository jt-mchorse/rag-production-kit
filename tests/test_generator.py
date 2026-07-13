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

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith discovered penicillin [cite:A].",  # title
            "The U.S. economy grew last year [cite:A].",  # dotted initialism
            "Founded by e.g. Amodei and others [cite:A].",  # Latin editorial
            "Acme Inc. shipped it on time [cite:A].",  # org abbreviation
            "It happened at 9 a.m. sharp [cite:A].",  # time abbreviation
            "Compare vs. the baseline result [cite:A].",  # editorial vs.
            "Dr. J. Smith wrote the paper [cite:A].",  # title + single-letter initial
        ],
    )
    def test_keeps_abbreviation_sentence_intact(self, text: str) -> None:
        # An abbreviation's period is not a sentence boundary. Splitting on it
        # stranded a claim-less fragment ("Dr.", "The U.S.") that survived the
        # alphanumeric filter and then demanded its own [cite:...] marker,
        # falsely refusing an otherwise fully-cited answer (#110). Each of these
        # is one claim carrying one marker and must stay a single sentence.
        assert split_sentences(text) == [text]

    def test_merges_multiple_abbreviations_in_one_sentence(self) -> None:
        # The merge pass is iterative: several abbreviations in one claim all
        # collapse back into the single sentence they belong to.
        text = "Mr. Poe met Dr. Smith in the U.S. yesterday [cite:A]."
        assert split_sentences(text) == [text]

    def test_does_not_over_merge_real_sentence_boundaries(self) -> None:
        # Guard against the fix being too greedy: a period after an ordinary word
        # (not an abbreviation or a lone capital initial) is still a real
        # boundary, so genuinely separate claims must each stay their own
        # sentence and remain individually under citation enforcement. A numeric
        # section marker ("Section 5.") is likewise not treated as an
        # abbreviation.
        assert split_sentences("The cat sat [cite:A]. The dog ran [cite:B].") == [
            "The cat sat [cite:A].",
            "The dog ran [cite:B].",
        ]
        assert split_sentences("See Section 5. The rest follows [cite:A].") == [
            "See Section 5.",
            "The rest follows [cite:A].",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "It contains vitamin C. It improves immunity [cite:A].",  # common word ending
            "The patient has hepatitis B. Treatment began [cite:A].",
            "This is grade A. It ships today [cite:A].",
            "Vitamin C. It helps [cite:A].",  # capitalized preceding word (sentence-initial)
        ],
    )
    def test_single_capital_letter_word_is_a_real_boundary(self, text: str) -> None:
        # A claim ending in a lone capital letter ("vitamin C.", "hepatitis B.")
        # is a genuine sentence boundary, NOT a name initial. Merging it into the
        # next sentence let the first (potentially fabricated, uncited) claim ride
        # on the following sentence's [cite:...] marker, bypassing enforcement
        # (#126). It must split into two sentences so each stays under citation
        # enforcement in its own right.
        assert len(split_sentences(text)) == 2

    def test_still_merges_name_initial_after_a_title(self) -> None:
        # The name-context exception is preserved: a single-letter initial that
        # follows a title abbreviation ("Dr. J.") or another initial is still not
        # a boundary, so "Dr. J. Smith ..." stays one cited claim (#110/#126).
        assert split_sentences("Dr. J. Smith wrote the paper [cite:A].") == [
            "Dr. J. Smith wrote the paper [cite:A]."
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is no. The sky is blue [cite:A].",  # "no" as the English word
            "The result is a clear no. It ships today [cite:A].",
        ],
    )
    def test_numeric_reference_abbreviation_word_sense_is_a_real_boundary(self, text: str) -> None:
        # #130 (sibling of #126): "no" is a numeric-reference abbreviation ("No. 5")
        # but ALSO a common English word that ends a claim. Treating the word sense
        # as a non-boundary merged an uncited claim into the next sentence, letting
        # it ride on that sentence's [cite:...] marker and bypass enforcement. The
        # word sense (no digit following) must stay a real boundary so each claim
        # remains individually under citation enforcement.
        assert len(split_sentences(text)) == 2

    @pytest.mark.parametrize(
        "text",
        [
            "See No. 5 for the details [cite:A].",  # "No." + number = numeric reference
            "It is described in Vol. 3 of the series [cite:A].",
            "The proof is in pp. 12 of the appendix [cite:A].",
        ],
    )
    def test_numeric_reference_abbreviation_numeric_sense_still_merges(self, text: str) -> None:
        # The legitimate numeric sense — the abbreviation immediately followed by a
        # number — is still a non-boundary, so "No. 5" / "Vol. 3" / "pp. 12" stays
        # one sentence carrying its single marker rather than stranding a claim-less
        # "No." fragment that would falsely refuse the answer (#110 posture).
        assert split_sentences(text) == [text]

    def test_no_word_bypass_is_refused_end_to_end(self) -> None:
        # End-to-end #130: an uncited claim ending in "no" followed by a cited
        # sentence must be REFUSED, not silently accepted by riding on the cited
        # sentence's marker. Mirrors the #126 capital-letter end-to-end guard.
        retrieved = [_result("doc1", "The sky is blue.")]
        with pytest.raises(CitationError) as excinfo:
            enforce_citations("The answer is no. The sky is blue [cite:doc1].", retrieved)
        assert excinfo.value.reason == "unparseable_output"

    @pytest.mark.parametrize(
        "text",
        [
            "The p50 latency was 5 ms. Requests were rare [cite:A].",  # integer unit
            "The mean was 1.5 ms. It held steady [cite:A].",  # decimal unit
            "It took 500 ms. Done [cite:A].",  # large integer unit
        ],
    )
    def test_unit_collision_abbreviation_unit_sense_is_a_real_boundary(self, text: str) -> None:
        # #138 (sibling of #126/#130): "ms" is the title "Ms." but ALSO the
        # milliseconds unit — the dominant claim-ending in a latency/telemetry
        # kit. Treating "N ms." as a non-boundary merged an uncited measurement
        # claim into the next sentence, letting it ride on that sentence's
        # [cite:...] marker and bypass enforcement. The unit sense (a number
        # precedes) must stay a real boundary so each claim is individually
        # under citation enforcement.
        assert len(split_sentences(text)) == 2

    @pytest.mark.parametrize(
        "text",
        [
            "Ms. Smith reported the outage [cite:A].",  # title + name
            "Ms. J. Smith signed off on it [cite:A].",  # title + initial + name
        ],
    )
    def test_unit_collision_abbreviation_title_sense_still_merges(self, text: str) -> None:
        # The legitimate title sense — "Ms." with no preceding number, followed
        # by a name — is still a non-boundary, so "Ms. Smith ..." stays one
        # sentence carrying its single marker rather than stranding a claim-less
        # "Ms." fragment that would falsely refuse the answer (#110 posture).
        assert split_sentences(text) == [text]

    def test_ms_unit_bypass_is_refused_end_to_end(self) -> None:
        # End-to-end #138: an uncited measurement claim ending in "N ms."
        # followed by a cited sentence must be REFUSED, not silently accepted by
        # riding on the cited sentence's marker. Mirrors the #130 "no" guard.
        retrieved = [_result("doc1", "Requests slower than that were rare.")]
        with pytest.raises(CitationError) as excinfo:
            enforce_citations(
                "The p50 latency was 5 ms. Requests were rare [cite:doc1].", retrieved
            )
        assert excinfo.value.reason == "unparseable_output"

    def test_ms_title_answer_is_not_falsely_refused_end_to_end(self) -> None:
        # The title sense must not regress into a false refusal: a fully-cited
        # "Ms. Smith ..." answer stays one enforced claim and is accepted.
        retrieved = [_result("doc1", "Ms. Smith reported the outage.")]
        citations = enforce_citations("Ms. Smith reported the outage [cite:doc1].", retrieved)
        assert [c.external_id for c in citations] == ["doc1"]

    @pytest.mark.parametrize(
        "text",
        [
            "The outage started at 5 p.m. The root cause was a config error [cite:A].",
            "The alert fired at 3 a.m. Engineers were paged [cite:A].",
        ],
    )
    def test_time_abbreviation_boundary_sense_is_a_real_boundary(self, text: str) -> None:
        # #142 (sibling of #126/#130/#139): "a.m"/"p.m" very commonly END a claim
        # in a telemetry kit ("... at 5 p.m."). Treating them as unconditional
        # non-boundaries merged an uncited time-of-day claim into the next
        # sentence, letting it ride on that sentence's [cite:...] marker and
        # bypass enforcement. A capitalized follow-on is a real boundary so each
        # claim is individually under citation enforcement.
        assert len(split_sentences(text)) == 2

    @pytest.mark.parametrize(
        "text",
        [
            "It happened at 9 a.m. sharp [cite:A].",  # lowercase continuation
            "The batch ran at 2 p.m. and finished quickly [cite:A].",
        ],
    )
    def test_time_abbreviation_mid_sentence_sense_still_merges(self, text: str) -> None:
        # The legitimate mid-sentence sense — a.m./p.m. followed by a lowercase
        # continuation ("sharp", "and ...") — stays a non-boundary, so it stays
        # one sentence carrying its single marker rather than stranding a
        # claim-less "... a.m." fragment that would falsely refuse (#110 posture).
        assert split_sentences(text) == [text]

    def test_time_abbreviation_bypass_is_refused_end_to_end(self) -> None:
        # End-to-end #142: an uncited time-of-day claim ending in "N p.m."
        # followed by a cited sentence must be REFUSED, not silently accepted by
        # riding on the cited sentence's marker. Mirrors the #138/#130 guards.
        retrieved = [_result("doc1", "The root cause was a config error.")]
        with pytest.raises(CitationError) as excinfo:
            enforce_citations(
                "The outage started at 5 p.m. The root cause was a config error [cite:doc1].",
                retrieved,
            )
        assert excinfo.value.reason == "unparseable_output"

    def test_time_abbreviation_mid_sentence_answer_is_not_falsely_refused_end_to_end(self) -> None:
        # The mid-sentence sense must not regress into a false refusal: a
        # fully-cited "... 9 a.m. sharp" answer stays one enforced claim.
        retrieved = [_result("doc1", "It happened at 9 a.m. sharp.")]
        citations = enforce_citations("It happened at 9 a.m. sharp [cite:doc1].", retrieved)
        assert [c.external_id for c in citations] == ["doc1"]


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

    @pytest.mark.parametrize(
        "marker",
        ["[cite: A]", "[cite:A ]", "[cite:  A  ]", "[cite:\tA]"],
    )
    def test_accepts_marker_with_incidental_whitespace(self, marker: str) -> None:
        # #88: the LLM path emits incidental spaces inside an otherwise-valid
        # marker. Stripping the captured id means a fully-grounded answer is no
        # longer falsely refused as a dangling/unparseable citation.
        retrieved = [_result("A", "alpha")]
        citations = enforce_citations(f"Alpha is the first letter {marker}.", retrieved)
        assert [c.external_id for c in citations] == ["A"]

    def test_whitespace_variants_dedupe_to_one_citation(self) -> None:
        # Padded and unpadded references to the same id are the same citation.
        retrieved = [_result("A", "alpha")]
        text = "Claim one [cite:A]. Claim two [cite: A ]. Claim three [cite:A ]."
        citations = enforce_citations(text, retrieved)
        assert len(citations) == 1
        assert citations[0].external_id == "A"

    def test_stripping_still_rejects_a_genuinely_unknown_id(self) -> None:
        # The leniency must not mask a real dangling reference: an id that's
        # unknown even after stripping still refuses (guards against a false
        # accept). Mirror of test_rejects_dangling_citation_id, with padding.
        retrieved = [_result("A", "alpha")]
        with pytest.raises(CitationError) as exc:
            enforce_citations("Alpha is the first letter [cite: Z ].", retrieved)
        assert "dangling" in exc.value.detail
        assert "'Z'" in exc.value.detail  # message reports the stripped id


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

    def test_cited_answer_with_abbreviation_is_not_falsely_refused(self) -> None:
        # End-to-end regression for #110: before the abbreviation-aware merge in
        # split_sentences, `enforce_citations` split "Dr. Smith discovered
        # penicillin [cite:A]." into a claim-less "Dr." fragment with no marker
        # and raised CitationError -> the generator returned a Refusal for a
        # fully-grounded, correctly-cited answer. It must now validate cleanly.
        retrieved = [_result("A", "Dr. Smith discovered penicillin", fused=0.4)]
        client = _FakeClient("Dr. Smith discovered penicillin in the U.S. [cite:A].")
        gen = AnthropicGenerator(client=client)
        result = gen.generate("who discovered penicillin?", retrieved, threshold=0.05)
        assert isinstance(result, GeneratedAnswer)
        assert [c.external_id for c in result.citations] == ["A"]

    def test_uncited_claim_ending_in_single_capital_letter_is_refused(self) -> None:
        # End-to-end regression for #126: a fabricated, uncited first claim that
        # ends in a lone capital letter ("... contains vitamin C.") must NOT be
        # able to merge into the following cited sentence and slip past
        # enforcement. It is now a real boundary, so the marker-less first claim
        # raises CitationError and the generator refuses.
        retrieved = [_result("A", "The product improves immunity", fused=0.4)]
        client = _FakeClient(
            "This cures cancer and contains vitamin C. It improves immunity [cite:A]."
        )
        gen = AnthropicGenerator(client=client)
        with pytest.raises(CitationError):
            enforce_citations(
                "This cures cancer and contains vitamin C. It improves immunity [cite:A].",
                retrieved,
            )
        # And the same through the generator's public path -> Refusal, not answer.
        result = gen.generate("does it cure cancer?", retrieved, threshold=0.05)
        assert isinstance(result, Refusal)

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
