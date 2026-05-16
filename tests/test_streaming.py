"""Hermetic tests for the streaming pipeline (#5).

What we verify here:

1. Event ordering — retrieving → retrieved → reranking → reranked →
   generating → token* → generated → done is exact and complete for
   every combination of optional phases.
2. SSE wire format — `to_sse()` produces a valid SSE frame the browser
   `EventSource` parser accepts.
3. Error path — exceptions raised by a component become a clean
   `error` terminal event; the generator does not raise out.
4. PhaseTimings percentiles — linear-interp matches NIST type-7 on
   small examples (no numpy dep needed).

No Postgres or LLM is touched. We use stub `RetrieverLike` and
`Reranker` implementations that return canned data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import pytest

from rag_kit.reranker import Candidate, Reranker, ScoredCandidate
from rag_kit.retriever import RetrievalResult
from rag_kit.streaming import (
    PhaseTimings,
    StreamEvent,
    StreamingPipeline,
    to_sse,
)

# ---------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------


class FakeRetriever:
    """Returns canned RetrievalResults without touching Postgres."""

    def __init__(self, chunks: list[tuple[str, str]]) -> None:
        # chunks is [(external_id, text), ...]
        self._chunks = chunks
        self.last_k: int | None = None
        self.calls = 0

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        reranker: Reranker | None = None,
    ) -> list[RetrievalResult]:
        self.last_k = k
        self.calls += 1
        out: list[RetrievalResult] = []
        for i, (ext_id, text) in enumerate(self._chunks[:k]):
            out.append(
                RetrievalResult(
                    external_id=ext_id,
                    text=text,
                    metadata={"i": i},
                    fused_score=1.0 / (i + 1),
                    ranks={"lexical": i + 1, "dense": i + 1},
                )
            )
        return out


class ReverseReranker:
    """Deterministic reranker that reverses input order. Lets us assert that
    the rerank phase actually mutated the result list."""

    def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> list[ScoredCandidate]:
        out: list[ScoredCandidate] = []
        n = len(candidates)
        for new_idx, c in enumerate(reversed(candidates)):
            out.append(
                ScoredCandidate(
                    external_id=c.external_id,
                    text=c.text,
                    metadata=c.metadata,
                    rerank_score=float(n - new_idx),
                    rerank_rank=new_idx + 1,
                )
            )
        return out


def _tokens_for(query: str, retrieved: Sequence[RetrievalResult]) -> Iterable[str]:
    # Yields a small fixed sequence so test assertions are stable.
    yield "Answer:"
    yield " "
    yield "ok"


# ---------------------------------------------------------------------
# Event ordering
# ---------------------------------------------------------------------


def _types(events: list[StreamEvent]) -> list[str]:
    return [e.type for e in events]


def test_retrieve_only_emits_retrieving_retrieved_done() -> None:
    pipe = StreamingPipeline(FakeRetriever([("a", "alpha"), ("b", "beta")]))
    events = list(pipe.run("q", k=2))
    assert _types(events) == ["retrieving", "retrieved", "done"]


def test_retrieve_plus_rerank_emits_full_rerank_phase() -> None:
    pipe = StreamingPipeline(
        FakeRetriever([("a", "alpha"), ("b", "beta"), ("c", "gamma")]),
        reranker=ReverseReranker(),
    )
    events = list(pipe.run("q", k=2))
    assert _types(events) == [
        "retrieving",
        "retrieved",
        "reranking",
        "reranked",
        "done",
    ]


def test_retrieve_plus_rerank_plus_generate_emits_full_pipeline() -> None:
    pipe = StreamingPipeline(
        FakeRetriever([("a", "alpha"), ("b", "beta")]),
        reranker=ReverseReranker(),
        token_stream=_tokens_for,
    )
    events = list(pipe.run("q", k=2))
    # 3 tokens come from _tokens_for; everything else is fixed.
    assert _types(events) == [
        "retrieving",
        "retrieved",
        "reranking",
        "reranked",
        "generating",
        "token",
        "token",
        "token",
        "generated",
        "done",
    ]


def test_reranker_actually_reorders_chunks() -> None:
    fr = FakeRetriever([("a", "alpha"), ("b", "beta"), ("c", "gamma")])
    pipe = StreamingPipeline(fr, reranker=ReverseReranker())
    events = list(pipe.run("q", k=3))
    reranked = next(e for e in events if e.type == "reranked")
    ids_after = [c["external_id"] for c in reranked.payload["chunks"]]
    assert ids_after == ["c", "b", "a"]


def test_overfetch_applied_only_when_reranker_present() -> None:
    fr_solo = FakeRetriever([(f"x{i}", "t") for i in range(40)])
    list(StreamingPipeline(fr_solo).run("q", k=3))
    assert fr_solo.last_k == 3, "no reranker → no over-fetch"

    fr_rr = FakeRetriever([(f"y{i}", "t") for i in range(40)])
    list(StreamingPipeline(fr_rr, reranker=ReverseReranker()).run("q", k=3))
    assert fr_rr.last_k == 12, "reranker present → over-fetch by 4×"


def test_retrieved_payload_carries_chunk_metadata() -> None:
    pipe = StreamingPipeline(FakeRetriever([("a", "alpha")]))
    events = list(pipe.run("q", k=1))
    retrieved = next(e for e in events if e.type == "retrieved")
    assert retrieved.payload["count"] == 1
    assert retrieved.payload["chunks"][0]["external_id"] == "a"
    assert retrieved.payload["chunks"][0]["text"] == "alpha"
    assert "phase_ms" in retrieved.payload
    assert retrieved.payload["phase_ms"] >= 0.0


def test_token_payload_carries_text() -> None:
    pipe = StreamingPipeline(
        FakeRetriever([("a", "alpha")]),
        token_stream=_tokens_for,
    )
    tokens = [e for e in pipe.run("q", k=1) if e.type == "token"]
    assert [t.payload["text"] for t in tokens] == ["Answer:", " ", "ok"]


def test_generated_payload_concatenates_tokens() -> None:
    pipe = StreamingPipeline(
        FakeRetriever([("a", "alpha")]),
        token_stream=_tokens_for,
    )
    events = list(pipe.run("q", k=1))
    generated = next(e for e in events if e.type == "generated")
    assert generated.payload["text"] == "Answer: ok"


def test_done_payload_has_total_ms() -> None:
    pipe = StreamingPipeline(FakeRetriever([("a", "alpha")]))
    events = list(pipe.run("q", k=1))
    done = events[-1]
    assert done.type == "done"
    assert done.payload["total_ms"] >= 0.0
    # `elapsed_ms` on `done` equals the total payload number — both
    # measured at the same instant so they're identical, not just close.
    assert done.elapsed_ms == done.payload["total_ms"]


# ---------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------


class ExplodingRetriever:
    def search(
        self, query: str, k: int = 5, *, reranker: Reranker | None = None
    ) -> list[RetrievalResult]:
        raise RuntimeError("simulated retrieval failure")


def test_error_during_retrieval_emits_error_event_and_returns_cleanly() -> None:
    pipe = StreamingPipeline(ExplodingRetriever())
    # No exception bubbles up — the generator finishes cleanly.
    events = list(pipe.run("q"))
    assert _types(events) == ["retrieving", "error"]
    err = events[-1]
    assert err.payload["exception"] == "RuntimeError"
    assert "simulated retrieval failure" in err.payload["message"]


class ExplodingReranker:
    def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> list[ScoredCandidate]:
        raise ValueError("bad rerank")


def test_error_during_reranking_emits_error_after_retrieved() -> None:
    pipe = StreamingPipeline(
        FakeRetriever([("a", "alpha")]),
        reranker=ExplodingReranker(),
    )
    events = list(pipe.run("q", k=1))
    assert _types(events) == ["retrieving", "retrieved", "reranking", "error"]


def test_invalid_k_raises_before_event_loop_starts() -> None:
    pipe = StreamingPipeline(FakeRetriever([("a", "alpha")]))
    with pytest.raises(ValueError, match="k must be positive"):
        list(pipe.run("q", k=0))


def test_empty_query_raises() -> None:
    pipe = StreamingPipeline(FakeRetriever([("a", "alpha")]))
    with pytest.raises(ValueError, match="query must be non-empty"):
        list(pipe.run("", k=1))


# ---------------------------------------------------------------------
# SSE wire format
# ---------------------------------------------------------------------


def test_to_sse_emits_event_line_and_json_data() -> None:
    e = StreamEvent("retrieving", {"query": "hello", "k": 5}, 0.0)
    frame = to_sse(e)
    assert frame.startswith("event: retrieving\n")
    assert frame.endswith("\n\n")
    # Round-trip: pull data line, parse JSON, assert shape.
    data_line = next(
        ln for ln in frame.splitlines() if ln.startswith("data: ")
    )
    parsed = json.loads(data_line[len("data: ") :])
    assert parsed["payload"]["query"] == "hello"
    assert parsed["elapsed_ms"] == 0.0


def test_to_sse_serializes_unicode_without_escapes() -> None:
    # The browser's EventSource parses UTF-8; preserving non-ASCII as
    # literal chars (ensure_ascii=False) keeps payloads compact and
    # human-readable in dev tools.
    e = StreamEvent("token", {"text": "café — espresso"}, 1.0)
    frame = to_sse(e)
    assert "café — espresso" in frame
    # And it still parses as JSON.
    data_line = next(
        ln for ln in frame.splitlines() if ln.startswith("data: ")
    )
    parsed = json.loads(data_line[len("data: ") :])
    assert parsed["payload"]["text"] == "café — espresso"


def test_to_sse_handles_unjsonifiable_via_default_str() -> None:
    # If a payload value isn't directly JSON-serializable, fall back to
    # str() rather than raising — keeps streaming alive at the cost of
    # exact round-tripping.
    class WeirdMeta:
        def __str__(self) -> str:
            return "WEIRD-META"

    e = StreamEvent("token", {"meta": WeirdMeta()}, 0.5)
    frame = to_sse(e)
    assert "WEIRD-META" in frame


# ---------------------------------------------------------------------
# PhaseTimings
# ---------------------------------------------------------------------


def test_phase_timings_records_per_phase() -> None:
    t = PhaseTimings()
    t.record("retrieving", 10.0)
    t.record("retrieving", 20.0)
    t.record("reranking", 5.0)
    assert t.retrieving == [10.0, 20.0]
    assert t.reranking == [5.0]


def test_phase_timings_percentile_handles_empty() -> None:
    t = PhaseTimings()
    assert t.percentile("retrieving", 50) is None


def test_phase_timings_percentile_known_values() -> None:
    t = PhaseTimings()
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        t.record("retrieving", v)
    # NIST type-7 / numpy default: 50th of 5 values = 30.0 exactly.
    assert t.percentile("retrieving", 50) == pytest.approx(30.0)
    # 95th of 5 values: rank = 0.95 * 4 = 3.8 → 40 + 0.8*(50-40) = 48.
    assert t.percentile("retrieving", 95) == pytest.approx(48.0)


def test_phase_timings_percentile_clamps_edges() -> None:
    t = PhaseTimings()
    t.record("retrieving", 5.0)
    t.record("retrieving", 15.0)
    assert t.percentile("retrieving", 0) == 5.0
    assert t.percentile("retrieving", 100) == 15.0
    # Out-of-range percentiles clamp rather than raising — match numpy's
    # well-behaved default; saves a fragile try/except in the bench script.
    assert t.percentile("retrieving", -5) == 5.0
    assert t.percentile("retrieving", 110) == 15.0


def test_phase_timings_summary_shape() -> None:
    t = PhaseTimings()
    t.record("retrieving", 10.0)
    s = t.summary()
    assert set(s.keys()) == {"retrieving", "reranking", "generating", "total"}
    assert s["retrieving"]["n"] == 1
    assert s["retrieving"]["p50_ms"] == 10.0
    assert s["reranking"]["n"] == 0
    assert s["reranking"]["p50_ms"] is None


def test_phase_timings_recorded_during_run() -> None:
    timings = PhaseTimings()
    pipe = StreamingPipeline(
        FakeRetriever([("a", "alpha"), ("b", "beta")]),
        reranker=ReverseReranker(),
        token_stream=_tokens_for,
        timings=timings,
    )
    list(pipe.run("q", k=2))
    list(pipe.run("q", k=2))
    assert len(timings.retrieving) == 2
    assert len(timings.reranking) == 2
    assert len(timings.generating) == 2
    assert len(timings.total) == 2
    # All numbers are >= 0 and total >= sum-of-parts (slack from event-yield work).
    for r, rr, g, t in zip(
        timings.retrieving,
        timings.reranking,
        timings.generating,
        timings.total,
        strict=True,
    ):
        assert r >= 0
        assert rr >= 0
        assert g >= 0
        assert t >= r + rr + g - 1e-6


def test_phase_timings_rejects_unknown_phase() -> None:
    t = PhaseTimings()
    with pytest.raises(ValueError, match="unknown phase"):
        t.record("invalid_phase_name", 1.0)
    with pytest.raises(ValueError, match="unknown phase"):
        t.percentile("invalid_phase_name", 50)
