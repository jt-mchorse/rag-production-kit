"""Streaming intermediate events for the RAG pipeline (issue #5).

The pipeline is a *sync generator* that runs retrieve → optional rerank
→ optional generate and yields a typed `StreamEvent` at every phase
boundary. Consumers either format events as Server-Sent Events for a
browser frontend or consume them programmatically (e.g., in tests or
benchmarks).

Design choices (logged as D-010, D-011 in MEMORY/):

- **Sync generators, not asyncio** (D-010). The retriever and reranker
  are sync today; an async layer here would force a colored API for no
  win at the current scale. SSE serialization is a thin wrapper at the
  HTTP boundary, where the demo server handles bytes-out directly.
- **Stdlib-only.** No FastAPI/Starlette dependency in the base install
  (D-011 + D-002). The demo server uses `http.server`; a FastAPI adapter
  is a documented one-liner in the README.
- **Generator is a seam, not a hard dep.** Issue #4's generator lands
  via the `TokenStream` protocol — any callable that yields strings
  works. This lets streaming ship before #4 merges and lets evals
  (#7) swap in a different generator without touching the pipeline.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .reranker import Candidate, Reranker
from .retriever import RetrievalResult

EventType = Literal[
    "retrieving",  # phase start: retrieving from PG
    "retrieved",   # phase end: chunks + phase_ms
    "reranking",   # phase start (only if reranker is set)
    "reranked",    # phase end: chunks + phase_ms
    "generating",  # phase start (only if token_stream is set)
    "token",       # one token from the generator
    "generated",   # phase end: full text + phase_ms
    "done",        # whole pipeline finished cleanly
    "error",       # pipeline failed; payload has message + exception type
]


@dataclass(frozen=True)
class StreamEvent:
    """One typed event emitted by `StreamingPipeline.run()`.

    `elapsed_ms` is wall-clock since the pipeline started — what a
    frontend uses to render "200 ms" next to a phase card.
    `payload` schema is per-`type` and stable; see the EventType docstring
    above for the wire shape of each phase.
    """

    type: EventType
    payload: dict[str, Any]
    elapsed_ms: float


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# A retriever is anything that exposes `.search(query, k, *, reranker)`.
# Defining it as a Protocol lets tests use lightweight fakes without a
# real Postgres connection, and keeps `StreamingPipeline` from importing
# the concrete `Retriever` for typing alone.
class RetrieverLike(Protocol):
    def search(
        self,
        query: str,
        k: int = 5,
        *,
        reranker: Reranker | None = None,
    ) -> list[RetrievalResult]: ...


class TokenStream(Protocol):
    """A callable that, given the query + retrieved chunks, yields generated tokens.

    The seam where issue #4's generator (or any other) plugs in. The
    pipeline doesn't care whether tokens come from Anthropic, a stub,
    or a template — it just streams whatever strings the callable
    yields, in order.
    """

    def __call__(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
    ) -> Iterable[str]: ...


@dataclass
class PhaseTimings:
    """Tracks elapsed-ms per phase across one or more runs.

    Used by `scripts/bench_streaming.py` and any caller that wants
    p50/p95 numbers. Linear-interpolation percentile (NIST type 7); for
    sample sizes around 100–1000 this matches numpy's default within a
    fraction of a millisecond, with no numpy dep.
    """

    retrieving: list[float] = field(default_factory=list)
    reranking: list[float] = field(default_factory=list)
    generating: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)

    _PHASES = ("retrieving", "reranking", "generating", "total")

    def record(self, phase: str, ms: float) -> None:
        if phase not in self._PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        getattr(self, phase).append(ms)

    def percentile(self, phase: str, p: float) -> float | None:
        """Return p-th percentile of recorded ms for `phase`, or None if empty."""
        if phase not in self._PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        values = sorted(getattr(self, phase))
        if not values:
            return None
        if p <= 0:
            return values[0]
        if p >= 100:
            return values[-1]
        rank = (p / 100.0) * (len(values) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(values) - 1)
        frac = rank - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        return {
            phase: {
                "n": len(getattr(self, phase)),
                "p50_ms": self.percentile(phase, 50),
                "p95_ms": self.percentile(phase, 95),
            }
            for phase in self._PHASES
        }


def _chunk_to_event(r: RetrievalResult) -> dict[str, Any]:
    """Serialize one RetrievalResult into the SSE payload shape."""
    return {
        "external_id": r.external_id,
        "text": r.text,
        "metadata": r.metadata,
        "fused_score": r.fused_score,
        "ranks": r.ranks,
        "rerank_score": r.rerank_score,
        "rerank_rank": r.rerank_rank,
    }


class StreamingPipeline:
    """Run retrieve → optional rerank → optional generate, yielding phase events.

    Skipping a phase is just passing `None` for its component:
    `StreamingPipeline(retriever)` emits retrieving/retrieved/done only;
    add `reranker=` to get reranking/reranked; add `token_stream=` to
    get generating/token/generated. Errors anywhere are caught and
    emitted as a final `error` event so an SSE client always sees a
    clean terminal frame.
    """

    # When a reranker is set, over-fetch from retrieval so the reranker
    # has more candidates than the final `k` to choose from. Matches the
    # multiplier baked into `Retriever.search` (#2).
    RERANK_OVERFETCH = 4

    def __init__(
        self,
        retriever: RetrieverLike,
        *,
        reranker: Reranker | None = None,
        token_stream: TokenStream | None = None,
        timings: PhaseTimings | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.token_stream = token_stream
        self.timings = timings

    def run(self, query: str, k: int = 5) -> Iterator[StreamEvent]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not query:
            raise ValueError("query must be non-empty")

        t0 = _now_ms()
        try:
            yield StreamEvent("retrieving", {"query": query, "k": k}, 0.0)

            over_fetch = k * self.RERANK_OVERFETCH if self.reranker else k
            t_ret_start = _now_ms()
            retrieved = self.retriever.search(query, k=over_fetch, reranker=None)
            t_ret_end = _now_ms()
            ret_ms = t_ret_end - t_ret_start
            if self.timings is not None:
                self.timings.record("retrieving", ret_ms)
            yield StreamEvent(
                "retrieved",
                {
                    "count": len(retrieved),
                    "chunks": [_chunk_to_event(r) for r in retrieved],
                    "phase_ms": ret_ms,
                },
                t_ret_end - t0,
            )

            if self.reranker is not None:
                yield StreamEvent(
                    "reranking",
                    {"candidates": len(retrieved)},
                    _now_ms() - t0,
                )
                t_rr_start = _now_ms()
                cands = [
                    Candidate(external_id=r.external_id, text=r.text, metadata=r.metadata)
                    for r in retrieved
                ]
                scored = self.reranker.rerank(query, cands)[:k]
                t_rr_end = _now_ms()
                rr_ms = t_rr_end - t_rr_start
                if self.timings is not None:
                    self.timings.record("reranking", rr_ms)
                base_by_id = {r.external_id: r for r in retrieved}
                final: list[RetrievalResult] = []
                for sc in scored:
                    base = base_by_id[sc.external_id]
                    final.append(
                        RetrievalResult(
                            external_id=sc.external_id,
                            text=sc.text,
                            metadata=sc.metadata,
                            fused_score=base.fused_score,
                            ranks=base.ranks,
                            rerank_score=sc.rerank_score,
                            rerank_rank=sc.rerank_rank,
                        )
                    )
                yield StreamEvent(
                    "reranked",
                    {
                        "count": len(final),
                        "chunks": [_chunk_to_event(r) for r in final],
                        "phase_ms": rr_ms,
                    },
                    t_rr_end - t0,
                )
            else:
                final = retrieved[:k]

            if self.token_stream is not None:
                yield StreamEvent(
                    "generating",
                    {"context_chunks": len(final)},
                    _now_ms() - t0,
                )
                t_gen_start = _now_ms()
                parts: list[str] = []
                for tok in self.token_stream(query, final):
                    parts.append(tok)
                    yield StreamEvent("token", {"text": tok}, _now_ms() - t0)
                t_gen_end = _now_ms()
                gen_ms = t_gen_end - t_gen_start
                if self.timings is not None:
                    self.timings.record("generating", gen_ms)
                yield StreamEvent(
                    "generated",
                    {"text": "".join(parts), "phase_ms": gen_ms},
                    t_gen_end - t0,
                )

            total_ms = _now_ms() - t0
            if self.timings is not None:
                self.timings.record("total", total_ms)
            yield StreamEvent("done", {"total_ms": total_ms}, total_ms)

        except Exception as e:
            yield StreamEvent(
                "error",
                {"message": str(e), "exception": type(e).__name__},
                _now_ms() - t0,
            )
            return


def to_sse(event: StreamEvent) -> str:
    """Format a `StreamEvent` as one Server-Sent Events frame.

    Per https://html.spec.whatwg.org/multipage/server-sent-events.html
    a frame is `event: <type>\\ndata: <json>\\n\\n`. The browser's
    `EventSource` parses this directly; for the JS-free demo we also
    accept plain `fetch()` and a streamed text decoder.
    """
    payload_obj = {"payload": event.payload, "elapsed_ms": event.elapsed_ms}
    data = json.dumps(payload_obj, default=str, ensure_ascii=False)
    return f"event: {event.type}\ndata: {data}\n\n"
