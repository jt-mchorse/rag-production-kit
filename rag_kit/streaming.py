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
import math
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .io_utils import atomic_write_text
from .reranker import Candidate, Reranker
from .retriever import RetrievalResult

EventType = Literal[
    "retrieving",  # phase start: retrieving from PG
    "retrieved",  # phase end: chunks + phase_ms
    "reranking",  # phase start (only if reranker is set)
    "reranked",  # phase end: chunks + phase_ms
    "generating",  # phase start (only if token_stream is set)
    "token",  # one token from the generator
    "generated",  # phase end: full text + phase_ms
    "done",  # whole pipeline finished cleanly
    "error",  # pipeline failed; payload has message + exception type
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
        # Finiteness guard (#63), parity with `CostRecord.build`'s
        # `total_latency_ms` check in `rag_kit.telemetry`: a NaN/Inf/negative
        # ms is otherwise appended silently and later poisons `percentile`
        # (sorting a list containing NaN is undefined — all NaN comparisons
        # are False), so the p50/p95/p99 read back silently wrong. Fail at the
        # ingestion site instead. `bool` is intentionally not rejected here —
        # the telemetry sibling accepts it (bool-is-int) and parity is the
        # whole point of this guard.
        if not math.isfinite(ms) or ms < 0:
            raise ValueError(f"ms must be a finite non-negative number; got {ms!r}")
        getattr(self, phase).append(ms)

    def percentile(self, phase: str, p: float) -> float | None:
        """Return p-th percentile of recorded ms for `phase`, or None if empty."""
        if phase not in self._PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        # Validate p at the boundary. Before this, NaN slipped both
        # `p <= 0` (NaN <= 0 is False) and `p >= 100` (NaN >= 100 is False),
        # reached `int(rank)` with rank=NaN, and surfaced a far-from-call-site
        # `ValueError: cannot convert float NaN to integer`. `True/False`
        # were silently interpreted as 1/0 percentiles via bool-is-int.
        # Out-of-range finite values continue to clamp (existing contract,
        # matches numpy's default) — see `test_phase_timings_percentile_clamps_edges`.
        if not isinstance(p, (int, float)) or isinstance(p, bool) or math.isnan(p):
            raise ValueError(f"p must be a finite number; got {p!r}")
        # Validate the *values* too, not just `p` (#168). `record` guards
        # finiteness at ingestion (#63), but this is a dataclass whose four
        # phase lists are public init fields, so `PhaseTimings(total=[...])`
        # — rebuilding timings from a persisted summary, merging across runs
        # via `combined.total.extend(other.total)` — never touches `record`,
        # and a later `pt.total.append(x)` bypasses it as well. That is the
        # same argument `telemetry.percentile` used to add its own guard in
        # #80 *despite* `CostRecord.build` already guarding at ingestion, and
        # that docstring names this method as the thing it must "agree on the
        # number" with. It didn't: `sorted()` leaves a NaN in an
        # implementation-defined slot (every NaN comparison is False), so the
        # same multiset in a different order returned a different percentile
        # (p50 of 20.0 / 40.0 / 20.0 for three orderings of one sample), and
        # a +Inf silently became the maximum and egressed through
        # `summary()` -> `to_dict()` -> `dump_summary_json` as the bare token
        # `Infinity` — invalid JSON a strict log-tailer rejects whole.
        # Read-boundary, not `__post_init__`: only this side sees an append.
        raw = getattr(self, phase)
        if any(not math.isfinite(v) for v in raw):
            raise ValueError(f"values must all be finite numbers; got {list(raw)!r}")
        values = sorted(raw)
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

    def to_dict(self) -> dict[str, dict[str, float | int | None]]:
        """JSON-stable dict for observability/logging sinks (#58).

        Canonical alias for `summary()` under the portfolio's
        observability surface name. Mirrors `Aggregate.to_dict` (#50)
        in this repo and the runtime trio in llm-cost-optimizer
        (CacheTelemetry #50, CacheStats #52, RouterStats #62). Pairs
        with `dump_summary_json` for the on-disk path; metric backends
        like statsd/prometheus consume the in-process dict directly.
        """
        return self.summary()

    def dump_summary_json(self, path: str | Path) -> None:
        """Write the current per-phase summary to ``path`` as JSON.

        Atomic on POSIX — uses ``rag_kit.io_utils.atomic_write_text``
        so a Ctrl-C / disk-full / OOM between truncate and flush can't
        leave a log-tailer reading a half-written file. Byte-shape
        parity with `TelemetryStore.dump_aggregate_json` (#50) and the
        llm-cost-optimizer runtime trio: sorted keys, indent=2,
        trailing newline. Operators can tail / diff the file across
        restarts.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"
        atomic_write_text(path, payload)


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
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
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


_UNREPRESENTABLE = "\ufffd"
"""U+FFFD REPLACEMENT CHARACTER, substituted for text with no UTF-8 encoding.

Deliberately the *opposite* call from `llm-eval-harness#215`, which rejects an
unencodable input outright (D-017). That seam writes a file that has to be
faithful, and there is no faithful spelling of a lone surrogate to write. This
seam's documented contract is "stream alive, don't raise", and a replacement
character in one metadata field beats a torn connection with no diagnostic.
"""


def _safe_text(text: str) -> str:
    """Return *text* with any character that has no UTF-8 encoding replaced.

    A lone surrogate is legal JSON escape syntax and Python decodes it happily,
    but it has no UTF-8 encoding -- so `json.dumps(..., ensure_ascii=False)`
    produces a `str` that looks fine and then dies at
    `to_sse(event).encode("utf-8")` in `demo/streaming/server.py`, *after* the
    200 and the headers have gone out. The client sees a truncated
    `text/event-stream` with no `error` and no `done` frame, which is
    byte-indistinguishable from a network drop (#188).

    The fast path allocates nothing: the overwhelming majority of frames are
    already encodable and return the same object. Only a string that actually
    fails is rebuilt, character by character, so a single bad codepoint costs
    the surrounding text nothing.

    Built explicitly rather than via `errors="replace"`, which substitutes
    `"?"` on the *encode* side -- U+FFFD is only what the *decode* side
    produces. `"?"` is a character a caller can legitimately have written, so
    it would make a substitution indistinguishable from real data.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return "".join(ch if _encodable(ch) else _UNREPRESENTABLE for ch in text)
    return text


def _encodable(ch: str) -> bool:
    try:
        ch.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _safe_key(key: Any) -> str:
    """Return the JSON object name `json.dumps` should emit for *key*.

    `json.dumps` coerces `int` / `float` / `bool` / `None` keys itself and
    raises `TypeError: keys must be str, int, float, bool or None` on anything
    else -- *before* consulting `default=`, which is only ever called for
    values. So `default=str` cannot rescue a key, and a `tuple` or `frozenset`
    key raised straight out of `to_sse` (#188).

    Doing the coercion here rather than leaving it to `json.dumps` buys two
    things beyond not raising:

    - non-finite floats are handled at the key position too. `_json_safe` maps
      them to `null` as *values* (#106) and passed them through as *keys*, so
      `{float("inf"): "x"}` reached the wire as the string key `"Infinity"`.
    - a coerced collision resolves to one key. `{1: "a", "1": "b"}` is two
      entries in Python and one JSON name, and `json.dumps` emitted **both**:
      `{"1": "a", "1": "b"}`. RFC 8259 leaves duplicate names undefined and
      `JSON.parse` keeps the last, so an entry vanished with no diagnostic.
      Building the dict here keeps the wire semantics identical (last-in wins)
      while making the frame well-defined JSON.
    """
    if isinstance(key, str):
        return _safe_text(key)
    # `bool` before `int`, which it subclasses, so `True` stays `"true"` rather
    # than becoming `"1"` -- json.dumps' own spelling, preserved.
    if key is True:
        return "true"
    if key is False:
        return "false"
    if key is None:
        return "null"
    if isinstance(key, int):
        return str(key)
    if isinstance(key, float):
        # Same rule the value position already applies (#106): a non-finite
        # float has no JSON spelling. `null` is what `JSON.stringify` produces
        # and what `_json_safe` already puts in the value position.
        return "null" if not math.isfinite(key) else repr(key)
    return _safe_text(str(key))


_CIRCULAR = "<circular reference>"

_MAX_DEPTH = 50
"""Deepest nesting `_json_safe` will reproduce; below it, the subtree is a marker.

Making `_json_safe` iterative stopped *this* function from blowing the stack,
but `json.dumps` is still recursive under the hood -- `to_sse` passes
`default=str`, which disqualifies the C encoder and selects the pure-Python
`_make_iterencode`. And *where* that gives out is a property of the interpreter,
not of this code:

    CPython 3.14 (recursionlimit 1000)   handles ~14690 levels
    CPython 3.11 (recursionlimit 1000)   raised RecursionError at 3000

Python 3.12 decoupled pure-Python frames from the C stack, so a depth that
serializes locally can fail on an older runner. A guarantee that "every frame
parses" cannot be conditional on which Python is running it, so the limit is
pinned here instead of inherited. 50 is far below the smallest observed
interpreter limit and far above any real payload -- the event schema is three or
four levels deep and `metadata` is free-form but not a tree.

Beyond it the subtree becomes a marker, exactly as a cycle does, for the same
reason: this seam's contract is "stream alive, don't raise" (D-017).
"""

_TOO_DEEP = f"<nesting deeper than {_MAX_DEPTH} levels>"


def _json_safe(obj: Any) -> Any:
    """Return a copy of *obj* that `json.dumps` can always render as valid JSON.

    Three rules, all enforced at every depth and at both the key and the value
    position:

    - **non-finite floats become `null`.** `json.dumps` defaults to
      `allow_nan=True`, emitting the bare tokens `NaN` / `Infinity` /
      `-Infinity`, which are **invalid JSON**, so a browser's `EventSource`
      (which runs `JSON.parse` on the `data:` line) rejects the whole frame.
      `null` matches JavaScript's own `JSON.stringify(NaN)` (#106).
    - **keys are coerced to the name `json.dumps` would emit**, so a key type
      it rejects cannot raise and a coerced collision cannot put a duplicate
      name on the wire. See `_safe_key` (#188).
    - **text with no UTF-8 encoding is replaced**, so the frame survives
      `.encode("utf-8")` at the write seam. See `_safe_text` (#188, D-017).

    `metadata` / `rerank_score` flow verbatim from free-form caller data --
    `RetrieverLike` is a documented Protocol and a caller supplying its own
    `RetrievalResult`s controls `metadata` entirely -- so the wire serializer is
    the correct single chokepoint to enforce frame validity.

    **Iterative, with cycle detection, on purpose.** The recursive version was
    strictly less robust than the `json.dumps` call it exists to protect:

        json.dumps alone, circular dict   -> ValueError: Circular reference detected
        recursive _json_safe, circular    -> RecursionError

    So a helper added to *guarantee* frame validity blew the Python stack before
    `json.dumps` was reached, and turned a diagnosable named error into an
    opaque one. Depth is handled by `_MAX_DEPTH` rather than by out-recursing
    `json.dumps`, because how deep `json.dumps` itself can go is a property of
    the interpreter version, not of this module. Same reason `llm-eval-harness#213` and `llm-cost-optimizer#192`
    walk iteratively. A back-reference becomes `_CIRCULAR` rather than raising,
    because this seam's contract is "stream alive, don't raise" -- the cycle is
    a caller bug, and naming it in the frame tells the operator far more than a
    torn connection does.

    `to_sse` is total: for any input, it returns a string that `json.loads`
    accepts and that `.encode("utf-8")` accepts. `tests/test_sse_frame_totality.py`
    runs that property over a table rather than restating it in prose.
    """
    root, container = _new_container(obj)
    if container is None:
        return root
    # (source node, destination container, ancestor ids on the path to it)
    stack: list[tuple[Any, Any, frozenset[int]]] = [(obj, root, frozenset({id(obj)}))]
    while stack:
        src, dst, ancestors = stack.pop()
        items = src.items() if isinstance(src, dict) else enumerate(src)
        for raw_key, value in items:
            key = _safe_key(raw_key) if isinstance(src, dict) else raw_key
            if isinstance(value, (dict, list, tuple)) and id(value) in ancestors:
                _place(dst, key, _CIRCULAR)
                continue
            if isinstance(value, (dict, list, tuple)) and len(ancestors) >= _MAX_DEPTH:
                # `json.dumps` is recursive below us and gives out at a
                # version-dependent depth (see `_MAX_DEPTH`). Truncate here so
                # the guarantee is ours rather than the interpreter's.
                _place(dst, key, _TOO_DEEP)
                continue
            child, child_container = _new_container(value)
            _place(dst, key, child)
            if child_container is not None:
                stack.append((value, child, ancestors | {id(value)}))
    return root


def _new_container(value: Any) -> tuple[Any, Any]:
    """Return ``(node, container_or_None)`` -- the scalar, or an empty shell to fill."""
    if isinstance(value, float):
        return (value if math.isfinite(value) else None), None
    if isinstance(value, str):
        return _safe_text(value), None
    if isinstance(value, dict):
        shell: Any = {}
        return shell, shell
    if isinstance(value, (list, tuple)):
        # tuple -> list, matching what `json.dumps` does anyway.
        shell = []
        return shell, shell
    return value, None


def _place(dst: Any, key: Any, value: Any) -> None:
    if isinstance(dst, list):
        dst.append(value)
    else:
        dst[key] = value


def to_sse(event: StreamEvent) -> str:
    """Format a `StreamEvent` as one Server-Sent Events frame.

    Per https://html.spec.whatwg.org/multipage/server-sent-events.html
    a frame is `event: <type>\\ndata: <json>\\n\\n`. The browser's
    `EventSource` parses this directly; for the JS-free demo we also
    accept plain `fetch()` and a streamed text decoder.

    Non-finite floats anywhere in the payload are mapped to JSON ``null`` by
    `_json_safe` so the emitted frame is always valid JSON (#106); everything
    else, including the `default=str` fallback for unjsonifiable objects, is
    unchanged.
    """
    payload_obj = {"payload": event.payload, "elapsed_ms": event.elapsed_ms}
    data = json.dumps(_json_safe(payload_obj), default=str, ensure_ascii=False)
    return f"event: {event.type}\ndata: {data}\n\n"
