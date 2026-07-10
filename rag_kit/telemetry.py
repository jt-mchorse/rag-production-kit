"""Cost telemetry: per-request token / dollar / latency capture + a small SQLite store.

Three pieces in this module:

- ``CostRecord`` — the per-request blob (timestamp, query, model, token
  counts, USD breakdown, total latency, per-phase latency dict).
- ``PriceTable`` — operator-supplied mapping of model id →
  ``(prompt_per_million, completion_per_million)`` USD prices. **No
  defaults are shipped** (D-015); querying an unconfigured model
  raises ``UnknownModelError``. Same rationale as D-013 — the
  no-fabricated-benchmarks rule extends to no-fabricated-prices.
- ``TelemetryStore`` — thin stdlib ``sqlite3`` wrapper, schema below.
  Single file, no daemon, dep-free; consistent with D-002 (only
  required runtime dep is ``psycopg``; everything else opt-in).

Aggregate latency percentiles use the same linear-interp NIST type-7
math as ``PhaseTimings`` in ``rag_kit.streaming`` so the two layers
report comparable numbers when the operator looks at a 24-hour window.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_kit.io_utils import atomic_write_text

# ----------------------------------------------------------------------
# Price table
# ----------------------------------------------------------------------


class UnknownModelError(KeyError):
    """Raised by ``PriceTable.cost`` when the model id is not configured.

    Inherits ``KeyError`` for backwards compatibility with callers that
    only catch ``KeyError``; the explicit subclass lets newer callers
    pattern-match on the more specific type.
    """


@dataclass(frozen=True)
class ModelPrice:
    """USD-per-million-tokens for one model."""

    prompt_per_million: float
    completion_per_million: float

    def __post_init__(self) -> None:
        # D-015 prevents silent-zero via UnknownModelError; this guard extends
        # the same posture to silent-negative and silent-NaN/Infinity (#38).
        # A NaN rate propagates through cost() into CostRecord.total_usd = NaN,
        # which sums NaN through aggregate() and renders as "NaN" on the cost
        # dashboard — same shape of silent corruption as the silent-zero case
        # D-015 addressed, just one layer of arithmetic later. Tightened from
        # sign-only to finiteness mirroring the portfolio's contract-tightening
        # sweep (llm-cost-optimizer#37, llm-eval-harness#43, etc).
        for name, value in (
            ("prompt_per_million", self.prompt_per_million),
            ("completion_per_million", self.completion_per_million),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite number >= 0.0; got {value}")

    def cost(self, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
        """Return ``(prompt_usd, completion_usd)`` rounded to 6 decimal places."""
        # Non-negative *integer* counts, not sign-only: a sign-only check lets a
        # NaN through (`NaN < 0` is False), which then propagates
        # `NaN * rate / 1_000_000` → prompt_usd → CostRecord.total_usd →
        # aggregate().total_usd, and `dump_aggregate_json` serializes it as the
        # bare token `NaN` — invalid JSON that a strict log-tailer rejects whole.
        # Same silent-corruption shape the rate guard above (#38) and percentile
        # (#80) already close; this is the token-count seam that sweep missed.
        # Reject bool too (it is an int subclass). Mirrors the dim/max_chunks/k
        # non-negative-int contracts elsewhere in the kit.
        for name, value in (("prompt", prompt_tokens), ("completion", completion_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"token counts must be non-negative integers; got {name}={value!r}"
                )
        prompt_usd = round(prompt_tokens * self.prompt_per_million / 1_000_000, 6)
        completion_usd = round(completion_tokens * self.completion_per_million / 1_000_000, 6)
        return prompt_usd, completion_usd


class PriceTable:
    """Operator-supplied mapping of model id → :class:`ModelPrice`.

    No default entries (D-015). Public-list prices change frequently
    and the repo can't be the source of truth for a downstream
    deployment's actual contract. Configure explicitly:

        prices = PriceTable({
            "claude-opus-4-7":  ModelPrice(15.0, 75.0),  # operator values, not shipped defaults
            "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
        })
    """

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None) -> None:
        self._prices: dict[str, ModelPrice] = dict(prices or {})

    def add(self, model: str, prompt_per_million: float, completion_per_million: float) -> None:
        self._prices[model] = ModelPrice(prompt_per_million, completion_per_million)

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
        """Return ``(prompt_usd, completion_usd)``. Raises ``UnknownModelError`` for unconfigured models."""
        try:
            entry = self._prices[model]
        except KeyError as e:
            raise UnknownModelError(
                f"no price configured for model {model!r}; call PriceTable.add() "
                "or pass an entry to PriceTable()"
            ) from e
        return entry.cost(prompt_tokens, completion_tokens)

    def known_models(self) -> tuple[str, ...]:
        return tuple(self._prices.keys())


# ----------------------------------------------------------------------
# Per-request record
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CostRecord:
    """One per-request telemetry blob.

    Latency fields are floats in milliseconds. ``per_phase_ms`` is the
    phase-name → ms mapping used by ``rag_kit.streaming.PhaseTimings``
    so a downstream consumer can chart per-phase latency over a window
    without an extra schema.
    """

    ts: float  # seconds since epoch, UTC
    query: str
    model: str
    retrieved_count: int
    prompt_tokens: int
    completion_tokens: int
    prompt_usd: float
    completion_usd: float
    total_usd: float
    total_latency_ms: float
    per_phase_ms: Mapping[str, float] = field(default_factory=dict)

    @staticmethod
    def build(
        *,
        ts: float | None,
        query: str,
        model: str,
        retrieved_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_latency_ms: float,
        per_phase_ms: Mapping[str, float] | None,
        price_table: PriceTable,
    ) -> CostRecord:
        """Build a record from raw inputs, computing USD from the price table.

        Raises ``UnknownModelError`` if the model is not in the price table.
        ``ts`` defaults to ``time.time()`` when ``None`` (test injection point).
        """
        # Finiteness guard (#38): NaN latency propagates through percentile()
        # which sorts a list with NaN — Python's sort is stable but NaN
        # comparisons are all false, so the returned percentile is implementation-
        # defined and silently wrong. Mirrors the ModelPrice guard above.
        if not math.isfinite(total_latency_ms) or total_latency_ms < 0:
            raise ValueError(
                f"total_latency_ms must be a finite non-negative number; got {total_latency_ms}"
            )
        # Per-phase value guard (#108): the same finiteness/sign contract as
        # total_latency_ms above, applied to each per_phase_ms value. Without
        # it a non-finite phase value reaches TelemetryStore.record, which
        # persists the map via json.dumps(... allow_nan=True) — writing the
        # bare token NaN/Infinity (invalid JSON). since() then swallows the
        # whole row's phases on JSONDecodeError, silently dropping the data.
        # bool is an int subclass; reject it so a stray True/False can't pose
        # as a millisecond value.
        for phase, value in (per_phase_ms or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"per_phase_ms[{phase!r}] must be a finite non-negative number; got {value!r}"
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"per_phase_ms[{phase!r}] must be a finite non-negative number; got {value}"
                )
        prompt_usd, completion_usd = price_table.cost(model, prompt_tokens, completion_tokens)
        return CostRecord(
            ts=ts if ts is not None else time.time(),
            query=query,
            model=model,
            retrieved_count=retrieved_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_usd=prompt_usd,
            completion_usd=completion_usd,
            total_usd=round(prompt_usd + completion_usd, 6),
            total_latency_ms=total_latency_ms,
            per_phase_ms=dict(per_phase_ms or {}),
        )


# ----------------------------------------------------------------------
# SQLite store
# ----------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cost_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    model TEXT NOT NULL,
    retrieved_count INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    prompt_usd REAL NOT NULL,
    completion_usd REAL NOT NULL,
    total_usd REAL NOT NULL,
    total_latency_ms REAL NOT NULL,
    per_phase_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cost_records_ts_idx ON cost_records (ts);
"""


class TelemetryStore:
    """Thin SQLite wrapper for ``CostRecord`` persistence.

    Schema is owned by this module — there's no migration story, just a
    single table that grows. For demo / dashboard purposes that's fine;
    a production deployment with retention requirements can build on
    top by writing to a managed store and using this as a reference.

    The store is not safe for concurrent writers (SQLite locks the file
    on write); that's an acceptable simplification for a single-process
    demo. A multi-process production setup should use a dedicated
    metrics backend.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TelemetryStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def record(self, rec: CostRecord) -> int:
        """Insert one record; return its assigned id."""
        cur = self._conn.execute(
            """
            INSERT INTO cost_records (
                ts, query, model, retrieved_count, prompt_tokens, completion_tokens,
                prompt_usd, completion_usd, total_usd, total_latency_ms, per_phase_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.ts,
                rec.query,
                rec.model,
                rec.retrieved_count,
                rec.prompt_tokens,
                rec.completion_tokens,
                rec.prompt_usd,
                rec.completion_usd,
                rec.total_usd,
                rec.total_latency_ms,
                json.dumps(dict(rec.per_phase_ms)),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def since(self, ts_floor: float) -> list[CostRecord]:
        """Return all records with ``ts >= ts_floor`` in chronological order."""
        cur = self._conn.execute(
            """
            SELECT ts, query, model, retrieved_count, prompt_tokens, completion_tokens,
                   prompt_usd, completion_usd, total_usd, total_latency_ms, per_phase_json
            FROM cost_records
            WHERE ts >= ?
            ORDER BY ts ASC
            """,
            (ts_floor,),
        )
        rows = cur.fetchall()
        out: list[CostRecord] = []
        for row in rows:
            ts, query, model, n_retr, p_tok, c_tok, p_usd, c_usd, tot_usd, lat_ms, phase_json = row
            try:
                phases = json.loads(phase_json) if phase_json else {}
            except json.JSONDecodeError:
                phases = {}
            out.append(
                CostRecord(
                    ts=ts,
                    query=query,
                    model=model,
                    retrieved_count=n_retr,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    prompt_usd=p_usd,
                    completion_usd=c_usd,
                    total_usd=tot_usd,
                    total_latency_ms=lat_ms,
                    per_phase_ms=phases,
                )
            )
        return out

    def last_24h(self, now: float | None = None) -> list[CostRecord]:
        cutoff = (now if now is not None else time.time()) - 24 * 3600
        return self.since(cutoff)

    def dump_aggregate_json(self, path: str | Path, *, since_ts: float | None = None) -> None:
        """Write the current window aggregate to ``path`` as JSON (#50).

        Atomic on POSIX via ``rag_kit.io_utils.atomic_write_text`` —
        the same helper the eval-action's composite sticky comment
        relies on (#44) — so a SIGINT / disk-full / OOM between
        truncate and flush can't leave the log-tailer (or the
        ``scripts/telemetry_dashboard.py`` build artifact) reading a
        half-written file.

        ``since_ts`` defaults to last-24h (the same window
        ``last_24h()`` returns and the dashboard renders), so an
        operator wiring a cron job that simply writes
        ``telemetry.json`` every minute gets the rolling 24-hour
        aggregate by default. Pass an explicit ``since_ts`` to widen
        or narrow the window.

        On-disk shape is ``Aggregate.to_dict()`` with sorted keys,
        indent=2, trailing newline — byte-shape parity with the
        cost-optimizer's ``dump_aggregate_json`` and
        ``dump_stats_json`` so one log-parsing config consumes all
        portfolio observability artifacts.
        """
        floor = since_ts if since_ts is not None else (time.time() - 24 * 3600)
        records = self.since(floor)
        agg = aggregate(records)
        payload = json.dumps(agg.to_dict(), sort_keys=True, indent=2) + "\n"
        atomic_write_text(path, payload)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """Window aggregate: counts, totals, and latency percentiles."""

    n: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_usd: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-stable dict for observability/logging sinks (#50).

        Mirrors the pattern set by ``CacheTelemetry.to_dict`` in
        ``llm-cost-optimizer`` (#50 there) and ``CacheStats.to_dict``
        (#52 there): one stable-keys dict per long-lived state object,
        consumed by metric backends or written to disk via
        ``TelemetryStore.dump_aggregate_json``. The seven fields match
        the dataclass exactly so a future addition without serializer
        update is caught loud by the field-set lock test.
        """
        return {
            "n": self.n,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_usd": self.total_usd,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }


def percentile(values: Sequence[float], q: float) -> float:
    """NIST type-7 (linear interp) percentile, no numpy dependency.

    Matches `rag_kit.streaming.PhaseTimings.percentile` so a 24-hour
    aggregate window and a streaming-pipeline snapshot agree on the
    number when given the same sample. Edges (q=0, q=1) clamp.

    A non-finite value (NaN / +/-Infinity) is rejected (#80). `CostRecord.build`
    already guards `total_latency_ms` finiteness at ingestion (#38), but a
    record constructed directly (the dataclass has no `__post_init__`) bypasses
    it, and this function is exported on its own. Unguarded, `sorted()` leaves
    the NaN in an implementation-defined slot (all NaN comparisons are False),
    so the returned percentile is silently wrong and position-dependent, and
    `dump_aggregate_json` then serializes the `nan` as the bare token `NaN` —
    invalid JSON a strict log-tailer rejects whole. Fail loud at the metric
    boundary, the same posture as the q-range guard below and `PhaseTimings.record`.
    """
    if not values:
        return 0.0
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"values must all be finite numbers; got {list(values)!r}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0.0, 1.0]; got {q}")
    s = sorted(values)
    if q == 0.0:
        return s[0]
    if q == 1.0:
        return s[-1]
    idx = q * (len(s) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return s[int(idx)]
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def aggregate(records: Iterable[CostRecord]) -> Aggregate:
    rs = list(records)
    if not rs:
        return Aggregate(
            n=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_usd=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
        )
    latencies = [r.total_latency_ms for r in rs]
    costs = [r.total_usd for r in rs]
    # `total_usd` fails loud at the metric boundary, the cost sibling of the
    # `total_latency_ms` percentile guard below (#80). `CostRecord.build` guards
    # cost finiteness at ingestion (#38/#58), but a record constructed directly
    # (the dataclass has no `__post_init__`) bypasses it, and a ±Infinity survives
    # the store round-trip — SQLite's `total_usd REAL NOT NULL` only rejects NaN
    # (stored as NULL), it passes ±Infinity through `record()`/`since()` to here.
    # Unguarded, `dump_aggregate_json` then serializes the non-finite sum as the
    # bare token `Infinity` / `NaN` (json.dumps defaults to allow_nan=True) —
    # invalid JSON that breaks the "one log-parsing config consumes all portfolio
    # observability artifacts" contract. Reject it here rather than emit an
    # unparseable artifact, the same posture as the latency guard (#80); #135's
    # dashboard `/json` egress sibling sanitizes at the presentation boundary.
    if any(not math.isfinite(c) for c in costs):
        raise ValueError(f"total_usd values must all be finite numbers; got {costs!r}")
    return Aggregate(
        n=len(rs),
        total_prompt_tokens=sum(r.prompt_tokens for r in rs),
        total_completion_tokens=sum(r.completion_tokens for r in rs),
        total_usd=round(sum(costs), 6),
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
        latency_p99_ms=percentile(latencies, 0.99),
    )


__all__ = [
    "Aggregate",
    "CostRecord",
    "ModelPrice",
    "PriceTable",
    "TelemetryStore",
    "UnknownModelError",
    "aggregate",
    "percentile",
]
