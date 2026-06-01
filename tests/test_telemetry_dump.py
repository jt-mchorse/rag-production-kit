"""Tests for ``Aggregate.to_dict`` and ``TelemetryStore.dump_aggregate_json`` (#50).

Mirrors the locks in the cost-optimizer repo's
``tests/test_cache_wrapper_dump.py`` / ``tests/test_semantic_cache_dump.py``
so the three observability surfaces (prompt-cache wrapper, semantic
cache, rag-kit telemetry) all share the same shape contract.

Coverage matrix:

- ``Aggregate.to_dict`` returns the seven dataclass fields exhaustively
  (field-set lock via ``dataclasses.fields`` catches a future field
  added without a serializer update).
- Round-trips through ``json.dumps``.
- Zero-state output: empty store → percentiles 0.0 (not NaN), totals 0,
  ``n=0`` — every key still present so consumers can read without a
  KeyError on the first observation.
- ``TelemetryStore.dump_aggregate_json``: on-disk shape with sorted
  keys + trailing newline; parent directory auto-create; atomic
  overwrite with no tempfile leftovers; ``since_ts`` window
  correctness (a record before the floor is excluded).
- Default-window behavior: a store with one fresh record and one
  >24h-old record → default dump includes only the fresh one.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from rag_kit.telemetry import (
    Aggregate,
    CostRecord,
    ModelPrice,
    PriceTable,
    TelemetryStore,
)


def _make_record(*, ts: float, latency_ms: float = 100.0, total_usd: float = 0.001) -> CostRecord:
    return CostRecord(
        ts=ts,
        query="q",
        model="m",
        retrieved_count=3,
        prompt_tokens=100,
        completion_tokens=50,
        prompt_usd=total_usd / 2,
        completion_usd=total_usd / 2,
        total_usd=total_usd,
        total_latency_ms=latency_ms,
        per_phase_ms={"retrieve": latency_ms / 2, "generate": latency_ms / 2},
    )


# --- Aggregate.to_dict -----------------------------------------------------


def test_to_dict_returns_full_field_set() -> None:
    """Field set must match the dataclass exactly. A future field
    added to Aggregate without to_dict updating would silently drop
    the new value from every observability sink — lock it loud."""
    a = Aggregate(
        n=10,
        total_prompt_tokens=1000,
        total_completion_tokens=500,
        total_usd=0.05,
        latency_p50_ms=120.0,
        latency_p95_ms=300.0,
        latency_p99_ms=500.0,
    )
    payload = a.to_dict()
    expected = {f.name for f in fields(a)}
    assert set(payload) == expected
    assert payload == {
        "n": 10,
        "total_prompt_tokens": 1000,
        "total_completion_tokens": 500,
        "total_usd": 0.05,
        "latency_p50_ms": 120.0,
        "latency_p95_ms": 300.0,
        "latency_p99_ms": 500.0,
    }


def test_to_dict_round_trips_through_json_dumps() -> None:
    """Every value must survive a ``json.dumps`` / ``json.loads`` cycle."""
    a = Aggregate(
        n=3,
        total_prompt_tokens=300,
        total_completion_tokens=120,
        total_usd=0.012,
        latency_p50_ms=80.0,
        latency_p95_ms=150.0,
        latency_p99_ms=180.0,
    )
    serialized = json.dumps(a.to_dict(), sort_keys=True)
    parsed = json.loads(serialized)
    assert parsed == a.to_dict()


def test_to_dict_on_empty_store_aggregate_is_well_defined(tmp_path: Path) -> None:
    """Empty store → ``aggregate([])`` returns Aggregate(n=0, ...). The
    ``to_dict()`` shape must still carry every key (so a consumer
    scanning ``payload["total_usd"]`` doesn't KeyError on the first
    observation) and percentiles must be ``0.0`` not NaN."""
    from rag_kit.telemetry import aggregate

    payload = aggregate([]).to_dict()
    assert payload == {
        "n": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_usd": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "latency_p99_ms": 0.0,
    }


# --- TelemetryStore.dump_aggregate_json ------------------------------------


def test_dump_aggregate_json_writes_file_with_aggregate_shape(tmp_path: Path) -> None:
    """Writer produces the dict shape on disk, sorted keys, trailing
    newline — the file is a self-contained JSON document a log-tailer
    can parse."""
    db = tmp_path / "tele.db"
    out = tmp_path / "agg.json"
    with TelemetryStore(db) as store:
        now = 1_700_000_000.0  # in-window for any reasonable since_ts
        store.record(_make_record(ts=now, latency_ms=100.0, total_usd=0.001))
        store.record(_make_record(ts=now + 1, latency_ms=200.0, total_usd=0.002))
        store.dump_aggregate_json(out, since_ts=now - 1)

    body = out.read_text(encoding="utf-8")
    assert body.endswith("\n")
    parsed_keys = list(json.loads(body))
    assert parsed_keys == sorted(parsed_keys), "keys must be sorted on disk"

    payload = json.loads(body)
    assert payload["n"] == 2
    assert payload["total_prompt_tokens"] == 200
    assert payload["total_completion_tokens"] == 100
    assert payload["total_usd"] == 0.003


def test_dump_aggregate_json_creates_parent_dirs(tmp_path: Path) -> None:
    """``atomic_write_text`` does ``parent.mkdir(parents=True)``;
    confirm the writer inherits that behavior."""
    db = tmp_path / "tele.db"
    out = tmp_path / "nested" / "sink" / "agg.json"
    with TelemetryStore(db) as store:
        store.dump_aggregate_json(out, since_ts=0.0)
    assert out.exists()
    assert out.parent.is_dir()


def test_dump_aggregate_json_overwrites_atomically(tmp_path: Path) -> None:
    """Two successive dumps leave the second payload — not the
    concatenation, not a half-written file. ``os.replace`` semantics
    make this atomic on POSIX; no tempfile leftovers in the parent."""
    db = tmp_path / "tele.db"
    out = tmp_path / "agg.json"
    with TelemetryStore(db) as store:
        store.dump_aggregate_json(out, since_ts=0.0)
        body1 = out.read_text(encoding="utf-8")

        store.record(_make_record(ts=1_700_000_000.0, latency_ms=100.0, total_usd=0.001))
        store.dump_aggregate_json(out, since_ts=0.0)
        body2 = out.read_text(encoding="utf-8")

    assert body1 != body2
    payload2 = json.loads(body2)
    assert payload2["n"] == 1
    # No tempfiles left in the parent.
    leftovers_tmp = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers_tmp == [], leftovers_tmp
    leftovers_dot = [p.name for p in tmp_path.iterdir() if p.name.startswith(".agg.json.")]
    assert leftovers_dot == [], leftovers_dot


def test_dump_aggregate_json_zero_state_writes_well_defined_shape(tmp_path: Path) -> None:
    """An empty store still produces a valid JSON document with every
    key present — useful for canary observability checks where the
    sink must always be parseable."""
    db = tmp_path / "tele.db"
    out = tmp_path / "agg.json"
    with TelemetryStore(db) as store:
        store.dump_aggregate_json(out, since_ts=0.0)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {
        "n": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_usd": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "latency_p99_ms": 0.0,
    }


def test_dump_aggregate_json_since_ts_filters_old_records(tmp_path: Path) -> None:
    """A record older than ``since_ts`` must be excluded from the
    dumped aggregate — the ``since_ts`` window controls which rows
    are summarized."""
    db = tmp_path / "tele.db"
    out = tmp_path / "agg.json"
    with TelemetryStore(db) as store:
        # One old record, one new — only the new should be in the window.
        old_ts = 1_000_000.0
        new_ts = 1_700_000_000.0
        store.record(_make_record(ts=old_ts, latency_ms=999.0, total_usd=0.999))
        store.record(_make_record(ts=new_ts, latency_ms=100.0, total_usd=0.001))
        store.dump_aggregate_json(out, since_ts=new_ts - 1)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n"] == 1
    assert payload["total_usd"] == 0.001  # the new record


def test_dump_aggregate_json_default_window_is_last_24h(tmp_path: Path) -> None:
    """When ``since_ts`` is omitted, the writer defaults to the last
    24 hours — the same window ``last_24h()`` returns and the
    dashboard renders. A record beyond 24h must be excluded."""
    db = tmp_path / "tele.db"
    out = tmp_path / "agg.json"
    with TelemetryStore(db) as store:
        import time

        now = time.time()
        old_ts = now - (25 * 3600)  # 25 hours ago — outside the default window
        fresh_ts = now - 600  # 10 minutes ago — inside the default window
        store.record(_make_record(ts=old_ts, latency_ms=200.0, total_usd=0.005))
        store.record(_make_record(ts=fresh_ts, latency_ms=100.0, total_usd=0.001))
        store.dump_aggregate_json(out)  # default since_ts

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n"] == 1, "default window should be last 24h"
    assert payload["total_usd"] == 0.001


# --- Backwards-compat: existing telemetry tests untouched ------------------


def test_module_still_exports_aggregate_function() -> None:
    """A regression check that the new methods didn't break the
    function-level ``aggregate`` export path used by
    ``scripts/telemetry_dashboard.py``."""
    from rag_kit.telemetry import aggregate

    # Smoke: call with empty input.
    result = aggregate([])
    assert isinstance(result, Aggregate)
    # And the price table API is still present (sanity check that
    # the import surface wasn't accidentally shuffled).
    assert isinstance(PriceTable({"foo": ModelPrice(1.0, 1.0)}), PriceTable)
