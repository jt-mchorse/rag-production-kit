"""Tests for the cost telemetry layer (#6, D-015).

Three concerns:
1. ``PriceTable`` math is correct on known inputs; missing-model raises;
   no defaults ship.
2. ``CostRecord.build`` composes prompt+completion USD into a total
   that round-trips through SQLite via ``TelemetryStore`` losslessly.
3. ``aggregate`` percentile math matches NIST type-7 known values
   (same shape as ``rag_kit.streaming.PhaseTimings.percentile``) so a
   24-hour aggregate view and a streaming-pipeline snapshot agree.
"""

from __future__ import annotations

import math

import pytest

from rag_kit.telemetry import (
    Aggregate,
    CostRecord,
    ModelPrice,
    PriceTable,
    TelemetryStore,
    UnknownModelError,
    aggregate,
    percentile,
)

# ----------------------------------------------------------------------
# PriceTable
# ----------------------------------------------------------------------


def _fixture_prices() -> PriceTable:
    """Fixture prices — explicitly synthetic; not real list prices."""
    return PriceTable(
        {
            "fake-big": ModelPrice(prompt_per_million=10.0, completion_per_million=40.0),
            "fake-small": ModelPrice(prompt_per_million=1.0, completion_per_million=4.0),
        }
    )


def test_price_table_ships_no_defaults():
    """D-015: empty constructor means empty table."""
    pt = PriceTable()
    assert pt.known_models() == ()


def test_price_table_cost_known_model_math():
    pt = _fixture_prices()
    prompt_usd, completion_usd = pt.cost("fake-big", 1_000_000, 500_000)
    assert prompt_usd == pytest.approx(10.0)
    assert completion_usd == pytest.approx(20.0)


def test_price_table_cost_rounds_to_6_decimals():
    pt = _fixture_prices()
    prompt_usd, _ = pt.cost("fake-small", 1, 0)
    assert prompt_usd == pytest.approx(0.000001)


def test_price_table_cost_unknown_model_raises_unknown_model_error():
    pt = _fixture_prices()
    with pytest.raises(UnknownModelError, match="fake-missing"):
        pt.cost("fake-missing", 100, 100)
    # And it remains a KeyError for backwards-compatible callers.
    with pytest.raises(KeyError):
        pt.cost("fake-missing", 100, 100)


def test_price_table_cost_rejects_negative_tokens():
    pt = _fixture_prices()
    with pytest.raises(ValueError, match="non-negative"):
        pt.cost("fake-big", -1, 10)


def test_price_table_add_overrides_existing():
    pt = _fixture_prices()
    pt.add("fake-big", 5.0, 20.0)  # overwrite
    prompt_usd, _ = pt.cost("fake-big", 1_000_000, 0)
    assert prompt_usd == pytest.approx(5.0)


# ----------------------------------------------------------------------
# CostRecord.build
# ----------------------------------------------------------------------


def test_costrecord_build_composes_total_usd():
    pt = _fixture_prices()
    rec = CostRecord.build(
        ts=1_700_000_000.0,
        query="hello",
        model="fake-big",
        retrieved_count=3,
        prompt_tokens=2_000_000,
        completion_tokens=1_000_000,
        total_latency_ms=420.0,
        per_phase_ms={"retrieving": 50.0, "generating": 370.0},
        price_table=pt,
    )
    assert rec.prompt_usd == pytest.approx(20.0)
    assert rec.completion_usd == pytest.approx(40.0)
    assert rec.total_usd == pytest.approx(60.0)


def test_costrecord_build_uses_now_when_ts_is_none(monkeypatch):
    monkeypatch.setattr("rag_kit.telemetry.time.time", lambda: 1_700_000_999.0)
    pt = _fixture_prices()
    rec = CostRecord.build(
        ts=None,
        query="q",
        model="fake-small",
        retrieved_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_latency_ms=12.5,
        per_phase_ms=None,
        price_table=pt,
    )
    assert rec.ts == 1_700_000_999.0


def test_costrecord_build_unknown_model_propagates():
    pt = _fixture_prices()
    with pytest.raises(UnknownModelError):
        CostRecord.build(
            ts=0.0,
            query="q",
            model="not-configured",
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_latency_ms=12.5,
            per_phase_ms=None,
            price_table=pt,
        )


def test_costrecord_build_rejects_negative_latency():
    pt = _fixture_prices()
    with pytest.raises(ValueError, match="total_latency_ms"):
        CostRecord.build(
            ts=0.0,
            query="q",
            model="fake-big",
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_latency_ms=-1.0,
            per_phase_ms=None,
            price_table=pt,
        )


# ----------------------------------------------------------------------
# TelemetryStore (SQLite)
# ----------------------------------------------------------------------


def _rec(ts: float, lat_ms: float = 100.0, model: str = "fake-big") -> CostRecord:
    return CostRecord.build(
        ts=ts,
        query=f"q@{ts}",
        model=model,
        retrieved_count=3,
        prompt_tokens=1000,
        completion_tokens=200,
        total_latency_ms=lat_ms,
        per_phase_ms={"retrieving": lat_ms * 0.1, "generating": lat_ms * 0.9},
        price_table=_fixture_prices(),
    )


def test_telemetry_store_roundtrip_record(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry.db")
    rec = _rec(ts=1_700_000_000.0)
    rid = store.record(rec)
    assert rid > 0
    got = store.since(1_699_999_999.0)
    assert len(got) == 1
    g = got[0]
    assert g.ts == rec.ts
    assert g.query == rec.query
    assert g.model == rec.model
    assert g.prompt_usd == pytest.approx(rec.prompt_usd)
    assert g.total_usd == pytest.approx(rec.total_usd)
    assert g.per_phase_ms == rec.per_phase_ms
    store.close()


def test_telemetry_store_since_filters_by_ts(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry.db")
    for ts in (100.0, 200.0, 300.0):
        store.record(_rec(ts=ts))
    got = store.since(200.0)
    assert [r.ts for r in got] == [200.0, 300.0]


def test_telemetry_store_orders_chronologically(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry.db")
    store.record(_rec(ts=300.0))
    store.record(_rec(ts=100.0))
    store.record(_rec(ts=200.0))
    got = store.since(0.0)
    assert [r.ts for r in got] == [100.0, 200.0, 300.0]


def test_telemetry_store_last_24h_uses_injected_now(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry.db")
    store.record(_rec(ts=1_000_000.0))  # older than 24h relative to `now=now_synth`
    store.record(_rec(ts=2_000_000.0))  # within 24h
    now_synth = 2_000_000.0 + 3600  # 1h after the second record
    got = store.last_24h(now=now_synth)
    assert [r.ts for r in got] == [2_000_000.0]


def test_telemetry_store_context_manager(tmp_path):
    with TelemetryStore(tmp_path / "telemetry.db") as store:
        store.record(_rec(ts=1.0))
        assert len(store.since(0.0)) == 1


# ----------------------------------------------------------------------
# percentile + aggregate
# ----------------------------------------------------------------------


def test_percentile_nist_type_7_known_values():
    # Classic NIST example: sorted [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], q=0.5 → 5.5.
    data = list(range(1, 11))
    assert percentile(data, 0.5) == pytest.approx(5.5)
    assert percentile(data, 0.95) == pytest.approx(9.55)
    assert percentile(data, 0.99) == pytest.approx(9.91)


def test_percentile_edge_cases():
    assert percentile([], 0.5) == 0.0
    assert percentile([42.0], 0.5) == 42.0
    assert percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 1.0) == 3.0


def test_percentile_rejects_out_of_range_q():
    with pytest.raises(ValueError, match="0.0, 1.0"):
        percentile([1.0, 2.0], -0.1)
    with pytest.raises(ValueError, match="0.0, 1.0"):
        percentile([1.0, 2.0], 1.5)


def test_aggregate_empty_returns_zeros():
    agg = aggregate([])
    assert agg == Aggregate(0, 0, 0, 0.0, 0.0, 0.0, 0.0)


def test_aggregate_totals_and_percentiles():
    pt = _fixture_prices()
    # 10 records with deterministic latencies 1..10 ms, identical USD per record.
    recs = [
        CostRecord.build(
            ts=float(i),
            query=f"q{i}",
            model="fake-big",
            retrieved_count=1,
            prompt_tokens=100,
            completion_tokens=50,
            total_latency_ms=float(i),
            per_phase_ms=None,
            price_table=pt,
        )
        for i in range(1, 11)
    ]
    agg = aggregate(recs)
    assert agg.n == 10
    assert agg.total_prompt_tokens == 1000
    assert agg.total_completion_tokens == 500
    # 100 * 10 USD/M + 50 * 40 USD/M, times 10 records.
    expected_per_record = round(100 * 10 / 1_000_000 + 50 * 40 / 1_000_000, 6)
    assert agg.total_usd == pytest.approx(round(expected_per_record * 10, 6))
    assert agg.latency_p50_ms == pytest.approx(5.5)
    assert agg.latency_p95_ms == pytest.approx(9.55)
    assert agg.latency_p99_ms == pytest.approx(9.91)


def test_aggregate_matches_streaming_phase_timings_percentile():
    """`telemetry.percentile` must agree with `PhaseTimings.percentile` on identical input.

    `PhaseTimings.percentile(phase, p)` takes ``p`` in [0, 100] and operates on
    ms directly; the math is the same NIST type-7 linear-interp as
    ``telemetry.percentile(values, q)`` where ``q`` is in [0, 1]. With matched
    inputs they must return the same number.
    """
    from rag_kit.streaming import PhaseTimings

    pt_streaming = PhaseTimings()
    samples = [10.0, 50.0, 100.0, 150.0, 200.0]
    for s in samples:
        pt_streaming.record("retrieving", s)
    expected_p95 = pt_streaming.percentile("retrieving", 95)
    got_p95 = percentile(samples, 0.95)
    assert expected_p95 is not None
    assert math.isclose(got_p95, expected_p95, rel_tol=1e-9)
