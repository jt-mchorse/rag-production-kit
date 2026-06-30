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


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1.0, True])
@pytest.mark.parametrize("side", ["prompt", "completion"])
def test_price_table_cost_rejects_non_integer_or_non_finite_tokens(bad, side):
    # #104: the sign-only guard let a NaN through (`NaN < 0` is False), so
    # `NaN * rate` produced a NaN USD that serialized as the bare token `NaN`
    # (invalid JSON) in the dashboard aggregate — the token-count seam the
    # #38/#80 finiteness sweep missed. floats (1.0) and bool (True, an int
    # subclass) are rejected for the same "genuine non-negative int" contract.
    # Pre-fix `cost("fake-big", nan, 10)` returned `(nan, ...)`; this is the
    # inverse safety net for that.
    pt = _fixture_prices()
    prompt, completion = (bad, 10) if side == "prompt" else (10, bad)
    with pytest.raises(ValueError, match="non-negative integers"):
        pt.cost("fake-big", prompt, completion)


def test_price_table_cost_valid_int_tokens_still_finite():
    # Over-rejection guard: ordinary non-negative int counts still compute
    # finite USD (the tightened guard must not reject the happy path).
    pt = _fixture_prices()
    prompt_usd, completion_usd = pt.cost("fake-big", 1000, 500)
    assert math.isfinite(prompt_usd)
    assert math.isfinite(completion_usd)
    assert prompt_usd >= 0.0
    assert completion_usd >= 0.0


def test_price_table_add_overrides_existing():
    pt = _fixture_prices()
    pt.add("fake-big", 5.0, 20.0)  # overwrite
    prompt_usd, _ = pt.cost("fake-big", 1_000_000, 0)
    assert prompt_usd == pytest.approx(5.0)


# Issue #36: ModelPrice rejects negative per-million rates so a misconfigured
# operator can't silently invert the sign of total_usd in cost dashboards.
# Extends D-015's "no silent zero" posture to "no silent negative".
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prompt_per_million", -0.01),
        ("prompt_per_million", -10.0),
        ("completion_per_million", -0.01),
        ("completion_per_million", -100.0),
    ],
)
def test_model_price_rejects_negative_rate(field: str, bad_value: float):
    kwargs: dict[str, float] = {
        "prompt_per_million": 1.0,
        "completion_per_million": 1.0,
    }
    kwargs[field] = bad_value
    # Message tightened in #38 to "must be a finite number >= 0.0".
    with pytest.raises(ValueError, match=rf"{field} must be a finite number >= 0\.0"):
        ModelPrice(**kwargs)


def test_model_price_accepts_zero_rates():
    # Zero is meaningful: free-tier or synthetic-workload model. Inclusive
    # bound is part of the contract.
    p = ModelPrice(prompt_per_million=0.0, completion_per_million=0.0)
    assert p.cost(1_000_000, 1_000_000) == (0.0, 0.0)


def test_price_table_add_rejects_negative_rate_via_wrap_through():
    # PriceTable.add wraps ModelPrice construction, so the guard fires
    # through the realistic operator-supplies-bad-config path.
    pt = PriceTable()
    with pytest.raises(ValueError, match="prompt_per_million must be a finite number >= 0.0"):
        pt.add("fake-bad", -1.0, 1.0)
    with pytest.raises(ValueError, match="completion_per_million must be a finite number >= 0.0"):
        pt.add("fake-bad", 1.0, -1.0)


# Issue #38: extend ModelPrice sign-only check to finiteness. A NaN rate
# propagates through cost() → CostRecord.total_usd = NaN → aggregate() sums
# NaN → cost dashboard renders "NaN" silently. Same harm shape as D-015's
# silent-zero, one arithmetic layer downstream.
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prompt_per_million", float("nan")),
        ("prompt_per_million", float("inf")),
        ("prompt_per_million", float("-inf")),
        ("completion_per_million", float("nan")),
        ("completion_per_million", float("inf")),
        ("completion_per_million", float("-inf")),
    ],
)
def test_model_price_rejects_non_finite_rate(field: str, bad_value: float):
    kwargs: dict[str, float] = {
        "prompt_per_million": 1.0,
        "completion_per_million": 1.0,
    }
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=rf"{field} must be a finite number >= 0\.0"):
        ModelPrice(**kwargs)


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


# Issue #38: extend total_latency_ms sign-only check to finiteness. NaN
# latency propagates through percentile(values, q) where the sort over NaN
# is implementation-defined and p95/p99 silently report a meaningless number.
@pytest.mark.parametrize(
    "bad_latency",
    [float("nan"), float("inf"), float("-inf")],
)
def test_costrecord_build_rejects_non_finite_latency(bad_latency: float):
    pt = _fixture_prices()
    with pytest.raises(ValueError, match=r"total_latency_ms must be a finite non-negative number"):
        CostRecord.build(
            ts=0.0,
            query="q",
            model="fake-big",
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_latency_ms=bad_latency,
            per_phase_ms=None,
            price_table=pt,
        )


# Issue #108: extend the total_latency_ms finiteness/sign contract to each
# per_phase_ms value. A non-finite phase value reaches TelemetryStore.record,
# which persists the map via json.dumps(... allow_nan=True) — writing the bare
# token NaN/Infinity (invalid JSON) that since() then swallows on
# JSONDecodeError, silently dropping the row's phases.
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_costrecord_build_rejects_non_finite_per_phase_value(bad_value: float):
    pt = _fixture_prices()
    with pytest.raises(
        ValueError, match=r"per_phase_ms\['retrieve'\] must be a finite non-negative number"
    ):
        CostRecord.build(
            ts=0.0,
            query="q",
            model="fake-big",
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_latency_ms=100.0,
            per_phase_ms={"retrieve": bad_value, "generate": 50.0},
            price_table=pt,
        )


def test_costrecord_build_rejects_negative_per_phase_value():
    pt = _fixture_prices()
    with pytest.raises(
        ValueError, match=r"per_phase_ms\['generate'\] must be a finite non-negative number"
    ):
        CostRecord.build(
            ts=0.0,
            query="q",
            model="fake-big",
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_latency_ms=100.0,
            per_phase_ms={"retrieve": 50.0, "generate": -1.0},
            price_table=pt,
        )


@pytest.mark.parametrize("bad_value", ["50.0", None, True])
def test_costrecord_build_rejects_non_numeric_per_phase_value(bad_value):
    # A str/None/bool millisecond value is not a number; bool is an int
    # subclass so it must be rejected explicitly, not silently coerced.
    pt = _fixture_prices()
    with pytest.raises(
        ValueError, match=r"per_phase_ms\['retrieve'\] must be a finite non-negative number"
    ):
        CostRecord.build(
            ts=0.0,
            query="q",
            model="fake-big",
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_latency_ms=100.0,
            per_phase_ms={"retrieve": bad_value},
            price_table=pt,
        )


def test_telemetry_store_per_phase_json_is_strict_valid_json(tmp_path):
    # A finite per-phase map persists as strict JSON — no bare NaN/Infinity
    # tokens reach the store. parse_constant fires only on JS-style constants
    # (NaN/Infinity/-Infinity), so a raised value there proves invalid JSON.
    import json
    import sqlite3

    store = TelemetryStore(tmp_path / "telemetry.db")
    store.record(_rec(ts=1_700_000_000.0))
    store.close()

    conn = sqlite3.connect(tmp_path / "telemetry.db")
    (raw,) = conn.execute("SELECT per_phase_json FROM cost_records").fetchone()
    conn.close()

    def _reject(_constant):  # pragma: no cover - only invoked on invalid JSON
        raise AssertionError("stored per_phase_json contains a non-finite JSON constant")

    parsed = json.loads(raw, parse_constant=_reject)
    assert parsed == {"retrieving": pytest.approx(10.0), "generating": pytest.approx(90.0)}


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


# Issue #80: percentile guards empty + q-range but not values finiteness. A NaN
# in `values` sorts into an implementation-defined slot (all NaN comparisons are
# False), so the returned percentile is silently wrong and position-dependent,
# and dump_aggregate_json then emits the bare token `NaN` (invalid JSON). Reject
# it at the metric boundary, same posture as PhaseTimings.record / CostRecord.build.
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_percentile_rejects_non_finite_values(bad: float):
    with pytest.raises(ValueError, match="finite numbers"):
        percentile([10.0, bad, 30.0], 0.5)


def test_percentile_still_accepts_all_finite_values():
    # Regression guard: the finiteness check must not reject legitimate samples.
    assert percentile([10.0, 20.0, 30.0], 0.5) == pytest.approx(20.0)


def test_aggregate_with_non_finite_latency_raises_not_nan_metric():
    # A CostRecord constructed directly (no __post_init__) bypasses the
    # build() latency guard; aggregate must fail loud rather than report a
    # `nan` latency_p50_ms and serialize an invalid-JSON `NaN` token.
    def _rec(lat: float) -> CostRecord:
        return CostRecord(
            ts=1.0,
            query="q",
            model="m",
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            prompt_usd=0.001,
            completion_usd=0.001,
            total_usd=0.002,
            total_latency_ms=lat,
            per_phase_ms={},
        )

    with pytest.raises(ValueError, match="finite numbers"):
        aggregate([_rec(10.0), _rec(20.0), _rec(math.nan), _rec(40.0)])


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
