"""The read-side floors and ints past float range refuse in words, too (#392).

#387 range-checked the ints `build` and `record` bind, but not the floors `since`
and `dump_aggregate_json` bind into `WHERE ts >= ?`, and each check it added ran
after `math.isfinite`, which converts an int to float and raises first. Measured
on main (1ff10c8):

    since(2**63), since(-2**63-1), dump_aggregate_json(since_ts=2**63)
        -> OverflowError: Python int too large to convert to SQLite INTEGER
    since(10**400), dump_aggregate_json(since_ts=10**400), last_24h(now=10**400)
        -> OverflowError: int too large to convert to float
    last_24h(now=-2**63)  (a `ts` that `record` stores)
        -> OverflowError at bind, from the int cutoff now - 86400
    CostRecord.build(ts=10**400 | total_latency_ms=10**400 | per_phase_ms={"a": 10**400})
        -> OverflowError: int too large to convert to float
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from rag_kit import CostRecord, TelemetryStore
from rag_kit.telemetry import ModelPrice, PriceTable

TABLE = PriceTable({"m": ModelPrice(3.0, 15.0)})
INT_MAX = 2**63 - 1
INT_MIN = -(2**63)
HUGE = 10**400  # past float range: float(HUGE) raises OverflowError
RANGE = "must fit SQLite's signed 64-bit INTEGER range"


def _build(**overrides: Any) -> CostRecord:
    kwargs: dict[str, Any] = {
        "ts": 1_700_000_000.0,
        "query": "q",
        "model": "m",
        "retrieved_count": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_latency_ms": 12.0,
        "per_phase_ms": {},
    }
    return CostRecord.build(**{**kwargs, **overrides}, price_table=TABLE)


@pytest.fixture
def store(tmp_path: Path):
    with TelemetryStore(tmp_path / "t.db") as s:
        yield s


# --- read side: the floors --------------------------------------------------


@pytest.mark.parametrize("floor", [INT_MAX + 1, INT_MIN - 1, HUGE, -HUGE])
def test_since_refuses_an_int_floor_sqlite_cannot_bind(store: TelemetryStore, floor: int) -> None:
    with pytest.raises(ValueError, match=rf"ts_floor {RANGE}"):
        store.since(floor)


@pytest.mark.parametrize("floor", [INT_MAX + 1, HUGE])
def test_dump_aggregate_json_refuses_an_int_floor_sqlite_cannot_bind(
    store: TelemetryStore, tmp_path: Path, floor: int
) -> None:
    out = tmp_path / "agg.json"
    with pytest.raises(ValueError, match=rf"since_ts {RANGE}"):
        store.dump_aggregate_json(out, since_ts=floor)
    assert not out.exists()


def test_last_24h_refuses_a_now_past_float_range(store: TelemetryStore) -> None:
    with pytest.raises(ValueError, match=rf"now {RANGE}"):
        store.last_24h(now=HUGE)


def test_last_24h_at_the_lowest_storable_ts_finds_the_row_stored_there(
    store: TelemetryStore,
) -> None:
    # `now=-2**63` passes `_checked_ts`, and `record` stores a row at that `ts`;
    # its 24h window must be answerable, not escape at bind.
    store.record(_build(ts=INT_MIN))
    (row,) = store.last_24h(now=INT_MIN)
    assert row.ts == float(INT_MIN)


# Controls: floors at the exact bounds, and non-int floors, bind as before.
@pytest.mark.parametrize("floor", [INT_MAX, INT_MIN, float(2**63), float("inf"), float("-inf")])
def test_since_accepts_a_floor_sqlite_can_bind(store: TelemetryStore, floor: float) -> None:
    store.record(_build(ts=0))
    assert len(store.since(floor)) == (1 if floor <= 0 else 0)


@pytest.mark.parametrize("now", [INT_MAX, INT_MIN + 24 * 3600, 1_700_000_000])
def test_last_24h_accepts_a_now_in_range(store: TelemetryStore, now: int) -> None:
    assert store.last_24h(now=now) == []


# --- build: the range check runs before the float conversion ----------------


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"ts": HUGE}, "ts"),
        ({"ts": -HUGE}, "ts"),
        ({"total_latency_ms": HUGE}, "total_latency_ms"),
        ({"per_phase_ms": {"a": HUGE}}, r"per_phase_ms\['a'\]"),
    ],
    ids=["ts", "negative-ts", "total_latency_ms", "per_phase_ms"],
)
def test_build_refuses_an_int_past_float_range_naming_the_field(
    overrides: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValueError, match=rf"{field} {RANGE}"):
        _build(**overrides)


def test_record_refuses_a_ts_past_float_range_from_the_bare_constructor(
    store: TelemetryStore,
) -> None:
    # `record` re-runs `_checked_ts` for records that skipped `build`; the same
    # ordering bug made it raise OverflowError there too.
    rec = dataclasses.replace(_build(), ts=HUGE)
    with pytest.raises(ValueError, match=rf"ts {RANGE}"):
        store.record(rec)


def test_an_int_too_long_to_print_still_names_the_field() -> None:
    # `str()` of an int past 4300 digits raises its own ValueError, which would
    # drop the field name; the message names the int by its size instead.
    with pytest.raises(ValueError, match=rf"ts {RANGE}.*an int of \d+ bits"):
        _build(ts=10**5000)


# Controls: exact bounds and ordinary values still build.
@pytest.mark.parametrize(
    "overrides",
    [
        {"ts": INT_MAX},
        {"ts": INT_MIN},
        {"total_latency_ms": INT_MAX},
        {"per_phase_ms": {"a": INT_MAX, "b": 0, "c": 1e300}},
    ],
    ids=["ts-max", "ts-min", "total_latency_ms-max", "per_phase_ms-max"],
)
def test_build_accepts_ints_at_the_bounds(overrides: dict[str, Any]) -> None:
    _build(**overrides)
