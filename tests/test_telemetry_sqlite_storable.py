"""Everything `build` accepts, `record` can store and `since()` reads back (#387).

`build` checked type and sign but not what SQLite can hold. Measured on a2a3187,
each accepted by `build`:

    ts / total_latency_ms / prompt_tokens / retrieved_count = 2**63
        -> record: raw OverflowError (Python int too large to convert to SQLite INTEGER)
    per_phase_ms={("a", "b"): 1.0}   -> record: raw TypeError from json.dumps
    per_phase_ms={1: 5.0, "1": 6.0}  -> read back as {"1": 6.0}; the 5.0 is lost
    per_phase_ms={None: 5.0}         -> read back as {"null": 5.0}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_kit import CostRecord, TelemetryStore
from rag_kit.telemetry import ModelPrice, PriceTable

GOOD_TS = 1_700_000_000.0
TABLE = PriceTable({"m": ModelPrice(3.0, 15.0)})
INT_MAX = 2**63 - 1
INT_MIN = -(2**63)

BUILD_DEFAULTS: dict[str, Any] = {
    "ts": GOOD_TS,
    "query": "q",
    "model": "m",
    "retrieved_count": 1,
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_latency_ms": 12.0,
    "per_phase_ms": {},
}

# Spelled out rather than read from `telemetry`, so this file collects against
# a tree without the check and its arms go red instead of erroring.
REAL_FIELDS = ("ts", "total_latency_ms", "prompt_usd", "completion_usd", "total_usd")
INTEGER_FIELDS = ("retrieved_count", "prompt_tokens", "completion_tokens")
BUILD_NUMERIC_ARGS = ("ts", "total_latency_ms") + INTEGER_FIELDS


def _build(**overrides: Any) -> CostRecord:
    return CostRecord.build(**{**BUILD_DEFAULTS, **overrides}, price_table=TABLE)


def _bare(**overrides: Any) -> CostRecord:
    kwargs: dict[str, Any] = {
        "ts": GOOD_TS,
        "query": "q",
        "model": "m",
        "retrieved_count": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "prompt_usd": 0.01,
        "completion_usd": 0.02,
        "total_usd": 0.03,
        "total_latency_ms": 12.0,
    }
    kwargs.update(overrides)
    return CostRecord(**kwargs)


def _round_trip(tmp_path: Path, rec: CostRecord) -> CostRecord:
    with TelemetryStore(tmp_path / "t.db") as store:
        store.record(rec)
        (back,) = store.since(float("-inf"))
    return back


# --- integers outside SQLite's signed 64-bit range ---------------------------


@pytest.mark.parametrize("name", BUILD_NUMERIC_ARGS)
def test_build_refuses_an_int_above_the_sqlite_range(name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} must fit SQLite's signed 64-bit"):
        _build(**{name: INT_MAX + 1})


def test_build_refuses_a_ts_below_the_sqlite_range() -> None:
    # `ts` is the one numeric argument allowed to be negative.
    with pytest.raises(ValueError, match=r"ts must fit SQLite's signed 64-bit"):
        _build(ts=INT_MIN - 1)


@pytest.mark.parametrize("name", REAL_FIELDS + INTEGER_FIELDS)
def test_record_refuses_a_bare_record_int_above_the_sqlite_range(tmp_path: Path, name: str) -> None:
    with TelemetryStore(tmp_path / "t.db") as store:
        with pytest.raises(ValueError, match=rf"{name} must fit SQLite's signed 64-bit"):
            store.record(_bare(**{name: INT_MAX + 1}))
        assert store.since(float("-inf")) == []


# Controls: the exact bounds are storable and must stay accepted.
@pytest.mark.parametrize("name", INTEGER_FIELDS)
def test_integer_column_at_the_max_bound_round_trips_exactly(tmp_path: Path, name: str) -> None:
    back = _round_trip(tmp_path, _build(**{name: INT_MAX}))
    assert getattr(back, name) == INT_MAX


@pytest.mark.parametrize("value", [INT_MAX, INT_MIN, float(2**63), 1e300])
def test_ts_at_or_past_the_int_bounds_is_accepted_and_stored_as_real(
    tmp_path: Path, value: float
) -> None:
    # A REAL column holds the float value; only an *int* has to fit the range.
    back = _round_trip(tmp_path, _build(ts=value))
    assert back.ts == float(value)


# --- per_phase_ms keys --------------------------------------------------------


@pytest.mark.parametrize(
    "phases",
    [
        {("a", "b"): 1.0},
        {1: 5.0, "1": 6.0},
        {None: 5.0},
        {1: 5.0},
        {b"retrieve": 5.0},
    ],
    ids=["tuple", "int-colliding-with-str", "none", "int", "bytes"],
)
def test_build_refuses_a_non_str_phase_key(phases: dict[Any, float]) -> None:
    with pytest.raises(ValueError, match=r"per_phase_ms keys must be str"):
        _build(per_phase_ms=phases)


def test_str_phase_keys_round_trip_unchanged(tmp_path: Path) -> None:
    # Control: the keys json.dumps would never have coerced.
    phases = {"retrieve": 1.0, "": 2.0, "null": 3.0, "1": 4.0, "fusión": 5.0}
    back = _round_trip(tmp_path, _build(per_phase_ms=phases))
    assert back.per_phase_ms == phases


# --- parity -------------------------------------------------------------------

# Edge values per `build` argument. Each one `build` accepts must round-trip.
EDGES: dict[str, list[Any]] = {
    "ts": [0, -1.5, INT_MAX, INT_MIN, INT_MAX + 1, INT_MIN - 1, 2**64, float(2**63), 1e300],
    "total_latency_ms": [0, 0.0, INT_MAX, INT_MAX + 1, 2**64, 1e300],
    "retrieved_count": [0, INT_MAX, INT_MAX + 1, 2**64],
    "prompt_tokens": [0, INT_MAX, INT_MAX + 1, 2**64],
    "completion_tokens": [0, INT_MAX, INT_MAX + 1, 2**64],
    "per_phase_ms": [
        {},
        {"a": 0, "b": INT_MAX + 1, "c": 1e300},
        {("a", "b"): 1.0},
        {1: 5.0, "1": 6.0},
        {None: 5.0},
        {True: 1.0},
        {1.5: 1.0},
    ],
}


@pytest.mark.parametrize(
    ("name", "value"),
    [(name, value) for name, values in EDGES.items() for value in values],
    ids=lambda v: repr(v)[:40],
)
def test_everything_build_accepts_round_trips_through_record(
    tmp_path: Path, name: str, value: Any
) -> None:
    try:
        rec = _build(**{name: value})
    except ValueError:
        return  # refused at the boundary: the contract holds
    back = _round_trip(tmp_path, rec)
    got = getattr(back, name)
    if name in REAL_FIELDS:
        # A REAL column stores the float value of an int (the field is `float`).
        assert got == float(value)
    else:
        assert got == value
