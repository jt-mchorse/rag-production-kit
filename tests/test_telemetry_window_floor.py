"""A window floor that is not a number is refused, not an empty window (#354).

#184/#245 refuse a non-finite or non-numeric `ts` on write. The read side's
floor reached `WHERE ts >= ?` unchecked, and SQLite sorts every REAL below every
TEXT and binds NaN as NULL. Measured on main with three records from the last
three minutes:

    since("2020-01-01T00:00:00")               -> 0
    since(datetime.now() - timedelta(hours=1)) -> 0
    since(nan)                                 -> 0
    since(True)                                -> 3 (a 1970 floor)
    dump_aggregate_json(out, since_ts="2020-01-01") -> n=0, total_usd=0.0
"""

from __future__ import annotations

import datetime
import json
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

from rag_kit.telemetry import CostRecord, ModelPrice, PriceTable, TelemetryStore

NOW = time.time()

BAD_FLOORS = {
    "iso-string": "2020-01-01T00:00:00",
    "datetime": datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    "nan": float("nan"),
    "bool": True,
    "none-as-floor": None,
    "decimal-string": "1700000000",
}


@pytest.fixture
def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    table = PriceTable({"m": ModelPrice(3.0, 15.0)})
    s = TelemetryStore(tmp_path / "t.db")
    for i in range(3):
        s.record(
            CostRecord.build(
                ts=NOW - 60 * (i + 1),
                query="q",
                model="m",
                retrieved_count=1,
                prompt_tokens=1000,
                completion_tokens=100,
                total_latency_ms=5.0,
                per_phase_ms=None,
                price_table=table,
            )
        )
    yield s
    s.close()


@pytest.mark.parametrize("floor", list(BAD_FLOORS.values()), ids=list(BAD_FLOORS))
def test_since_refuses_it_naming_the_parameter(store: TelemetryStore, floor: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no sqlite3 adapter deprecation slips through first
        with pytest.raises(ValueError, match=r"^ts_floor must "):
            store.since(floor)


@pytest.mark.parametrize(
    "floor",
    [v for k, v in BAD_FLOORS.items() if k != "none-as-floor"],
    ids=[k for k in BAD_FLOORS if k != "none-as-floor"],
)
def test_dump_aggregate_json_refuses_it_and_writes_nothing(
    store: TelemetryStore, tmp_path: Path, floor: Any
) -> None:
    out = tmp_path / "agg.json"
    with pytest.raises(ValueError, match=r"^since_ts must "):
        store.dump_aggregate_json(out, since_ts=floor)
    assert not out.exists()


@pytest.mark.parametrize(
    "now", [float("nan"), float("inf"), float("-inf"), "2026-10-16", True], ids=repr
)
def test_last_24h_refuses_a_now_that_is_not_a_finite_timestamp(
    store: TelemetryStore, now: Any
) -> None:
    with pytest.raises(ValueError, match=r"^now must be a finite number"):
        store.last_24h(now=now)


def test_an_infinite_floor_means_what_it_says(store: TelemetryStore) -> None:
    assert len(store.since(float("-inf"))) == 3
    assert store.since(float("inf")) == []


def test_numeric_floors_are_unchanged(store: TelemetryStore, tmp_path: Path) -> None:
    assert len(store.since(NOW - 3600)) == 3
    assert len(store.since(int(NOW) - 150)) == 2
    assert len(store.last_24h(now=NOW)) == 3
    out = tmp_path / "agg.json"
    store.dump_aggregate_json(out, since_ts=NOW - 3600)
    assert json.loads(out.read_text(encoding="utf-8"))["n"] == 3
    store.dump_aggregate_json(out)  # the default window
    assert json.loads(out.read_text(encoding="utf-8"))["n"] == 3
