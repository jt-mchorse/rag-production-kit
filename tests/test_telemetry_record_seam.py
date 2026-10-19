"""`TelemetryStore.record` holds a bare `CostRecord` to `build`'s rules (#245).

#184 closed `ts` at `CostRecord.build` and named its tests for the store --
`test_no_sqlite_integrity_error_can_escape_record`, `..._resident_in_every_window`,
`..._absent_from_every_window` -- while exercising `build` only, "the only way
a caller can" construct a row. The bare constructor is public in
`rag_kit.__all__` and validates nothing. Measured on `main` at `2ff3f6f`,
records built with `CostRecord(...)` and passed to `record`::

    ts=nan                -> sqlite3.IntegrityError: NOT NULL constraint failed: cost_records.ts
    total_latency_ms=nan  -> sqlite3.IntegrityError (SQLite stores NaN as NULL)
    ts=inf, ts="2026-08-24" -> stored; last_24h(now=9e9) == ['inf', 'str']
    ts=-inf               -> stored; since(-1e300) does not return it

`ts` cannot be guarded at a read boundary: `since()` filters and orders on it
in SQL, before any Python sees the row.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from rag_kit import CostRecord, TelemetryStore, telemetry
from rag_kit.telemetry import ModelPrice, PriceTable

ROOT = Path(__file__).resolve().parent.parent
GOOD_TS = 1_700_000_000.0


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


def _build(ts: Any) -> CostRecord:
    return CostRecord.build(
        ts=ts,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_latency_ms=12.0,
        per_phase_ms={},
        price_table=PriceTable({"m": ModelPrice(3.0, 15.0)}),
    )


# ----------------------------------------------------------------------
# #184's three properties, through the bare constructor
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ts", [math.nan, math.inf, -math.inf, "2026-08-24", True, [1.0]], ids=repr)
def test_record_refuses_what_build_refuses_in_the_same_words(tmp_path: Path, ts: Any) -> None:
    with pytest.raises(ValueError) as built:  # noqa: PT011 - compared below
        _build(ts)
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        with pytest.raises(ValueError) as recorded:  # noqa: PT011 - compared below
            store.record(_bare(ts=ts))
        assert store.since(-math.inf) == []
    assert str(recorded.value) == str(built.value)


def test_a_bare_record_cannot_be_resident_in_every_window(tmp_path: Path) -> None:
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        store.record(_bare(query="real"))
        with pytest.raises(ValueError, match="ts must be a finite number of seconds"):
            store.record(_bare(ts=math.inf, query="inf"))
        assert store.last_24h(now=9_000_000_000.0) == []


def test_a_bare_record_cannot_be_absent_from_every_window(tmp_path: Path) -> None:
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        with pytest.raises(ValueError, match="ts must be a finite number of seconds"):
            store.record(_bare(ts=-math.inf))
        store.record(_bare(query="real"))
        assert [r.query for r in store.since(-1e300)] == ["real"]


# ----------------------------------------------------------------------
# No sqlite3.IntegrityError escapes record
# ----------------------------------------------------------------------


def _real_columns() -> list[str]:
    """Every `REAL NOT NULL` column in the schema, from the schema text."""
    return re.findall(r"^\s*(\w+) REAL NOT NULL", telemetry._SCHEMA_SQL, re.M)


def test_the_schema_walk_finds_the_real_columns() -> None:
    """Control: the population arm below is vacuous if this finds nothing."""
    assert "ts" in _real_columns()
    assert len(_real_columns()) >= 5


def test_every_real_column_is_nan_checked() -> None:
    """A NaN in any REAL NOT NULL column reaches the constraint as NULL. `ts`
    has its own fuller rule; every other REAL column must be in the list."""
    assert set(_real_columns()) - {"ts"} == set(telemetry._NAN_CHECKED_REAL_FIELDS)


@pytest.mark.parametrize("field", ["ts", *telemetry._NAN_CHECKED_REAL_FIELDS])
def test_a_nan_in_any_real_column_is_a_value_error_naming_the_field(
    tmp_path: Path, field: str
) -> None:
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        # A raw `sqlite3.IntegrityError` is not a `ValueError`, so the defect
        # fails this as an error rather than passing it.
        with pytest.raises(ValueError, match=field):
            store.record(_bare(**{field: math.nan}))
        assert store.since(-1e300) == []


@pytest.mark.parametrize("value", [math.inf, -1.0, 0.0])
def test_only_nan_is_refused_in_the_other_columns(tmp_path: Path, value: float) -> None:
    """Scope: `aggregate` guards infinities and negatives at the read boundary
    on purpose (#80, #213); the write seam closes only what SQLite cannot store."""
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        store.record(_bare(total_latency_ms=value))
        (row,) = store.since(0.0)
        assert row.total_latency_ms == value


# ----------------------------------------------------------------------
# The REAL columns' type, not only their NaN (#368)
# ----------------------------------------------------------------------

# Spelled out rather than read from `telemetry`, so the population lock above
# covers drift and these arms stay meaningful against any tree.
REAL_FIELDS = ("prompt_usd", "completion_usd", "total_usd", "total_latency_ms")
BAD_REAL = {
    "str": "0.03 USD",
    "bytes": b"\x00",
    "none": None,
    "decimal": Decimal("0.03"),
    "bool": True,
}


@pytest.mark.parametrize("label", list(BAD_REAL))
@pytest.mark.parametrize("field", REAL_FIELDS)
def test_a_wrong_typed_real_column_is_a_value_error_naming_the_field(
    tmp_path: Path, field: str, label: str
) -> None:
    """Measured on main: a str/bytes was stored (affinity is a preference),
    True was stored as 1.0, None/Decimal escaped as raw sqlite3 errors."""
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        with pytest.raises(ValueError, match=rf"^{field} must be a real number; got "):
            store.record(_bare(**{field: BAD_REAL[label]}))
        assert store.since(-1e300) == []


@pytest.mark.parametrize("value", [3, 0, -1, math.inf, 0.03])
@pytest.mark.parametrize("field", REAL_FIELDS)
def test_every_real_number_is_still_stored(tmp_path: Path, field: str, value: float) -> None:
    """Control: an int, a negative and an infinity are real numbers; `aggregate`
    owns their meaning on read (#80, #213)."""
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        store.record(_bare(**{field: value}))
        (row,) = store.since(0.0)
        assert getattr(row, field) == value


def test_the_dashboard_renders_whatever_record_accepted(tmp_path: Path) -> None:
    """The outcome the issue measured: a stored str `total_usd` raised a
    TypeError in the HTML renderer on every request for the whole window."""
    spec = importlib.util.spec_from_file_location(
        "telemetry_dashboard", ROOT / "scripts" / "telemetry_dashboard.py"
    )
    assert spec is not None
    assert spec.loader is not None
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        store.record(_bare())
        for field, bad in (("total_usd", "0.03 USD"), ("total_latency_ms", "12ms")):
            with pytest.raises(ValueError, match=f"^{field} must be a real number"):
                store.record(_bare(**{field: bad}))
        records = store.since(0.0)
    assert len(records) == 1
    assert "<html" in dashboard._render_dashboard_html(records).lower()
    assert dashboard._json_response_body(records)


# ----------------------------------------------------------------------
# Reading what is already stored is unchanged
# ----------------------------------------------------------------------


def test_a_legacy_row_with_a_bad_ts_still_reads_back(tmp_path: Path) -> None:
    """Why the check is at `record` and not in `__post_init__`: `since()`
    rebuilds every stored row through the constructor, and a store written
    before #245 may already hold one."""
    path = tmp_path / "t.sqlite"
    with TelemetryStore(path) as store:
        store.record(_bare(query="real"))
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO cost_records (ts, query, model, retrieved_count, prompt_tokens, "
        "completion_tokens, prompt_usd, completion_usd, total_usd, total_latency_ms, "
        "per_phase_json) VALUES (?, 'legacy', 'm', 1, 1, 1, 0, 0, 0, 1, '{}')",
        (math.inf,),
    )
    conn.commit()
    conn.close()
    with TelemetryStore(path) as store:
        assert [r.query for r in store.since(0.0)] == ["real", "legacy"]
