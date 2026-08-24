"""`CostRecord.ts` is validated at the write seam, and unformattable rows
do not take the dashboard down with them (#184).

`CostRecord` carries four float fields. Three were guarded for finiteness, each
with an argument in the source: `total_latency_ms` (#38, "NaN latency propagates
through `percentile()` ... the returned percentile is implementation-defined and
silently wrong"), every `per_phase_ms` value (#108, "the same finiteness/sign
contract as `total_latency_ms` above ... bool is an int subclass"), and
`total_usd` (`aggregate` raises on a non-finite cost).

The fourth was `ts`, and nothing validated it -- while being the only one the
whole query surface is keyed on. `since()` filters `WHERE ts >= ?` and orders
`ORDER BY ts ASC`; `last_24h()` is defined in terms of it.

Measured on `main`, every value accepted by `build`::

    nan          -> record() raised sqlite3.IntegrityError (NOT NULL on ts)
    inf          -> stored; in EVERY last_24h() window, forever
    -inf         -> stored; in NO window, ever
    '2026-08-24' -> stored as SQLite `text`; in every window, forever; and
                    round-trips out of since() as a `str` in a float field
    True         -> stored as 1.0, i.e. 1970-01-01T00:00:01Z

and three raw exception types escaping the dashboard renderer -- `OverflowError`
(inf), `TypeError` (str), `OSError` (1e18) -- each of which killed the *whole*
page, so a good record next to a bad one could not be rendered either.

The split is deliberate. The first four are closed at the write seam with the
rule the module already states twice. The fifth (`1e18`, from `time.time_ns()`
in place of `time.time()`) is *finite*, and the range `time.gmtime` can
represent is a property of the platform's `time_t` -- so it is guarded at the
outcome in the renderer instead, and **no test here asserts where that platform
bound lies**.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from rag_kit.telemetry import CostRecord, ModelPrice, PriceTable, TelemetryStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.telemetry_dashboard import (  # noqa: E402
    _format_ts,
    _render_chart_svg,
    _render_dashboard_html,
)

PRICES = PriceTable({"m": ModelPrice(3.0, 15.0)})
GOOD_TS = 1_700_000_000.0


def _build(ts, query: str = "q") -> CostRecord:
    return CostRecord.build(
        ts=ts,
        query=query,
        model="m",
        retrieved_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_latency_ms=12.0,
        per_phase_ms={},
        price_table=PRICES,
    )


# ----------------------------------------------------------------------
# The write seam
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_ts",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "2026-08-24",
        "1700000000",
        None.__class__,  # a type object — anything non-numeric
        [1_700_000_000.0],
    ],
)
def test_build_rejects_a_ts_the_read_paths_cannot_use(bad_ts: object) -> None:
    with pytest.raises(ValueError, match=r"ts must be a finite number of seconds"):
        _build(bad_ts)


def test_the_error_names_the_field_and_says_what_it_is_for() -> None:
    """ "2026-08-24" is the mistake being rejected, so the message has to say
    "seconds since the Unix epoch" — naming only the type would leave the
    caller with no idea what to pass instead."""
    with pytest.raises(ValueError) as exc:  # noqa: PT011 - the message IS the assertion
        _build("2026-08-24")
    msg = str(exc.value)
    assert "ts" in msg
    assert "seconds since the Unix epoch" in msg
    assert "2026-08-24" in msg


@pytest.mark.parametrize("ok_ts", [GOOD_TS, 0.0, -1_000_000_000.0, 5, 1.7e18])
def test_build_accepts_every_finite_real_timestamp(ok_ts: float) -> None:
    """Negative is allowed on purpose — a pre-1970 timestamp is unusual but
    well-defined, and this module has no business deciding an operator's
    backfill window is wrong. `1.7e18` is allowed because it is *finite*; the
    harm it causes is a rendering one and is guarded there."""
    assert _build(ok_ts).ts == ok_ts


def test_ts_none_still_defaults_to_now() -> None:
    """The documented test-injection point must survive the guard."""
    rec = _build(None)
    assert isinstance(rec.ts, float)
    assert math.isfinite(rec.ts)


# ----------------------------------------------------------------------
# What the guard prevents downstream
# ----------------------------------------------------------------------


def test_no_sqlite_integrity_error_can_escape_record() -> None:
    """On `main`, `ts=nan` reached `record()` and came back as
    `sqlite3.IntegrityError: NOT NULL constraint failed: cost_records.ts`
    (SQLite stores NaN as NULL) — an exception type outside this module's
    contract, naming a column rather than the caller's mistake.

    Asserted by constructing the row the only way a caller can, and confirming
    the failure happens at `build` instead.
    """
    with pytest.raises(ValueError, match="ts must be a finite number of seconds"):
        _build(float("nan"))


def test_a_record_cannot_be_resident_in_every_window(tmp_path: Path) -> None:
    """`inf` was in `last_24h()` for any `now`, including the year 2255 — the
    dashboard's headline 24h spend permanently inflated by a row that is not
    from the last 24 hours and never will be."""
    with pytest.raises(ValueError, match="ts must be a finite number of seconds"):
        _build(float("inf"))

    # And the property the rejection buys, stated over a real store.
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        store.record(_build(GOOD_TS, "real"))
        assert [r.query for r in store.last_24h(now=GOOD_TS + 60)] == ["real"]
        assert store.last_24h(now=9_000_000_000.0) == []


def test_a_record_cannot_be_absent_from_every_window(tmp_path: Path) -> None:
    """The mirror case: `-inf` was stored and visible to no window at all."""
    with pytest.raises(ValueError, match="ts must be a finite number of seconds"):
        _build(float("-inf"))


def test_ts_round_trips_as_a_float_not_a_str(tmp_path: Path) -> None:
    """`ts REAL NOT NULL` does not stop a string: SQLite type affinity is a
    preference, not a constraint, so `typeof(ts)` came back `'text'` and the
    value round-tripped out of `since()` as a `str` in a field annotated
    `float`. The guard is what makes the annotation true."""
    with pytest.raises(ValueError, match="ts must be a finite number of seconds"):
        _build("2026-08-24")

    with TelemetryStore(tmp_path / "t.sqlite") as store:
        store.record(_build(GOOD_TS))
        (row,) = store.since(0.0)
        assert isinstance(row.ts, float)
        (stored_type,) = store._conn.execute("SELECT typeof(ts) FROM cost_records").fetchone()
        assert stored_type == "real"


def test_bool_is_rejected_for_the_reason_per_phase_ms_already_gives(tmp_path: Path) -> None:
    """`per_phase_ms`' guard says "bool is an int subclass; reject it so a stray
    True/False can't pose as a millisecond value". A timestamp has the identical
    problem, and `True` meant 1970-01-01T00:00:01Z."""
    with pytest.raises(ValueError, match="ts must be a finite number of seconds"):
        _build(True)
    # The sibling guard this one was extended from still holds.
    with pytest.raises(ValueError, match=r"per_phase_ms\['retrieve'\]"):
        CostRecord.build(
            ts=GOOD_TS,
            query="q",
            model="m",
            retrieved_count=1,
            prompt_tokens=1,
            completion_tokens=1,
            total_latency_ms=1.0,
            per_phase_ms={"retrieve": True},
            price_table=PRICES,
        )


# ----------------------------------------------------------------------
# The renderer: blast radius, not cosmetics
# ----------------------------------------------------------------------


def test_format_ts_falls_back_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driven through a stubbed `gmtime`, not through a magic constant, so this
    test asserts a property of the code and not of the CI host's `time_t`.
    """
    import scripts.telemetry_dashboard as dash

    def _boom(_ts: float):
        raise OSError(84, "Value too large to be stored in data type")

    monkeypatch.setattr(dash.time, "gmtime", _boom)
    out = _format_ts(1.7e18)
    assert "unrepresentable" in out
    assert "1.7e+18" in out


@pytest.mark.parametrize(
    "exc", [OSError(84, "too large"), OverflowError("out of range"), ValueError("bad")]
)
def test_format_ts_catches_every_shape_gmtime_can_raise(
    exc: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which of the three `gmtime` raises for an out-of-range value differs by
    platform and by magnitude, so catching one of them would reintroduce the
    defect on the others."""
    import scripts.telemetry_dashboard as dash

    def _boom(_ts: float):
        raise exc

    monkeypatch.setattr(dash.time, "gmtime", _boom)
    assert "unrepresentable" in _format_ts(GOOD_TS)


def test_one_unformattable_row_does_not_kill_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual harm. Before this change, a store holding one ordinary record
    and one with `ts=1e18` made `_render_dashboard_html` raise `OSError`, so the
    operator lost *every* good row in the window to one bad one — and got a
    traceback naming "data type" rather than the record at fault.
    """
    import scripts.telemetry_dashboard as dash

    real_gmtime = dash.time.gmtime

    def _selective(ts: float):
        if ts > 1e15:
            raise OSError(84, "Value too large to be stored in data type")
        return real_gmtime(ts)

    monkeypatch.setattr(dash.time, "gmtime", _selective)

    with TelemetryStore(tmp_path / "t.sqlite") as store:
        store.record(_build(GOOD_TS, "good-row"))
        store.record(_build(1.7e18, "ns-instead-of-s"))
        records = store.since(float("-inf"))

    page = _render_dashboard_html(records)
    assert "good-row" in page
    assert "ns-instead-of-s" in page
    assert "unrepresentable" in page

    svg = _render_chart_svg(records)
    assert "unrepresentable" in svg
    assert "nan" not in svg.lower()


def test_an_ordinary_page_is_unchanged(tmp_path: Path) -> None:
    """No fallback marker anywhere when every timestamp is representable — this
    is a guard on an exceptional path, not a change to the normal rendering."""
    with TelemetryStore(tmp_path / "t.sqlite") as store:
        for i in range(5):
            store.record(_build(GOOD_TS + i * 60, f"q{i}"))
        records = store.since(0.0)

    page = _render_dashboard_html(records)
    assert "unrepresentable" not in page
    assert "2023-11-14" in page  # GOOD_TS formatted, proving the normal path ran
