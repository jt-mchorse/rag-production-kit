"""`build` and `record` refuse a TEXT or INTEGER column of the wrong type (#352).

#245 checked `ts` and the REAL columns; SQLite affinity is a preference, not a
constraint, so the TEXT and INTEGER columns took anything. Measured on main:

    build(query=b"raw bytes query")  -> stored, typeof(query) == 'blob'; the
        dashboard's `/` and `/api/last_24h` then both return an empty reply
        (html.escape / json.dumps refuse bytes)
    build(retrieved_count=None)      -> record: sqlite3.IntegrityError
"""

from __future__ import annotations

import importlib.util
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from rag_kit import CostRecord, TelemetryStore, telemetry
from rag_kit.telemetry import ModelPrice, PriceTable

ROOT = Path(__file__).resolve().parent.parent
GOOD_TS = 1_700_000_000.0
TABLE = PriceTable({"m": ModelPrice(3.0, 15.0)})

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


# Spelled out here rather than read from `telemetry`, so this file collects
# against a tree without the check and its arms go red instead of erroring.
TEXT_FIELDS = ("query", "model")
INTEGER_FIELDS = ("retrieved_count", "prompt_tokens", "completion_tokens")


def _columns(affinity: str) -> list[str]:
    return re.findall(rf"^\s*(\w+) {affinity} NOT NULL", telemetry._SCHEMA_SQL, re.M)


def test_the_schema_walk_finds_the_text_and_integer_columns() -> None:
    assert "query" in _columns("TEXT")
    assert "retrieved_count" in _columns("INTEGER")


def test_every_text_column_is_checked_except_the_one_record_writes_itself() -> None:
    assert set(_columns("TEXT")) - {"per_phase_json"} == set(TEXT_FIELDS)
    assert set(getattr(telemetry, "_TEXT_CHECKED_FIELDS", ())) == set(TEXT_FIELDS)


def test_every_integer_column_is_checked() -> None:
    assert set(_columns("INTEGER")) == set(INTEGER_FIELDS)
    assert set(getattr(telemetry, "_INTEGER_CHECKED_FIELDS", ())) == set(INTEGER_FIELDS)


BAD_TEXT = {"bytes": b"raw bytes query", "none": None, "int": 7}
BAD_INTEGER = {"none": None, "float": 1.0, "str": "3", "bool": True, "negative": -1}


def _cases() -> list[Any]:
    out = []
    for field in TEXT_FIELDS:
        for label, value in BAD_TEXT.items():
            out.append(pytest.param(field, value, id=f"{field}-{label}"))
    for field in INTEGER_FIELDS:
        for label, value in BAD_INTEGER.items():
            out.append(pytest.param(field, value, id=f"{field}-{label}"))
    return out


@pytest.mark.parametrize(("field", "value"), _cases())
def test_build_and_record_refuse_it_in_the_same_words(
    tmp_path: Path, field: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=rf"^{field} must be a ") as via_build:
        _build(**{field: value})
    with TelemetryStore(tmp_path / "t.db") as store:
        with pytest.raises(ValueError, match=rf"^{field} must be a ") as via_record:
            store.record(_bare(**{field: value}))
        assert store.since(0.0) == []
    assert str(via_build.value) == str(via_record.value)


SURROGATES = {"high": "caf" + chr(0xD800), "low": chr(0xDFFF) + "x", "escaped-byte": chr(0xDC80)}


@pytest.mark.parametrize("label", list(SURROGATES))
@pytest.mark.parametrize("field", TEXT_FIELDS)
def test_a_lone_surrogate_is_refused_by_build_and_record_alike(
    tmp_path: Path, field: str, label: str
) -> None:
    """#368: build accepted it and record raised a raw UnicodeEncodeError
    (SQLite binds text as UTF-8). Codepoints built with chr(), so the source
    holds no literal surrogate."""
    value = SURROGATES[label]
    pattern = rf"^{field} must be a str SQLite can store; it contains a lone surrogate"
    with pytest.raises(ValueError, match=pattern) as via_build:
        _build(**{field: value})
    with TelemetryStore(tmp_path / "t.db") as store:
        with pytest.raises(ValueError, match=pattern) as via_record:
            store.record(_bare(**{field: value}))
        assert store.since(0.0) == []
    assert str(via_build.value) == str(via_record.value)


def test_a_nul_and_non_bmp_text_still_round_trip(tmp_path: Path) -> None:
    """Control: SQLite stores a NUL and an astral character intact, so the
    surrogate rule must not grow into the indexer's NUL rule."""
    query = "a" + chr(0) + "b " + chr(0x1F600)
    with TelemetryStore(tmp_path / "t.db") as store:
        store.record(_bare(query=query))
        (back,) = store.since(0.0)
    assert back.query == query


def test_a_bytes_query_never_reaches_the_database(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with TelemetryStore(db) as store:
        with pytest.raises(ValueError, match="^query must be a str; got bytes"):
            store.record(_bare(query=b"raw bytes query"))
        store.record(_build(query="ok"))
    rows = sqlite3.connect(db).execute("SELECT typeof(query) FROM cost_records").fetchall()
    assert rows == [("text",)]


def test_good_values_still_round_trip(tmp_path: Path) -> None:
    rec = _build(query="", retrieved_count=0, prompt_tokens=0, completion_tokens=0)
    with TelemetryStore(tmp_path / "t.db") as store:
        store.record(rec)
        (back,) = store.since(0.0)
    assert (back.query, back.retrieved_count, back.prompt_tokens) == ("", 0, 0)


def test_a_row_written_before_the_check_still_reads_back(tmp_path: Path) -> None:
    # `since()` rebuilds rows through the bare constructor, which stays
    # unchecked so existing databases keep loading.
    db = tmp_path / "t.db"
    TelemetryStore(db).close()
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO cost_records (ts, query, model, retrieved_count, prompt_tokens, "
        "completion_tokens, prompt_usd, completion_usd, total_usd, total_latency_ms, "
        "per_phase_json) VALUES (?, ?, 'm', 1, 10, 5, 0.01, 0.02, 0.03, 12.0, '{}')",
        (GOOD_TS, b"legacy blob"),
    )
    con.commit()
    con.close()
    with TelemetryStore(db) as store:
        (back,) = store.since(0.0)
    assert back.query == b"legacy blob"


def test_the_dashboard_renders_what_record_accepts(tmp_path: Path) -> None:
    # The outcome the issue measured: with the seam closed, every row in the
    # store renders through both endpoints' serialisers.
    spec = importlib.util.spec_from_file_location(
        "telemetry_dashboard", ROOT / "scripts" / "telemetry_dashboard.py"
    )
    assert spec is not None
    assert spec.loader is not None
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    with TelemetryStore(tmp_path / "t.db") as store:
        for bad in (b"raw bytes query", None):
            with pytest.raises(ValueError, match="^query must be a str"):
                store.record(_bare(query=bad))
        store.record(_build(query="what is <b>rrf</b>?"))
        records = store.since(0.0)
    html = dashboard._render_dashboard_html(records)
    assert "what is &lt;b&gt;rrf&lt;/b&gt;?" in html
    assert dashboard._json_response_body(records)
