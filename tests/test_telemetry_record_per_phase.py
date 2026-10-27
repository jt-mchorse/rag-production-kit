"""`TelemetryStore.record` holds a bare record's `per_phase_ms` to `build`'s rule (#406).

#387 (PR #390) made `build` refuse non-`str` phase keys, and #108/#392 had already
given each phase value the finite non-negative rule. `record` never checked the
field, and the bare `CostRecord(...)` constructor is public. Measured on origin/main
1b6ebd6 with `record` then `since(-inf)`:

    {("a", "b"): 1.0}            -> raw TypeError: keys must be str, int, float, bool or None
    {1: 5.0, "1": 6.0}           -> read back {'1': 6.0}; the 5.0 is lost
    {None: 5.0}                  -> read back {'null': 5.0}
    {"retrieving": "fast" | None | nan | inf | -3.0 | [1] | True | 2**70 | 10**400}
                                 -> stored and read back as given

The parity arm below takes `build`'s own refusal message for each value and
requires `record` to raise the same one and store nothing.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pytest

from rag_kit import CostRecord, TelemetryStore
from rag_kit.telemetry import ModelPrice, PriceTable

GOOD_TS = 1_700_000_000.0
TABLE = PriceTable({"m": ModelPrice(3.0, 15.0)})
INT_MAX = 2**63 - 1


def _build(per_phase_ms: Any) -> CostRecord:
    return CostRecord.build(
        ts=GOOD_TS,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_latency_ms=12.0,
        per_phase_ms=per_phase_ms,
        price_table=TABLE,
    )


def _bare(per_phase_ms: Any) -> CostRecord:
    return CostRecord(
        ts=GOOD_TS,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        prompt_usd=0.01,
        completion_usd=0.02,
        total_usd=0.03,
        total_latency_ms=12.0,
        per_phase_ms=per_phase_ms,
    )


# Every map here is one `build` refuses (asserted below, so a row that stops
# being refused there fails instead of passing vacuously).
REFUSED: dict[str, dict[Any, Any]] = {
    "tuple-key": {("a", "b"): 1.0},
    "int-key-colliding-with-str": {1: 5.0, "1": 6.0},
    "none-key": {None: 5.0},
    "int-key": {1: 5.0},
    "bytes-key": {b"retrieving": 5.0},
    "bool-key": {True: 1.0},
    "str-value": {"retrieving": "fast"},
    "none-value": {"retrieving": None},
    "list-value": {"retrieving": [1]},
    "bool-value": {"retrieving": True},
    "nan-value": {"retrieving": math.nan},
    "inf-value": {"retrieving": math.inf},
    "neg-inf-value": {"retrieving": -math.inf},
    "negative-value": {"retrieving": -3.0},
    "negative-int-value": {"retrieving": -1},
    "int-past-int64": {"retrieving": 2**70},
    "int-past-float-range": {"retrieving": 10**400},
    "bad-after-good": {"retrieving": 1.0, "generating": math.nan},
}


def _build_message(phases: dict[Any, Any]) -> str:
    with pytest.raises(ValueError, match=r"^per_phase_ms") as info:
        _build(phases)
    return str(info.value)


@pytest.mark.parametrize("label", list(REFUSED))
def test_record_refuses_what_build_refuses_in_the_same_words(tmp_path: Path, label: str) -> None:
    phases = REFUSED[label]
    expected = _build_message(phases)
    with TelemetryStore(tmp_path / "t.db") as store:
        # A raw TypeError is not a ValueError, so the unfixed tuple key errors
        # here rather than passing.
        with pytest.raises(ValueError, match=rf"^{re.escape(expected)}$"):
            store.record(_bare(phases))
        assert store.since(-math.inf) == []


def test_a_refused_record_does_not_block_the_next_one(tmp_path: Path) -> None:
    with TelemetryStore(tmp_path / "t.db") as store:
        with pytest.raises(ValueError, match=r"per_phase_ms keys must be str"):
            store.record(_bare({None: 5.0}))
        store.record(_bare({"retrieving": 5.0}))
        (row,) = store.since(-math.inf)
    assert row.per_phase_ms == {"retrieving": 5.0}


def test_a_map_mutated_after_construction_is_checked_at_record(tmp_path: Path) -> None:
    # The record is frozen but its dict is not; `record` checks what it stores.
    rec = _bare({"retrieving": 5.0})
    rec.per_phase_ms["generating"] = -1.0  # type: ignore[index]
    with TelemetryStore(tmp_path / "t.db") as store:
        with pytest.raises(ValueError, match=r"per_phase_ms\['generating'\] must be a finite"):
            store.record(rec)
        assert store.since(-math.inf) == []


# --- controls: what `build` accepts, a bare record still stores --------------

ACCEPTED: dict[str, dict[str, Any]] = {
    "empty": {},
    "floats": {"retrieving": 50.0, "reranking": 30.0, "generating": 340.0},
    "zero-int-and-float": {"a": 0, "b": 0.0},
    "int64-max": {"a": INT_MAX},
    "huge-float": {"a": 1e300},
    "odd-str-keys": {"": 1.0, "null": 2.0, "1": 3.0, "fusión": 4.0},
}


@pytest.mark.parametrize("label", list(ACCEPTED))
def test_a_bare_record_build_would_accept_still_round_trips(tmp_path: Path, label: str) -> None:
    phases = ACCEPTED[label]
    assert _build(phases).per_phase_ms == phases  # build accepts it too
    with TelemetryStore(tmp_path / "t.db") as store:
        store.record(_bare(phases))
        (row,) = store.since(-math.inf)
    assert row.per_phase_ms == phases
