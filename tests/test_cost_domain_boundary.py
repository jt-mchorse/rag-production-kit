"""`aggregate` refuses a negative `total_usd`, the clause #213 did not carry (#410).

`ModelPrice` refuses a negative rate ("a finite number >= 0.0", #36) and
`PriceTable.cost` a negative token count, so `CostRecord.build` can never produce
negative spend. `TelemetryStore.record` leaves the sign to the read side by design
(#245), and #213 carried the non-negative clause to `aggregate` for latency and
token counts -- not for cost. Measured on `49660aa`::

    aggregate([CostRecord(total_usd=-5.0), CostRecord(total_usd=0.002)]).total_usd
        -> -4.998
    record() -> since() -> dump_aggregate_json   -> "total_usd": -4.998
    dashboard headline                           -> total USD $-4.9980

The money fields, enumerated rather than taken from the issue: `CostRecord` has
`prompt_usd`, `completion_usd` and `total_usd`; `Aggregate`, the `/json` rows and
the dashboard read only `total_usd`. The population arm at the bottom pins that,
so a future field that sums `prompt_usd` has to come back here.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import sys
import time

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.telemetry import Aggregate, CostRecord, TelemetryStore, aggregate  # noqa: E402
from scripts.telemetry_dashboard import (  # noqa: E402
    _json_response_body,
    _render_dashboard_html,
)


def _rec(
    usd: float,
    *,
    ts: float = 1.0,
    prompt_usd: float = 0.001,
    completion_usd: float = 0.001,
) -> CostRecord:
    """A bare-constructed record: the seam `build`'s pricing guards do not cover."""
    return CostRecord(
        ts=ts,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        prompt_usd=prompt_usd,
        completion_usd=completion_usd,
        total_usd=usd,
        total_latency_ms=100.0,
        per_phase_ms={},
    )


@pytest.mark.parametrize("bad", [-5.0, -1e-9, -1e12, -1])
def test_aggregate_rejects_a_negative_total_usd(bad: float) -> None:
    with pytest.raises(ValueError, match="total_usd values must all be non-negative"):
        aggregate([_rec(bad), _rec(0.002)])


def test_the_issue_case_no_longer_publishes_negative_spend() -> None:
    """The measured repro from #410, as a value check: no -4.998 comes back."""
    with pytest.raises(ValueError, match=r"got \[-5\.0, 0\.002\]"):
        aggregate([_rec(-5.0), _rec(0.002)])


def test_a_negative_hidden_by_a_positive_sum_is_still_refused() -> None:
    """Per record, not on the sum: -5.0 + 10.0 publishes 5.0, and is just as wrong.

    This is the arm a guard on `sum(costs) < 0` goes red on.
    """
    with pytest.raises(ValueError, match="total_usd values must all be non-negative"):
        aggregate([_rec(10.0), _rec(-5.0)])


def test_zero_and_positive_costs_still_sum() -> None:
    """Control: exactly 0.0 is a valid cost (a cached or free call)."""
    assert aggregate([_rec(0.0)]).total_usd == 0.0
    assert aggregate([_rec(0.002), _rec(0.0), _rec(0.003)]).total_usd == pytest.approx(0.005)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_the_finiteness_message_is_unchanged(bad: float) -> None:
    """Control: #80's wording still wins for a non-finite cost, -inf included.

    `-inf < 0` is True, so a negativity check placed first would have taken the
    -inf case over with the wrong diagnosis.
    """
    with pytest.raises(ValueError, match="total_usd values must all be finite numbers"):
        aggregate([_rec(0.002), _rec(bad)])


def test_a_negative_cost_cannot_reach_the_aggregate_json(tmp_path: pathlib.Path) -> None:
    """End to end across SQLite, which keeps the row by design (#245)."""
    store = TelemetryStore(tmp_path / "t.db")
    now = time.time()
    store.record(_rec(-5.0, ts=now))
    store.record(_rec(0.002, ts=now))
    loaded = sorted(r.total_usd for r in store.since(now - 60))
    assert loaded == [-5.0, 0.002], (
        "record() is deliberately not the sign boundary (#245); if it starts "
        "refusing, the read-side guard is no longer what this test exercises"
    )
    out = tmp_path / "agg.json"
    with pytest.raises(ValueError, match="total_usd values must all be non-negative"):
        store.dump_aggregate_json(out, since_ts=now - 60)
    assert not out.exists(), "refuse before writing, not after"


def test_the_dashboard_names_the_problem_instead_of_a_negative_headline() -> None:
    """The headline goes; the raw value stays in the table (#213's posture)."""
    page = _render_dashboard_html([_rec(-5.0, ts=1000.0), _rec(0.002, ts=1001.0)])
    assert "Summary unavailable." in page
    assert "total_usd values must all be non-negative" in page
    assert "$-4.9980" not in page
    assert "total USD</div>" not in page
    assert "<td>$-5.000000</td>" in page


def test_the_dashboard_headline_is_unchanged_for_good_input() -> None:
    """Control for the arm above."""
    page = _render_dashboard_html([_rec(0.002, ts=1000.0), _rec(0.003, ts=1001.0)])
    assert "Summary unavailable." not in page
    assert '<div class="stat-value">$0.0050</div>' in page


# --- the money-field population ---------------------------------------------


def test_total_usd_is_the_only_money_field_anything_publishes() -> None:
    """Enumerated from the types and the serialised payloads, not from a list.

    `CostRecord` carries three money fields. If a new `Aggregate` field or `/json`
    key reads `prompt_usd` / `completion_usd`, this fails, and the new read path
    needs the same non-negative clause `aggregate` gives `total_usd`.
    """
    record_money = {f.name for f in dataclasses.fields(CostRecord) if "usd" in f.name}
    assert record_money == {"prompt_usd", "completion_usd", "total_usd"}

    aggregate_money = {f.name for f in dataclasses.fields(Aggregate) if "usd" in f.name}
    assert aggregate_money == {"total_usd"}
    assert {k for k in aggregate([_rec(0.002)]).to_dict() if "usd" in k} == {"total_usd"}

    rows = json.loads(_json_response_body([_rec(0.002)]))["records"]
    assert {k for k in rows[0] if "usd" in k} == {"total_usd"}


def test_a_negative_component_cost_is_not_published_anywhere() -> None:
    """Behavioural form of the arm above: a bare record with negative components
    and a valid total reaches no published number, so there is nothing to guard.
    """
    recs = [_rec(0.002, ts=1000.0, prompt_usd=-7.25, completion_usd=-3.5)]
    agg = aggregate(recs)
    assert agg.total_usd == 0.002
    published = (
        json.dumps(agg.to_dict())
        + _json_response_body(recs).decode()
        + _render_dashboard_html(recs)
    )
    assert "-7.25" not in published
    assert "-3.5" not in published
    assert all(math.isfinite(v) for v in agg.to_dict().values())
