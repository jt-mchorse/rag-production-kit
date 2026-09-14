"""Latency read boundaries enforce BOTH clauses of "finite non-negative" (#213).

Both of this repo's latency write boundaries state the same two-clause rule in
the same words -- `telemetry.CostRecord.build` ("total_latency_ms must be a
finite non-negative number") and `streaming.PhaseTimings.record` ("ms must be a
finite non-negative number"), and `per_phase_ms[*]` carries it too. The
**non-finite** clause was then swept to every read and egress boundary in the
package across #38, #58, #63, #80, #81, #82, #87, #106, #108, #135 and #168. The
**non-negative** clause was never swept past the two write boundaries it was
written at.

Both guards exist *because* the data can arrive without meeting them: `CostRecord`
and `PhaseTimings` are public dataclasses with no `__post_init__`, which is the
explicit premise of #80's and #168's own docstrings --

    `record` guards finiteness at ingestion (#63), but this is a dataclass whose
    four phase lists are public init fields, so `PhaseTimings(total=[...])` --
    rebuilding timings from a persisted summary, merging across runs via
    `combined.total.extend(other.total)` -- never touches `record`

-- and that reason covers everything `record` checks, not the one clause that got
ported.

Measured on `b74753c`::

    PhaseTimings.record('total', -5.0)                        ValueError
    PhaseTimings(total=[-5.0, 10.0]).percentile('total', 50)   ->  2.5
    telemetry.percentile([-5.0, 10.0], 0.5)                    ->  2.5

    PhaseTimings(total=[-5.0, 1.0]).dump_summary_json(p)
        -> {"total": {"n": 2, "p50_ms": -2.0, ...}}

    CostRecord(total_latency_ms=-5000.0) -> record() -> SQLite -> since()
        -> aggregate() -> dump_aggregate_json, latencies [-5000.0, 10.0, 12.0]

    _render_chart_svg, SVG viewport height 240:
      control 90..109 ms           y in [  16.0,   30.1]  in bounds
      one -5000 ms among real ones y in [  16.0, 9545.3]  OFF-CANVAS
      all negative                 y in [-572.0,   16.0]  OFF-CANVAS

Two faces, and the second is why a guard beats "it would be obvious". With two
samples the negative *becomes* the reported number (`p50_ms = -2.0`). With twenty
it is invisible and still moves every percentile down -- 99.50/108.05/108.81
becomes 99.00/108.00/108.80 -- with nothing in the output naming the sample that
did it, so a latency SLO computed from the window reads better than reality.

**Where the clause deliberately does NOT go.** `telemetry.percentile` is a
general-purpose percentile over `Sequence[float]` with a public name, and
`percentile([-5.0, 5.0], 0.5) == 0.0` is the correct answer for a signed
quantity. A guard there would reject valid input, which is worse than the gap.
`test_the_general_percentile_helper_still_accepts_a_signed_sample` is a PASSING
control pinning that, so a later sweep cannot over-tighten it -- and it is the
arm the "guard it in `percentile` instead" neighbour goes red on.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import pathlib
import re
import sys
import time

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit import streaming as streaming_mod  # noqa: E402
from rag_kit import telemetry as telemetry_mod  # noqa: E402
from rag_kit.streaming import PhaseTimings  # noqa: E402
from rag_kit.telemetry import (  # noqa: E402
    CostRecord,
    ModelPrice,
    PriceTable,
    TelemetryStore,
    aggregate,
    percentile,
)
from scripts.telemetry_dashboard import _render_chart_svg  # noqa: E402

PRICES = PriceTable({"m": ModelPrice(3.0, 15.0)})

#: The two shapes each boundary must reject, and the one it must accept. Kept as
#: one table so a boundary that handles only half of it is visible as a gap in a
#: row rather than as a missing test nobody wrote.
BAD = [
    pytest.param(-5.0, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="pos-inf"),
    pytest.param(float("-inf"), id="neg-inf"),
]


def _rec(lat: float, *, ts: float = 1.0, phases: dict[str, float] | None = None) -> CostRecord:
    """A record built by bypassing `build`, which is the documented seam.

    `CostRecord` has no `__post_init__` *by design* -- `TelemetryStore.since()`
    rebuilds records from SQLite rows -- so this is the ordinary construction
    path for a loaded record, not a contrived one.
    """
    return CostRecord(
        ts=ts,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        prompt_usd=0.001,
        completion_usd=0.001,
        total_usd=0.002,
        total_latency_ms=lat,
        per_phase_ms=dict(phases or {}),
    )


def _tok_rec(prompt: int, completion: int) -> CostRecord:
    """A record with hand-set token counts, bypassing `PriceTable.cost` as a
    SQLite-loaded record does."""
    rec = _rec(1.0)
    return CostRecord(
        ts=rec.ts,
        query=rec.query,
        model=rec.model,
        retrieved_count=rec.retrieved_count,
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_usd=rec.prompt_usd,
        completion_usd=rec.completion_usd,
        total_usd=rec.total_usd,
        total_latency_ms=rec.total_latency_ms,
        per_phase_ms=dict(rec.per_phase_ms),
    )


# --- the write boundaries, re-asserted so the comparison is in one file ------


@pytest.mark.parametrize("bad", BAD)
def test_phase_timings_record_rejects_both_clauses(bad: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        PhaseTimings().record("total", bad)


@pytest.mark.parametrize("bad", BAD)
def test_cost_record_build_rejects_both_clauses(bad: float) -> None:
    with pytest.raises(ValueError, match="total_latency_ms must be a finite non-negative number"):
        CostRecord.build(
            ts=1.0,
            query="q",
            model="m",
            prompt_tokens=1,
            completion_tokens=1,
            retrieved_count=1,
            total_latency_ms=bad,
            per_phase_ms={},
            price_table=PRICES,
        )


# --- site 1: PhaseTimings.percentile ----------------------------------------


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("phase", ["retrieving", "reranking", "generating", "total"])
def test_phase_timings_percentile_rejects_both_clauses(phase: str, bad: float) -> None:
    """Every phase, not just `total`: the four lists are four public init fields."""
    timings = PhaseTimings(**{phase: [bad, 1.0]})
    with pytest.raises(ValueError, match="finite non-negative"):
        timings.percentile(phase, 50)


@pytest.mark.parametrize("bad", BAD)
def test_the_egress_path_cannot_publish_a_bad_latency(bad: float, tmp_path: pathlib.Path) -> None:
    """`summary()` / `to_dict()` / `dump_summary_json` all route through `percentile`.

    Measured on `b74753c` the negative row published
    `{"total": {"n": 2, "p50_ms": -2.0}}`. Asserting at all three callers and not
    only at `percentile` is the point: the defect was visible as a *published
    number*, and a test of the predicate alone cannot see a caller that stops
    using it.
    """
    timings = PhaseTimings(total=[bad, 1.0])
    for label, call in (
        ("summary", timings.summary),
        ("to_dict", timings.to_dict),
        ("dump_summary_json", lambda: timings.dump_summary_json(tmp_path / "s.json")),
    ):
        with pytest.raises(ValueError, match="finite non-negative"):
            call()
        assert not (tmp_path / "s.json").exists(), (
            f"{label} must refuse before writing, not truncate the destination"
        )


def test_the_merge_path_the_168_docstring_names_is_guarded() -> None:
    """`combined.total.extend(other.total)` -- quoted verbatim from #168."""
    combined = PhaseTimings()
    combined.record("total", 10.0)
    combined.total.extend(PhaseTimings(total=[-5.0]).total)
    with pytest.raises(ValueError, match="finite non-negative"):
        combined.percentile("total", 50)


def test_phase_timings_percentile_still_answers_for_good_input() -> None:
    """Anti-vacuous: the guard must not have turned the function into a raiser."""
    assert PhaseTimings(total=[1.0, 2.0]).percentile("total", 50) == pytest.approx(1.5)
    assert PhaseTimings(total=[0.0, 0.0]).percentile("total", 50) == 0.0
    assert PhaseTimings(total=[]).percentile("total", 50) is None


# --- site 2: aggregate() ----------------------------------------------------


def test_aggregate_rejects_a_negative_latency() -> None:
    with pytest.raises(ValueError, match="total_latency_ms values must all be non-negative"):
        aggregate([_rec(10.0), _rec(-5000.0), _rec(12.0)])


def test_aggregate_still_reports_the_non_finite_case_with_its_own_message() -> None:
    """#80's contract, unchanged. Two tests quote this wording.

    This is the arm that keeps the negativity clause from being bolted onto the
    finiteness message: the two diagnoses are different operator actions, and the
    existing message is part of the package's contract.
    """
    with pytest.raises(ValueError, match="finite numbers"):
        aggregate([_rec(10.0), _rec(float("nan"))])


def test_a_negative_latency_cannot_reach_the_aggregate_json(tmp_path: pathlib.Path) -> None:
    """End to end across the SQLite boundary, which is where it used to survive."""
    store = TelemetryStore(tmp_path / "t.db")
    now = time.time()
    for lat in (-5000.0, 10.0, 12.0):
        store.record(_rec(lat, ts=now))
    loaded = [r.total_latency_ms for r in store.since(now - 60)]
    assert -5000.0 in loaded, (
        "the store is deliberately not a validation boundary -- if this stops "
        "being true the metric-boundary guard is no longer the thing under test"
    )
    out = tmp_path / "agg.json"
    with pytest.raises(ValueError, match="non-negative"):
        store.dump_aggregate_json(out, since_ts=now - 60)
    assert not out.exists(), "refuse before writing, not after"


def test_aggregate_rejects_a_negative_token_count() -> None:
    """The site the inventory lock found, not the hand list (#213).

    `PriceTable.cost` rejects "token counts must be non-negative integers" on the
    way in; the sums in `aggregate` had no guard. Measured on `b74753c`: two real
    records of 100 prompt tokens plus one `prompt_tokens=-5000` gave
    `total_prompt_tokens = -4800`, and `dump_aggregate_json` published
    `{"total_prompt_tokens": -4900}`. A negative duration can be read as a clock
    artifact; a negative token count cannot be read at all.
    """
    for prompt, completion in ((-5000, 50), (100, -2000), (-1, -1)):
        with pytest.raises(ValueError, match="token counts must all be non-negative"):
            aggregate([_tok_rec(100, 50), _tok_rec(prompt, completion)])


def test_aggregate_token_totals_are_right_for_good_input() -> None:
    """Anti-vacuous for the arm above, and it pins the sum rather than the guard."""
    agg = aggregate([_tok_rec(100, 50), _tok_rec(7, 3)])
    assert (agg.total_prompt_tokens, agg.total_completion_tokens) == (107, 53)
    assert aggregate([_tok_rec(0, 0)]).total_prompt_tokens == 0


def test_aggregate_still_answers_for_good_input() -> None:
    agg = aggregate([_rec(10.0), _rec(20.0), _rec(30.0)])
    assert agg.latency_p50_ms == pytest.approx(20.0)
    assert agg.n == 3
    assert aggregate([]).latency_p50_ms == 0.0


# --- the clause that must NOT be added --------------------------------------


def test_the_general_percentile_helper_still_accepts_a_signed_sample() -> None:
    """A PASSING control, and the arm the over-tightening neighbour fails.

    `telemetry.percentile` is a general percentile over `Sequence[float]` with a
    public name. A percentile of a signed quantity -- a delta, a drift, a
    difference between two windows -- is a correct use, and `0.0` is the right
    answer for `[-5.0, 5.0]`. Pushing the latency domain into this helper would
    reject valid input, which is worse than the gap #213 closes.
    """
    assert percentile([-5.0, 5.0], 0.5) == pytest.approx(0.0)
    assert percentile([-10.0, -5.0, -1.0], 0.5) == pytest.approx(-5.0)
    assert percentile([-1.0], 0.99) == pytest.approx(-1.0)
    # The non-finite clause DOES belong here and stays.
    with pytest.raises(ValueError, match="finite numbers"):
        percentile([float("nan"), 1.0], 0.5)


# --- site 3: the dashboard chart --------------------------------------------

_CHART_HEIGHT = 240
_CHART_WIDTH = 720
_POINTS = re.compile(r'points="([^"]*)"')

_CHART_CASES = [
    pytest.param([90.0 + i for i in range(8)], id="control-real-latencies"),
    pytest.param([100.0, 105.0, -5000.0, 98.0], id="one-negative-among-real"),
    pytest.param([-5.0, -10.0, -20.0], id="all-negative"),
    pytest.param([0.0, 0.0, 0.0], id="all-zero-falsy-lat-max"),
    pytest.param([-1.0], id="single-negative"),
    pytest.param([1e12, 1.0], id="huge-and-small"),
]


#: `(label, [(ts, latency_ms)])`. The timestamps matter as much as the latencies:
#: `_render_chart_svg` derives `ts_min`/`ts_max` from the FIRST and LAST element
#: and does not sort, so an out-of-order record drives the `x` axis out of the
#: plot box exactly as a negative latency drives `y` out. `TelemetryStore.since`
#: orders by `ts ASC`, so the real path is sorted and the clamp is a backstop for
#: this function's own signature, which promises nothing about order.
_CHART_TS_CASES = [
    pytest.param([(1000.0, 100.0), (5000.0, 101.0), (1001.0, 102.0)], id="ts-out-of-order-high"),
    pytest.param([(1000.0, 100.0), (1001.0, 101.0), (-9000.0, 102.0)], id="ts-out-of-order-low"),
    pytest.param([(1000.0, 100.0), (1000.0, 101.0)], id="ts-all-equal"),
    pytest.param([(1000.0, 100.0), (5000.0, -5000.0), (1001.0, 102.0)], id="both-axes-out-at-once"),
]


def _assert_points_in_viewport(records: list[CostRecord], label: object) -> None:
    svg = _render_chart_svg(records, width=_CHART_WIDTH, height=_CHART_HEIGHT)
    match = _POINTS.search(svg)
    assert match is not None, "chart emitted no polyline element"
    assert match.group(1), "chart emitted an empty polyline"
    for pair in match.group(1).split():
        xs, ys = pair.split(",")
        x, y = float(xs), float(ys)
        assert 0.0 <= x <= _CHART_WIDTH, f"x={x} outside [0, {_CHART_WIDTH}] for {label}"
        assert 0.0 <= y <= _CHART_HEIGHT, f"y={y} outside [0, {_CHART_HEIGHT}] for {label}"


@pytest.mark.parametrize("points", _CHART_TS_CASES)
def test_no_plotted_point_falls_outside_the_viewport_on_the_time_axis(
    points: list[tuple[float, float]],
) -> None:
    """The `x` half of the clamp, which a latency-only table cannot reach.

    This arm exists because the neighbour that clamps `y` and not `x` was built
    and run against this file and came back **0 red** -- the same
    one-operand-of-two shape #213 is about, in the fix for #213. Every row in
    `_CHART_CASES` keeps `ts` increasing, so none of them can see it.
    """
    _assert_points_in_viewport([_rec(lat, ts=ts) for ts, lat in points], points)


@pytest.mark.parametrize("latencies", _CHART_CASES)
def test_no_plotted_point_falls_outside_the_viewport(latencies: list[float]) -> None:
    """The `y` half: the latency axis, with `ts` held well-ordered."""
    records = [_rec(v, ts=1000.0 + i) for i, v in enumerate(latencies)]
    _assert_points_in_viewport(records, latencies)


def test_the_chart_clamp_does_not_flatten_a_real_window() -> None:
    """Anti-vacuous: a clamp that maps everything to one y passes the bounds test.

    The control window must still produce *distinct* y values, or the guard above
    is green on a renderer that draws a straight line.
    """
    records = [_rec(90.0 + i, ts=1000.0 + i) for i in range(8)]
    svg = _render_chart_svg(records, width=_CHART_WIDTH, height=_CHART_HEIGHT)
    ys = {p.split(",")[1] for p in _POINTS.search(svg).group(1).split()}
    assert len(ys) >= 5, f"a real window must not be flattened by the clamp; got {ys}"


def test_the_raw_value_is_still_reported_somewhere() -> None:
    """The clamp is geometry only.

    A chart that silently clamps and is the *only* account of the data would be
    its own lie. The table row carries the unclamped number, which is what makes
    clamping the honest choice here rather than rejecting at the renderer.
    """
    from scripts.telemetry_dashboard import _render_dashboard_html

    html = _render_dashboard_html([_rec(-5000.0, ts=1000.0)])
    assert "-5000.0ms" in html, (
        "the unclamped latency must remain visible in the table, or the clamp "
        "becomes the only account of what happened"
    )


# --- the discovered-population lock -----------------------------------------


def _non_negative_guard_sites() -> set[str]:
    """Every `ValueError` message in the package that states "non-negative".

    Discovered over the AST of both modules rather than hand-listed, because a
    hand list is precisely what produced #213: the non-finite clause was swept
    by following the guards, and the non-negative clause was never enumerated at
    all.
    """
    sites: set[str] = set()
    for mod in (streaming_mod, telemetry_mod):
        path = pathlib.Path(inspect.getfile(mod))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Raise) or node.exc is None:
                    continue
                text = ast.unparse(node.exc)
                if "non-negative" in text:
                    sites.add(f"{path.stem}.{func.name}")
    return sites


def test_every_non_negative_guard_is_accounted_for() -> None:
    """A hard pin on the inventory, so a new clause cannot go unswept.

    When this fails, the question to answer is not "update the set" but: the new
    write boundary states a two-clause rule — does its *read* boundary agree?
    That is the question nobody asked between #63 and #213.
    """
    assert _non_negative_guard_sites() == {
        # write boundaries
        "streaming.record",
        "telemetry.build",
        "telemetry.cost",
        # read / metric boundaries (#213). `telemetry.aggregate` carries two
        # clauses -- latency and token counts -- and the second of those is the
        # site THIS LOCK FOUND: #213 opened with a hand-written list of three
        # boundaries, this assertion came back with a fourth, and the fourth has
        # the least arguable consequence of them all (`dump_aggregate_json`
        # publishing `total_prompt_tokens: -4900`). That is the whole reason the
        # population is discovered here rather than listed.
        "streaming.percentile",
        "telemetry.aggregate",
    }, (
        "the set of 'non-negative' guards changed. For each new write boundary, "
        "check that the matching read boundary enforces the same clause; for each "
        "new read boundary, check that it does not over-tighten a general helper "
        f"(see test_the_general_percentile_helper_still_accepts_a_signed_sample). "
        f"Found: {sorted(_non_negative_guard_sites())}"
    )


def test_the_inventory_matcher_can_fail() -> None:
    """Anti-vacuous: the AST matcher must find the shape it was written for."""
    src = 'def f(x):\n    if x < 0:\n        raise ValueError("x must be non-negative")\n'
    found = [
        ast.unparse(n.exc)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Raise) and n.exc is not None and "non-negative" in ast.unparse(n.exc)
    ]
    assert len(found) == 1, found
    # And a docstring merely *mentioning* the phrase is not a guard.
    prose = 'def f():\n    """Values must be non-negative."""\n    return 1\n'
    assert [n for n in ast.walk(ast.parse(prose)) if isinstance(n, ast.Raise)] == []


def test_the_two_read_boundaries_agree_with_their_write_boundaries() -> None:
    """The behavioural form of the lock above, over the actual functions.

    A source inventory says the guard exists; this says it *fires*. Both are
    needed -- the inventory catches a new unswept clause, and this catches a
    guard that is present and wrong.
    """
    for value in (-1e-9, -1.0, -1e12):
        with pytest.raises(ValueError, match="non-negative"):
            PhaseTimings().record("total", value)
        with pytest.raises(ValueError, match="non-negative"):
            PhaseTimings(total=[value]).percentile("total", 50)
        with pytest.raises(ValueError, match="non-negative"):
            aggregate([_rec(value)])
    # And the boundary itself: exactly 0.0 is a valid duration on both sides.
    PhaseTimings().record("total", 0.0)
    assert PhaseTimings(total=[0.0]).percentile("total", 50) == 0.0
    assert aggregate([_rec(0.0)]).latency_p50_ms == 0.0


def test_json_egress_of_a_guarded_path_is_still_strict_json(tmp_path: pathlib.Path) -> None:
    """The good-input path must still produce parseable JSON (#106/#135 contract)."""
    timings = PhaseTimings(total=[1.0, 2.0], retrieving=[1.0])
    out = tmp_path / "s.json"
    timings.dump_summary_json(out)
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["total"]["p50_ms"] == pytest.approx(1.5)
    assert all(
        v is None or math.isfinite(v)
        for phase in parsed.values()
        for v in phase.values()
        if isinstance(v, (int, float))
    )
