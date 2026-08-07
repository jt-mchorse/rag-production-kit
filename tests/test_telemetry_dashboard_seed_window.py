"""`--seed N` must serve exactly N records, not N-1 (#170).

`_seed` distributed records as ``now - span_s + span_s * i / max(n - 1, 1)``,
putting the oldest on ``now - span_s`` exactly. `last_24h()` recomputes its
cutoff at *request* time — necessarily later than seed time — and `since()`
filters ``ts >= cutoff``, so the boundary record was always already outside
the window. The seeder printed "seeded 60 synthetic records" and the
dashboard then showed 59.

The `n == 1` case is the sharp one and the reason this test parametrizes
down to 1 rather than starting at the documented 60: with a single record,
`max(n - 1, 1)` is 1 and `i=0` still lands on the boundary, so *the whole
store* fell outside the window and the dashboard rendered its "no records
in window" empty state.

These assertions are anchored to the served count — the operator-visible
number — rather than to the arithmetic, so a later refactor of the
distribution formula can't quietly reintroduce a boundary record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.telemetry import TelemetryStore  # noqa: E402
from scripts.telemetry_dashboard import _seed  # noqa: E402

SPAN_S = 24 * 3600


def _store(tmp_path: Path) -> TelemetryStore:
    return TelemetryStore(str(tmp_path / "telemetry.db"))


@pytest.mark.parametrize("n", [1, 2, 3, 10, 60])
def test_seed_n_serves_exactly_n_records(tmp_path: Path, n: int) -> None:
    """The acceptance criterion: asked for N, the window holds N."""
    store = _store(tmp_path)
    seed_now = 1_760_000_000.0
    _seed(store, n=n, now=seed_now)

    # Any real reader runs *after* the seed. Half a second is generous —
    # pre-fix, even a microsecond dropped the boundary record.
    served = store.last_24h(now=seed_now + 0.5)
    assert len(served) == n, (
        f"--seed {n} must serve {n} records; got {len(served)}. "
        "The oldest record is on or outside the 24h boundary."
    )


def test_seed_one_is_not_an_empty_dashboard(tmp_path: Path) -> None:
    """`--seed 1` served zero records pre-fix — the worst case, not N-1."""
    store = _store(tmp_path)
    seed_now = 1_760_000_000.0
    _seed(store, n=1, now=seed_now)
    assert store.last_24h(now=seed_now + 0.5), (
        "a single seeded record must be inside the window; an empty result "
        "renders the dashboard's 'no records in window' state"
    )


def test_seed_survives_a_long_gap_between_seeding_and_the_request(tmp_path: Path) -> None:
    """The margin is real time, not epsilon.

    At the documented `--seed 60` the oldest record sits `span_s / n` = 24
    minutes inside the window, so an operator who seeds and then loads the
    page some minutes later still sees all 60.
    """
    store = _store(tmp_path)
    seed_now = 1_760_000_000.0
    _seed(store, n=60, now=seed_now)
    ten_minutes = 600.0
    assert len(store.last_24h(now=seed_now + ten_minutes)) == 60


def test_seeded_timestamps_are_deterministic_and_ordered(tmp_path: Path) -> None:
    """Same `now` ⇒ identical timestamps, ascending, all within the window."""
    seed_now = 1_760_000_000.0
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _store(tmp_path / "a")
    second = _store(tmp_path / "b")
    _seed(first, n=25, now=seed_now)
    _seed(second, n=25, now=seed_now)

    ts_a = [r.ts for r in first.last_24h(now=seed_now)]
    ts_b = [r.ts for r in second.last_24h(now=seed_now)]

    assert ts_a == ts_b, "seeding is documented as deterministic given the same `now`"
    assert ts_a == sorted(ts_a), "records must be chronological for the chart"
    assert ts_a[0] > seed_now - SPAN_S, "oldest record must be strictly inside the window"
    assert ts_a[-1] <= seed_now, "newest record must not be in the future"


def test_seeded_records_still_span_nearly_the_whole_window(tmp_path: Path) -> None:
    """The chart keeps its shape: the fix shifts records, it doesn't bunch them.

    With `n=60` the covered span is `span_s * (n - 1) / n` — 23h36m of the
    24h window. Asserting >90% keeps the intent (a full-width chart) without
    pinning the exact formula.
    """
    store = _store(tmp_path)
    seed_now = 1_760_000_000.0
    _seed(store, n=60, now=seed_now)
    ts = [r.ts for r in store.last_24h(now=seed_now)]
    assert (ts[-1] - ts[0]) / SPAN_S > 0.9
