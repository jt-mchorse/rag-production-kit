"""`telemetry.percentile` validates `q` before the empty-window return (#330).

The `q` check sat after `if not values: return 0.0`, so a bad `q` was 0.0 on
an empty window and an error only once the first record landed. And a bare
range check read `True`/`False` as q=1.0/0.0. `PhaseTimings.percentile`, which
this function's docstring says it matches, checks `p` first and refuses bool.
"""

from __future__ import annotations

import math

import pytest

from rag_kit.telemetry import percentile

_BAD_Q = [95, 1.5, -0.1, math.nan, math.inf, True, False, "0.5", None]


@pytest.mark.parametrize("q", _BAD_Q, ids=repr)
def test_a_bad_q_is_refused_on_an_empty_window(q: object) -> None:
    with pytest.raises(ValueError, match=r"^q must be"):
        percentile([], q)  # type: ignore[arg-type]


@pytest.mark.parametrize("q", _BAD_Q, ids=repr)
def test_a_bad_q_is_refused_on_a_populated_window(q: object) -> None:
    with pytest.raises(ValueError, match=r"^q must be"):
        percentile([1.0, 2.0, 3.0], q)  # type: ignore[arg-type]


@pytest.mark.parametrize("q", [0, 0.0, 0.5, 0.95, 1, 1.0])
def test_an_in_range_q_on_an_empty_window_is_still_zero(q: float) -> None:
    assert percentile([], q) == 0.0


def test_in_range_results_are_unchanged() -> None:
    data = [float(v) for v in range(1, 11)]
    assert percentile(data, 0) == 1.0
    assert percentile(data, 1) == 10.0
    assert percentile(data, 0.5) == pytest.approx(5.5)
    assert percentile(data, 0.95) == pytest.approx(9.55)
