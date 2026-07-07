"""Doc-vs-code lock for the telemetry dashboard stat tiles.

`Aggregate` computes `latency_p50_ms` / `latency_p95_ms` / `latency_p99_ms`,
and two doc surfaces promise the served dashboard renders all three:

- README.md: "renders the last 24 hours with p50 / p95 / p99 latency ..."
- docs/architecture.md: "aggregated into p50/p95/p99 percentiles, rendered by
  a stdlib-only HTTP dashboard"

The rendered HTML only carried p50 + p95 tiles, so the operator-facing page
silently dropped the p99 the docs advertise. This test pins every documented
percentile tile to the rendered HTML so the dashboard and its docs can't drift
apart again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.telemetry_dashboard import _render_dashboard_html  # noqa: E402


@pytest.mark.parametrize("label", ["p50 latency", "p95 latency", "p99 latency"])
def test_dashboard_renders_every_documented_percentile_tile(label: str) -> None:
    # Empty records still render the static stat tiles (with 0ms values), which
    # is all this doc-contract check needs.
    html = _render_dashboard_html([])
    assert label in html, f"dashboard HTML is missing the documented {label!r} tile"


def test_dashboard_stat_grid_width_matches_tile_count() -> None:
    # Five tiles (requests, total USD, p50, p95, p99) — the CSS grid must be
    # wide enough or the last tile wraps to a second row.
    html = _render_dashboard_html([])
    assert "repeat(5, 1fr)" in html
