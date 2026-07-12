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

import re
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


def test_readme_stat_card_count_matches_rendered_tiles() -> None:
    # README's Endpoints section hardcodes "<N> stat cards". #122 added the p99
    # tile (4 -> 5) and updated the p50/p95/p99 prose + the CSS-grid lock, but
    # left this count at the stale "4". Pin the prose count to the actual number
    # of `class="stat"` tiles the dashboard renders so it can't drift again.
    html = _render_dashboard_html([])
    rendered_tiles = html.count('class="stat"')
    assert rendered_tiles == 5  # sanity: keep this in step with the tile set

    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"(\d+)\s+stat cards", readme)
    assert m is not None, "README no longer states an 'N stat cards' count"
    claimed = int(m.group(1))
    assert claimed == rendered_tiles, (
        f"README claims {claimed} stat cards but the dashboard renders "
        f'{rendered_tiles} `class="stat"` tiles'
    )
