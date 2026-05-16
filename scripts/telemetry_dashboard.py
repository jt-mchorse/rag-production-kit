"""Minimal stdlib HTTP dashboard for the cost-telemetry SQLite store.

Serves one page with an inline-SVG chart of the last 24 hours of
records (total USD over time) plus a per-request latency view, all
dep-free — no Chart.js, no JS framework, no external CDNs. The chart
renders air-gapped because everything is computed server-side from
``rag_kit.telemetry.TelemetryStore.last_24h()``.

Usage:

    python -m scripts.telemetry_dashboard --db ./telemetry.db --port 8766
    # http://127.0.0.1:8766/

For demo / development only — single-threaded ``http.server`` is fine
for one operator looking at their own data on localhost; production
deployments should ship records to a managed metrics backend.

The ``--seed`` flag fills the database with deterministic synthetic
records so the dashboard can be exercised on a fresh machine without
running real queries first. Seed records are clearly labeled in their
``query`` field as ``synthetic-N`` so the operator can tell them apart
from real telemetry.
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.telemetry import (  # noqa: E402
    CostRecord,
    ModelPrice,
    PriceTable,
    TelemetryStore,
    aggregate,
)


def _seed(store: TelemetryStore, n: int = 60, now: float | None = None) -> None:
    """Insert N deterministic synthetic records spanning the last 24 hours."""
    now = now if now is not None else time.time()
    pt = PriceTable({"synthetic-model": ModelPrice(2.0, 8.0)})
    span_s = 24 * 3600
    for i in range(n):
        ts = now - span_s + (span_s * i / max(n - 1, 1))
        # Deterministic-but-varied latency curve so the chart shows shape.
        latency_ms = 80.0 + 40.0 * ((i * 7) % 13) / 12 + 20.0 * ((i * 11) % 7) / 6
        rec = CostRecord.build(
            ts=ts,
            query=f"synthetic-{i}",
            model="synthetic-model",
            retrieved_count=3,
            prompt_tokens=1200 + i * 5,
            completion_tokens=180 + (i * 3) % 50,
            total_latency_ms=latency_ms,
            per_phase_ms={
                "retrieving": latency_ms * 0.15,
                "reranking": latency_ms * 0.10,
                "generating": latency_ms * 0.75,
            },
            price_table=pt,
        )
        store.record(rec)


def _render_chart_svg(records: Sequence[CostRecord], width: int = 720, height: int = 240) -> str:
    """Per-request latency over time, as an inline SVG line chart."""
    if not records:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="50%" y="50%" text-anchor="middle" fill="#666">'
            "no records in window"
            "</text></svg>"
        )
    ts_min = records[0].ts
    ts_max = records[-1].ts if records[-1].ts > ts_min else ts_min + 1.0
    lat_max = max((r.total_latency_ms for r in records), default=1.0) or 1.0
    margin_l, margin_r, margin_t, margin_b = 40, 16, 16, 28
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def x(ts: float) -> float:
        return margin_l + plot_w * (ts - ts_min) / (ts_max - ts_min)

    def y(lat: float) -> float:
        return margin_t + plot_h * (1 - lat / lat_max)

    points = " ".join(f"{x(r.ts):.1f},{y(r.total_latency_ms):.1f}" for r in records)
    # Axis labels: latency max, 0; time start, time end.
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        'style="background:#fafafa;border:1px solid #eee">'
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#ccc"/>'
        f'<polyline points="{points}" fill="none" stroke="#1f6feb" stroke-width="1.5"/>'
        f'<text x="{margin_l - 6}" y="{margin_t + 10}" text-anchor="end" font-size="10" fill="#666">'
        f'{lat_max:.0f}ms</text>'
        f'<text x="{margin_l - 6}" y="{margin_t + plot_h}" text-anchor="end" font-size="10" fill="#666">'
        "0ms</text>"
        f'<text x="{margin_l}" y="{height - 8}" font-size="10" fill="#666">'
        f"{time.strftime('%H:%M', time.gmtime(ts_min))} UTC</text>"
        f'<text x="{width - margin_r}" y="{height - 8}" text-anchor="end" font-size="10" fill="#666">'
        f"{time.strftime('%H:%M', time.gmtime(ts_max))} UTC</text>"
        "</svg>"
    )


def _render_dashboard_html(records: Sequence[CostRecord]) -> str:
    agg = aggregate(records)
    chart = _render_chart_svg(records)
    rows_html = "\n".join(
        f"<tr><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(r.ts))} UTC</td>"
        f"<td>{html.escape(r.query)}</td>"
        f"<td>{html.escape(r.model)}</td>"
        f"<td>{r.prompt_tokens}</td>"
        f"<td>{r.completion_tokens}</td>"
        f"<td>${r.total_usd:.6f}</td>"
        f"<td>{r.total_latency_ms:.1f}ms</td></tr>"
        for r in records[-20:][::-1]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>rag-production-kit telemetry</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 800px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1 {{ font-size: 20px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0; }}
  .stat {{ background: #f4f4f4; padding: 10px 12px; border-radius: 6px; }}
  .stat-label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 18px; font-weight: 600; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 16px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }}
  th {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Cost telemetry — last 24 hours</h1>
<div class="stats">
  <div class="stat"><div class="stat-label">requests</div><div class="stat-value">{agg.n}</div></div>
  <div class="stat"><div class="stat-label">total USD</div><div class="stat-value">${agg.total_usd:.4f}</div></div>
  <div class="stat"><div class="stat-label">p50 latency</div><div class="stat-value">{agg.latency_p50_ms:.0f}ms</div></div>
  <div class="stat"><div class="stat-label">p95 latency</div><div class="stat-value">{agg.latency_p95_ms:.0f}ms</div></div>
</div>
<h2 style="font-size: 14px; color:#555">Per-request latency over time</h2>
{chart}
<h2 style="font-size: 14px; color:#555; margin-top: 24px">Most recent 20 records</h2>
<table>
  <thead><tr><th>Time</th><th>Query</th><th>Model</th><th>Prompt tok</th><th>Completion tok</th><th>USD</th><th>Latency</th></tr></thead>
  <tbody>
    {rows_html or '<tr><td colspan="7" style="text-align:center;color:#999">no records</td></tr>'}
  </tbody>
</table>
<p style="color:#888;font-size:11px;margin-top:24px">
Single-page dashboard — dep-free stdlib HTTP server, inline SVG, no external assets.
Served by <code>scripts/telemetry_dashboard.py</code>.
</p>
</body>
</html>
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "rag-kit-telemetry/0.1"
    db_path: str = ""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Quiet single-page dashboard; the dev sees the data, not access logs.
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._handle_dashboard()
        elif parsed.path == "/api/last_24h":
            self._handle_json()
        else:
            self.send_error(404, f"unknown path: {parsed.path}")

    def _handle_dashboard(self) -> None:
        with TelemetryStore(self.db_path) as store:
            records = store.last_24h()
        body = _render_dashboard_html(records).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_json(self) -> None:
        with TelemetryStore(self.db_path) as store:
            records = store.last_24h()
        payload = {
            "records": [
                {
                    "ts": r.ts,
                    "query": r.query,
                    "model": r.model,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_usd": r.total_usd,
                    "total_latency_ms": r.total_latency_ms,
                    "per_phase_ms": dict(r.per_phase_ms),
                }
                for r in records
            ],
        }
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a stdlib dashboard for the cost-telemetry SQLite store.")
    parser.add_argument("--db", default="./telemetry.db", help="Path to the telemetry SQLite file.")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port to listen on.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost).")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="If >0, insert this many deterministic synthetic records before serving.",
    )
    args = parser.parse_args(argv)

    if args.seed > 0:
        with TelemetryStore(args.db) as store:
            _seed(store, n=args.seed)
        print(f"seeded {args.seed} synthetic records into {args.db}", file=sys.stderr)

    _Handler.db_path = args.db
    server = http.server.ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"serving http://{args.host}:{args.port}/ from {args.db} (Ctrl-C to stop)",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
