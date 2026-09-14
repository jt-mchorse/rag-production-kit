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
import sqlite3
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.streaming import _json_safe  # noqa: E402
from rag_kit.telemetry import (  # noqa: E402
    Aggregate,
    CostRecord,
    ModelPrice,
    PriceTable,
    TelemetryStore,
    aggregate,
)


def _json_response_body(records: Sequence[CostRecord]) -> bytes:
    """Serialize the ``/json`` records payload as strict-valid UTF-8 JSON.

    ``per_phase_ms`` is a free-form phase→ms mapping. A record constructed
    directly (``CostRecord`` has no ``__post_init__``, so ``build``'s finiteness
    guard is bypassed) can carry a non-finite phase value, which ``json.dumps``
    (``allow_nan=True`` by default) emits as the bare tokens ``NaN`` /
    ``Infinity`` — invalid JSON that a browser's ``fetch().then(r => r.json())``
    rejects wholesale, blanking the dashboard. Route the payload through the
    same ``_json_safe`` chokepoint the SSE wire seam uses (#106): non-finite
    floats map to ``null``, matching JavaScript's own ``JSON.stringify(NaN)``,
    so every ``/json`` response parses. The egress sibling of the #81/#82/#87/
    #106/#108 non-finite-at-the-seam sweep.
    """
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
    return json.dumps(_json_safe(payload), indent=2).encode("utf-8")


def _seed(store: TelemetryStore, n: int = 60, now: float | None = None) -> None:
    """Insert N deterministic synthetic records *strictly inside* the last 24 hours.

    The interval is half-open — ``(now - 24h, now]`` — and that matters (#170).
    Distributing as ``i / (n - 1)`` puts the oldest record on ``now - span_s``
    exactly, i.e. on the boundary. ``last_24h()`` recomputes its cutoff at
    *request* time, which is necessarily later than seed time, and ``since()``
    filters ``ts >= cutoff``; so the boundary record had always expired by the
    time anything read it. ``--seed 60`` served 59, and ``--seed 1`` served 0 —
    the dashboard rendered its "no records in window" empty state for a store
    that had just been told to seed a record.

    ``(i + 1) / n`` instead puts the oldest one interval in and the newest on
    ``now``. Landing on ``now`` is safe: ``since()`` is a lower bound only, so
    there is no symmetric boundary to fall off at the top.

    The margin against elapsed wall-clock time is ``span_s / n`` — 24 minutes
    at the documented ``--seed 60``. It shrinks as N grows, so a pathological
    seed of ~86k records could still race the clock; that is out of proportion
    to a demo path and deliberately not defended against.
    """
    now = now if now is not None else time.time()
    pt = PriceTable({"synthetic-model": ModelPrice(2.0, 8.0)})
    span_s = 24 * 3600
    for i in range(n):
        ts = now - span_s + span_s * (i + 1) / n
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


def _format_ts(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format an epoch-seconds timestamp, falling back to its raw value.

    `CostRecord.build` rejects a non-finite, non-real or `bool` `ts` at the
    write seam (#184), so the values reaching here are finite numbers. A finite
    number can still be outside what `time.gmtime` can represent -- the classic
    case being `time.time_ns()` in place of `time.time()`, which yields
    `1.7e18` and raises `OSError: [Errno 84] Value too large to be stored in
    data type` on this platform.

    That bound is a property of the platform's `time_t`, not of this repo, so it
    cannot honestly be expressed as an input-domain rule in `telemetry.py` --
    an input ceiling there would pin a host property as a contract. Guard the
    outcome instead, which is the posture `main`'s `--host` classifier below
    already argues for ("classified here rather than pre-checked").

    The harm being closed is not the ugly cell, it is the *blast radius*.
    Measured before this helper, a store holding one ordinary record and one
    with `ts=1e18`::

        _render_dashboard_html(records) -> OSError: [Errno 84] ...

    The whole page died. The operator lost every good row in the window to one
    bad one, and got a raw traceback naming `data type` rather than the record
    at fault. Returning the raw value keeps the row visible *and* legible as
    wrong, which is what an operator needs in order to go delete it.

    `OverflowError` and `ValueError` are caught alongside `OSError` because
    which one `gmtime` raises for an out-of-range value differs by platform and
    by magnitude; catching one of the three would reintroduce the defect on the
    others.
    """
    try:
        return time.strftime(fmt, time.gmtime(ts))
    except (OSError, OverflowError, ValueError):
        return f"unrepresentable ts={ts!r}"


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
    # `max(...) or 1.0` replaced a *falsy* maximum, which is `0.0` only: a
    # negative maximum is truthy and went straight through as the divisor, so
    # `1 - lat / lat_max` read inverted (#213). Floor at 0 first, then apply the
    # zero-replacement, so an all-negative window scales against 1.0 rather than
    # against a negative.
    lat_max = max(max((r.total_latency_ms for r in records), default=1.0), 0.0) or 1.0
    margin_l, margin_r, margin_t, margin_b = 40, 16, 16, 28
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def _clamp(value: float, lo: float, hi: float) -> float:
        # Geometry only, and both axes, not one (#213). `CostRecord` has no
        # `__post_init__` by design, so a directly-constructed record carrying a
        # negative `total_latency_ms` reaches this renderer -- `aggregate()`
        # rejects it at the metric boundary, but this chart draws the raw
        # records, not the aggregate. Unclamped, `y` reached 9545 in a 240px
        # viewport for one `-5000 ms` sample among real ones, and -572 for an
        # all-negative window: the polyline was drawn outside the image.
        #
        # Clamped rather than dropped, and clamped *only here*: the raw value
        # stays in the table row (`<td>{r.total_latency_ms:.1f}ms</td>`) and in
        # the `/json` payload, so the chart staying in bounds never becomes the
        # only account of what happened. That is #135's posture -- sanitize at
        # the presentation boundary -- applied to the other clause of the same
        # sentence. A NaN is not reachable here (`math.isnan` comparisons are all
        # False, so a clamp would pass it through); #135 already maps non-finite
        # to `null` on the `/json` side and the axis-label path below renders it
        # as text, which is why this function guards the *range* and not the
        # *finiteness*.
        return lo if value < lo else hi if value > hi else value

    def x(ts: float) -> float:
        raw = margin_l + plot_w * (ts - ts_min) / (ts_max - ts_min)
        return _clamp(raw, float(margin_l), float(margin_l + plot_w))

    def y(lat: float) -> float:
        raw = margin_t + plot_h * (1 - lat / lat_max)
        return _clamp(raw, float(margin_t), float(margin_t + plot_h))

    points = " ".join(f"{x(r.ts):.1f},{y(r.total_latency_ms):.1f}" for r in records)
    # Axis labels: latency max, 0; time start, time end.
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        'style="background:#fafafa;border:1px solid #eee">'
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#ccc"/>'
        f'<polyline points="{points}" fill="none" stroke="#1f6feb" stroke-width="1.5"/>'
        f'<text x="{margin_l - 6}" y="{margin_t + 10}" text-anchor="end" font-size="10" fill="#666">'
        f"{lat_max:.0f}ms</text>"
        f'<text x="{margin_l - 6}" y="{margin_t + plot_h}" text-anchor="end" font-size="10" fill="#666">'
        "0ms</text>"
        f'<text x="{margin_l}" y="{height - 8}" font-size="10" fill="#666">'
        f"{_format_ts(ts_min, '%H:%M')} UTC</text>"
        f'<text x="{width - margin_r}" y="{height - 8}" text-anchor="end" font-size="10" fill="#666">'
        f"{_format_ts(ts_max, '%H:%M')} UTC</text>"
        "</svg>"
    )


def _render_dashboard_html(records: Sequence[CostRecord]) -> str:
    # `aggregate` is a *metric* boundary and refuses to compute a summary over
    # invalid data -- a non-finite `total_usd` (#80) or, since #213, a negative
    # `total_latency_ms`. That is right for `dump_aggregate_json`, which must not
    # publish a number it cannot stand behind, and wrong to propagate from here:
    # an unhandled `ValueError` in `_handle_dashboard` becomes a bare 500 and a
    # blank page, which is the exact failure mode #135 added `_json_safe` to
    # prevent on the `/json` side. Surfaced by
    # `tests/test_latency_domain_boundaries.py` while landing #213's guard.
    #
    # So: name the problem in place of the summary row, and still render the
    # chart and the table, which carry the raw per-record values. The operator
    # ends up with MORE information than before #213 (when the summary was
    # silently computed from the bad datum) and more than a 500 would give.
    try:
        agg: Aggregate | None = aggregate(records)
        agg_error = ""
    except ValueError as e:
        agg = None
        agg_error = str(e)
    chart = _render_chart_svg(records)
    rows_html = "\n".join(
        f"<tr><td>{html.escape(_format_ts(r.ts))} UTC</td>"
        f"<td>{html.escape(r.query)}</td>"
        f"<td>{html.escape(r.model)}</td>"
        f"<td>{r.prompt_tokens}</td>"
        f"<td>{r.completion_tokens}</td>"
        f"<td>${r.total_usd:.6f}</td>"
        f"<td>{r.total_latency_ms:.1f}ms</td></tr>"
        for r in records[-20:][::-1]
    )
    if agg is None:
        stats_html = (
            '<p style="background:#fdf0ed;border:1px solid #e7b9ae;color:#8a3322;'
            'padding:10px 12px;border-radius:6px;font-size:12px">'
            "<strong>Summary unavailable.</strong> "
            + html.escape(agg_error)
            + " &mdash; the per-request table and chart below are unaffected and show "
            "the raw values."
            "</p>"
        )
    else:
        stats_html = (
            '<div class="stats">'
            f'<div class="stat"><div class="stat-label">requests</div>'
            f'<div class="stat-value">{agg.n}</div></div>'
            f'<div class="stat"><div class="stat-label">total USD</div>'
            f'<div class="stat-value">${agg.total_usd:.4f}</div></div>'
            f'<div class="stat"><div class="stat-label">p50 latency</div>'
            f'<div class="stat-value">{agg.latency_p50_ms:.0f}ms</div></div>'
            f'<div class="stat"><div class="stat-label">p95 latency</div>'
            f'<div class="stat-value">{agg.latency_p95_ms:.0f}ms</div></div>'
            f'<div class="stat"><div class="stat-label">p99 latency</div>'
            f'<div class="stat-value">{agg.latency_p99_ms:.0f}ms</div></div>'
            "</div>"
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
  .stats {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 12px 0; }}
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
{stats_html}
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
        body = _json_response_body(records)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a stdlib dashboard for the cost-telemetry SQLite store."
    )
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

    # An out-of-range port reached `ThreadingHTTPServer`'s bind() and came
    # back as a raw `OverflowError` traceback at exit 1 (#176) — the wrong
    # code for a usage error, and a diagnostic pointing at the socket layer
    # rather than at the flag the operator typed. The sibling scripts
    # already exit 2 with a flag-named message for this class (#114).
    if not 0 <= args.port <= 65535:
        parser.error(f"--port must be in 0-65535; got {args.port}")

    if args.seed > 0:
        # `--db` in a directory that doesn't exist came back as a raw
        # `sqlite3.OperationalError: unable to open database file` traceback at
        # exit 1 (#178) — reachable through the documented `--seed` path, and a
        # diagnostic naming SQLite rather than the flag the operator typed.
        # `sqlite3.Error` is NOT an `OSError` subclass, so it needs its own arm
        # alongside the `OSError` one (a path component that is a file, a
        # permission denial).
        try:
            with TelemetryStore(args.db) as store:
                _seed(store, n=args.seed)
        except (OSError, sqlite3.Error) as e:
            print(f"::error::--db {args.db!r} is not usable: {e}", file=sys.stderr)
            return 2
        print(f"seeded {args.seed} synthetic records into {args.db}", file=sys.stderr)

    _Handler.db_path = args.db
    # The `--port` range check above covers one operand of this bind tuple, and
    # its comment states the contract for both: a usage error must not surface
    # as a raw traceback at exit 1 with a diagnostic pointing at the socket
    # layer. Measured on the unguarded call (#178):
    #
    #   --host 'not a host'     -> socket.gaierror: [Errno 8] nodename nor
    #                              servname provided, or not known  (exit 1)
    #   --port <in-use port>    -> OSError: [Errno 48] Address already in use
    #                              (exit 1)
    #
    # Classified here rather than pre-checked. A hostname cannot be validated
    # ahead of the bind without reimplementing the resolver — `localhost`, a
    # `.local` name, an IPv6 literal and a bare `""` (all interfaces) are all
    # valid — so a pre-check carries false-positive risk on working setups
    # where a post-failure classifier carries none (the llm-eval-harness#194
    # posture). `socket.gaierror` subclasses `OSError`, so one arm covers both.
    #
    # Starting the dashboard twice, or on a port something else already holds,
    # is the routine case and the one where a clear message matters most.
    try:
        server = http.server.ThreadingHTTPServer((args.host, args.port), _Handler)
    except OSError as e:
        print(
            f"::error::could not bind --host {args.host!r} --port {args.port}: {e}",
            file=sys.stderr,
        )
        return 2
    # From here the socket is open, so every exit path must close it — not just
    # `serve_forever`'s. The `print` below can raise (a closed or full stdout),
    # and wrapping only `serve_forever` would leak the listener.
    try:
        print(
            f"serving http://{args.host}:{args.port}/ from {args.db} (Ctrl-C to stop)",
            file=sys.stderr,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
