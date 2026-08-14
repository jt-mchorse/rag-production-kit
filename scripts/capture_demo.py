#!/usr/bin/env python3
"""Deterministic capture orchestrator for the rag-production-kit 60-second demo.

Sequences the two demo surfaces from issue #25's spec (terminal SSE
server + Next.js frontend) under explicit stage banners + configurable
inter-stage pause. STAGE 1 also includes a hermetic in-process preview
of the streaming pipeline so the recording captures the exact phase
sequence without depending on a live HTTP server.

Stages:

- **STAGE 1 (auto, hermetic).** Composes `StreamingPipeline` directly
  with the same `FakeRetriever` / `SlowReranker` / `_stub_token_stream`
  the SSE server uses (`demo.streaming.server`), iterates the events,
  and prints each one as an SSE frame. The recording sees the exact
  `retrieving → retrieved → reranking → reranked → generating →
  token* → generated → done` sequence — no port allocation, no
  client-server race.
- **STAGE 2 (operator-action).** Cheat-sheet for the actual SSE server
  the README cites: `python -m demo.streaming.server` and `curl -N
  'http://localhost:8765/stream?q=postgres+tuning'`. The `--launch-server`
  flag subprocess-spawns the server and curls it for the operator;
  off by default because the server is a long-running process that
  can't run hermetically in CI.
- **STAGE 3 (operator-action).** Cheat-sheet for the Next.js dev
  server: `cd demo/nextjs && npm run dev` and the URL the recording
  navigates to, plus a click checklist (citation chips → retrieved-
  chunks panel → phase pills).

Usage:

    python scripts/capture_demo.py [--pause-seconds 2.0] [--no-open]
                                   [--query 'postgres tuning']
                                   [--launch-server]
                                   [--skip-server-cheatsheet]
                                   [--skip-nextjs-cheatsheet]

Closes the AC3 row on #25. AC1 (committed GIF/MP4) and AC2 (README
embed) remain operator-only. Locked by
`tests/test_capture_demo_smoke.py`, same hermetic contract as
`tests/test_streaming.py`.
"""

from __future__ import annotations

import argparse
import importlib
import math
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUERY = "postgres tuning"
DEFAULT_K = 3

SSE_SERVER_URL = "http://localhost:8765"
SSE_STREAM_PATH = "/stream?q=postgres+tuning"
NEXTJS_DEV_URL = "http://localhost:3000"


def _banner(stage: int, title: str) -> str:
    line = "=" * 72
    return f"\n{line}\n  STAGE {stage}  {title}\n{line}\n"


def _validate_pause_seconds(seconds: float) -> str | None:
    """Return an error message for an unusable ``--pause-seconds``, else ``None``.

    ``type=float`` is not validation, and both directions of the unguarded
    domain were live (#176). This was the portfolio's last unguarded
    ``capture_demo`` pause; the shape is ported from the identical fix in
    llm-eval-harness#198, including the ``bool`` exclusion (``bool``
    subclasses ``int``, so ``True`` would otherwise pass as ``1.0`` for a
    caller reaching ``main(argv=...)`` programmatically).

    - ``inf`` is the loud half: ``time.sleep(inf)`` raises ``OverflowError``
      ("timestamp out of range for platform time_t"), but only at the first
      ``_pause`` — which is *after* STAGE 1 has already run its streaming
      preview, so the operator loses a partial capture to a usage error that
      is free to reject here. ``1e400`` reaches the same place while looking
      like a finite, in-range literal: ``float("1e400")`` is ``inf``.
    - ``nan`` and negatives are the quiet half, and the worse one. ``_pause``
      guards ``if seconds > 0``; ``nan > 0`` and ``-1 > 0`` are both ``False``,
      so every stage prints and the script exits 0 having taken no pause at
      all. The inter-stage pauses are this script's only reason to exist (the
      module docstring calls them the recorder's cue points), so a silent
      exit-0 run whose recording is unusable beats a crash for badness.

    ``0`` stays valid — it is the documented CI value.
    """
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return f"--pause-seconds must be a number; got {seconds!r}"
    if math.isnan(seconds) or math.isinf(seconds):
        return f"--pause-seconds must be finite; got {seconds!r}"
    if seconds < 0:
        return f"--pause-seconds must be >= 0; got {seconds!r}"
    return None


def _pause(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _import_streaming_demo_internals():
    """Import the stubs already shipped in `demo.streaming.server` —
    `FakeRetriever`, `SlowReranker`, `_stub_token_stream`, `_CORPUS` —
    plus `StreamingPipeline` and `to_sse` from `rag_kit.streaming`.

    Sharing the stubs with the server (rather than redefining them in
    the capture script) means a future change to the demo's corpus or
    timing automatically flows through to the recorded sequence —
    they can't drift apart.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if "demo.streaming.server" in sys.modules:
        del sys.modules["demo.streaming.server"]
    server_mod = importlib.import_module("demo.streaming.server")
    streaming_mod = importlib.import_module("rag_kit.streaming")
    return {
        "FakeRetriever": server_mod.FakeRetriever,
        "SlowReranker": server_mod.SlowReranker,
        "_stub_token_stream": server_mod._stub_token_stream,  # noqa: SLF001 — intentional reuse
        "_CORPUS": server_mod._CORPUS,  # noqa: SLF001
        "StreamingPipeline": streaming_mod.StreamingPipeline,
        "to_sse": streaming_mod.to_sse,
    }


def _run_streaming_preview(query: str, k: int) -> list[str]:
    """Run the streaming pipeline in-process and return the list of
    event-type names actually emitted, in order. Prints each event
    as an SSE frame so the recording matches the on-wire format the
    real server emits.
    """
    parts = _import_streaming_demo_internals()
    pipe = parts["StreamingPipeline"](
        parts["FakeRetriever"](parts["_CORPUS"]),
        reranker=parts["SlowReranker"](),
        token_stream=parts["_stub_token_stream"],
    )
    event_types: list[str] = []
    for event in pipe.run(query, k=k):
        event_types.append(event.type)
        frame = parts["to_sse"](event)
        # Trailing newline is part of the SSE frame; print without an
        # extra one so the terminal output matches what `curl -N` shows
        # against the live server.
        print(frame, end="")
    return event_types


def _server_cheatsheet(query: str) -> str:
    return (
        "# SSE-server demo (STAGE 2) — operator steps.\n"
        "# The hermetic STAGE 1 preview above shows the exact event\n"
        "# sequence the server emits over the wire. This stage runs\n"
        "# the actual HTTP server so the recording captures the\n"
        "# command the README cites.\n"
        "#\n"
        "# 1. Start the SSE server in a separate terminal:\n"
        "#      python -m demo.streaming.server\n"
        "#\n"
        f"# 2. Curl the stream endpoint and pipe through `cat -e` so\n"
        f"#    the SSE frame boundaries are visible in the recording:\n"
        f"#      curl -N '{SSE_SERVER_URL}{SSE_STREAM_PATH.replace(' ', '+')}'\n"
        f"#    (URL-encode the query; default in the README is\n"
        f"#    `q=postgres+tuning`. This stage's --query override is\n"
        f"#    only applied to STAGE 1's hermetic preview.)\n"
        "#\n"
        "# 3. Ctrl-C the server when the recording cuts to STAGE 3."
    )


def _nextjs_cheatsheet() -> str:
    return (
        "# Next.js frontend tour (STAGE 3) — operator steps.\n"
        "# Long-running dev server; not auto-launched.\n"
        "#\n"
        "# 1. Start the Next.js dev server in a separate terminal:\n"
        "#      cd demo/nextjs && npm install   # first run only\n"
        "#      cd demo/nextjs && npm run dev\n"
        "#\n"
        f"# 2. Open the URL the recording navigates to:\n"
        f"#      {NEXTJS_DEV_URL}\n"
        "#\n"
        "# 3. Recording click checklist (in order, so the GIF is\n"
        "#    reproducible across re-captures):\n"
        "#      a. Type `postgres tuning` into the search input and\n"
        "#         submit; the phase pills tick through retrieve →\n"
        "#         rerank → generate as the SSE stream arrives.\n"
        "#      b. Hover a citation chip in the streamed answer;\n"
        "#         the corresponding retrieved-chunks panel row\n"
        "#         highlights.\n"
        "#      c. Click the retrieved-chunks panel header to\n"
        "#         collapse / re-expand it; show the side-by-side\n"
        "#         layout collapsing cleanly."
    )


def _maybe_launch_sse_server() -> subprocess.Popen[bytes] | None:
    """Spawn `python -m demo.streaming.server` as a child. Returns the
    child for the operator to terminate when the recording is done.
    Returns ``None`` if the spawn failed (rare — the server is
    stdlib-only).
    """
    try:
        return subprocess.Popen(  # noqa: S603 — sys.executable, no shell.
            [sys.executable, "-m", "demo.streaming.server"],
            cwd=REPO_ROOT,
        )
    except OSError:
        return None


def _maybe_run_curl(query: str) -> None:
    """Run `curl -N <stream-url>?q=<query>` to consume the SSE stream
    so the operator's recording captures the wire frames. No-op if
    `curl` isn't on PATH (rare on macOS/linux dev machines).
    """
    if shutil.which("curl") is None:
        print(
            "[capture] --launch-server was passed but `curl` is not on "
            "PATH; falling back to the cheat-sheet."
        )
        return
    url = f"{SSE_SERVER_URL}/stream?q={query.replace(' ', '+')}"
    # `-N` disables curl's output buffering so SSE frames land
    # in-order in the recording terminal.
    subprocess.run(  # noqa: S603 — absolute resolution of curl, no shell.
        [shutil.which("curl") or "curl", "-N", url],
        cwd=REPO_ROOT,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic 60-second demo capture orchestrator for rag-production-kit."
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        help="Pause between stages so the screen recorder has cue points. Default 2.0.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help=(
            "Skip launching the system browser on the Next.js URL when "
            "--launch-server is in effect (the Next.js cheat-sheet's "
            "browser hint is always informational). Default behavior is "
            "to honor the cheat-sheets without auto-launching anything."
        ),
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=(
            "Query for STAGE 1's hermetic preview. Default 'postgres tuning' "
            "to match the README and the cheat-sheets."
        ),
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help=(
            "Subprocess-spawn the SSE server and curl `/stream?q=...` for "
            "the operator. Off by default — the server is a long-running "
            "process that can't run hermetically in CI."
        ),
    )
    parser.add_argument(
        "--skip-server-cheatsheet",
        action="store_true",
        help="Suppress the STAGE 2 cheat-sheet print. Useful for CI/tests.",
    )
    parser.add_argument(
        "--skip-nextjs-cheatsheet",
        action="store_true",
        help="Suppress the STAGE 3 cheat-sheet print. Useful for CI/tests.",
    )
    args = parser.parse_args(argv)

    # Before STAGE 1, not at the first `_pause`. Rejecting here costs
    # nothing; rejecting later costs the operator a half-finished capture.
    # `parser.error` exits 2 and names the flag, matching the
    # `ap.error("--n must be positive")` form bench_streaming.py (#114) and
    # bench_rewriter.py already use for the same class of usage error.
    if (msg := _validate_pause_seconds(args.pause_seconds)) is not None:
        parser.error(msg)

    # STAGE 1 — in-process streaming preview, hermetic.
    print(
        _banner(
            1,
            f"Streaming pipeline preview (hermetic, query={args.query!r})",
        )
    )
    event_types = _run_streaming_preview(args.query, DEFAULT_K)
    print(f"\n[capture] events emitted (in order): {event_types}")
    _pause(args.pause_seconds)

    # STAGE 2 — SSE server cheat-sheet (operator-action, optional spawn).
    print(_banner(2, "SSE server demo (python -m demo.streaming.server)"))

    streamlit_child = None  # placeholder name; actually the SSE child.
    if args.launch_server:
        streamlit_child = _maybe_launch_sse_server()
        if streamlit_child is None:
            print("[capture] failed to spawn the SSE server subprocess; cheat-sheet only.")
        else:
            print(
                f"[capture] spawned SSE server (pid {streamlit_child.pid}); "
                "Ctrl-C / terminate when the recording is done."
            )
            # Small grace period so the server's listen() is up before curl.
            time.sleep(1.0)
            _maybe_run_curl(args.query)

    if not args.skip_server_cheatsheet:
        print(_server_cheatsheet(args.query))

    _pause(args.pause_seconds)

    # STAGE 3 — Next.js cheat-sheet (operator-action only).
    print(_banner(3, "Next.js frontend tour (cd demo/nextjs && npm run dev)"))
    if args.launch_server and not args.no_open:
        webbrowser.open(NEXTJS_DEV_URL)

    if not args.skip_nextjs_cheatsheet:
        print(_nextjs_cheatsheet())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
