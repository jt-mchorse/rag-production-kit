"""Smoke test for ``scripts/capture_demo.py``.

Same hermetic contract as ``tests/test_streaming.py`` — composes the
streaming pipeline with the stubs in `demo.streaming.server`, no
Postgres, no LLM, no HTTP server spawn. Asserts STAGE 1's in-process
preview emits the exact phase sequence the streaming test contract
defines (`retrieving` first, full phase set present, `done` last)
and that the two cheat-sheets print under the default flags.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path


def _load_capture_module():
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if "capture_demo" in sys.modules:
        del sys.modules["capture_demo"]
    import capture_demo  # noqa: WPS433 — dynamic import is the point here.

    return capture_demo


def test_capture_demo_stage1_emits_full_phase_sequence() -> None:
    capture_demo = _load_capture_module()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = capture_demo.main(
            [
                "--pause-seconds",
                "0",
                "--no-open",
                "--skip-server-cheatsheet",
                "--skip-nextjs-cheatsheet",
            ]
        )
    out = buf.getvalue()

    assert rc == 0, f"capture_demo exited {rc}; stdout:\n{out}"

    # STAGE 1 banner present.
    assert "STAGE 1" in out

    # The summary line lists the event-type sequence; pull it out and
    # assert against the contract from tests/test_streaming.py.
    summary_marker = "[capture] events emitted (in order): "
    assert summary_marker in out, (
        "expected the STAGE 1 summary line listing the event types; got:\n" + out[-600:]
    )
    summary = out[out.rindex(summary_marker) + len(summary_marker) :].splitlines()[0]
    # `summary` is a python list literal like ['retrieving','retrieved',...]
    assert "retrieving" in summary, "first emitted event must be `retrieving`"
    assert "retrieved" in summary
    assert "reranking" in summary
    assert "reranked" in summary
    assert "generating" in summary
    assert "generated" in summary
    assert "done" in summary, "terminal event must be `done`"
    # `done` must be last; `retrieving` must be first.
    assert summary.index("retrieving") < summary.index("done")
    # At least one token event between `generating` and `generated`.
    assert "token" in summary

    # SSE frame format hits the wire — `data:` lines + a blank line
    # between frames, matching `to_sse()`.
    assert "data:" in out, "STAGE 1 should print SSE frames containing `data:` lines"


def test_capture_demo_prints_cheatsheets_by_default() -> None:
    capture_demo = _load_capture_module()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = capture_demo.main(
            [
                "--pause-seconds",
                "0",
                "--no-open",
            ]
        )
    assert rc == 0
    out = buf.getvalue()

    # STAGE 2 cheat-sheet content.
    assert "python -m demo.streaming.server" in out
    assert "curl -N" in out
    assert "8765" in out  # the server port the README cites

    # STAGE 3 cheat-sheet content.
    assert "cd demo/nextjs" in out
    assert "npm run dev" in out
    assert "localhost:3000" in out
    # Click checklist anchors.
    assert "citation chip" in out
    assert "phase pills" in out


def test_capture_demo_skip_flags_suppress_cheatsheets() -> None:
    capture_demo = _load_capture_module()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = capture_demo.main(
            [
                "--pause-seconds",
                "0",
                "--no-open",
                "--skip-server-cheatsheet",
                "--skip-nextjs-cheatsheet",
            ]
        )
    assert rc == 0
    out = buf.getvalue()

    # Banners still appear (the recording cuts on them); body content
    # is what's suppressed. Note the STAGE 2 banner title itself
    # contains `python -m demo.streaming.server`, so we anchor on
    # cheat-sheet-only markers instead.
    assert "STAGE 2" in out
    assert "STAGE 3" in out
    assert "Start the SSE server in a separate terminal" not in out
    assert "Start the Next.js dev server in a separate terminal" not in out
    assert "click checklist" not in out.lower()


def test_capture_demo_exposes_main_callable() -> None:
    capture_demo = _load_capture_module()
    assert hasattr(capture_demo, "main")
    import inspect

    sig = inspect.signature(capture_demo.main)
    assert "argv" in sig.parameters, f"main() must accept argv; got: {sig}"
