"""``capture_demo --launch-server`` records the server it started, then stops it (#274).

The script slept a fixed second, curled port 8765 and returned 0 without ever
stopping its child, which outlived it with ppid 1. The next capture's server
died of EADDRINUSE -- its traceback in the take -- while curl read the
previous run's orphan, and the script still said "spawned SSE server" and
exited 0. It now waits for its own child's listen line, refuses when that
never comes, and stops the child at the end of STAGE 2.

The arms that bind 8765 skip (with the reason) when something else already
holds it: the server's port is hard-coded, and these tests must not touch a
process they did not start.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ARGS = [
    "--pause-seconds",
    "0",
    "--no-open",
    "--launch-server",
    "--skip-server-cheatsheet",
    "--skip-nextjs-cheatsheet",
]


def _load():
    for p in (REPO_ROOT / "scripts", REPO_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.modules.pop("capture_demo", None)
    import capture_demo  # noqa: WPS433

    return capture_demo


def _child(code: str) -> subprocess.Popen[str]:
    return subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, text=True)


def _port_8765_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", 8765)) != 0


class TestWaitForListenLine:
    def test_returns_the_childs_listen_line_after_other_output(self) -> None:
        cd = _load()
        child = _child(
            "import time; print('warming up'); "
            "print('streaming demo on http://127.0.0.1:8765/'); time.sleep(30)"
        )
        try:
            assert cd._wait_for_listen_line(child, timeout=10) == (
                "streaming demo on http://127.0.0.1:8765/"
            )
        finally:
            cd._stop(child)
        assert child.returncode is not None

    def test_a_child_that_exits_first_is_none_without_waiting_out_the_timeout(self) -> None:
        cd = _load()
        child = _child(
            "import sys; print('OSError: [Errno 48] Address already in use'); sys.exit(1)"
        )
        t = threading.Event()
        result: list[object] = []
        threading.Thread(
            target=lambda: (result.append(cd._wait_for_listen_line(child, timeout=30)), t.set())
        ).start()
        assert t.wait(10), "the helper waited out its timeout on a child that had exited"
        assert result == [None]

    def test_a_child_that_never_reports_times_out_to_none(self) -> None:
        cd = _load()
        child = _child("import time; print('something else'); time.sleep(30)")
        try:
            assert cd._wait_for_listen_line(child, timeout=0.5) is None
        finally:
            cd._stop(child)


@pytest.mark.skipif(not _port_8765_free(), reason="127.0.0.1:8765 is held by another process")
class TestAgainstPort8765:
    def test_a_held_port_fails_the_capture_and_the_holder_is_never_curled(self) -> None:
        cd = _load()
        hits: list[str] = []

        class Holder(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                hits.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"event: done\ndata: {}\n\n")

            def log_message(self, *args: object) -> None:
                pass

        holder = ThreadingHTTPServer(("127.0.0.1", 8765), Holder)
        threading.Thread(target=holder.serve_forever, daemon=True).start()
        try:
            assert cd.main(CAPTURE_ARGS) != 0
        finally:
            holder.shutdown()
            holder.server_close()
        assert hits == [], "the capture curled a server it did not start"

    def test_a_clean_run_streams_and_leaves_nothing_on_the_port(self, capfd) -> None:
        cd = _load()
        assert cd.main(CAPTURE_ARGS) == 0
        out = capfd.readouterr().out
        assert "streaming demo on http://127.0.0.1:8765/" in out
        assert out.count("event: done") >= 2  # the in-process preview and the curled stream
        assert _port_8765_free(), "the SSE server outlived the capture"
