"""Smoke test for `scripts/capture_demo.sh` (issue #25).

The capture script is the deterministic driver for the 60-second README demo.
JT records the GIF/video while it runs; CI runs it with
`CAPTURE_PACE_SECONDS=0` and `CAPTURE_LAUNCH_NEXTJS=0` so the SSE surface
is exercised without trying to bring up the Next.js dev server.

Contract this test pins:

1. The script exits 0 on a fresh clone with no API key and no Postgres.
2. The SSE curl tour produces all eight expected phase events in order
   (retrieving → retrieved → reranking → reranked → generating →
    token... → generated → done).
3. The Next.js dashboard section describes the launch path even when
   the launch itself is skipped.
4. The background SSE server is cleanly reaped — port 8765 is free
   after the script exits.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "capture_demo.sh"


def _port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="module")
def capture_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the capture script once and reuse its stdout across assertions.

    Picks a free port off the OS so two parallel test runs (or a stray
    leftover server from a manual capture) don't collide on 8765.
    """
    if not SCRIPT.exists():
        pytest.fail(f"missing {SCRIPT}")
    if shutil.which("bash") is None:
        pytest.skip("bash not available")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = dict(os.environ)
    env["CAPTURE_PACE_SECONDS"] = "0"
    env["CAPTURE_LAUNCH_NEXTJS"] = "0"
    env["CAPTURE_SSE_TIMEOUT"] = "8"
    env["CAPTURE_DEMO_PORT"] = str(port)
    venv_bin = Path(sys.executable).parent
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"capture_demo.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    # Give the OS a brief moment to free the socket before the
    # teardown assertion below — TIME_WAIT can otherwise show the
    # port as still bound for a second or two after SIGTERM.
    for _ in range(10):
        if not _port_is_listening(port):
            break
        time.sleep(0.1)

    return {
        "stdout": result.stdout,
        "port": port,
    }


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"


def test_surface_1_sse_phase_events_in_order(capture_run: dict[str, object]) -> None:
    """All eight pipeline phases must land on stdout in order.

    `retrieving → retrieved → reranking → reranked → generating →
    token (at least one) → generated → done`."""
    stdout = capture_run["stdout"]
    assert isinstance(stdout, str)
    assert "1/2 · streaming SSE server" in stdout

    phase_markers = [
        "event: retrieving",
        "event: retrieved",
        "event: reranking",
        "event: reranked",
        "event: generating",
        "event: token",
        "event: generated",
        "event: done",
    ]
    indices = []
    for marker in phase_markers:
        idx = stdout.find(marker)
        assert idx != -1, (
            f"expected SSE phase {marker!r} in capture output; missing.\n"
            f"first 1200 chars:\n{stdout[:1200]}"
        )
        indices.append(idx)
    assert indices == sorted(indices), (
        "SSE phases must arrive in pipeline order: retrieving → retrieved → "
        "reranking → reranked → generating → token → generated → done; got "
        f"indices {indices} for {phase_markers}"
    )


def test_surface_2_describes_nextjs_launch_path(capture_run: dict[str, object]) -> None:
    stdout = capture_run["stdout"]
    assert isinstance(stdout, str)
    assert "2/2 · Next.js frontend" in stdout
    assert "demo/nextjs" in stdout
    # With LAUNCH=0 the script prints the skip notice; either way the
    # `npm install && npm run dev` invocation must appear so the
    # recording instructions are reproducible.
    assert "npm run dev" in stdout


def test_background_server_is_cleanly_reaped(capture_run: dict[str, object]) -> None:
    """The script must not leave the SSE server running on the test port."""
    port = capture_run["port"]
    assert isinstance(port, int)
    assert not _port_is_listening(port), (
        f"port {port} is still bound after capture_demo.sh exited — "
        "the EXIT trap should have reaped the background `python -m "
        "demo.streaming.server` process."
    )
