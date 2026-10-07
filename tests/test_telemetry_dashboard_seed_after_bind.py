"""A run that cannot bind writes nothing to the dashboard's database (#278).

`--seed N` inserted its rows before the bind, so a second instance on a busy
port -- the "started it twice" case the bind's own comment calls routine --
wrote N more rows into the running dashboard's DB and only then exited 2.
Measured: two `--seed 5` runs on one port left 10 rows. The port is held by a
real socket, as in `test_telemetry_dashboard_bind_errors.py`.
"""

from __future__ import annotations

import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "telemetry_dashboard.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )


def _rows(db: Path) -> int:
    if not db.exists():
        return 0
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
        return sum(conn.execute(f"select count(*) from {t}").fetchone()[0] for t in tables)  # noqa: S608


def test_a_busy_port_fails_before_any_row_is_seeded(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        proc = _run("--db", str(db), "--host", "127.0.0.1", "--port", str(port), "--seed", "5")
    assert proc.returncode == 2, proc.stderr
    assert "could not bind" in proc.stderr
    assert "seeded" not in proc.stderr
    assert _rows(db) == 0


def test_an_unusable_db_still_exits_two_and_closes_the_listener(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    bad_db = tmp_path / "missing-dir" / "t.db"
    proc = _run("--db", str(bad_db), "--host", "127.0.0.1", "--port", str(port), "--seed", "3")
    assert proc.returncode == 2, proc.stderr
    assert "is not usable" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr
    # The listener the run bound was closed on its way out: the port is free.
    with socket.socket() as again:
        again.bind(("127.0.0.1", port))
