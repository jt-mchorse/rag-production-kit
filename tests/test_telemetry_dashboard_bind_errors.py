"""The `--port` range guard covered one operand of the bind tuple (#178).

#177 added `if not 0 <= args.port <= 65535: parser.error(...)` and stated the
contract in its comment: an out-of-range port "came back as a raw
`OverflowError` traceback at exit 1 — the wrong code for a usage error, and a
diagnostic pointing at the socket layer rather than at the flag the operator
typed."

That was true of `--port`'s *range* and of nothing else in `main()`.
`ThreadingHTTPServer((args.host, args.port), _Handler)` sits one line below the
guard, and `args.host` — also operator input — is the first element of the same
tuple. Measured on `main` @ 5ca7a77:

    --host 'not a host'         exit=1  socket.gaierror: [Errno 8] nodename nor
                                        servname provided, or not known
    --port <an in-use port>     exit=1  OSError: [Errno 48] Address already in use
    --db /nonexistent/t.db --seed 3
                                exit=1  sqlite3.OperationalError: unable to open
                                        database file

The in-use port is produced here by **binding a real socket**, not by patching
`ThreadingHTTPServer`. A mock would assert my model of the failure; a real bind
asserts the OS's. Same for the unresolvable host — a genuinely unresolvable
string rather than a patched `socket` module.

`main()` is exercised in a subprocess rather than in-process because the
success path calls `serve_forever()` and would never return. Every assertion
below is on the child's exit code and stderr, which is exactly the surface an
operator sees.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "telemetry_dashboard.py"


def _run(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_REPO_ROOT,
    )


def _assert_clean_usage_failure(proc: subprocess.CompletedProcess[str]) -> None:
    """Exit 2, an `::error::` line, and no traceback.

    The absence of a traceback is asserted explicitly: exit 2 alone would pass
    even if the guard printed a message and *then* let the exception through.
    """
    assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}\n{proc.stderr}"
    assert "::error::" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


# ----------------------------------------------------------------------
# The other operand of the bind tuple
# ----------------------------------------------------------------------


@pytest.mark.parametrize("host", ["not a host", "no-such-host.invalid"])
def test_unresolvable_host_exits_two_naming_the_flag(tmp_path: Path, host: str) -> None:
    # Both cases are network-independent. A host containing a space cannot be a
    # valid name at all, so it fails without a lookup; `.invalid` is reserved by
    # RFC 2606 precisely so it can never resolve, and on a runner with no DNS at
    # all the lookup still raises `gaierror`, just with a different errno.
    # Deliberately NOT `999.999.999.999`: it is syntactically a valid hostname,
    # so a network with a wildcard resolver could answer it and flake CI.
    proc = _run("--host", host, "--port", "0", "--db", str(tmp_path / "t.db"))
    _assert_clean_usage_failure(proc)
    assert "--host" in proc.stderr
    # The OS's own text is carried through rather than replaced — the operator
    # needs to know it was a *resolve* failure, not a permission or range one.
    assert host in proc.stderr


def test_an_already_bound_port_exits_two_naming_the_flag(tmp_path: Path) -> None:
    # A real listening socket, not a patched constructor. Port 0 lets the OS
    # pick a free one, so this cannot collide with anything else on the machine.
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        proc = _run("--host", "127.0.0.1", "--port", str(port), "--db", str(tmp_path / "t.db"))
    finally:
        holder.close()
    _assert_clean_usage_failure(proc)
    assert "--port" in proc.stderr
    assert str(port) in proc.stderr
    # This is the routine case — starting the dashboard twice — and the reason
    # the message must name the flag rather than the socket layer.
    assert "in use" in proc.stderr.lower()


# ----------------------------------------------------------------------
# The --seed write seam
# ----------------------------------------------------------------------


def test_db_in_a_nonexistent_directory_exits_two_naming_the_flag(tmp_path: Path) -> None:
    # `sqlite3.OperationalError` is not an `OSError` subclass, so this needs its
    # own except arm — catching `OSError` alone would leave it exposed. Reachable
    # through the documented `--seed` path.
    bad = tmp_path / "no-such-dir" / "t.db"
    proc = _run("--port", "0", "--db", str(bad), "--seed", "3")
    _assert_clean_usage_failure(proc)
    assert "--db" in proc.stderr
    assert "unable to open database file" in proc.stderr


def test_db_path_that_is_a_directory_exits_two(tmp_path: Path) -> None:
    # The `OSError` half of the same arm: a directory sitting where the database
    # file should be. Distinct from the case above, which is a sqlite3 error.
    d = tmp_path / "telemetry.db"
    d.mkdir()
    proc = _run("--port", "0", "--db", str(d), "--seed", "3")
    _assert_clean_usage_failure(proc)
    assert "--db" in proc.stderr


def test_a_failed_seed_does_not_reach_the_bind(tmp_path: Path) -> None:
    # Ordering: the seed step runs before the bind, so a bad `--db` must return
    # 2 without ever opening a socket. If it fell through, an operator with a
    # typo'd `--db` would get a server serving an empty database.
    bad = tmp_path / "no-such-dir" / "t.db"
    proc = _run("--port", "0", "--db", str(bad), "--seed", "3")
    assert proc.returncode == 2
    assert "serving http://" not in proc.stderr


# ----------------------------------------------------------------------
# What must not change
# ----------------------------------------------------------------------


def test_the_port_range_guard_still_uses_parser_error(tmp_path: Path) -> None:
    # A pure argument-domain check stays an argparse usage error: it prints the
    # program name and the flag message, not an `::error::` line. Keeping the
    # two forms distinct is deliberate — printing the usage block for "address
    # already in use" would be noise.
    proc = _run("--port", "99999", "--db", str(tmp_path / "t.db"))
    assert proc.returncode == 2
    assert "--port must be in 0-65535; got 99999" in proc.stderr
    assert "::error::" not in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_a_successful_bind_still_serves_and_the_banner_is_unchanged(tmp_path: Path) -> None:
    # The success path must be untouched. `--port 0` binds to an OS-assigned
    # port, so the banner reports `:0` — the point is that it is printed and the
    # process is serving, i.e. it did not exit.
    db = tmp_path / "t.db"
    proc = subprocess.Popen(
        [sys.executable, str(_SCRIPT), "--host", "127.0.0.1", "--port", "0", "--db", str(db)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=_REPO_ROOT,
    )
    try:
        # The banner is written before `serve_forever`, so it arrives promptly;
        # a `communicate` timeout is the signal that the process is still alive.
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=6)
    finally:
        proc.terminate()
        _out, err = proc.communicate(timeout=15)
    assert f"serving http://127.0.0.1:0/ from {db} (Ctrl-C to stop)" in err
    assert "::error::" not in err


def test_seed_success_still_reports_the_count(tmp_path: Path) -> None:
    # The seed guard must not swallow the success message, and a writable `--db`
    # must still be seeded before the bind.
    db = tmp_path / "t.db"
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        # Deliberately bind-failing so the process exits instead of serving; the
        # seed has already run and printed by then, which is what we assert.
        proc = _run("--host", "127.0.0.1", "--port", str(port), "--db", str(db), "--seed", "5")
    finally:
        holder.close()
    assert "seeded 5 synthetic records into" in proc.stderr
    assert db.exists(), "the seed really wrote the database before the bind was attempted"
    assert proc.returncode == 2  # ...and then the bind failed cleanly, as above
