"""The dashboard binds an IPv6 `--host`, and its banner brackets it (#291).

`ThreadingHTTPServer` always opens an `AF_INET` socket, so `--host ::1` exited 2
with `nodename nor servname provided`, though the bind comment in `main` lists
an IPv6 literal as a valid host. The serving test runs the real script in a
subprocess and follows the URL in its banner, the same way
`test_telemetry_dashboard_bind_errors.py` does for IPv4. It is skipped on a host
with no IPv6 loopback, where nothing could bind `::1`.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "telemetry_dashboard.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from scripts.telemetry_dashboard import _address_family, _url_host  # noqa: E402


def _ipv6_loopback_available() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback on this host")
def test_an_ipv6_host_serves_at_the_url_in_the_banner(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    proc = subprocess.Popen(
        [sys.executable, str(_SCRIPT), "--host", "::1", "--port", "0", "--db", str(db)],
        stderr=subprocess.PIPE,
        text=True,
        cwd=_REPO_ROOT,
    )
    try:
        assert proc.stderr is not None
        line = proc.stderr.readline()  # the banner, or the bind error on main
        m = re.search(r"serving (http://\[::1\]:(\d+)/) from ", line)
        assert m, line
        assert int(m.group(2)) != 0, line
        with urllib.request.urlopen(m.group(1), timeout=10) as resp:
            assert resp.status == 200
    finally:
        proc.terminate()
        _out, err = proc.communicate(timeout=15)
    assert "::error::" not in err


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("::1", "[::1]"),
        ("fe80::1%lo0", "[fe80::1%lo0]"),
        ("127.0.0.1", "127.0.0.1"),
        ("localhost", "localhost"),
    ],
)
def test_the_banner_brackets_an_ipv6_host_only(host: str, expected: str) -> None:
    assert _url_host(host) == expected


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "", "localhost"])
def test_a_host_with_an_ipv4_address_still_binds_ipv4(host: str) -> None:
    # `localhost` resolves to ::1 first on some systems; it must stay on
    # 127.0.0.1, where it has always been served.
    assert _address_family(host, 0) == socket.AF_INET


def test_an_ipv6_literal_binds_ipv6() -> None:
    assert _address_family("::1", 0) == socket.AF_INET6
