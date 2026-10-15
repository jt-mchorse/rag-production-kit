"""`--host ""` advertises the address it bound, not an empty host (#344).

A bare `""` binds every interface, and the bind comment in `main` lists it as
a valid host, but the banner printed `serving http://:<port>/`: a URL with no
host, which curl ("No host part in the URL") and the WHATWG parser browsers use
both reject. #268 made the banner's port the bound one and #291 bracketed an
IPv6 host; this is the host half. The test runs the real script in a
subprocess and follows the URL in its banner, as the IPv4 and IPv6 banner tests
do.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "telemetry_dashboard.py"


def _banner(host: str, db: Path) -> tuple[subprocess.Popen[str], str]:
    proc = subprocess.Popen(
        [sys.executable, str(_SCRIPT), "--host", host, "--port", "0", "--db", str(db)],
        stderr=subprocess.PIPE,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.stderr is not None
    return proc, proc.stderr.readline()


def test_an_empty_host_banner_names_the_bound_address_and_serves(tmp_path: Path) -> None:
    proc, line = _banner("", tmp_path / "t.db")
    try:
        m = re.match(r"serving (http://\S+) ", line)
        assert m, line
        url = m.group(1)
        parts = urlsplit(url)
        assert parts.hostname == "0.0.0.0", line
        assert parts.port not in (None, 0), line
        with urllib.request.urlopen(url, timeout=10) as resp:
            assert resp.status == 200
    finally:
        proc.terminate()
        _out, err = proc.communicate(timeout=15)
    assert "::error::" not in err


def test_an_empty_host_and_0000_print_the_same_banner_host(tmp_path: Path) -> None:
    # Both bind the same all-interfaces IPv4 socket, so they advertise it alike.
    hosts = []
    for host in ("", "0.0.0.0"):
        proc, line = _banner(host, tmp_path / "t.db")
        try:
            m = re.match(r"serving (http://\S+) ", line)
            assert m, line
            hosts.append(urlsplit(m.group(1)).hostname)
        finally:
            proc.terminate()
            proc.communicate(timeout=15)
    assert hosts == ["0.0.0.0", "0.0.0.0"]
