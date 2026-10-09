"""``capture_demo --launch-server`` curls the query the operator typed (#293).

The curl URL encoded spaces only, so the server's `parse_qs` read ``R&D budget``
as ``R`` and ``c++ tuning`` as ``c   tuning``. ``#1 postgres tip`` got a 400,
because curl dropped everything after ``#``, and non-ASCII text arrived as
mojibake. The script still exited 0, so the live take in STAGE 2 streamed a
different query from STAGE 1's preview. The unit arm checks that the URL is the
inverse of `parse_qs`. The end-to-end arm runs the real server and curl, and
skips if curl is missing or 8765 is held, because the tests must not touch a
process they did not start.
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
from itertools import pairwise
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    for p in (REPO_ROOT / "scripts", REPO_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.modules.pop("capture_demo", None)
    import capture_demo  # noqa: WPS433

    return capture_demo


def _port_8765_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", 8765)) != 0


QUERIES = [
    "postgres tuning",
    "R&D budget",
    "c++ tuning",
    "#1 postgres tip",
    "100% recall",
    "k=5 rerank",
    "ошибка импорта",
    "a  b",
]


@pytest.mark.parametrize("query", QUERIES)
def test_the_server_reads_back_exactly_the_query(query: str) -> None:
    url = _load()._stream_url(query)
    split = urlsplit(url)
    assert split.fragment == "", url
    assert split.path == "/stream", url
    # `keep_blank_values` so a query that decodes to "" can't vanish quietly.
    assert parse_qs(split.query, keep_blank_values=True) == {"q": [query]}, url
    assert url.isascii(), url


def test_the_default_query_keeps_the_documented_form() -> None:
    cd = _load()
    assert cd._stream_url(cd.DEFAULT_QUERY) == f"{cd.SSE_SERVER_URL}/stream?q=postgres+tuning"


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl is not on PATH")
@pytest.mark.skipif(not _port_8765_free(), reason="127.0.0.1:8765 is held by another process")
def test_the_live_stream_carries_the_typed_query(capfd) -> None:
    cd = _load()
    query = "R&D budget"
    rc = cd.main(
        [
            "--pause-seconds",
            "0",
            "--no-open",
            "--launch-server",
            "--skip-server-cheatsheet",
            "--skip-nextjs-cheatsheet",
            "--query",
            query,
        ]
    )
    out = capfd.readouterr().out
    assert rc == 0
    # Every `retrieving` frame names its query: one from the in-process
    # preview and one from the curled live stream.
    seen = [
        json.loads(line[len("data: ") :])["payload"]["query"]
        for prev, line in pairwise(out.splitlines())
        if prev == "event: retrieving" and line.startswith("data: ")
    ]
    assert seen == [query, query], seen
    assert _port_8765_free(), "the SSE server outlived the capture"
