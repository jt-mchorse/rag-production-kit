"""Lock tests for #166: `demo/streaming/server.py`'s query-string handling.

`_serve_stream` validated `q` before starting the response but only
half-validated `k`. `StreamingPipeline.run` enforces the repo's positive-int
contract (#41) and is a *generator*, so its `ValueError` for `k <= 0` landed on
the first `next()` — inside the `for`, after `send_response(200)` and the SSE
headers had already gone out. The client got a 200 with a zero-byte body:
`resp.ok` true, reader immediately `done`, nothing rendered, and `app.js`'s
`HTTP <status>` error card unreachable for the one input that needed it.

These drive the real handler over a real socket on an ephemeral port. That is
the point — the defect was in *when* the response was committed relative to the
validation, which no test of a pure function can see.

The server's own error stream is captured and asserted empty, because the
pre-fix symptom on the wire (an empty 200 body) is indistinguishable from a
client that disconnected early; the traceback is what names it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import socket
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

_DEMO_SERVER = Path(__file__).resolve().parents[1] / "demo" / "streaming" / "server.py"


def _load_demo_server():
    """Import `demo/streaming/server.py` by path.

    `demo/` is not a package on the install path, and the module inserts the
    repo root into `sys.path` at import time for its own sibling imports.
    """
    spec = importlib.util.spec_from_file_location("_demo_streaming_server", _DEMO_SERVER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo_server = _load_demo_server()


class _Captured(ThreadingHTTPServer):
    """Collects what the handler writes to stderr instead of letting it escape.

    `BaseHTTPRequestHandler` swallows a handler exception into a traceback on
    stderr and drops the connection, so an unhandled error is invisible from
    the client side beyond a truncated body. Routing it here makes it
    assertable.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.errors = io.StringIO()
        super().__init__(*args, **kwargs)

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001
        import traceback

        traceback.print_exc(file=self.errors)


@pytest.fixture
def server() -> Iterator[_Captured]:
    srv = _Captured(("127.0.0.1", 0), demo_server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _url(server: _Captured, path: str) -> str:
    host, port = server.server_address[0], server.server_address[1]
    return f"http://{host}:{port}{path}"


def _get(server: _Captured, path: str, timeout: float = 10.0) -> tuple[int, bytes]:
    """GET `path`, reading a streamed body until its terminal frame.

    The SSE response carries `Connection: keep-alive` and no `Content-Length`,
    so a plain `resp.read()` blocks until the peer closes — which it does not.
    Read incrementally and stop at the `done` frame (or EOF for a finite body).
    """
    try:
        with urllib.request.urlopen(_url(server, path), timeout=timeout) as resp:
            if "text/event-stream" not in (resp.headers.get("Content-Type") or ""):
                return resp.status, resp.read()
            # `read1` returns what has arrived rather than waiting for a full
            # buffer, which `read(n)` does on a body with no Content-Length.
            body = b""
            while b"event: done" not in body:
                chunk = resp.read1(4096)
                if not chunk:
                    break
                body += chunk
            return resp.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _assert_no_traceback(server: _Captured) -> None:
    captured = server.errors.getvalue()
    assert "Traceback" not in captured, f"handler raised:\n{captured}"


# --- the bug: an unusable k must fail before the response is committed ------


@pytest.mark.parametrize("k", ["0", "-1", "1.5", "abc", "0x10", "1e3", "3,4"])
def test_unusable_k_is_a_400_with_no_partial_stream(server: _Captured, k: str) -> None:
    status, body = _get(server, f"/stream?q=refund&k={k}")
    assert status == 400
    # Not a truncated 200: the client's `!resp.ok` branch has to be the one
    # that fires, since that is the only path `app.js` renders an error from.
    assert b"event: retrieving" not in body
    _assert_no_traceback(server)


def test_zero_k_does_not_reach_the_pipeline(server: _Captured) -> None:
    # The specific regression: pre-fix this returned 200 with an empty body and
    # left a `ValueError: k must be a positive integer, got 0` traceback in the
    # server log, because `run` is a generator and raised on the first next().
    status, body = _get(server, "/stream?q=refund&k=0")
    assert status == 400
    assert body != b""  # a 400 carries an explanatory body; the pre-fix 200 did not
    _assert_no_traceback(server)


def test_missing_q_is_still_a_400(server: _Captured) -> None:
    # The branch that was already correct — pinned so the k fix can't disturb it.
    status, _ = _get(server, "/stream?q=")
    assert status == 400
    _assert_no_traceback(server)


# --- the success paths the fix must not disturb -----------------------------


@pytest.mark.parametrize("query", ["/stream?q=refund&k=3", "/stream?q=refund"])
def test_valid_request_streams_the_full_event_sequence(server: _Captured, query: str) -> None:
    # Explicit k and an omitted k (defaulting to 3) both stream.
    status, body = _get(server, query)
    assert status == 200
    text = body.decode("utf-8")
    for event in ("retrieving", "retrieved", "reranking", "reranked", "generating", "done"):
        assert f"event: {event}\n" in text, f"missing {event!r} frame"
    _assert_no_traceback(server)


def test_large_k_is_accepted_and_clamped_by_the_corpus(server: _Captured) -> None:
    # `k` is only required to be a positive int; the retriever returns what the
    # corpus has. Pinned so the new validation isn't tightened into a range cap
    # that the library itself doesn't impose.
    status, body = _get(server, "/stream?q=refund&k=500")
    assert status == 200
    assert b"event: done" in body
    _assert_no_traceback(server)


@pytest.mark.parametrize("path", ["/", "/index.html", "/app.js"])
def test_static_paths_still_serve(server: _Captured, path: str) -> None:
    status, body = _get(server, path)
    assert status == 200
    assert body
    _assert_no_traceback(server)


def test_unknown_path_is_a_404(server: _Captured) -> None:
    status, _ = _get(server, "/nope")
    assert status == 404
    _assert_no_traceback(server)


# --- the contract this rests on --------------------------------------------


def test_pipeline_still_rejects_a_non_positive_k_by_raising() -> None:
    """The library side is deliberately unchanged.

    `run`'s validation stays *outside* its own `except Exception`, so a bad
    argument raises at the call site instead of becoming a yielded `error`
    event. That is the right split — a caller's bad argument is a different
    category from a mid-stream failure — and it is exactly why the demo server
    has to validate before committing the response. Pinned here so a future
    change to either side has to confront the pairing.
    """
    from rag_kit.streaming import StreamingPipeline

    pipe = StreamingPipeline(demo_server.FakeRetriever(demo_server._CORPUS))
    with pytest.raises(ValueError, match="k must be a positive integer"):
        next(iter(pipe.run("refund", k=0)))


def test_client_only_renders_an_error_from_a_non_ok_status_or_an_error_frame() -> None:
    """Why a 400 (and not a 200 with an empty body) is the fix.

    `app.js` reads the stream with `fetch()` + `TextDecoder`, not
    `EventSource`, so its only two error surfaces are `!resp.ok` and an
    `error` frame. A 200 with no body hits neither. If the client is ever
    rewritten to use `EventSource`, this test fails and points at the
    server-side assumption that depends on it.
    """
    app_js = (_DEMO_SERVER.parent / "app.js").read_text(encoding="utf-8")
    assert "!resp.ok" in app_js
    assert 'case "error":' in app_js
    assert "new EventSource" not in app_js


def test_module_docstring_names_the_mechanism_app_js_actually_uses() -> None:
    # The docstring said the page "wires `EventSource('/stream?q=...')`" while
    # `app.js` has used `fetch()` + `TextDecoder` since it was written. That
    # difference is load-bearing for how a failure reaches the user, which is
    # the whole subject of #166 — so it is worth pinning rather than just
    # correcting once.
    app_js = (_DEMO_SERVER.parent / "app.js").read_text(encoding="utf-8")
    docstring = _DEMO_SERVER.read_text(encoding="utf-8").split('"""')[1]
    assert "new EventSource" not in app_js
    assert "fetch" in docstring


def test_no_socket_left_listening_after_shutdown() -> None:
    # The fixture tears down between tests; if it ever stopped doing so these
    # would start failing on a bound port instead of silently sharing state.
    srv = _Captured(("127.0.0.1", 0), demo_server.Handler)
    port = srv.server_address[1]
    srv.server_close()
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", port))  # free again — raises OSError if not
