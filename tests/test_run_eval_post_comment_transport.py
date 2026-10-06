"""`--post-comment` cannot hang on GitHub, and every transport failure is legible (#276).

Neither `urlopen` in `_post_composite_comment` had a timeout, so a connection
that was accepted and never answered blocked until the CI job's cap: through a
local proxy that accepts and never replies, the call was still blocked after
45 s. #174's `except URLError` arms covered "never reached the API" but not a
`TimeoutError` raised while reading, and the "non-fatal" list arm did not cover
a 200 whose body is not JSON.
"""

from __future__ import annotations

import io
import socket
import threading
import time
import urllib.error

import pytest

from evals import run_eval


class _Resp(io.BytesIO):
    status = 201

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_a_silent_github_connection_times_out_instead_of_hanging(monkeypatch) -> None:
    # A "proxy" that accepts and never answers. Every connection it accepts is
    # kept open, so the client sees a live socket and nothing else.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                held.append(listener.accept()[0])
            except OSError:
                continue

    threading.Thread(target=accept, daemon=True).start()
    proxy = f"http://127.0.0.1:{listener.getsockname()[1]}"
    for var in ("HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(var, proxy)
    monkeypatch.setattr(run_eval, "_GITHUB_TIMEOUT_S", 0.5)
    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(run_eval._post_composite_comment("a/b", 1, {}, "ghp_fake")),
        daemon=True,
    )
    started = time.monotonic()
    try:
        worker.start()
        worker.join(15)
        assert not worker.is_alive(), "the comment post was still blocked after 15 s"
        assert result == [False]
        assert time.monotonic() - started < 15
    finally:
        stop.set()
        for c in held:
            c.close()
        listener.close()


def test_a_non_json_list_response_is_a_warning_and_the_post_still_happens(
    monkeypatch, capsys
) -> None:
    calls: list[str] = []

    def fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        calls.append(req.get_method())
        assert timeout == run_eval._GITHUB_TIMEOUT_S
        if req.get_method() == "GET":
            return _Resp(b"<html>proxy login</html>")
        return _Resp(b"{}")

    monkeypatch.setattr(run_eval.urllib.request, "urlopen", fake_urlopen)
    assert run_eval._post_composite_comment("a/b", 1, {}, "ghp_fake") is True
    assert calls == ["GET", "POST"]
    assert "warning: failed to list PR comments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(TimeoutError("The read operation timed out"), id="read-timeout"),
        pytest.param(ConnectionResetError(54, "reset by peer"), id="reset"),
        pytest.param(urllib.error.URLError("refused"), id="urlerror"),
    ],
)
def test_a_transport_failure_on_the_write_is_an_error_line_not_a_traceback(
    monkeypatch, capsys, exc: BaseException
) -> None:
    def fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        if req.get_method() == "GET":
            return _Resp(b"[]")
        raise exc

    monkeypatch.setattr(run_eval.urllib.request, "urlopen", fake_urlopen)
    assert run_eval._post_composite_comment("a/b", 1, {}, "ghp_fake") is False
    assert "::error::failed to post PR comment:" in capsys.readouterr().err
