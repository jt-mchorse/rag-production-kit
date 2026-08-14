"""Failure-seam coverage for ``evals/run_eval.py`` (#174).

The script speaks a 0/2 exit contract (unknown ``--suite`` → 2,
``--post-comment`` without ``--repo``/``--pr`` → 2), but four seams used to
bypass it: the results write escaped as a raw ``OSError`` traceback at exit 1,
the comment-create call was unguarded while the comment-list call beside it was
wrapped, that list guard caught ``HTTPError`` rather than its parent
``URLError``, and a ``resp.status >= 300`` branch was unreachable because
``urlopen`` raises for every status ``urlopen`` would have reported.

Every test here drives a real seam rather than asserting on a mock's call
count: the HTTP cases run against a stub ``http.server`` on localhost, and the
write case blocks the output path on a real filesystem.
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import threading
import urllib.request
from collections.abc import Iterator

import pytest

from evals import run_eval

DELTAS = {"faithfulness": "f", "recall_at_5": "r", "correctness": "c"}


class _StubHandler(http.server.BaseHTTPRequestHandler):
    """GitHub-comments API stub. ``post_status`` is set per test."""

    post_status = 201
    list_status = 200
    posted_bodies: list[bytes] = []

    def _reply(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._reply(type(self).list_status, b"[]")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length") or 0)
        type(self).posted_bodies.append(self.rfile.read(length))
        self._reply(
            type(self).post_status,
            b'{"message":"Resource not accessible by integration"}',
        )

    def log_message(self, *args: object) -> None:  # silence the test log
        return


@pytest.fixture
def stub_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_StubHandler]]:
    """Serve the GitHub comments API from localhost and point the script at it."""
    _StubHandler.post_status = 201
    _StubHandler.list_status = 200
    _StubHandler.posted_bodies = []
    server = socketserver.TCPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _redirect_github_api(monkeypatch, f"http://127.0.0.1:{port}")
    try:
        yield _StubHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _redirect_github_api(monkeypatch: pytest.MonkeyPatch, base: str) -> None:
    """Rewrite the api.github.com host in every Request the script builds."""
    original = urllib.request.Request

    def _request(url: str, *args: object, **kwargs: object) -> urllib.request.Request:
        return original(url.replace("https://api.github.com", base), *args, **kwargs)

    monkeypatch.setattr(urllib.request, "Request", _request)


def _closed_port() -> int:
    """A port with nothing listening, so connecting raises ConnectionRefused."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


# --------------------------------------------------------------------------
# Seam 1 — the results write
# --------------------------------------------------------------------------


def test_unwritable_output_dir_exits_2_without_traceback(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """A path component that is a file used to raise NotADirectoryError at 1."""
    blocker = tmp_path / "notadir"
    blocker.write_text("i am a file\n")
    monkeypatch.setattr(run_eval, "CURRENT_DIR", blocker / "current")

    assert run_eval.main([]) == 2
    err = capsys.readouterr().err
    assert "::error::failed to write eval results" in err
    assert "Traceback" not in err


def test_successful_write_still_returns_0(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The guard must not swallow the happy path."""
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path / "current")

    assert run_eval.main([]) == 0
    assert (tmp_path / "current" / "faithfulness.json").exists()


# --------------------------------------------------------------------------
# Seam 2 — the comment-create call
# --------------------------------------------------------------------------


def test_post_403_reports_cleanly_and_returns_false(stub_api, capsys):
    """A fork PR's read-only GITHUB_TOKEN returns 403 on create."""
    stub_api.post_status = 403

    assert run_eval._post_composite_comment("o/r", 1, DELTAS, "tok") is False
    err = capsys.readouterr().err
    assert "::error::failed to post PR comment: HTTP 403" in err
    # The response body carries GitHub's explanation; surfacing it is the
    # difference between a legible CI failure and a bare status number.
    assert "Resource not accessible by integration" in err
    assert "Traceback" not in err


def test_post_success_returns_true(stub_api):
    assert run_eval._post_composite_comment("o/r", 1, DELTAS, "tok") is True
    assert stub_api.posted_bodies, "the stub should have received the comment body"


@pytest.fixture
def eval_harness_on_path(monkeypatch: pytest.MonkeyPatch):
    """Satisfy main's `[eval]`-extra pre-check without installing the extra.

    The delta rendering itself is stubbed: these tests are about main's exit
    code, and `_diff_markdown`'s own behaviour is covered separately below.
    """
    monkeypatch.setattr(run_eval.shutil, "which", lambda _name: "/usr/bin/eval-harness")
    monkeypatch.setattr(run_eval, "_diff_markdown", lambda _cur, _base: "delta")


def test_main_exits_1_when_the_comment_fails_to_post(
    stub_api, tmp_path, monkeypatch: pytest.MonkeyPatch, eval_harness_on_path
):
    """A swallowed 403 used to leave CI green with the delta silently missing."""
    stub_api.post_status = 403
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path / "current")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    assert run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"]) == 1


def test_main_exits_0_when_the_comment_posts(
    stub_api, tmp_path, monkeypatch: pytest.MonkeyPatch, eval_harness_on_path
):
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path / "current")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    assert run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"]) == 0


# --------------------------------------------------------------------------
# Seam 5 — the `[eval]`-extra binary `check=False` does not cover
# --------------------------------------------------------------------------


def test_missing_eval_harness_exits_2_before_writing_anything(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """`eval-harness` ships in the `[eval]` extra, not the base install."""
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path / "current")
    monkeypatch.setattr(run_eval.shutil, "which", lambda _name: None)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    assert run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"]) == 2
    err = capsys.readouterr().err
    assert "pip install -e '.[eval]'" in err
    assert "Traceback" not in err


def test_diff_markdown_reports_a_missing_binary_instead_of_raising(tmp_path):
    """`check=False` covers a non-zero exit, not a missing executable."""
    import subprocess as _subprocess

    original = _subprocess.run

    def _run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "eval-harness":
            raise FileNotFoundError(2, "No such file or directory", "eval-harness")
        return original(cmd, *args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(run_eval.subprocess, "run", _run)
    try:
        rendered = run_eval._diff_markdown(tmp_path / "cur.json", tmp_path / "base.json")
    finally:
        monkey.undo()

    assert "pip install -e '.[eval]'" in rendered


# --------------------------------------------------------------------------
# Seam 3 — URLError vs its HTTPError subclass
# --------------------------------------------------------------------------


def test_connection_refused_does_not_escape(monkeypatch: pytest.MonkeyPatch, capsys):
    """Nothing is listening, so both calls raise bare URLError, not HTTPError.

    The list call's old `except HTTPError` did not catch this — the one
    condition the warning existed for was the one it missed.
    """
    _redirect_github_api(monkeypatch, f"http://127.0.0.1:{_closed_port()}")

    assert run_eval._post_composite_comment("o/r", 1, DELTAS, "tok") is False
    err = capsys.readouterr().err
    assert "warning: failed to list PR comments" in err
    assert "::error::failed to post PR comment" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# Seam 4 — the previously dead status branch
# --------------------------------------------------------------------------


def test_unexpected_2xx_status_is_reported(stub_api, capsys):
    """204 means the write was accepted but created nothing — not a success.

    The old check read `resp.status >= 300`, which `urlopen` makes unreachable:
    it raises HTTPError for >= 400 and follows 3xx transparently. Checking for
    the statuses the API *should* return makes the branch fire again.
    """
    stub_api.post_status = 204

    assert run_eval._post_composite_comment("o/r", 1, DELTAS, "tok") is False
    assert "unexpected status 204" in capsys.readouterr().err


def test_dry_run_without_token_still_reports_success(capsys):
    assert run_eval._post_composite_comment("o/r", 1, DELTAS, None) is True
    assert "dry-run" in capsys.readouterr().out
