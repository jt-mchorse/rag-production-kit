"""A failed `eval-harness diff-json` is not posted as a suite's delta (#299).

`_diff_markdown` returned `out.stdout or out.stderr`, so when the diff failed
(a corrupt or unreadable suite JSON) its traceback became that suite's section
of the PR's eval comment, and `run_eval --post-comment` exited 0. The pinned
eval-harness exits 1 for a crash as well as for a flagged row, so the check is
on the shape (a 0/1 exit with markdown on stdout), not on the exit code alone.

The stub below is a real executable on PATH, so `_diff_markdown`'s own
`subprocess.run` call is exercised rather than replaced.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from evals import run_eval

_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "eval_harness/runner.py", line 378, in load_run_result_from_json\n'
    "json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes"
)

_STUB = """#!/bin/sh
printf '%s' "$STUB_STDOUT"
printf '%s' "$STUB_STDERR" >&2
exit "$STUB_RC"
"""


@pytest.fixture
def stub_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "eval-harness"
    exe.write_text(_STUB)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    def configure(rc: int, stdout: str = "", stderr: str = "") -> None:
        monkeypatch.setenv("STUB_RC", str(rc))
        monkeypatch.setenv("STUB_STDOUT", stdout)
        monkeypatch.setenv("STUB_STDERR", stderr)

    return configure


@pytest.mark.parametrize(
    ("rc", "stdout", "stderr"),
    [
        # The pinned harness (2398cc3): an uncaught exception on bad input.
        (1, "", _TRACEBACK),
        # A harness with a real bad-input code.
        (2, "", "error: cannot read baseline"),
        # Any other non-0/1 exit, even with something on stdout.
        (3, "partial", "boom"),
        # A 0/1 exit that rendered nothing is not a delta either.
        (0, "  \n", ""),
    ],
    ids=["crash-exit-1", "bad-input-exit-2", "other-exit", "empty-stdout"],
)
def test_a_failed_diff_raises_instead_of_returning_stderr(
    stub_harness, tmp_path: Path, rc: int, stdout: str, stderr: str
) -> None:
    stub_harness(rc, stdout, stderr)
    with pytest.raises(run_eval.DiffFailedError) as info:
        run_eval._diff_markdown(tmp_path / "cur.json", tmp_path / "base.json")
    assert f"exited {rc}" in str(info.value)
    if stderr:
        assert stderr.strip().splitlines()[-1] in str(info.value)


@pytest.mark.parametrize("rc", [0, 1], ids=["clean", "flagged"])
def test_a_rendered_delta_is_returned_for_exit_0_and_1(stub_harness, tmp_path: Path, rc: int):
    md = "# Eval delta\n[=] mean delta +0.000 · flagged 0\n"
    stub_harness(rc, md, "")
    assert run_eval._diff_markdown(tmp_path / "cur.json", tmp_path / "base.json") == md


def test_main_posts_nothing_and_exits_2_when_a_diff_fails(
    stub_harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    stub_harness(1, "", _TRACEBACK)
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path / "current")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    rc = run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"])

    out, err = capsys.readouterr()
    assert rc == 2
    assert "::error::faithfulness:" in err
    assert "JSONDecodeError" in err
    # The dry-run body (what would have been posted) is never produced.
    assert "dry-run" not in out
    assert "Traceback" not in out
    assert "Eval delta" not in out


def test_main_still_posts_a_flagged_delta(
    stub_harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    stub_harness(1, "FLAGGED-DELTA-MARKDOWN\n", "")
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path / "current")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"]) == 0
    out = capsys.readouterr().out
    assert out.count("FLAGGED-DELTA-MARKDOWN") == len(run_eval.SUITES)


@pytest.mark.skipif(shutil.which("eval-harness") is None, reason="needs the [eval] extra")
def test_the_installed_harness_on_a_corrupt_baseline_is_a_failed_diff(tmp_path: Path) -> None:
    # The real binary, not the stub: whatever exit code this harness version
    # uses for bad input, it renders no delta, and that must not be posted.
    good = run_eval.BASELINES_DIR / "recall_at_5.json"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    probe = subprocess.run(
        ["eval-harness", "diff-json", "--current", str(good), "--baseline", str(bad)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0
    with pytest.raises(run_eval.DiffFailedError):
        run_eval._diff_markdown(good, bad)
