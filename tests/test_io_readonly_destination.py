"""`atomic_write_text` refuses a read-only existing destination (#432).

`os.replace` swaps the directory entry, so the target's own write bit was
never consulted and a ``chmod 444`` file was silently rewritten -- while
`Path.write_text`, the call the helper replaced, raises ``PermissionError``.
The write seams in `evals/run_eval.py` and `scripts/bench_streaming.py`
document an unwritable target as an error at exit 2. Helper cases run both
writers on the same layout, so the lock is parity with `Path.write_text`.
Port of prompt-regression-suite#318.

Root bypasses permission bits (`Path.write_text` succeeds too), so the
read-only arms skip as root. GitHub's ubuntu runners run as the non-root
``runner`` user, so CI exercises them.
"""

from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from evals import run_eval
from rag_kit.io_utils import atomic_write_text, check_writable

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits"),
    pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses file permission bits, so a 0444 file stays writable",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(writer: str, path: Path, text: str) -> None:
    if writer == "atomic":
        atomic_write_text(path, text)
    else:
        path.write_text(text, encoding="utf-8")


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def frozen(tmp_path: Path) -> Iterator[Path]:
    """A 0444 file holding ``old``; made writable again for tmp_path cleanup."""
    path = tmp_path / "r.json"
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o444)
    yield path
    path.chmod(0o644)


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_readonly_file_is_refused(frozen: Path, writer: str) -> None:
    with pytest.raises(PermissionError) as excinfo:
        _write(writer, frozen, "new\n")
    assert excinfo.value.errno == errno.EACCES
    assert frozen.read_text(encoding="utf-8") == "old\n"
    assert _mode(frozen) == 0o444


def test_refusal_leaves_no_temp_file(frozen: Path) -> None:
    with pytest.raises(PermissionError):
        atomic_write_text(frozen, "new\n")
    assert sorted(p.name for p in frozen.parent.iterdir()) == ["r.json"]


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_readonly_file_behind_a_symlink_is_refused(tmp_path: Path, writer: str) -> None:
    # The check runs on the RESOLVED file (#296): a writable link name over a
    # read-only file must not get through.
    real = tmp_path / "real.json"
    real.write_text("old\n", encoding="utf-8")
    real.chmod(0o444)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    try:
        with pytest.raises(PermissionError):
            _write(writer, link, "new\n")
        assert real.read_text(encoding="utf-8") == "old\n"
        assert link.is_symlink()
    finally:
        real.chmod(0o644)


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_writable_and_missing_targets_still_write(tmp_path: Path, writer: str) -> None:
    # Controls: the refusal is scoped to an existing unwritable file.
    owner_only = tmp_path / "owner.json"
    owner_only.write_text("old\n", encoding="utf-8")
    owner_only.chmod(0o600)
    _write(writer, owner_only, "new\n")
    assert owner_only.read_text(encoding="utf-8") == "new\n"
    assert _mode(owner_only) == 0o600

    missing = tmp_path / "new.json"
    _write(writer, missing, "fresh\n")
    assert missing.read_text(encoding="utf-8") == "fresh\n"


def test_bench_streaming_out_on_a_readonly_file_exits_2(frozen: Path) -> None:
    # Subprocess, as in tests/test_bench_streaming_exit_contract.py: the exit
    # code is only real once the module guard has had its say (#172).
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_streaming", "--n", "3", "--out", str(frozen)],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert f"::error::failed to write {frozen}" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert frozen.read_text(encoding="utf-8") == "old\n"


def test_run_eval_on_a_readonly_suite_file_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "current"
    out_dir.mkdir()
    suite_file = out_dir / "faithfulness.json"
    suite_file.write_text("old\n", encoding="utf-8")
    suite_file.chmod(0o444)
    monkeypatch.setattr(run_eval, "CURRENT_DIR", out_dir)
    try:
        assert run_eval.main([]) == 2
        err = capsys.readouterr().err
        assert "::error::failed to write eval results" in err
        assert suite_file.read_text(encoding="utf-8") == "old\n"
    finally:
        suite_file.chmod(0o644)


@pytest.mark.parametrize("readonly", run_eval.SUITES)
def test_run_eval_refuses_before_replacing_any_suite_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readonly: str
) -> None:
    # `write_runs` writes the three suites in turn. With the refusal only in
    # the writer, a read-only LAST file was refused after the first two had
    # been replaced: exit 2 over a half-new results directory.
    out_dir = tmp_path / "current"
    out_dir.mkdir()
    for suite in run_eval.SUITES:
        (out_dir / f"{suite}.json").write_text("old\n", encoding="utf-8")
    (out_dir / f"{readonly}.json").chmod(0o444)
    monkeypatch.setattr(run_eval, "CURRENT_DIR", out_dir)
    try:
        assert run_eval.main([]) == 2
        assert {
            suite: (out_dir / f"{suite}.json").read_text(encoding="utf-8")
            for suite in run_eval.SUITES
        } == dict.fromkeys(run_eval.SUITES, "old\n")
    finally:
        (out_dir / f"{readonly}.json").chmod(0o644)


def test_check_writable_agrees_with_the_writer(frozen: Path, tmp_path: Path) -> None:
    with pytest.raises(PermissionError) as excinfo:
        check_writable(frozen)
    assert excinfo.value.errno == errno.EACCES
    link = tmp_path / "link.json"
    link.symlink_to(frozen)
    with pytest.raises(PermissionError):
        check_writable(link)
    # Controls: a missing target and a writable one are both fine.
    check_writable(tmp_path / "missing.json")
    writable = tmp_path / "w.json"
    writable.write_text("x", encoding="utf-8")
    check_writable(writable)
