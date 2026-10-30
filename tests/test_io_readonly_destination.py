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


# -- #436: what `check_writable` must refuse beyond a read-only file ----------
#
# Each layout is one the writer fails on AFTER its caller has replaced the
# files before it: a directory target (the rename raises), and a symlink whose
# RESOLVED parent the temp file cannot be created in. A link's resolved parent
# is its own, so "a parent problem fails on the first write" does not hold.


@pytest.fixture
def readonly_dir(tmp_path: Path) -> Iterator[Path]:
    """A 0555 directory holding ``c.json``; made writable again for cleanup."""
    path = tmp_path / "ro"
    path.mkdir()
    (path / "c.json").write_text("old\n", encoding="utf-8")
    path.chmod(0o555)
    yield path
    path.chmod(0o755)


def test_directory_target_is_refused_like_the_writer(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.mkdir()
    with pytest.raises(IsADirectoryError):
        check_writable(target)
    # Parity: the writer raises the same class, only later (at the rename).
    with pytest.raises(IsADirectoryError):
        atomic_write_text(target, "new\n")
    assert target.is_dir()


def test_symlink_to_a_directory_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "d"
    real.mkdir()
    link = tmp_path / "out.json"
    link.symlink_to(real)
    with pytest.raises(IsADirectoryError):
        check_writable(link)


@pytest.mark.parametrize("name", ["c.json", "missing.json"])
def test_symlink_into_a_readonly_dir_is_refused_like_the_writer(
    tmp_path: Path, readonly_dir: Path, name: str
) -> None:
    # Existing file (writable itself) and dangling link: either way the temp
    # file goes beside the resolved path, in a directory it cannot be made in.
    link = tmp_path / "out.json"
    link.symlink_to(readonly_dir / name)
    with pytest.raises(PermissionError) as excinfo:
        check_writable(link)
    assert excinfo.value.errno == errno.EACCES
    with pytest.raises(PermissionError):
        atomic_write_text(link, "new\n")
    assert (readonly_dir / "c.json").read_text(encoding="utf-8") == "old\n"
    assert sorted(p.name for p in readonly_dir.iterdir()) == ["c.json"]


def test_missing_parents_are_checked_at_the_first_existing_one(
    tmp_path: Path, readonly_dir: Path
) -> None:
    # The writer makes missing directories, so the question goes to the
    # nearest one that exists: a read-only one is refused like the writer.
    nested = readonly_dir / "a" / "b" / "r.json"
    with pytest.raises(PermissionError):
        check_writable(nested)
    with pytest.raises(PermissionError):
        atomic_write_text(nested, "new\n")
    # Control: under a writable directory the create case passes.
    check_writable(tmp_path / "x" / "y" / "r.json")
    atomic_write_text(tmp_path / "x" / "y" / "r.json", "new\n")


def test_a_file_in_the_chain_is_refused(tmp_path: Path) -> None:
    blocker = tmp_path / "f"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        check_writable(blocker / "r.json")
    with pytest.raises(NotADirectoryError):
        check_writable(blocker / "a" / "r.json")
    # The writer fails here too, with the class its `mkdir` raises.
    with pytest.raises((FileExistsError, NotADirectoryError)):
        atomic_write_text(blocker / "r.json", "new\n")


def test_a_symlink_loop_still_raises_its_own_error(tmp_path: Path) -> None:
    # Control: the read-only check's `os.stat` raises ELOOP first.
    link = tmp_path / "loop.json"
    link.symlink_to(link)
    with pytest.raises(OSError, match="symbolic links") as excinfo:
        check_writable(link)
    assert excinfo.value.errno == errno.ELOOP


def _suite_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "current"
    out_dir.mkdir()
    for suite in run_eval.SUITES:
        (out_dir / f"{suite}.json").write_text("old\n", encoding="utf-8")
    return out_dir


def _assert_untouched(out_dir: Path, skip: str) -> None:
    assert {
        suite: (out_dir / f"{suite}.json").read_text(encoding="utf-8")
        for suite in run_eval.SUITES
        if suite != skip
    } == {suite: "old\n" for suite in run_eval.SUITES if suite != skip}


@pytest.mark.parametrize("blocked", run_eval.SUITES)
def test_run_eval_refuses_a_directory_suite_target_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: str
) -> None:
    out_dir = _suite_dir(tmp_path)
    (out_dir / f"{blocked}.json").unlink()
    (out_dir / f"{blocked}.json").mkdir()
    monkeypatch.setattr(run_eval, "CURRENT_DIR", out_dir)
    assert run_eval.main([]) == 2
    _assert_untouched(out_dir, skip=blocked)


@pytest.mark.parametrize("blocked", run_eval.SUITES)
def test_run_eval_refuses_a_link_into_a_readonly_dir_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readonly_dir: Path, blocked: str
) -> None:
    out_dir = _suite_dir(tmp_path)
    (out_dir / f"{blocked}.json").unlink()
    (out_dir / f"{blocked}.json").symlink_to(readonly_dir / "c.json")
    monkeypatch.setattr(run_eval, "CURRENT_DIR", out_dir)
    assert run_eval.main([]) == 2
    _assert_untouched(out_dir, skip=blocked)
    assert (readonly_dir / "c.json").read_text(encoding="utf-8") == "old\n"
