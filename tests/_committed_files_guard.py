"""Suite-wide guard: no test may rewrite a committed file (portfolio-ops#79).

Ported from python-async-llm-pipelines#115, where a test ran a bench script with
its `--out-md` in `tmp_path` and no `--out-json`, so the JSON half defaulted to
a committed artifact. On ext4, which accepts the test's surrogate byte in a
filename, the run completed and every CI run overwrote that file. On APFS the
name is refused and the script exits first, so it never reproduced on a Mac.
The class is "a test wrote outside `tmp_path`", and nothing in a green run says
it happened.

This snapshots every git-tracked file before the session and fails the session
if any of them changed or vanished. **Every** tracked file, not one directory:
pyasync guards `docs/` because that is where its artifacts live, but a repo's
committed outputs sit in `results/`, `evals/`, `datasets/`, `docs/` and the
README, and a full run of each Python repo's suite modified zero tracked files
(portfolio-ops#79, 2026-10-01), so the wider rule costs nothing and needs no
per-repo list to drift.

Self-contained on purpose: `tests/test_committed_files_guard.py` copies this
file into a throwaway git repository as that session's `conftest.py` and checks
that a writer test fails it.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest


def repo_root(start: Path) -> Path | None:
    """The git work tree containing *start*, or None outside one (an sdist)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out)


def tracked_snapshot(root: Path) -> dict[str, str]:
    """sha256 of every git-tracked regular file under *root*, by path.

    `-z` so a tracked name with a space or a newline is one entry, not several.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.split(b"\0")
    snapshot = {}
    for raw in listed:
        if not raw:
            continue
        name = raw.decode("utf-8", "surrogateescape")
        path = root / name
        if path.is_file() and not path.is_symlink():
            snapshot[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def changed_since(before: dict[str, str], root: Path) -> list[str]:
    """Tracked files whose content differs from *before*, or that are gone."""
    changed = []
    for name, digest in before.items():
        path = root / name
        try:
            now = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            now = None
        if now != digest:
            changed.append(name)
    return sorted(changed)


@pytest.fixture(scope="session", autouse=True)
def committed_files_are_untouched() -> Iterator[None]:
    root = repo_root(Path(__file__).resolve().parent)
    if root is None:
        yield
        return
    before = tracked_snapshot(root)
    yield
    changed = changed_since(before, root)
    assert not changed, (
        f"the test session rewrote committed files: {changed}. A test must write "
        f"only under tmp_path (portfolio-ops#79)."
    )
