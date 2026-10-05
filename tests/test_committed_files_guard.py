"""The committed-files guard fires, stays quiet, and is wired (portfolio-ops#79).

A guard nobody has seen fail is a guard nobody has tested. These run a real
inner pytest session in a throwaway git repository whose `conftest.py` is the
guard module verbatim, so the arm exercises the same file this suite loads.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _committed_files_guard as guard

_GUARD_SOURCE = Path(guard.__file__)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "artifact.json").write_text('{"rows": 3}\n', encoding="utf-8")
    (root / "name with space.md").write_text("kept\n", encoding="utf-8")
    shutil.copy(_GUARD_SOURCE, root / "conftest.py")
    _git(root, "init", "-q")
    _git(root, "add", "artifact.json", "name with space.md", "conftest.py")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def _inner_session(root: Path, body: str) -> subprocess.CompletedProcess[str]:
    (root / "test_inner.py").write_text(body, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "test_inner.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_test_that_rewrites_a_tracked_file_fails_the_session(repo: Path) -> None:
    r = _inner_session(
        repo,
        "from pathlib import Path\n"
        "def test_writer():\n"
        "    Path('artifact.json').write_text('{\"rows\": 1}\\n')\n",
    )
    assert r.returncode != 0, r.stdout
    assert "rewrote committed files: ['artifact.json']" in r.stdout


def test_a_deleted_tracked_file_fails_the_session(repo: Path) -> None:
    r = _inner_session(
        repo,
        "from pathlib import Path\ndef test_deleter():\n    Path('name with space.md').unlink()\n",
    )
    assert r.returncode != 0, r.stdout
    assert "name with space.md" in r.stdout


def test_a_test_that_writes_under_tmp_path_passes(repo: Path) -> None:
    # The control: without it, a guard that failed every session would pass
    # both arms above.
    r = _inner_session(
        repo,
        "def test_writer(tmp_path):\n    (tmp_path / 'artifact.json').write_text('x')\n",
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_an_untracked_new_file_is_not_a_change(repo: Path, tmp_path: Path) -> None:
    before = guard.tracked_snapshot(repo)
    (repo / "scratch.txt").write_text("new", encoding="utf-8")
    assert guard.changed_since(before, repo) == []
    assert set(before) == {"artifact.json", "name with space.md", "conftest.py"}


def test_this_suite_runs_under_the_guard(request: pytest.FixtureRequest) -> None:
    # Wiring, behaviourally: the fixture is active in THIS session, so the
    # import in `tests/conftest.py` is not decorative.
    assert "committed_files_are_untouched" in request.fixturenames


def test_this_checkout_is_what_the_guard_snapshots() -> None:
    # The corpus, not just the result: a snapshot of nothing would make every
    # session pass. This file is tracked, so it must be in the population.
    root = guard.repo_root(Path(__file__).resolve().parent)
    if root is None:
        pytest.skip("not a git checkout")
    assert "tests/test_committed_files_guard.py" in guard.tracked_snapshot(root)
