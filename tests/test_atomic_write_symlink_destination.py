"""`atomic_write_text` writes THROUGH a symlinked destination (#296).

`os.replace` renames onto the link itself. A symlinked destination used to
become a regular file, and the file it pointed at kept its old contents.
`Path.write_text`, which this helper replaced and whose file-mode behaviour
#247 restored, writes through the link. Each write-through case runs
`Path.write_text` on an identical layout as well, so the lock checks parity
with it instead of a hand-written expectation. Sibling of
python-async-llm-pipelines#157.
"""

from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from rag_kit.io_utils import atomic_write_text

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink creation needs privileges on Windows"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _layout(root: Path, *, absolute: bool, name: str = "bench.json") -> tuple[Path, Path]:
    real_dir = root / "real"
    real_dir.mkdir(parents=True)
    real = real_dir / name
    real.write_text("old\n")
    real.chmod(0o640)
    link = root / "link.json"
    link.symlink_to(real if absolute else Path("real") / name)
    return link, real


@pytest.mark.parametrize("absolute", [False, True], ids=["relative-link", "absolute-link"])
@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_write_goes_through_the_link(tmp_path: Path, absolute: bool, writer: str) -> None:
    link, real = _layout(tmp_path, absolute=absolute)
    if writer == "atomic":
        atomic_write_text(link, "new\n")
    else:
        link.write_text("new\n")
    assert link.is_symlink(), "the link was replaced by a regular file"
    assert real.read_text() == "new\n", "the linked file kept its old contents"
    assert link.read_text() == "new\n"
    # The linked file's mode is kept (#247), on the file that was written.
    assert stat.S_IMODE(os.stat(real).st_mode) == 0o640
    # No temp file left behind beside the link or beside the linked file.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link.json", "real"]
    assert sorted(p.name for p in real.parent.iterdir()) == ["bench.json"]


@pytest.mark.parametrize("writer", ["atomic", "write_text"])
def test_dangling_link_creates_its_target(tmp_path: Path, writer: str) -> None:
    (tmp_path / "real").mkdir()
    real = tmp_path / "real" / "new.json"
    link = tmp_path / "link.json"
    link.symlink_to(Path("real") / "new.json")
    if writer == "atomic":
        atomic_write_text(link, "fresh\n")
    else:
        link.write_text("fresh\n")
    assert link.is_symlink()
    assert real.read_text() == "fresh\n"


def test_link_loop_raises_oserror_and_leaves_no_temp(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.symlink_to("b.json")
    b.symlink_to("a.json")
    with pytest.raises(OSError, match="symbolic links") as exc:
        atomic_write_text(a, "x\n")
    assert exc.value.errno == errno.ELOOP
    assert a.is_symlink()
    assert b.is_symlink()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.json"]


def test_long_linked_name_still_gets_a_capped_temp_name(tmp_path: Path) -> None:
    """The temp name is built from the RESOLVED basename, so the #128 cap applies to it."""
    link, real = _layout(tmp_path, absolute=False, name="r" * 250 + ".json")
    atomic_write_text(link, "new\n")
    assert link.is_symlink()
    assert real.read_text() == "new\n"


def test_plain_destination_is_unchanged_behaviour(tmp_path: Path) -> None:
    dest = tmp_path / "plain.json"
    dest.write_text("old\n")
    dest.chmod(0o640)
    atomic_write_text(dest, "new\n")
    assert not dest.is_symlink()
    assert dest.read_text() == "new\n"
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o640


def test_bench_streaming_out_through_a_link_updates_the_linked_file(tmp_path: Path) -> None:
    """End to end: `bench_streaming --out link.json` updates the file the link names."""
    link, real = _layout(tmp_path, absolute=False)
    proc = subprocess.run(
        [sys.executable, "scripts/bench_streaming.py", "--n", "3", "--out", str(link)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert link.is_symlink()
    assert real.read_text().startswith("{")
    assert stat.S_IMODE(os.stat(real).st_mode) == 0o640
