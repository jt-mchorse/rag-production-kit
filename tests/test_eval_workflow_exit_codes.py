"""eval.yml's preview loop tolerates exit 1 (flagged) and fails on 2 (bad input) (#289).

The loop ended each `diff-json` in `|| echo "...regression flagged — continuing"`,
which swallows every non-zero exit. `diff-json` exits 1 when a row is flagged
and 2 on bad input, so a missing or unreadable suite JSON was previewed as if
it had merely regressed. Sibling of llm-eval-harness #310. The step's real
`run:` script runs under `bash -e`, as Actions does, with a PATH stub standing
in for `eval-harness`.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "eval.yml"


def _preview_step() -> dict:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    steps = [s for job in jobs.values() for s in job["steps"]]
    (step,) = [s for s in steps if "run" in s and "eval-harness diff-json" in s["run"]]
    return step


def _run(codes: list[int], tmp_path: Path) -> tuple[int, int]:
    """Run the step with a stub that exits codes[i] on its i-th call."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    counter = tmp_path / "calls"
    counter.write_text("0", encoding="utf-8")
    cases = " ".join(f"{i}) exit {c};;" for i, c in enumerate(codes))
    stub = bindir / "eval-harness"
    stub.write_text(
        "#!/bin/bash\n"
        f'n=$(cat "{counter}"); echo $((n + 1)) > "{counter}"\n'
        f'case "$n" in {cases} *) exit 0;; esac\n',
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    script = tmp_path / "step.sh"
    script.write_text(_preview_step()["run"], encoding="utf-8")
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    rc = subprocess.run(
        ["bash", "-e", str(script)], env=env, capture_output=True, timeout=30
    ).returncode
    return rc, int(counter.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("codes", "expected_rc", "expected_calls"),
    [
        ([0, 0, 0], 0, 3),
        ([1, 1, 1], 0, 3),  # flagged rows: every suite is still previewed
        ([0, 2, 0], 2, 2),  # bad input on suite 2: stop there
    ],
    ids=["clean", "flagged", "bad-input"],
)
def test_the_preview_loop(
    codes: list[int], expected_rc: int, expected_calls: int, tmp_path: Path
) -> None:
    assert _run(codes, tmp_path) == (expected_rc, expected_calls)
