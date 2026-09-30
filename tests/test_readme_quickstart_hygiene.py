"""The README's commands, run from a fresh clone, do what they say and leave no untracked files (#239).

Two findings from running the Quickstart on a fresh clone:

- ``docker compose up -d`` returns once the container has *started*. The
  comment under it said it "waits for pg_isready", but the compose
  ``healthcheck`` only gates anything under ``--wait``. On a first start the
  image still runs ``initdb`` and the mounted schema, so the Python snippet
  that follows could hit connection refused or a missing table.
- ``TelemetryStore("./telemetry.db")`` and ``scripts.telemetry_dashboard
  --db ./telemetry.db`` write into the working directory, which is the
  checkout root when you run them as documented. ``.gitignore`` did not cover
  it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
COMPOSE = REPO_ROOT / "docker-compose.yml"

_FENCE = re.compile(r"^```(\w*)\s*$")


def _fenced_lines(langs: set[str]) -> list[str]:
    lines: list[str] = []
    lang: str | None = None
    for line in README.read_text(encoding="utf-8").splitlines():
        fence = _FENCE.match(line)
        if fence:
            lang = None if lang is not None else fence.group(1)
            continue
        if lang in langs:
            lines.append(line)
    return lines


def test_compose_up_waits_for_the_healthcheck() -> None:
    # The premise: the compose file declares a healthcheck. If it stops
    # doing so, --wait only waits for "running" and this test's reason is gone.
    assert "healthcheck:" in COMPOSE.read_text(encoding="utf-8")
    ups = [ln for ln in _fenced_lines({"bash"}) if re.match(r"\s*docker compose up\b", ln)]
    assert ups, "no `docker compose up` in a README bash fence; the pattern went stale"
    missing = [ln.strip() for ln in ups if "--wait" not in ln.split()]
    assert not missing, (
        f"README starts compose without --wait: {missing}. `up -d` returns before "
        "the pg_isready healthcheck passes, so the next command can race initdb."
    )


def test_every_documented_db_path_is_gitignored() -> None:
    paths: set[str] = set()
    # A path is a quoted string literal (python) or a `--db` value (bash).
    # Not a bare `\w+.db`, which also matches the module in `from rag_kit.db import`.
    for ln in _fenced_lines({"python"}):
        paths.update(re.findall(r"""["'](?:\./)?([\w.-]+\.db)["']""", ln))
    for ln in _fenced_lines({"bash"}):
        paths.update(re.findall(r"--db[ =](?:\./)?([\w.-]+\.db)\b", ln))
    # Three documented sites today, one path: `./telemetry.db`.
    assert paths, "no *.db path in a README bash/python fence; the pattern went stale"
    # --no-index: judge the ignore rules alone, so a path that was committed by
    # mistake is still reported as "should be ignored" rather than hidden.
    unignored = sorted(
        p
        for p in paths
        if subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", "--no-index", p],
            check=False,
        ).returncode
        != 0
    )
    assert not unignored, (
        f"README commands write {unignored} into the working directory and "
        ".gitignore does not cover it; a documented run leaves it for `git add -A`."
    )
