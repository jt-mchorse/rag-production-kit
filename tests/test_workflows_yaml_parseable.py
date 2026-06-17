"""Lock that every workflow YAML in this repo parses cleanly.

Propagation of `portfolio-ops/tests/test_workflows_yaml_parseable.py` (issue
portfolio-ops#30 / PR portfolio-ops#31). The original lock was written
because PR portfolio-ops#27 / portfolio-ops#28 closed a 21-day silent CI
outage caused by a single unquoted colon-space in a `run:` value:

    - name: Verify D-001 baseline decision exists
      run: grep -q "id: D-001" MEMORY/core_decisions_ai.md

`yaml.safe_load()` rejected the line with
`ScannerError: mapping values are not allowed here`. GitHub Actions'
parser was lenient enough to *complete* the workflow run with zero jobs
and `conclusion=failure`. `statusCheckRollup` stayed empty so PR
auto-merge in Phase A couldn't tell that no CI ran.

`rag-production-kit`'s workflows use the `run: |` block-scalar form today
and are YAML-safe, so this lock is the inverse safety net: it makes the
unparseable-shape failure *cannot* land here either. A `jobs:` non-empty
assertion catches the broader "valid YAML, no actual workflow" failure
mode in case GitHub Actions silently absorbs another shape the same way.

Related: portfolio-ops#27, portfolio-ops#28, portfolio-ops#30,
llm-eval-harness#60, this repo's #52.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _all_workflow_files() -> list[Path]:
    if not ACTIVE_WORKFLOWS_DIR.is_dir():
        return []
    return sorted(ACTIVE_WORKFLOWS_DIR.glob("*.yml"))


def _ids(files: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO_ROOT)) for p in files]


WORKFLOW_FILES = _all_workflow_files()


def test_at_least_one_workflow_file_exists() -> None:
    # Smoke check: if this fails, the parametrized tests below silently degrade
    # to a no-op. The fixture-discovery boundary is its own assertion.
    assert WORKFLOW_FILES, (
        f"No *.yml files found under {ACTIVE_WORKFLOWS_DIR}. "
        "If the workflows were intentionally removed, delete this lock test."
    )


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=_ids(WORKFLOW_FILES))
def test_workflow_yaml_parses_cleanly(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - assertion message is the point
        rel = path.relative_to(REPO_ROOT)
        pytest.fail(
            f"{rel} failed yaml.safe_load:\n{exc}\n"
            "GitHub Actions' parser is lenient enough to *complete* a workflow "
            "with an unparseable file, emitting zero jobs and `conclusion=failure` "
            "with an empty `statusCheckRollup` — the exact silent-CI shape that "
            "blocked portfolio-ops for 21 days (#27). Fix the YAML, do not skip this lock."
        )
    assert isinstance(parsed, dict), (
        f"{path.relative_to(REPO_ROOT)} parsed to {type(parsed).__name__}, "
        "expected a top-level mapping. A workflow file should be a YAML mapping "
        "with at least `name`, `on`, and `jobs` keys."
    )


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=_ids(WORKFLOW_FILES))
def test_workflow_has_non_empty_jobs(path: Path) -> None:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    rel = path.relative_to(REPO_ROOT)
    jobs = parsed.get("jobs")
    # Split the assertion (ruff PT018): isinstance check first, non-empty second.
    # A workflow with no jobs is the broader shape of the phantom-failure bug
    # — valid YAML, but GitHub Actions still emits a completed/failure run
    # with zero work. If a file is intentionally a re-usable workflow with
    # only `on:` and a callable surface, exempt it explicitly in this test.
    assert isinstance(jobs, dict), (
        f"{rel}: expected `jobs:` to be a mapping, got {type(jobs).__name__}."
    )
    assert jobs, f"{rel}: `jobs:` mapping is empty."
