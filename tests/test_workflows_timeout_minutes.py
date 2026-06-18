"""Lock that every workflow job has a sensible `timeout-minutes` bound.

Companion to `test_workflows_yaml_parseable.py` (this repo's #52, propagated
from portfolio-ops#30 / #31) — same silent-rot prevention arc, different
failure mode.

The failure mode this catches: GitHub Actions defaults to 360 minutes
(6 hours) per job when no `timeout-minutes` is set. A hung job — network
stall during `pip install`, infinite test loop, stuck rerank call, pgvector
container hang during health-check — therefore burns the full 6-hour
ceiling before the runner kills it. That's quota the operator pays for
whether the run produced anything or not.

`llm-eval-harness` shipped the canonical first hop in PR #63. This file
is the propagation hop for `rag-production-kit` per issue #54. The
policy band [1, 30] matches llm-eval-harness's; per-job values land at
15 for everything except `integration-pg`, which gets 20 because the
postgres+pgvector container spin-up plus full `pytest -m pg` integration
suite is the longest-running job and deserves slightly wider headroom.

Spec / origin: this repo's #54.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Policy band for this repo. Tight enough to flag accidental long bounds;
# wide enough for `integration-pg`'s pg container start + integration suite.
# Bumping the ceiling is intentional and should land with a comment naming
# the workload that forced the change (e.g., a future benchmark job).
MIN_TIMEOUT_MINUTES = 1
MAX_TIMEOUT_MINUTES = 30


def _all_workflow_files() -> list[Path]:
    if not ACTIVE_WORKFLOWS_DIR.is_dir():
        return []
    return sorted(ACTIVE_WORKFLOWS_DIR.glob("*.yml"))


def _all_jobs() -> list[tuple[str, str, dict[str, Any]]]:
    """Return (workflow_filename, job_id, job_body) for every job.

    Flattened across all workflow files so pytest parametrization
    surfaces each missing or out-of-band timeout as its own failure,
    not a single "one of N jobs is broken" summary line.
    """
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for path in _all_workflow_files():
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            continue
        jobs = parsed.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_id, body in jobs.items():
            if isinstance(body, dict):
                rows.append((path.name, str(job_id), body))
    return rows


ALL_JOBS = _all_jobs()


def test_at_least_one_job_discovered() -> None:
    # Smoke check: parametrization silently degrades to a no-op if the
    # discovery fixture returns []. Make that loud.
    assert ALL_JOBS, (
        f"No jobs discovered under {ACTIVE_WORKFLOWS_DIR}. Either the "
        "workflow files were removed or YAML discovery is broken; this "
        "lock should not silently pass in either case."
    )


@pytest.mark.parametrize(
    ("workflow", "job_id", "body"),
    ALL_JOBS,
    ids=[f"{wf}::{jid}" for (wf, jid, _) in ALL_JOBS],
)
def test_job_has_timeout_minutes(workflow: str, job_id: str, body: dict[str, Any]) -> None:
    timeout = body.get("timeout-minutes")
    assert timeout is not None, (
        f"{workflow}::{job_id} has no `timeout-minutes` set. GitHub "
        f"Actions defaults to 360 min/job when this is missing — a hung "
        f"job (network stall, infinite loop, stuck API call, pgvector "
        f"health-check loop) burns the full 6-hour ceiling before the "
        f"runner kills it. Set `timeout-minutes:` on this job. For this "
        f"repo's workloads, 15 is the policy default for pure-python jobs "
        f"and 20 for `integration-pg`; tighten or relax to match the job, "
        f"but stay in [{MIN_TIMEOUT_MINUTES}, {MAX_TIMEOUT_MINUTES}]."
    )


@pytest.mark.parametrize(
    ("workflow", "job_id", "body"),
    ALL_JOBS,
    ids=[f"{wf}::{jid}" for (wf, jid, _) in ALL_JOBS],
)
def test_job_timeout_is_int(workflow: str, job_id: str, body: dict[str, Any]) -> None:
    timeout = body.get("timeout-minutes")
    if timeout is None:
        pytest.skip("covered by test_job_has_timeout_minutes")
    msg = (
        f"{workflow}::{job_id} has `timeout-minutes: {timeout!r}` "
        f"({type(timeout).__name__}); GitHub Actions requires an integer. "
        "A YAML string like `'15'` is parsed but rejected at workflow-load "
        "time, producing a silent failure shape similar to #52."
    )
    # `bool` is a subclass of `int` in Python; reject it explicitly so a
    # stray `timeout-minutes: true` (parsed as 1) doesn't sneak past.
    assert isinstance(timeout, int), msg
    assert not isinstance(timeout, bool), msg


@pytest.mark.parametrize(
    ("workflow", "job_id", "body"),
    ALL_JOBS,
    ids=[f"{wf}::{jid}" for (wf, jid, _) in ALL_JOBS],
)
def test_job_timeout_in_policy_band(workflow: str, job_id: str, body: dict[str, Any]) -> None:
    timeout = body.get("timeout-minutes")
    if not isinstance(timeout, int) or isinstance(timeout, bool):
        pytest.skip("covered by test_job_timeout_is_int")
    assert MIN_TIMEOUT_MINUTES <= timeout <= MAX_TIMEOUT_MINUTES, (
        f"{workflow}::{job_id} has `timeout-minutes: {timeout}` outside the "
        f"policy band [{MIN_TIMEOUT_MINUTES}, {MAX_TIMEOUT_MINUTES}]. Values "
        f"above the ceiling reintroduce most of the unbounded-job quota burn; "
        f"values at 0 disable the timeout entirely (GitHub Actions semantics). "
        f"If this job genuinely needs a wider bound (e.g., a future benchmark "
        f"job), bump MAX_TIMEOUT_MINUTES with a comment naming the workload."
    )
