"""Lock that every workflow file has a top-level `concurrency:` group.

Companion to `test_workflows_yaml_parseable.py` and
`test_workflows_timeout_minutes.py` — same silent-rot prevention
arc, different failure mode.

The failure mode this catches: without a `concurrency:` group, a rapid
push-on-push (rebased session branch force-pushed, PR chain merged in
quick succession, contributor amending in flight) burns one full CI run
per push even though the in-flight run is immediately superseded. The
wasted runs eat operator quota and obscure which run is the "real"
check for a PR.

The lock matches the audit-side fingerprint shipped in portfolio-ops #41
(`audit_phase_a.py --check missing-concurrency`); each repo's per-repo
lock catches the regression at PR test time, the audit catches it after
the fact for direct-to-main commits and operator-disabled CI.

Spec / origin: this repo's #56. Propagated from llm-eval-harness #64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _all_workflow_files() -> list[Path]:
    if not ACTIVE_WORKFLOWS_DIR.is_dir():
        return []
    return sorted(ACTIVE_WORKFLOWS_DIR.glob("*.yml"))


def _all_workflows() -> list[tuple[str, dict[str, Any]]]:
    """Return (workflow_filename, parsed_yaml) for every workflow file.

    Flattened so pytest parametrization surfaces each missing or
    malformed concurrency block as its own failure, not a single
    "one of N workflows is broken" summary line.
    """
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in _all_workflow_files():
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            rows.append((path.name, parsed))
    return rows


ALL_WORKFLOWS = _all_workflows()


def test_at_least_one_workflow_discovered() -> None:
    # Smoke check: parametrization silently degrades to a no-op if the
    # discovery fixture returns []. Make that loud — a moved or deleted
    # workflows dir should fail the lock, not silently pass it.
    assert ALL_WORKFLOWS, (
        f"No workflow files discovered under {ACTIVE_WORKFLOWS_DIR}. "
        "Either the workflow files were removed or YAML discovery is "
        "broken; this lock should not silently pass in either case."
    )


@pytest.mark.parametrize(
    ("workflow", "parsed"),
    ALL_WORKFLOWS,
    ids=[wf for (wf, _) in ALL_WORKFLOWS],
)
def test_workflow_has_concurrency(workflow: str, parsed: dict[str, Any]) -> None:
    concurrency = parsed.get("concurrency")
    assert concurrency is not None, (
        f"{workflow} has no top-level `concurrency:` block. Without one, "
        f"a rapid push-on-push burns one full CI run per push even when "
        f"the in-flight run is immediately superseded. Add "
        f"`concurrency: {{ group: '<workflow>-${{{{ github.ref }}}}', "
        f"cancel-in-progress: true }}` at the top level. See "
        f".github/workflows/ci.yml in this repo for the template."
    )


@pytest.mark.parametrize(
    ("workflow", "parsed"),
    ALL_WORKFLOWS,
    ids=[wf for (wf, _) in ALL_WORKFLOWS],
)
def test_concurrency_group_is_nonempty_string(workflow: str, parsed: dict[str, Any]) -> None:
    concurrency = parsed.get("concurrency")
    if not isinstance(concurrency, dict):
        pytest.skip("covered by test_workflow_has_concurrency")
    group = concurrency.get("group")
    msg = (
        f"{workflow} has `concurrency.group: {group!r}` "
        f"({type(group).__name__}); must be a non-empty string. GitHub "
        f"Actions evaluates the group at runtime; an empty or missing "
        f"group falls back to a default that doesn't dedupe — silently "
        f"reintroducing the failure mode this lock exists to prevent."
    )
    assert isinstance(group, str), msg
    assert group.strip(), msg


@pytest.mark.parametrize(
    ("workflow", "parsed"),
    ALL_WORKFLOWS,
    ids=[wf for (wf, _) in ALL_WORKFLOWS],
)
def test_concurrency_cancel_in_progress_is_true_bool(workflow: str, parsed: dict[str, Any]) -> None:
    concurrency = parsed.get("concurrency")
    if not isinstance(concurrency, dict):
        pytest.skip("covered by test_workflow_has_concurrency")
    cancel = concurrency.get("cancel-in-progress")
    msg = (
        f"{workflow} has `concurrency.cancel-in-progress: {cancel!r}` "
        f"({type(cancel).__name__}); must be the YAML bool `true`. A "
        f"string `'true'` is parsed but produces the inverse semantics "
        f"under some GitHub Actions paths (queue rather than cancel), "
        f"and `false` defeats the lock's purpose — the prior run would "
        f"complete, burning the quota the lock exists to save."
    )
    # `bool` is a subclass of `int` in Python, so check bool first.
    assert isinstance(cancel, bool), msg
    assert cancel is True, msg
