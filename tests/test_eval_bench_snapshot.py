"""Snapshot test for the README's eval mean-score table.

The README's "First baseline" table (under "Benchmarks / Results") shows three
rows of measured mean scores for the `faithfulness`, `recall_at_5`, and
`correctness` suites produced by `evals/run_eval.py`. The orchestrator is
deterministic: `TemplateGenerator` is dep-free, the dataset and corpus are
committed under `evals/dataset/`, the judge is `deterministic-stub-v1`, and
the retriever is the in-memory token-overlap one used by the rewriter bench
(D-013). The README's numbers are therefore reproducible — and worth locking
against drift.

Same hygiene pattern as `test_rewriter_bench_snapshot.py` in this repo,
the `savings.{json,md}` snapshot in `llm-cost-optimizer`, and the
`regression_demo.html` snapshot in `prompt-regression-suite`. Filed as the
follow-up #19 explicitly recommended.

When the snapshot fails, regenerate with:

    python -m evals.run_eval --write-baselines

…then update the README's mean-score column and inspect with
`git diff README.md` before committing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evals.run_eval import run_all_suites

_REPO_ROOT = Path(__file__).resolve().parents[1]
README = _REPO_ROOT / "README.md"

SUITES = ("faithfulness", "recall_at_5", "correctness")

REGEN_HINT = (
    "Regenerate the eval table:\n"
    "  python -m evals.run_eval --write-baselines\n"
    "Update the README's mean-score column and inspect with\n"
    "`git diff README.md` before committing."
)

# Header signature: a row that opens with `| suite | mean score | reproducer`.
# Whitespace inside cells is permissive; the regex only anchors the column
# names so cosmetic edits (extra spacing, column widening) don't trip it.
_TABLE_HEADER_RE = re.compile(r"^\|\s*suite\s*\|\s*mean score\s*\|\s*reproducer\s*\|")


def _extract_readme_eval_table() -> dict[str, float]:
    """Parse the README's eval mean-score table; return {suite: mean}.

    Locating the table by its header signature keeps the test insensitive to
    edits in the surrounding prose. The table is short (3 rows today) and
    pipe-delimited. The suite name and mean score are the first two cells of
    each row; the third (`reproducer`) is parsed but unused — its presence
    keeps the column count check honest.

    Raises AssertionError when the header isn't found so the failure mode is
    loud rather than a silent empty-dict pass.
    """
    lines = README.read_text(encoding="utf-8").splitlines()
    header_index: int | None = None
    for i, line in enumerate(lines):
        if _TABLE_HEADER_RE.match(line):
            header_index = i
            break
    assert header_index is not None, (
        "Could not locate the eval mean-score table header in README.md. "
        "The test expects a row beginning `| suite | mean score | reproducer |`. "
        "If the README structure changed intentionally, update _TABLE_HEADER_RE."
    )

    rows: dict[str, float] = {}
    # Data rows follow the separator (header + 1). Stop on the first non-pipe line.
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Expected columns: [suite, mean_score, reproducer]
        if len(cells) != 3:
            continue  # tolerate stray separator-looking lines
        suite, mean_text, _reproducer = cells
        try:
            mean = float(mean_text)
        except ValueError:
            continue
        rows[suite] = mean

    assert rows, "No data rows parsed from the README eval mean-score table."
    return rows


@pytest.fixture(scope="module")
def live_means() -> dict[str, float]:
    """Single invocation of the orchestrator shared across parametrized rows.

    `run_all_suites` is hermetic but iterates the full 8-example dataset
    three times (once per scorer pass-through inside the loop), so caching
    one run keeps the test fast.
    """
    runs = run_all_suites()
    return {
        run.suite: run.to_run_result(dataset_version="rag-qa-v0.1")["mean_score"] for run in runs
    }


@pytest.mark.parametrize("suite", SUITES)
def test_readme_eval_table_row_matches_live_orchestrator(
    suite: str, live_means: dict[str, float]
) -> None:
    """Each README row's mean must match what `run_all_suites()` produces today."""
    readme_means = _extract_readme_eval_table()
    assert suite in readme_means, (
        f"README eval table is missing the `{suite}` row.\n"
        f"Available rows: {sorted(readme_means)}\n{REGEN_HINT}"
    )

    # README rounds means to 2 decimals (1.00, 0.90). Tolerance 5e-3
    # accommodates the worst-case half-round error.
    assert readme_means[suite] == pytest.approx(live_means[suite], abs=5e-3), (
        f"README mean for `{suite}` ({readme_means[suite]}) doesn't match "
        f"`run_all_suites()` output ({live_means[suite]:.4f}).\n{REGEN_HINT}"
    )


def test_readme_eval_table_has_expected_suite_set() -> None:
    """Guard against silently dropping or adding a suite row in the README."""
    readme_rows = _extract_readme_eval_table()
    assert set(readme_rows) == set(SUITES), (
        f"README eval table suites {sorted(readme_rows)} differ from "
        f"the expected set {sorted(SUITES)}. If `evals.run_eval.SUITES` "
        f"changed intentionally, update both this test and the README.\n"
        f"{REGEN_HINT}"
    )
