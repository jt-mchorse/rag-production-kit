"""Snapshot test for the README's rewriter recall@k table.

The README's "Rewriter recall@k on a synthetic multi-hop fixture" table shows
five rows of measured numbers for k ∈ {1, 2, 3, 4, 5} produced by
`scripts/bench_rewriter.py`. The script is deterministic (`TemplateRewriter`
is dep-free, the corpus + questions are committed in-process, the retriever
is the in-memory token-overlap one used by the eval suite for hermetic CI),
so the README's numbers are reproducible — and worth locking against drift.

Same hygiene pattern as the snapshot tests in `llm-cost-optimizer` (for
`docs/savings.{json,md}` + README table) and `prompt-regression-suite` (for
`docs/regression_demo.html`).

When the snapshot fails, regenerate with:

    python -m scripts.bench_rewriter --k <K> --output md

…then `git diff README.md` before committing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.bench_rewriter import _run, _summary  # noqa: E402

README = _REPO_ROOT / "README.md"

REGEN_HINT = (
    "Regenerate the rewriter table:\n"
    "  python -m scripts.bench_rewriter --k 1 --output md\n"
    "  python -m scripts.bench_rewriter --k 2 --output md\n"
    "  python -m scripts.bench_rewriter --k 3 --output md\n"
    "  python -m scripts.bench_rewriter --k 4 --output md\n"
    "  python -m scripts.bench_rewriter --k 5 --output md\n"
    "Update the README rows and inspect with `git diff README.md` before committing."
)

_TABLE_HEADER_RE = re.compile(
    r"^\| k \| mean recall \(baseline\) \| mean recall \(rewriter\) \| "
    r"improvements / regressions \|"
)


def _extract_readme_rewriter_table() -> dict[int, dict[str, float]]:
    """Parse the README's rewriter table; return {k: {fields}}.

    Locating the table by its header signature keeps the test insensitive
    to edits in the surrounding prose. The table is short (5 rows today)
    and pipe-delimited; cells are stripped and converted to floats /
    ints where appropriate.

    Raises AssertionError when the header isn't found so the failure mode
    is loud rather than a silent empty-dict pass.
    """
    lines = README.read_text(encoding="utf-8").splitlines()
    header_index: int | None = None
    for i, line in enumerate(lines):
        if _TABLE_HEADER_RE.match(line):
            header_index = i
            break
    assert header_index is not None, (
        "Could not locate the rewriter table header in README.md. The test "
        "expects a row beginning `| k | mean recall (baseline) | mean recall "
        "(rewriter) | improvements / regressions |`. If the README structure "
        "changed intentionally, update _TABLE_HEADER_RE."
    )

    rows: dict[int, dict[str, float]] = {}
    # Data rows follow the separator. Stop on the first non-pipe line.
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Expected columns: [k, mean_baseline, mean_rewriter, "X / Y"]
        if len(cells) != 4:
            continue  # tolerate stray separator-looking lines
        k_text, base_text, rew_text, deltas_text = cells
        try:
            k = int(k_text)
            base = float(base_text)
            rew = float(rew_text)
        except ValueError:
            continue
        improved_text, regressed_text = (s.strip() for s in deltas_text.split("/", 1))
        rows[k] = {
            "mean_recall_baseline": base,
            "mean_recall_rewriter": rew,
            "queries_with_improvement": int(improved_text),
            "queries_with_regression": int(regressed_text),
        }

    assert rows, "No data rows parsed from the README rewriter table."
    return rows


# k=1 and k=4 joined in #379: k=1 is the only k with a regressing query and
# k=4 ties k=3 for the largest gap, so dropping either hid a published claim.
_TABLE_KS = [1, 2, 3, 4, 5]
_HALF_ROUND = 5e-4 + 1e-9


@pytest.mark.parametrize("k", _TABLE_KS)
def test_readme_rewriter_table_row_matches_bench_output(k: int) -> None:
    """Each README row must match what `bench_rewriter._run(k)` produces today."""
    readme_rows = _extract_readme_rewriter_table()
    assert k in readme_rows, (
        f"README rewriter table is missing the k={k} row.\n"
        f"Available rows: {sorted(readme_rows)}\n{REGEN_HINT}"
    )
    readme_row = readme_rows[k]

    bench_summary = _summary(_run(k))

    # README rounds means to 3 decimals (0.625, 0.688, …). Tolerance 5e-4
    # accommodates the worst-case half-round error, plus 1e-9 because that
    # worst case is reached exactly on this fixture: 0.4375 -> "0.438" (k=1,
    # #379) differs by 5e-4 and float subtraction lands just above it.
    assert readme_row["mean_recall_baseline"] == pytest.approx(
        bench_summary["mean_recall_baseline"], abs=_HALF_ROUND
    ), (
        f"README mean baseline for k={k} ({readme_row['mean_recall_baseline']}) "
        f"doesn't match bench_rewriter output "
        f"({bench_summary['mean_recall_baseline']}).\n{REGEN_HINT}"
    )
    assert readme_row["mean_recall_rewriter"] == pytest.approx(
        bench_summary["mean_recall_rewriter"], abs=_HALF_ROUND
    ), (
        f"README mean rewriter for k={k} ({readme_row['mean_recall_rewriter']}) "
        f"doesn't match bench_rewriter output "
        f"({bench_summary['mean_recall_rewriter']}).\n{REGEN_HINT}"
    )
    # Improvement / regression counts are integers; no rounding slack.
    assert readme_row["queries_with_improvement"] == bench_summary["queries_with_improvement"], (
        f"README improvement count for k={k} "
        f"({readme_row['queries_with_improvement']}) doesn't match bench output "
        f"({bench_summary['queries_with_improvement']}).\n{REGEN_HINT}"
    )
    assert readme_row["queries_with_regression"] == bench_summary["queries_with_regression"], (
        f"README regression count for k={k} "
        f"({readme_row['queries_with_regression']}) doesn't match bench output "
        f"({bench_summary['queries_with_regression']}).\n{REGEN_HINT}"
    )


def test_readme_rewriter_table_has_expected_row_count() -> None:
    """Guard against silently dropping (or adding) a k row from the README."""
    readme_rows = _extract_readme_rewriter_table()
    expected_ks = set(_TABLE_KS)
    assert set(readme_rows) == expected_ks, (
        f"README rewriter table k-values {sorted(readme_rows)} differ from "
        f"the expected set {sorted(expected_ks)}. If the bench's reported "
        f"k-values changed intentionally, update this test.\n{REGEN_HINT}"
    )


# Every k at which the bench reports anything: from k=6 up both paths sit at
# 1.000 recall with 0 / 0, so this range covers each k that can regress.
_PROSE_KS = range(1, 7)


def _readme_rewriter_section() -> str:
    """The README's rewriter-bench subsection, from its bold lead to the next `## `."""
    text = README.read_text(encoding="utf-8")
    start = text.index("**Rewriter recall@k on a synthetic multi-hop fixture**")
    end = text.index("\n## ", start)
    # Collapse the README's hard wraps so a phrase split across lines (the old
    # "No query regresses\non this fixture") still matches.
    return " ".join(text[start:end].split())


@pytest.mark.parametrize("k", _PROSE_KS)
def test_readme_names_every_regressing_query(k: int) -> None:
    """#379: the README said "No query regresses on this fixture" while k=1
    regressed one query. Every query the bench regresses, at any k that can
    move, must be named in the subsection; this is driven by the bench, not by
    the README table, so dropping the k=1 row cannot hide the regression."""
    section = _readme_rewriter_section()
    for r in _run(k):
        if r.rewriter_recall < r.baseline_recall:
            assert f"`{r.query}`" in section, (
                f"bench_rewriter regresses {r.query!r} at k={k} "
                f"({r.baseline_recall:.2f} -> {r.rewriter_recall:.2f}) but the "
                f"README rewriter section does not name it."
            )
            assert f"k={k}" in section, f"README rewriter section never mentions k={k}."


def test_readme_has_no_unscoped_no_regression_claim() -> None:
    """#379: the old sentence "No query regresses on this fixture" was true
    only for the rows shown. Once any k regresses, that wording is false."""
    any_regression = any(_summary(_run(k))["queries_with_regression"] for k in _PROSE_KS)
    if any_regression:
        assert "No query regresses on this fixture" not in _readme_rewriter_section()


def test_readme_largest_gap_claim_matches_table() -> None:
    """#379: k=4 ties k=3 for the largest gap; the prose must name every tied k."""
    gaps = {
        k: round(
            _summary(_run(k))["mean_recall_rewriter"] - _summary(_run(k))["mean_recall_baseline"], 6
        )
        for k in _TABLE_KS
    }
    best = max(gaps.values())
    tied = sorted(k for k, g in gaps.items() if g == best)
    m = re.search(r"The gap is largest at ([^,(]+)", _readme_rewriter_section())
    assert m, "README rewriter section lost its 'The gap is largest at' sentence."
    claimed = sorted(int(x) for x in re.findall(r"k=(\d+)", m.group(1)))
    assert claimed == tied, (
        f"README claims largest gap at k={claimed}; bench says k={tied} ({gaps})."
    )
