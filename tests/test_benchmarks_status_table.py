"""`docs/benchmarks.md`'s Status table says where each number lives (#233).

It marked all five rows **pending** against #2, #6 and #7 -- all three closed --
while the README published the eval baselines those rows asked for. Nothing
read the table, so nothing noticed. These arms tie each row that has a
committed artifact to that artifact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOC = (_ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")


def _status_rows() -> list[list[str]]:
    section = _DOC.split("## Status", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| ") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "Metric":
            continue
        rows.append(cells)
    return rows


def test_the_table_is_found() -> None:
    assert len(_status_rows()) >= 7


@pytest.mark.parametrize("suite", ["recall_at_5", "faithfulness", "correctness"])
def test_a_row_backed_by_a_committed_baseline_quotes_it(suite: str) -> None:
    """The value is the baseline's `mean_score` at two places, and the row is
    not pending: the artifact exists, so the number does."""
    baseline = json.loads((_ROOT / "evals" / "baselines" / f"{suite}.json").read_text())
    rows = [r for r in _status_rows() if f"evals/baselines/{suite}.json" in r[1]]
    assert len(rows) == 1, f"no single Status row cites evals/baselines/{suite}.json"
    (row,) = rows
    assert "pending" not in " ".join(row).lower()
    assert row[2] == f"{baseline['mean_score']:.2f}"


def test_only_rows_without_an_artifact_are_pending_and_each_cites_an_issue() -> None:
    pending = [r for r in _status_rows() if "pending" in r[1].lower()]
    assert pending, "the reranker row is genuinely unmeasured; a table with no pending row lies"
    for row in pending:
        assert "evals/" not in row[1]
        ref = re.search(r"\[#(\d+)\]", row[1])
        assert ref, f"pending row cites no issue: {row}"
        # The link is defined in the doc, so it renders.
        assert f"[#{ref.group(1)}]: https://github.com/" in _DOC
