"""Regression: `bench_rewriter._print_md` must escape `|` in the free-form
query cell so a piped query can't corrupt the GFM results table (#116).

Backticks do not protect a literal pipe — GFM splits table cells on unescaped
pipes before parsing inline-code spans — so an unescaped `|` in a query injects
a spurious column. This is the same recurring class fixed in the sibling
markdown emitters (#130 / #134 / #79 / #100).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.bench_rewriter import _print_md, _QueryResult  # noqa: E402


def _result(query: str) -> _QueryResult:
    return _QueryResult(
        query=query,
        gold=(),
        baseline_top=[],
        rewriter_top=[],
        baseline_recall=0.5,
        rewriter_recall=0.8,
        sub_queries=(),
    )


def _data_row(output: str) -> str:
    rows = [ln for ln in output.splitlines() if ln.startswith("| `")]
    assert len(rows) == 1, f"expected exactly one data row, got {rows}"
    return rows[0]


def test_piped_query_yields_four_gfm_cells(capsys) -> None:
    # Pre-fix, a query containing `|` split into 5 cells vs. the 4-column
    # header, shifting every downstream column.
    _print_md([_result("compare cats | dogs")], k=5, elapsed_ms=1.0)
    row = _data_row(capsys.readouterr().out)

    # GFM counts only *unescaped* pipes as cell delimiters. Strip the escaped
    # ones first, then split on the structural pipes.
    structural = row.replace("\\|", "")
    cells = [c for c in structural.strip().strip("|").split("|")]
    assert len(cells) == 4, f"row must have 4 GFM cells, got {len(cells)}: {row!r}"
    # The literal pipe survives (escaped) inside the query cell.
    assert "cats \\| dogs" in row


def test_pipe_free_query_is_unchanged(capsys) -> None:
    _print_md([_result("what is a vector database")], k=5, elapsed_ms=1.0)
    row = _data_row(capsys.readouterr().out)
    assert "\\|" not in row
    assert row == "| `what is a vector database` | 0.50 | 0.80 | +0.30 |"
