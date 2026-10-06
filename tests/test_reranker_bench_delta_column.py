"""The Δ column is the difference of the two recalls printed beside it (#264).

`render_markdown` took Δ from the unrounded recalls and printed all three at
`.3f`. Measured on main (acab504) with `--k 1..10 --candidates 20`, the
multi-hop k=2 row read `0.625 | 0.688 | +0.062`: 0.6875 - 0.625 = 0.0625 is a
rounding tie that formats `.062`, while 0.6875 alone formats `.688`. The
default k=1,3,5 happened to agree.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.bench_reranker as bench  # noqa: E402
from rag_kit.reranker import LexicalOverlapReranker  # noqa: E402


def _table(ks: tuple[int, ...]) -> list[list[str]]:
    fx = bench.fixtures()
    rows = [
        row for f in fx for row in bench.measure(f, LexicalOverlapReranker(), ks=ks, candidates=20)
    ]
    md = bench.render_markdown(fx, rows, candidates=20)
    out = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[1].isdigit():
            out.append(cells)
    return out


def test_every_delta_is_the_difference_of_its_printed_recalls() -> None:
    table = _table(tuple(range(1, 11)))
    assert len(table) == 20  # two fixtures x ten k
    for fixture, k, fused, reranked, delta, _ in table:
        assert Fraction(delta) == Fraction(reranked) - Fraction(fused), (fixture, k)


def test_the_measured_row() -> None:
    rows = {(c[0], c[1]): c for c in _table(tuple(range(1, 11)))}
    assert rows[("multi-hop", "2")][2:5] == ["0.625", "0.688", "+0.063"]
