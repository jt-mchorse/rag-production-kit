"""Docs whose RRF scores are exactly equal tie, whatever ranks produced them (#358).

#205 summed each doc's terms with `math.fsum`, which correctly rounds the sum of
the *float* terms -- each already rounded from `1/(k+rank)` -- so it equalised
docs with the same multiset of terms only. Measured on main at k=60: doc `a`
at dense 3 / lexical 80 and doc `b` at dense 24 / lexical 30 are exactly tied
(1/63 + 1/140 == 1/84 + 1/90), but `b` scored 0.023015873015873017 and `a`
0.023015873015873014, so `b` ranked first and the doc-id tie-break never ran.
Over ranks 1-300, 134 of 519 groups of exact ties were split this way.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

from rag_kit.fusion import DEFAULT_K, reciprocal_rank_fusion


def _rankings(placements: dict[str, tuple[int, int]], depth: int = 300) -> dict[str, list[str]]:
    dense = [f"x{i}" for i in range(1, depth + 1)]
    lexical = [f"y{i}" for i in range(1, depth + 1)]
    for doc, (d_rank, l_rank) in placements.items():
        dense[d_rank - 1] = doc
        lexical[l_rank - 1] = doc
    return {"dense": dense, "lexical": lexical}


def test_the_issues_pair_ties_and_the_doc_id_decides() -> None:
    assert Fraction(1, 63) + Fraction(1, 140) == Fraction(1, 84) + Fraction(1, 90)
    out = reciprocal_rank_fusion(_rankings({"b": (24, 30), "a": (3, 80)}), k=60)
    pair = [row for row in out if row[0] in {"a", "b"}]
    assert [row[0] for row in pair] == ["a", "b"]
    assert pair[0][1] == pair[1][1]


def _exact_tie_pairs(
    limit: int = 150, k: int = DEFAULT_K
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    by_score: dict[Fraction, list[tuple[int, int]]] = {}
    for d, lex in itertools.combinations(range(1, limit + 1), 2):
        by_score.setdefault(Fraction(1, k + d) + Fraction(1, k + lex), []).append((d, lex))
    return [(g[0], g[1]) for g in by_score.values() if len(g) > 1]


def test_every_exact_tie_among_small_ranks_prints_equal_and_orders_by_doc_id() -> None:
    pairs = _exact_tie_pairs()
    split_by_floats = 0
    for p, q in pairs:
        if 1.0 / (DEFAULT_K + p[0]) + 1.0 / (DEFAULT_K + p[1]) != 1.0 / (DEFAULT_K + q[0]) + 1.0 / (
            DEFAULT_K + q[1]
        ):
            split_by_floats += 1
        # Put the later doc id on the ranks whose float sum would come out higher.
        out = reciprocal_rank_fusion(_rankings({"zz": p, "aa": q}, depth=200))
        rows = [row for row in out if row[0] in {"aa", "zz"}]
        assert [row[0] for row in rows] == ["aa", "zz"], (p, q)
        assert rows[0][1] == rows[1][1], (p, q)
    # Anti-vacuity: the brute force found exact ties, and the plain float sums
    # split some of them -- the population the fix is for.
    assert len(pairs) >= 10
    assert split_by_floats >= 1


def test_non_tied_docs_keep_their_order() -> None:
    out = reciprocal_rank_fusion(_rankings({"a": (1, 1), "b": (2, 2)}, depth=10))
    assert [row[0] for row in out][:2] == ["a", "b"]
    assert out[0][1] > out[1][1]
