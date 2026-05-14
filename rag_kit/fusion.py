"""Reciprocal Rank Fusion.

Single function so the math is reviewable in isolation and the retriever
stays focused on the SQL/IO side.

Reference: Cormack, Clarke & Buettcher (SIGIR 2009), "Reciprocal Rank
Fusion outperforms Condorcet and Individual Rank Learning Methods."
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

DEFAULT_K = 60
"""The k constant in 1 / (k + rank). 60 is the original paper's choice."""


def reciprocal_rank_fusion(
    rankings: Mapping[str, Iterable[str]],
    *,
    k: int = DEFAULT_K,
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse multiple per-method rankings into one ranked list.

    Args:
        rankings: mapping of method-name -> iterable of document ids ordered
            best-to-worst. Methods with no results contribute nothing.
        k: the RRF constant. Higher k smooths the contribution of any
            single ranker.

    Returns:
        List of ``(doc_id, fused_score, per_method_ranks)`` tuples, sorted
        by ``fused_score`` descending. ``per_method_ranks`` exposes the
        1-indexed rank each method gave the doc (missing methods omitted)
        so the retriever can show callers exactly why a doc surfaced.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for method, ids in rankings.items():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(doc_id, {})[method] = rank

    fused = [(doc_id, scores[doc_id], ranks[doc_id]) for doc_id in scores]
    fused.sort(key=lambda row: row[1], reverse=True)
    return fused
