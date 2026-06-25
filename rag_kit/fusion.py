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
    # Pre-#40 sign-only: bool (True=1) silently shifted the RRF constant from
    # 60 to 1, and float (0.5) silently changed the 1/(k+rank) score curve.
    # Mirrors the portfolio-wide finiteness/integer-positive sweep (#38 here,
    # llm-eval-harness#42, etc.).
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for method, ids in rankings.items():
        # RRF contributes exactly one 1/(k+rank) term per (method, doc)
        # (Cormack et al. 2009). A method may still emit the same doc twice —
        # a union of two SQL paths, a hybrid surfacing one row via two routes,
        # or an upstream dedup bug. Counting both occurrences would double-add
        # the score and overwrite the recorded rank with the worse one, so we
        # keep only the first (best-rank) occurrence per method.
        seen_in_method: set[str] = set()
        for rank, doc_id in enumerate(ids, start=1):
            if doc_id in seen_in_method:
                continue
            seen_in_method.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(doc_id, {})[method] = rank

    # RRF ties are common: 1/(k+rank) sums collide for any symmetric rank
    # configuration. Sorting on score alone leaves tied docs in `scores`
    # insertion order, which depends on the incidental order methods appear in
    # `rankings` and the order doc ids appear within each list -- so the same
    # rankings can yield a different top-k just from caller method-ordering.
    # Break ties by doc id (ascending) for a stable, caller-order-independent
    # ranking. Same class as the chunking-strategies-lab cosine-tie fix (#69).
    fused = [(doc_id, scores[doc_id], ranks[doc_id]) for doc_id in scores]
    fused.sort(key=lambda row: (-row[1], row[0]))
    return fused
