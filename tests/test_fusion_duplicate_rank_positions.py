"""A deduplicated doc must not consume a rank position (#203).

#65 made a doc a method emits twice contribute one `1/(k+rank)` term at its
best rank. It skipped the duplicate but let `enumerate` keep counting it, so
the duplicate still consumed a rank position and shifted every later doc down
one.

All three of #65's tests use `["d1", "d2", "d1"]` — the duplicate **last**,
where it displaces nothing. `test_duplicate_within_method_counts_once_at_best_rank`
even asserts d2 is "unaffected at rank 2", which is true only because nothing
follows the duplicate. The population was "rankings containing a duplicate";
the tests sampled the one position in that population where the bug is
invisible. This file samples the rest of it.

Every existing test in `test_fusion.py` passes against both the old and the new
code, which is exactly why they could not have caught this.
"""

from __future__ import annotations

import math
import random

import pytest

from rag_kit.fusion import DEFAULT_K, reciprocal_rank_fusion


def _dedupe(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _order(rankings: dict[str, list[str]]) -> list[str]:
    return [doc for doc, _, _ in reciprocal_rank_fusion(rankings)]


def _ranks(rankings: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    return {doc: r for doc, _, r in reciprocal_rank_fusion(rankings)}


def _scores(rankings: dict[str, list[str]]) -> dict[str, float]:
    return {doc: s for doc, s, _ in reciprocal_rank_fusion(rankings)}


# --------------------------------------------------------------------------
# The positions #65 did not sample
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "ids"),
    [
        ("duplicate first", ["d1", "d1", "d2", "d3"]),
        ("duplicate in the middle", ["d1", "d2", "d2", "d3"]),
        ("duplicate last (the #65 shape)", ["d1", "d2", "d3", "d3"]),
        ("two duplicates", ["d1", "d1", "d2", "d2", "d3"]),
        ("duplicate far from its original", ["d1", "d2", "d3", "d1"]),
        ("everything duplicated", ["d1", "d1", "d2", "d2", "d3", "d3"]),
    ],
)
def test_recorded_ranks_are_a_dense_run_from_one(label: str, ids: list[str]) -> None:
    """`1..n` with no holes, for n distinct docs.

    Asserted as a *shape* rather than as three specific integers, because the
    defect was that the recorded ranks were neither raw positions nor distinct
    positions but a mix — and "a mix" is what a hole in the sequence is.
    """
    ranks = _ranks({"m": ids})
    got = sorted(r["m"] for r in ranks.values())
    assert got == list(range(1, len(_dedupe(ids)) + 1)), (
        f"{label}: recorded ranks {got} for {len(_dedupe(ids))} distinct docs"
    )


def test_the_65_shape_still_behaves_exactly_as_65_pinned_it() -> None:
    """The duplicate-last case is the one position that was already correct.

    Pinned here too, so a fix that renumbers *everything* — including the case
    #65 got right — cannot pass this file while quietly moving that behaviour.
    """
    fused = reciprocal_rank_fusion({"a": ["d1", "d2", "d1"]}, k=60)
    by_doc = {doc: score for doc, score, _ in fused}
    assert math.isclose(by_doc["d1"], 1 / 61, abs_tol=1e-12)
    assert math.isclose(by_doc["d2"], 1 / 62, abs_tol=1e-12)
    assert _ranks({"a": ["d1", "d2", "d1"]})["d1"] == {"a": 1}


def test_a_duplicate_does_not_deflate_the_score_of_a_later_doc() -> None:
    """The score half of the harm, stated on its own.

    `d3` is third in the method's distinct ranking either way, so it must be
    scored `1/(k+3)` either way.
    """
    with_dup = _scores({"m": ["d1", "d1", "d2", "d3"]})
    without = _scores({"m": ["d1", "d2", "d3"]})
    assert math.isclose(with_dup["d3"], 1 / (DEFAULT_K + 3), abs_tol=1e-12)
    assert with_dup == pytest.approx(without)


# --------------------------------------------------------------------------
# The invariant, as a property rather than a footnote
# --------------------------------------------------------------------------


def test_removing_a_duplicate_changes_nothing_about_the_fused_result() -> None:
    """Fusing a channel must not depend on how that channel was *built*.

    A duplicate comes from a union of two SQL paths or a row surfacing by two
    routes — an implementation detail of the channel, not a property of the
    documents. So for any rankings, fusing them must equal fusing them with
    the intra-method duplicates removed: same order, same scores, same ranks.

    The 19.5%-of-200k figure in #203 is just how often the old code broke this
    invariant; the invariant is the thing worth keeping.
    """
    rng = random.Random(20260904)
    docs = "abcdefg"
    checked = 0
    for _ in range(3000):
        rankings: dict[str, list[str]] = {}
        for m in range(rng.randint(2, 3)):
            picked = rng.sample(docs, rng.randint(2, 5))
            if m == 0 and len(picked) >= 2:
                pos = rng.randrange(1, len(picked))
                picked = picked[:pos] + [rng.choice(picked[:pos])] + picked[pos:]
            rankings[f"m{m}"] = picked
        deduped = {m: _dedupe(v) for m, v in rankings.items()}
        if deduped == rankings:
            continue
        checked += 1
        assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(deduped), (
            f"duplicate changed the fused result: {rankings}"
        )
    # Anti-vacuous: the loop must actually have exercised duplicated rankings.
    assert checked > 1000, f"only {checked} of 3000 trials contained a duplicate"


def test_the_minimal_ordering_flip_from_the_issue() -> None:
    """The smallest case found by that search, kept as a named regression.

    It changes the **top-1**, which is the result an end user sees.
    """
    as_built = {"m0": ["d", "d", "a", "e"], "m1": ["e", "b"], "m2": ["a", "d"]}
    deduped = {"m0": ["d", "a", "e"], "m1": ["e", "b"], "m2": ["a", "d"]}
    assert _order(as_built) == _order(deduped)


# --------------------------------------------------------------------------
# Three neighbouring fixes, each of which passes every test in test_fusion.py
# --------------------------------------------------------------------------


def _fuse_with(strategy: str, rankings: dict[str, list[str]], k: int = DEFAULT_K) -> tuple:
    """Reimplementation of the loop under three alternative strategies.

    `"shipped"` is kept alongside the neighbours so this helper is checked
    against the real module below rather than being a second, drifting copy.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for method, ids in rankings.items():
        seen: set[str] = set()
        accepted = 0
        for raw_rank, doc_id in enumerate(ids, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            accepted += 1
            if strategy == "shipped":
                score_rank = recorded_rank = accepted
            elif strategy == "ranks_only":
                score_rank, recorded_rank = raw_rank, accepted
            elif strategy == "score_only":
                score_rank, recorded_rank = accepted, raw_rank
            else:  # pragma: no cover - guard against a typo in a parametrize id
                raise AssertionError(strategy)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + score_rank)
            ranks.setdefault(doc_id, {})[method] = recorded_rank
    fused = [(d, scores[d], ranks[d]) for d in scores]
    fused.sort(key=lambda row: (-row[1], row[0]))
    return tuple(fused)


def test_the_local_reimplementation_agrees_with_the_module() -> None:
    """Otherwise the two neighbour tests below prove nothing about the module."""
    cases = [
        {"m": ["d1", "d1", "d2", "d3"]},
        {"m0": ["d", "d", "a", "e"], "m1": ["e", "b"], "m2": ["a", "d"]},
        {"a": ["d1", "d2", "d1"], "b": ["d2", "d3"]},
    ]
    for rankings in cases:
        assert _fuse_with("shipped", rankings) == tuple(reciprocal_rank_fusion(rankings))


def test_renumbering_only_the_recorded_ranks_is_not_enough() -> None:
    """Fixes the debugging surface D-004 cares about, leaves the scores deflated.

    It is the more tempting of the two, because #203's most legible symptom is
    a rank of 4 out of three docs — and this makes that symptom go away.
    """
    rankings = {"m": ["d1", "d1", "d2", "d3"]}
    neighbour = _fuse_with("ranks_only", rankings)
    # It genuinely fixes the ranks...
    assert sorted(r["m"] for _, _, r in neighbour) == [1, 2, 3]
    # ...and leaves d3 scored as if it were fourth.
    by_doc = {d: s for d, s, _ in neighbour}
    assert math.isclose(by_doc["d3"], 1 / (DEFAULT_K + 4), abs_tol=1e-12)
    assert not math.isclose(by_doc["d3"], 1 / (DEFAULT_K + 3), abs_tol=1e-12)


def test_renumbering_only_the_score_is_not_enough() -> None:
    """The mirror: correct ordering, and the ranks dict still reports a hole."""
    rankings = {"m": ["d1", "d1", "d2", "d3"]}
    neighbour = _fuse_with("score_only", rankings)
    by_doc = {d: s for d, s, _ in neighbour}
    assert math.isclose(by_doc["d3"], 1 / (DEFAULT_K + 3), abs_tol=1e-12)
    assert sorted(r["m"] for _, _, r in neighbour) == [1, 3, 4]


def test_a_global_seen_set_across_methods_would_be_much_worse() -> None:
    """The neighbour that looks like "just dedupe harder".

    A single seen-set hoisted out of the per-method loop passes every
    single-method test in `test_fusion.py`, and silently drops a doc from the
    second method that the first already ranked — destroying the multi-channel
    agreement RRF exists to measure.
    """
    rankings = {"lexical": ["d1", "d2"], "dense": ["d2", "d3"]}

    scores: dict[str, float] = {}
    seen: set[str] = set()  # hoisted out of the per-method loop -- the bug
    for _method, ids in rankings.items():
        rank = 0
        for doc_id in ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (DEFAULT_K + rank)

    # d2 is ranked by *both* methods and should carry two terms.
    assert math.isclose(scores["d2"], 1 / (DEFAULT_K + 2), abs_tol=1e-12)
    real = {d: s for d, s, _ in reciprocal_rank_fusion(rankings)}
    assert math.isclose(real["d2"], 1 / (DEFAULT_K + 2) + 1 / (DEFAULT_K + 1), abs_tol=1e-12)
    assert real["d2"] > scores["d2"]
