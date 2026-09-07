"""The fused ranking must be a function of the data, not of how the caller
assembled it (#205).

`fusion.py`'s tie-break comment states the property:

    Break ties by doc id (ascending) for a stable, caller-order-independent
    ranking.

That was true only for ties that are *exactly* equal in IEEE-754 — and #69's
own case, two docs whose RRF scores are **mathematically** equal, usually was
not one of those. Floating-point addition is not associative, so a running
`+=` summed each doc's terms in `rankings` iteration order, i.e. the caller's
dict insertion order. Two mathematically-tied docs landed up to an ULP apart,
in a direction that flipped when the caller reordered their methods, and the
doc-id tie-break never fired because `-row[1]` had already separated them.

Measured on the unfixed code over 4000 random rankings, comparing every
permutation of the caller's method dict against the base ordering: 0.75% fused
into a different order and 0.57% changed the top-1 document.

The assertions here are stated over **all permutations**, because the property
is universally quantified and a single hand-picked pair of orderings is a
sample of it. `test_the_reproduction_rests_on_an_exact_tie` pins the arithmetic
the reproduction depends on, so if a future edit changes the rank math this
file fails for the right reason rather than becoming a coincidence.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

import pytest

from rag_kit.fusion import DEFAULT_K, reciprocal_rank_fusion

# --- the reproduction from the issue ---------------------------------------

#: Five channels over six docs. `d2` and `d3` carry the same multiset of terms
#: — {1/61, 1/61, 1/62, 1/63} — so they are mathematically tied.
REPRO: dict[str, list[str]] = {
    "m0": ["d2", "d3", "d1"],
    "m1": ["d3", "d0", "d5"],
    "m2": ["d3", "d0", "d2", "d5", "d1"],
    "m3": ["d0", "d2", "d3", "d1", "d4", "d5"],
    "m4": ["d2", "d4"],
}


def _exact_scores(rankings: dict[str, list[str]], k: int = DEFAULT_K) -> dict[str, Fraction]:
    """The fused scores in exact rational arithmetic.

    Deliberately a re-implementation rather than a call into `fusion`: it is
    the independent oracle the float result is checked against, and an oracle
    that shares the code under test proves nothing.
    """
    out: dict[str, Fraction] = {}
    for method_ids in rankings.values():
        seen: set[str] = set()
        rank = 0
        for doc_id in method_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            out[doc_id] = out.get(doc_id, Fraction(0)) + Fraction(1, k + rank)
    return out


def _exactly_rounded_scores(rankings: dict[str, list[str]], k: int = DEFAULT_K) -> dict[str, float]:
    """What `math.fsum` promises: the correctly-rounded sum of the float terms.

    Note the two oracles differ and both are needed. `_exact_scores` works in
    rationals and answers "are these two docs mathematically tied?" — the
    question the reproduction rests on. This one sums the `Fraction` value of
    each *already-rounded* `1.0 / (k + rank)` float, which is what `fsum` is
    actually specified to return. Asserting the float result against the
    rational sum would overclaim: `1.0 / 61` is not `Fraction(1, 61)`.
    """
    out: dict[str, Fraction] = {}
    for method_ids in rankings.values():
        seen: set[str] = set()
        rank = 0
        for doc_id in method_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            out[doc_id] = out.get(doc_id, Fraction(0)) + Fraction(1.0 / (k + rank))
    return {doc: float(value) for doc, value in out.items()}


def test_the_reproduction_rests_on_an_exact_tie() -> None:
    """The anchor. Without this, the rest of the file is a coincidence.

    If a future change to the rank arithmetic breaks the `d2`/`d3` tie, the
    permutation assertions below would start passing for a reason that has
    nothing to do with `math.fsum`, and nobody would notice.
    """
    exact = _exact_scores(REPRO)
    assert exact["d2"] == exact["d3"]
    assert exact["d2"] == Fraction(15437, 238266)
    # The tie is between the *top two*, which is what makes the flip visible.
    ordered = sorted(exact, key=lambda d: (-exact[d], d))
    assert ordered[:2] == ["d2", "d3"]


def test_the_reproduction_is_stable_under_every_caller_ordering() -> None:
    """All 120 permutations of the five channels, not the two from the issue."""
    baseline = reciprocal_rank_fusion(REPRO)
    for perm in itertools.permutations(REPRO):
        reordered = reciprocal_rank_fusion({key: REPRO[key] for key in perm})
        assert reordered == baseline, perm


def test_the_two_tied_docs_now_land_on_the_identical_float() -> None:
    """The mechanism, not just its consequence.

    The tie-break can only run once `-row[1]` compares equal. Asserting the
    ordering alone would pass against a fix that merely reversed which of the
    two ULPs won.
    """
    scores = {doc: score for doc, score, _ in reciprocal_rank_fusion(REPRO)}
    assert scores["d2"] == scores["d3"]
    # And the shared value is the correctly-rounded sum of the float terms,
    # not one of the two neighbouring floats the old running sum produced.
    assert scores["d2"] == _exactly_rounded_scores(REPRO)["d2"]
    # The doc-id tie-break is what orders them, so it is `d2` before `d3`.
    order = [doc for doc, _, _ in reciprocal_rank_fusion(REPRO)]
    assert order.index("d2") < order.index("d3")


# --- the property, over a randomized corpus --------------------------------


def _random_rankings(rng: random.Random) -> dict[str, list[str]]:
    docs = [f"d{i}" for i in range(rng.randint(3, 6))]
    rankings: dict[str, list[str]] = {}
    for m in range(rng.randint(3, 5)):
        shuffled = docs[:]
        rng.shuffle(shuffled)
        rankings[f"m{m}"] = shuffled[: rng.randint(2, len(docs))]
    return rankings


#: Fixed seed: this is a *regression corpus*, not a fuzz run. A corpus that
#: changes every CI run makes a red build unreproducible.
_CORPUS = [_random_rankings(random.Random(11 + i)) for i in range(400)]


def test_the_corpus_actually_contains_mathematically_tied_pairs() -> None:
    """Anti-vacuous, and the arm that matters most in this file.

    A corpus with no tied pairs would satisfy every permutation assertion
    below **against the unfixed code**, because the defect only manifests
    where two docs are mathematically equal. This test is what makes the
    sweep evidence rather than decoration.
    """
    with_ties = 0
    for rankings in _CORPUS:
        exact = list(_exact_scores(rankings).values())
        if len(set(exact)) < len(exact):
            with_ties += 1
    assert with_ties >= 20, f"only {with_ties}/400 corpus entries contain a tie"


@pytest.mark.parametrize("index", range(len(_CORPUS)), ids=[f"c{i}" for i in range(len(_CORPUS))])
def test_every_permutation_of_the_caller_dict_fuses_identically(index: int) -> None:
    rankings = _CORPUS[index]
    baseline = reciprocal_rank_fusion(rankings)
    for perm in itertools.permutations(rankings):
        reordered = reciprocal_rank_fusion({key: rankings[key] for key in perm})
        assert reordered == baseline, (index, perm)


@pytest.mark.parametrize("index", range(0, len(_CORPUS), 8))
def test_the_float_score_is_the_correctly_rounded_exact_score(index: int) -> None:
    """Stronger than order-independence, and the reason `fsum` was chosen.

    Sorting each doc's terms before a running sum would also be
    order-independent, and would still be *wrong* — a different float from the
    exact answer. This pins the value, so that neighbour cannot pass.
    """
    rankings = _CORPUS[index]
    expected = _exactly_rounded_scores(rankings)
    for doc, score, _ in reciprocal_rank_fusion(rankings):
        assert score == expected[doc], doc


# --- the plausible neighbouring fix, built and run -------------------------


def _rounded_neighbour(rankings: dict[str, list[str]], *, places: int = 12) -> list[tuple]:
    """The fix that reads as correct: round the score before sorting.

    Written out from the **unfixed** running-sum read rather than wrapped
    around `reciprocal_rank_fusion`, because wrapping the shipped function
    would let the neighbour inherit the `fsum` it is supposed to lack.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for method, ids in rankings.items():
        seen: set[str] = set()
        rank = 0
        for doc_id in ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (DEFAULT_K + rank)
            ranks.setdefault(doc_id, {})[method] = rank
    fused = [(doc, scores[doc], ranks[doc]) for doc in scores]
    fused.sort(key=lambda row: (-round(row[1], places), row[0]))
    return fused


def test_the_rounded_neighbour_leaves_the_returned_score_order_dependent() -> None:
    """It orders stably and still hands the caller a different number.

    A consumer that persists `fused_score`, diffs two runs, or thresholds on
    it keeps the entire defect — the rounding exists only inside the sort key.
    """
    baseline = {doc: score for doc, score, _ in _rounded_neighbour(REPRO)}
    differed = False
    for perm in itertools.permutations(REPRO):
        reordered = {
            doc: score for doc, score, _ in _rounded_neighbour({key: REPRO[key] for key in perm})
        }
        if any(reordered[doc] != baseline[doc] for doc in baseline):
            differed = True
            break
    assert differed, "the neighbour was expected to return order-dependent scores"

    # What shipped does not.
    shipped = {doc: score for doc, score, _ in reciprocal_rank_fusion(REPRO)}
    for perm in itertools.permutations(REPRO):
        again = {
            doc: score
            for doc, score, _ in reciprocal_rank_fusion({key: REPRO[key] for key in perm})
        }
        assert again == shipped, perm


def test_the_rounded_neighbour_manufactures_ties_between_separated_docs() -> None:
    """And the rounding is a new defect, not merely an incomplete fix.

    Any fixed rounding step collapses two docs separated by less than that
    step into a tie, and the doc-id tie-break then silently overrules the
    arithmetic. A minimal case: one channel ranking `b` above `a`.

        b -> 1/61 = 0.0163934...
        a -> 1/62 = 0.0161290...

    They differ by 2.6e-4, so rounding to 3 places puts both at 0.016 and the
    doc-id tie-break reverses the only ranking signal in the input. `fusion`
    orders them correctly because it never rounds.
    """
    one_channel = {"m": ["b", "a"]}
    exact = _exact_scores(one_channel)
    assert exact["b"] > exact["a"], "the fixture must have a real, if small, margin"

    assert [doc for doc, _, _ in reciprocal_rank_fusion(one_channel)] == ["b", "a"]
    assert [doc for doc, _, _ in _rounded_neighbour(one_channel, places=3)] == ["a", "b"]

    # The step size is what does it: a fine enough rounding keeps them apart,
    # so this is a property of choosing *any* fixed precision, not of the 3.
    assert [doc for doc, _, _ in _rounded_neighbour(one_channel, places=12)] == ["b", "a"]


# --- the arithmetic guarantee `fsum` is relied on for ----------------------


def test_fsum_is_order_independent_over_the_term_shapes_this_module_produces() -> None:
    """The load-bearing property of the stdlib call, exercised on real terms.

    `math.fsum` returns the correctly-rounded value of the exact sum, so it
    depends on the multiset of terms and not their order. Asserted rather than
    assumed, on `1/(k+rank)` terms specifically.
    """
    rng = random.Random(29)
    for _ in range(2000):
        term_list = [1.0 / (DEFAULT_K + rng.randint(1, 50)) for _ in range(rng.randint(2, 8))]
        target = math.fsum(term_list)
        for _ in range(5):
            shuffled = term_list[:]
            rng.shuffle(shuffled)
            assert math.fsum(shuffled) == target
        # And a naive running sum is not: this is the contrast the fix rests
        # on, so it is pinned rather than asserted in a comment.
    naive_differs = False
    rng = random.Random(31)
    for _ in range(5000):
        term_list = [1.0 / (DEFAULT_K + rng.randint(1, 50)) for _ in range(rng.randint(3, 8))]
        running = 0.0
        for value in term_list:
            running += value
        reversed_running = 0.0
        for value in reversed(term_list):
            reversed_running += value
        if running != reversed_running:
            naive_differs = True
            break
    assert naive_differs, "a running sum was expected to be order-dependent on these terms"
