"""The relevance scale is a property of `before` alone (#217).

`rerank_delta_ndcg` scored relevance as `rel[before[i]] = n - i` with
`n = max(len(before_list), len(after_list))`. `n` is the *scale*, so a longer
`after` inflated every relevance to `n, n-1, n-2, …` for a large `n` and shrank
their relative differences. nDCG is a ratio of weighted sums of those
relevances, so the whole score compressed toward `1.0` — which this module
documents as "no change".

Measured on `["a","b","c"]` against a full reversal, varying only the padding
appended to `after`::

    pad=0      0.789998        pad=100     0.995410
    pad=1      0.854921        pad=1000    0.999532
    pad=10     0.961643        pad=10000   0.999953

The reversal is byte-identical in every row. A reranker could reverse its input
and report "kept the input order" to four nines.

**Why the existing `[0, 1]` lock is green on every one of those.** `ideal` and
`actual` are built from the *same* inflated `rel`, so the ratio stays in range
for any padding — the bug moves the value *within* the interval. An invariant
can be perfectly true and still certify the wrong number; the arms here assert
values and a property, never only the range.

**What is pinned is the property, not a literal.** The headline arm is that a
fixed reordering's displacement is invariant to padding. A single pinned number
would only say which padding the author happened to choose; the literal is
carried alongside it, taken from the *unpadded* case, which is the one that was
already correct before this change.

Complementary to #215, not a replacement: with the scale taken from `before`,
an empty `before` gives `ideal == 0`, which is exactly the case #215 raises on.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from rag_kit.reranker import RerankDelta, rerank_delta_ndcg

BEFORE = ["a", "b", "c"]
REVERSED = ["c", "b", "a"]
# The unpadded reversal. This value is unchanged by #217 -- it is the row that
# was always right, and every padded row must now equal it.
REVERSED_DISPLACEMENT = 0.7899980042460358


def _pad(after: list[str], m: int) -> list[str]:
    return after + [f"foreign-{i}" for i in range(m)]


# ----------------------------------------------------------------------
# The property
# ----------------------------------------------------------------------


@pytest.mark.parametrize("padding", [0, 1, 2, 3, 10, 100, 1000])
def test_displacement_is_invariant_to_padding_after(padding: int) -> None:
    """The headline. Only the padding changes; the reordering does not.

    Against the unfixed code this is red for every `padding >= 1`, and the
    parameters do different things: the reported value climbed monotonically
    toward the ceiling as the padding grew.
    """
    delta = rerank_delta_ndcg(BEFORE, _pad(REVERSED, padding), k=3)
    assert delta.ndcg_displacement == pytest.approx(REVERSED_DISPLACEMENT, abs=1e-15)


@pytest.mark.parametrize(
    ("name", "permutation"),
    [
        ("reversed", ["c", "b", "a"]),
        ("swap top two", ["b", "a", "c"]),
        ("rotate", ["b", "c", "a"]),
        ("identity", ["a", "b", "c"]),
    ],
)
def test_no_reordering_can_be_padded_toward_the_ceiling(name: str, permutation: list[str]) -> None:
    """Not specific to a reversal: every reordering converged, and none may now.

    Unfixed, `reversed` read 0.789998 / 0.961643 / 0.999532 / 0.999995 at
    paddings of 0 / 10 / 1000 / 100000, and `swap top two` and `rotate` did the
    same from their own starting points.
    """
    unpadded = rerank_delta_ndcg(BEFORE, permutation, k=3).ndcg_displacement
    for padding in (10, 1000):
        padded = rerank_delta_ndcg(BEFORE, _pad(permutation, padding), k=3)
        assert padded.ndcg_displacement == pytest.approx(unpadded, abs=1e-15), (
            f"{name} moved under padding={padding}"
        )


def test_the_relevance_scale_comes_from_before_not_from_the_longer_list() -> None:
    """The mechanism itself, stated as a number rather than as prose.

    With `n = len(before)` the scale is `a:3, b:2, c:1` no matter how long
    `after` is. The unfixed `max(...)` made it `a:1003, b:1002, c:1001` for a
    1000-id padding, which is the whole defect.
    """

    def ndcg_with_scale(n: int, after: list[str]) -> float:
        rel = {e: float(n - i) for i, e in enumerate(BEFORE)}

        def dcg(seq: list[str]) -> float:
            return sum(rel.get(e, 0.0) / math.log2(i + 2) for i, e in enumerate(seq))

        return dcg(after) / dcg(BEFORE)

    padded = _pad(REVERSED, 1000)
    shipped = rerank_delta_ndcg(BEFORE, padded, k=3).ndcg_displacement

    # Reproduced from the definition with the `before`-only scale...
    assert shipped == pytest.approx(ndcg_with_scale(len(BEFORE), padded), abs=1e-15)
    # ...and *not* with the scale the old `max(...)` produced.
    assert shipped != pytest.approx(ndcg_with_scale(len(padded), padded), abs=1e-6)
    assert ndcg_with_scale(len(padded), padded) == pytest.approx(0.999532, abs=1e-6)


# ----------------------------------------------------------------------
# The branch that had never been exercised with a value
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "after", "expected"),
    [
        # A foreign id appended moves nothing, and reads as nothing moved. That
        # is correct: the metric answers "how much did the input ordering move".
        # It is silent about the reranker having emitted an id it was never
        # given -- see the companion-count follow-up.
        ("foreign appended at the tail", ["a", "b", "c", "x"], 1.0),
        ("foreign at the head", ["x", "a", "b", "c"], 0.697934454765513),
        ("foreign at index 1", ["a", "x", "b", "c"], 0.9304509197357168),
        ("foreign at index 2", ["a", "b", "x", "c"], 0.9854419388428785),
    ],
)
def test_the_variant_table_where_after_is_longer(
    name: str, after: list[str], expected: float
) -> None:
    """Every pre-#217 test had `len(after) <= len(before)`.

    That is the branch where `max(...)` and `len(before_list)` disagree, so it
    is the entire untested surface. The monotone descent as the foreign id moves
    toward the head is the shape a displacement metric should have.
    """
    delta = rerank_delta_ndcg(BEFORE, after, k=3)
    assert delta.ndcg_displacement == pytest.approx(expected, abs=1e-15), name


def test_a_longer_after_does_not_change_the_other_telemetry_fields() -> None:
    """`n_input` counts inputs and `top_k_size` is capped by both lists.

    Pinned because they are the fields a reader might expect to reveal a longer
    `after`, and they do not -- which is the follow-up's premise. That follow-up
    is #218, and it landed: `n_foreign` now reveals it (D-019). The original
    claim is *unchanged and still true* -- the four fields below are exactly as
    blind as they were -- so this test keeps making it, over the four fields it
    was written about rather than over the whole dataclass.
    """
    delta = rerank_delta_ndcg(BEFORE, ["a", "b", "c", "x"], k=3)
    assert (delta.n_input, delta.top_k_overlap, delta.top_k_size) == (3, 3, 3)
    assert delta.ndcg_displacement == 1.0
    # Identical on those four to a run with no foreign id at all: the blindness
    # this test exists to document.
    clean = rerank_delta_ndcg(BEFORE, ["a", "b", "c"], k=3)
    assert clean == RerankDelta(n_input=3, top_k_overlap=3, top_k_size=3, ndcg_displacement=1.0)
    assert (delta.n_input, delta.top_k_overlap, delta.top_k_size, delta.ndcg_displacement) == (
        clean.n_input,
        clean.top_k_overlap,
        clean.top_k_size,
        clean.ndcg_displacement,
    )
    # And the field that ends the blindness. Asserted here too so the two
    # modules cannot drift on what this call means.
    assert delta.n_foreign == 1
    assert clean.n_foreign == 0


# ----------------------------------------------------------------------
# The invariants that must survive
# ----------------------------------------------------------------------


def test_the_unit_interval_holds_for_a_longer_after_too() -> None:
    """Searched, not argued.

    Foreign ids contribute `rel = 0` and push real ids to later positions, so
    `actual <= ideal` by the rearrangement inequality -- `before` is the
    arrangement that maximises the DCG. Brute-forced rather than left as that
    sentence: every `before` of length 1-5 over a 6-id alphabet, 0-3 foreign ids
    appended, six shuffles each.
    """
    rng = random.Random(5)
    ids = [chr(ord("a") + i) for i in range(6)]
    trials = 0
    lo, hi = 2.0, -2.0
    for n_before in range(1, 6):
        for before in itertools.permutations(ids, n_before):
            for extra in range(4):
                pool = list(before) + [f"F{i}" for i in range(extra)]
                for _ in range(6):
                    after = pool[:]
                    rng.shuffle(after)
                    d = rerank_delta_ndcg(list(before), after, k=3).ndcg_displacement
                    trials += 1
                    lo, hi = min(lo, d), max(hi, d)
                    assert 0.0 <= d <= 1.0, (before, after, d)
    assert trials == 29664
    # And the search is not vacuous at either end: it reaches the ceiling and
    # stays well clear of it elsewhere.
    assert hi == 1.0
    assert lo < 0.5


def test_the_n_zero_early_return_is_unchanged_in_meaning() -> None:
    """`n` used to be "the longer list"; it is now "`before`".

    Those coincide at zero *because of* #215's guard: `before` empty with a
    non-empty `after` raises before this line is reached, so `len(before) == 0`
    still implies both lists are empty.
    """
    both_empty = rerank_delta_ndcg([], [], k=5)
    assert both_empty == RerankDelta(
        n_input=0, top_k_overlap=0, top_k_size=0, ndcg_displacement=1.0
    )

    with pytest.raises(ValueError, match="before is empty while after is not"):
        rerank_delta_ndcg([], ["x"], k=3)


@pytest.mark.parametrize(
    ("name", "before", "after", "expected"),
    [
        ("identity", ["a", "b", "c"], ["a", "b", "c"], 1.0),
        ("reversed", ["a", "b", "c"], ["c", "b", "a"], REVERSED_DISPLACEMENT),
        ("all foreign, same length", ["a", "b", "c"], ["x", "y", "z"], 0.0),
        ("after empty", ["a", "b", "c"], [], 0.0),
        ("subset, dropped tail", ["a", "b", "c"], ["a", "b"], 0.894999002123018),
        ("both empty", [], [], 1.0),
    ],
)
def test_nothing_moves_where_the_two_scales_agree(
    name: str, before: list[str], after: list[str], expected: float
) -> None:
    """`max(...)` and `len(before_list)` are the same whenever `after` is no
    longer than `before` — which is every case the suite pinned before #217.
    These arms are green on both trees, deliberately: they are what makes the
    change a fix rather than a re-baselining."""
    delta = rerank_delta_ndcg(before, after, k=3)
    assert delta.ndcg_displacement == pytest.approx(expected, abs=1e-15), name
