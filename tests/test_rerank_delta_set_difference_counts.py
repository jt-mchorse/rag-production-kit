"""What the displacement score structurally cannot see (#218, D-019).

#217 made the relevance scale a property of `before` alone, so
`["a","b","c"] -> ["a","b","c","x"]` reports `1.0`. That is the *correct*
answer to the question the metric asks — "how much did the reranker move the
input ordering?" — and the answer is "not at all". The gap #218 reports is that
no field revealed `after` holding an id `before` never had.

`RerankDelta` now carries both set differences, and this module's job is to
show that each one is invisible in the other four fields rather than merely
under-emphasised:

- `test_padding_is_invisible_in_every_other_field` is AC2: two calls whose
  `ndcg_displacement` is the same `1.0`, one a clean identity and one an
  identity plus 1,000 invented ids.
- `test_a_dropped_id_is_invisible_in_every_other_field` is the same claim for
  the symmetric direction, and it needed a **search** to find. On a 5-id
  `before` at `k=3` every truncation lowers the displacement and an exhaustive
  walk of all 304 ordered subsets finds zero collisions — which is a true fact
  about that corpus and the wrong corpus to conclude from. At 7 ids and `k=5`
  there are several hundred classes where a truncating and a non-truncating
  output agree on all four fields *to the last bit* (360 on CPython
  3.14/arm64, 346 on 3.11/x86-64 — the exact count turns on exact float
  equality and is therefore a property of the host, so it is asserted as a
  floor). This module pins the smallest of them.

Both counts are reported, never raised. #215 deliberately kept
`["a","b","c"] -> ["x","y","z"]` reporting `0.0` as its contrast row, and
`rerank_delta_ndcg` is telemetry: a misbehaving reranker in production is
exactly when a caller wants a number rather than an exception.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import pytest

from rag_kit.reranker import RerankDelta, rerank_delta_ndcg

BEFORE = ["a", "b", "c"]


# ----------------------------------------------------------------------
# AC2: the same displacement, told apart
# ----------------------------------------------------------------------


def test_padding_is_invisible_in_every_other_field() -> None:
    """A reranker emitting 1,000 candidates it was never given, scoring a perfect 1.0.

    The four original fields are *identical* across these two calls, not merely
    close: `n_input` counts `before`, `top_k_size` is capped by both lists, and
    a foreign id at the tail contributes `rel = 0.0` at a position the top-k
    slice never reaches.
    """
    clean = rerank_delta_ndcg(BEFORE, ["a", "b", "c"], k=3)
    padded = rerank_delta_ndcg(BEFORE, ["a", "b", "c"] + [f"x{i}" for i in range(1000)], k=3)

    original_fields = ("n_input", "top_k_overlap", "top_k_size", "ndcg_displacement")
    for name in original_fields:
        assert getattr(clean, name) == getattr(padded, name), name
    assert clean.ndcg_displacement == 1.0

    # And the new field is what tells them apart.
    assert clean.n_foreign == 0
    assert padded.n_foreign == 1000


def test_a_dropped_id_is_invisible_in_every_other_field() -> None:
    """The symmetric case, and the reason it is a field rather than an omission.

    These two outputs are not merely similar. `RerankDelta.__eq__` on the four
    original fields is `True` and the displacement floats are bit-identical:
    one reranker silently lost document `e`, the other returned all seven and
    reordered the head, and the telemetry could not tell a reader which.
    """
    before = list("abcdefg")
    truncating = rerank_delta_ndcg(before, list("abcdfg"), k=5)  # dropped `e`
    reordering = rerank_delta_ndcg(before, list("cbadfge"), k=5)  # kept all seven

    # Bit-identical *to each other* is the claim, and it is platform-independent:
    # both values come out of the same arithmetic on the same inputs. The
    # absolute value is NOT pinned to its last bit -- it is 0.9374720354963293
    # on CPython 3.14/arm64 and ...91 on 3.11/x86-64, because `math.log2` and
    # float summation differ in the final ULP across libm builds. A literal
    # there is a host-environment assertion, not a test.
    assert truncating.ndcg_displacement.hex() == reordering.ndcg_displacement.hex()
    assert truncating.ndcg_displacement == pytest.approx(0.93747203549, rel=1e-10)
    for name in ("n_input", "top_k_overlap", "top_k_size"):
        assert getattr(truncating, name) == getattr(reordering, name), name

    assert truncating.n_dropped == 1
    assert reordering.n_dropped == 0
    assert truncating.n_foreign == reordering.n_foreign == 0


def test_the_collision_class_is_not_a_lucky_pair() -> None:
    """Searched, not argued — and re-run here so the claim cannot rot.

    This walks the same space as the docstring on `rerank_delta_ndcg` and
    checks that the *new* fields break every class it finds. A weaker version
    of this module would pin the one hand-picked pair above and stay green if
    the counts were computed over the top-k slice instead of the whole ranking.

    **A floor, not an equality, and the reason is measured.** Membership of a
    class turns on *exact* float equality, so the count is platform-dependent:
    360 on CPython 3.14/arm64, 346 on 3.11/x86-64, because `math.log2` and
    float summation differ in the last ULP across libm builds. An `== 360` here
    is a host-environment assertion that passes on the machine it was written
    on and reddens in CI -- which is exactly what it did. What is *not*
    platform-dependent is that the phenomenon is pervasive rather than a lucky
    pair, and that every class the search finds is separated by `n_dropped`.
    The floor is set well below both observed counts.
    """
    ids = [chr(ord("a") + i) for i in range(7)]
    buckets: dict[tuple[int, int, int, float], list[tuple[str, ...]]] = defaultdict(list)
    for size in range(1, len(ids) + 1):
        for combo in itertools.combinations(ids, size):
            for perm in itertools.permutations(combo):
                d = rerank_delta_ndcg(ids, list(perm), k=5)
                key = (d.n_input, d.top_k_overlap, d.top_k_size, d.ndcg_displacement)
                buckets[key].append(perm)

    spanning = [
        (key, members)
        for key, members in buckets.items()
        if len({len(p) for p in members}) > 1  # differing output lengths
    ]
    assert len(spanning) >= 300, (
        f"only {len(spanning)} collision classes found; the observed range is "
        "346 (3.11/x86-64) to 360 (3.14/arm64), so a number far below this "
        "floor means the search space or the metric changed, not the platform"
    )

    # Every one of them is now separable: within a class, the outputs that
    # differ in length differ in `n_dropped`.
    for _key, members in spanning:
        by_len = {len(p): p for p in members}
        deltas = {
            length: rerank_delta_ndcg(ids, list(perm), k=5) for length, perm in by_len.items()
        }
        assert len({d.n_dropped for d in deltas.values()}) == len(deltas)


# ----------------------------------------------------------------------
# What the counts are counted over
# ----------------------------------------------------------------------


def test_a_swap_is_one_foreign_and_one_dropped_not_a_length_difference() -> None:
    """Rejects a count computed from `len()` instead of set membership.

    `["a","b","c"] -> ["a","b","x"]` drops one id and invents another. The
    lengths are equal, so any `len(after) - len(before)` formulation reports
    `0` in both directions while two things went wrong.
    """
    delta = rerank_delta_ndcg(BEFORE, ["a", "b", "x"], k=3)
    assert delta.n_foreign == 1
    assert delta.n_dropped == 1


def test_the_counts_are_over_the_whole_ranking_not_the_top_k_window() -> None:
    """Rejects a count sliced to `eff_k`.

    Both counts describe the ranking the reranker returned. `k` is a knob the
    *caller* chooses for the overlap comparison, so a count that moved with it
    would make the same reranker behaviour report differently to two
    dashboards watching the same stream.
    """
    before = list("abcde")
    after = ["a", "b", "c", "d", "z"]  # one foreign id, at the tail
    at_k2 = rerank_delta_ndcg(before, after, k=2)
    at_k5 = rerank_delta_ndcg(before, after, k=5)

    assert at_k2.top_k_size != at_k5.top_k_size  # the window really does move
    assert at_k2.n_foreign == at_k5.n_foreign == 1
    assert at_k2.n_dropped == at_k5.n_dropped == 1


def test_len_after_is_recoverable_so_n_output_is_not_a_field() -> None:
    """The identity that makes the third field unnecessary.

    `len(after) == n_input - n_dropped + n_foreign`, exactly — the duplicate
    guard means each list's length equals its set cardinality. Checked over a
    spread of shapes rather than one, because an identity pinned on a single
    case is an example.
    """
    cases = [
        (list("abc"), list("abc")),
        (list("abc"), list("abc") + ["x", "y"]),
        (list("abc"), ["a", "b", "x"]),
        (list("abcde"), list("ab")),
        (list("abcde"), ["z"] + list("abcde")),
        (list("abc"), list("xyz")),
        (list("a"), list("a")),
    ]
    for before, after in cases:
        d = rerank_delta_ndcg(before, after, k=3)
        assert len(after) == d.n_input - d.n_dropped + d.n_foreign, (before, after)


# ----------------------------------------------------------------------
# Nothing that shipped moves
# ----------------------------------------------------------------------


def test_the_four_original_fields_are_untouched_across_a_spread_of_shapes() -> None:
    """#218 adds fields; it corrects no value. This is the arm that proves it.

    The expected numbers were read off the pre-change `rag_kit/reranker.py`
    (loaded from a copy of the file at its parent commit) and pasted verbatim,
    so a future edit that "improves" the displacement
    while adding a count goes red here rather than silently moving a published
    telemetry series.
    """
    expected = {
        ("abc", "abc"): (3, 3, 3, 1.0),
        ("abc", "abcx"): (3, 3, 3, 1.0),
        ("abc", "xabc"): (3, 2, 3, 0.697934454765513),
        ("abc", "cba"): (3, 3, 3, 0.7899980042460358),
        ("abc", "xyz"): (3, 0, 3, 0.0),
        ("abcde", "abc"): (5, 3, 3, 0.8784837378625933),
        ("abcde", "cde"): (5, 1, 3, 0.46358005301237865),
        ("abcde", "a"): (5, 1, 1, 0.4867636816216399),
    }
    for (before, after), (n_input, overlap, size, displacement) in expected.items():
        d = rerank_delta_ndcg(list(before), list(after), k=3)
        assert (d.n_input, d.top_k_overlap, d.top_k_size) == (n_input, overlap, size), before
        assert d.ndcg_displacement == pytest.approx(displacement, abs=1e-15), (before, after)


def test_the_new_fields_default_so_the_original_four_still_construct_alone() -> None:
    """AC4, and the reason both defaults can honestly be `0`.

    A zero here means "no such id" — a true statement about a delta built from
    four fields — rather than standing in for an unmeasured quantity. An
    `n_output` field could not do this: `0` there means "the reranker returned
    nothing", which is a real and alarming value, and defaulting to it would
    be the fabricated-extreme default ruled out in llm-cost-optimizer D-018.
    """
    hand_built = RerankDelta(n_input=3, top_k_overlap=3, top_k_size=3, ndcg_displacement=1.0)
    assert hand_built.n_foreign == 0
    assert hand_built.n_dropped == 0
    # And it still compares equal to a real clean run, which is what the
    # existing suite does with it.
    assert hand_built == rerank_delta_ndcg(BEFORE, ["a", "b", "c"], k=3)


def test_foreign_ids_are_reported_never_raised() -> None:
    """#215's contrast row stays a number.

    A wholly disjoint `after` still reports `0.0` rather than raising — that
    row is what made the empty-`before` `1.0` visibly wrong, and this is
    telemetry, where a misbehaving reranker is precisely when a caller needs a
    value. The counts annotate it; they do not gate it.
    """
    d = rerank_delta_ndcg(BEFORE, ["x", "y", "z"], k=3)
    assert d.ndcg_displacement == 0.0
    assert d.n_foreign == 3
    assert d.n_dropped == 3


def test_the_empty_ranking_early_return_reports_no_differences() -> None:
    """Both lists empty: there is nothing foreign and nothing dropped.

    The early return predates these fields and does not set them, so this arm
    is what pins that the defaults are the *right* values there rather than
    merely the ones that happen to appear.
    """
    d = rerank_delta_ndcg([], [], k=5)
    assert (d.n_foreign, d.n_dropped) == (0, 0)
    assert d == RerankDelta(n_input=0, top_k_overlap=0, top_k_size=0, ndcg_displacement=1.0)
