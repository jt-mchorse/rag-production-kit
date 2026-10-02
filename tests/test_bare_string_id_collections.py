"""A bare `str` where a collection of ids is expected is refused (#253).

A `str` is a `Sequence[str]` and an `Iterable[str]`, so the annotations on
`rerank_delta_ndcg` and `reciprocal_rank_fusion` admit one and mypy accepts the
call. Measured on `main` before this change:

    rerank_delta_ndcg("doc-7", "doc-7")
        -> n_input=5, top_k_overlap=5, ndcg_displacement=1.0      (one document)
    rerank_delta_ndcg("doc-7", ["doc-7"])
        -> ndcg_displacement=0.0, n_foreign=1, n_dropped=5          (D-019's alarms)
    rerank_delta_ndcg("doc-77", "doc-77")
        -> ValueError: before contains duplicate external_ids       (there are none)
    reciprocal_rank_fusion({"bm25": "doc-7", "vec": ["doc-7"]})
        -> d, doc-7, o, c, -, 7                                     (five letters fused)

Both functions are exported from `rag_kit`, so the arms import them from there.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_kit import reciprocal_rank_fusion, rerank_delta_ndcg

_SPLIT = "would be split into its characters"

# --- rerank_delta_ndcg ------------------------------------------------------


@pytest.mark.parametrize("bare", ["doc-7", b"doc-7", bytearray(b"d"), ""])
def test_rerank_delta_refuses_a_bare_before(bare: Any) -> None:
    with pytest.raises(ValueError, match=_SPLIT) as exc:
        rerank_delta_ndcg(bare, ["doc-7"])
    assert str(exc.value).startswith("before must be a sequence of ids")
    assert repr(bare) in str(exc.value)


@pytest.mark.parametrize("bare", ["doc-7", b"doc-7", ""])
def test_rerank_delta_refuses_a_bare_after(bare: Any) -> None:
    with pytest.raises(ValueError, match=_SPLIT) as exc:
        rerank_delta_ndcg(["doc-7"], bare)
    assert str(exc.value).startswith("after must be a sequence of ids")


def test_rerank_delta_message_shows_the_working_spelling() -> None:
    with pytest.raises(ValueError, match=_SPLIT) as exc:
        rerank_delta_ndcg("doc-7", "doc-7")
    assert "pass ['doc-7']" in str(exc.value)


def test_a_repeated_character_is_not_reported_as_duplicate_ids() -> None:
    # `"doc-77"` used to reach the duplicate-id check as six characters and be
    # told it held duplicates. The bare-string refusal runs first.
    with pytest.raises(ValueError, match=_SPLIT):
        rerank_delta_ndcg("doc-77", "doc-77")


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (["a", "b", "c"], ["c", "b", "a"]),
        (("a", "b", "c"), ("c", "b", "a")),
        (["a", "b", "c"], ("c", "b", "a")),
    ],
)
def test_rerank_delta_sequences_are_unchanged(before: Any, after: Any) -> None:
    got = rerank_delta_ndcg(before, after, k=2)
    assert got == rerank_delta_ndcg(list(before), list(after), k=2)
    assert got.n_input == 3


def test_rerank_delta_the_working_spelling_measures_one_document() -> None:
    got = rerank_delta_ndcg(["doc-7"], ["doc-7"])
    assert (got.n_input, got.top_k_size, got.ndcg_displacement) == (1, 1, 1.0)
    assert (got.n_foreign, got.n_dropped) == (0, 0)


def test_rerank_delta_empty_inputs_are_unchanged() -> None:
    # `[]` -> `[]` keeps its existing answer; `""` is a bare string and is
    # refused above, which is the one empty shape that changed.
    got = rerank_delta_ndcg([], [])
    assert got.n_input == 0


# --- reciprocal_rank_fusion -------------------------------------------------


@pytest.mark.parametrize("bare", ["doc-7", b"doc-7", ""])
def test_fusion_refuses_a_bare_string_ranking(bare: Any) -> None:
    with pytest.raises(ValueError, match=_SPLIT) as exc:
        reciprocal_rank_fusion({"vec": ["doc-7"], "bm25": bare})
    assert str(exc.value).startswith("rankings['bm25'] must be a sequence of ids")


def test_fusion_refuses_before_scoring_any_method() -> None:
    # The bad value is the SECOND method; the check walks every value before the
    # scoring loop starts, so there is no partial result to inspect -- the call
    # raises, and the first method's ids are never half-fused.
    with pytest.raises(ValueError, match=r"rankings\['bm25'\]"):
        reciprocal_rank_fusion({"vec": ["doc-7", "doc-8"], "bm25": "doc-7"})


@pytest.mark.parametrize(
    "rankings",
    [
        {"vec": ["a", "b"], "bm25": ["b", "a"]},
        {"vec": ("a", "b"), "bm25": ("b", "a")},
        {"vec": iter(["a", "b"]), "bm25": (x for x in ["b", "a"])},
        {"vec": [], "bm25": ["a"]},
    ],
)
def test_fusion_collections_are_unchanged(rankings: dict[str, Any]) -> None:
    fused = reciprocal_rank_fusion(rankings)
    assert {doc for doc, _, _ in fused} <= {"a", "b"}
    assert fused


def test_fusion_the_working_spelling_fuses_one_document() -> None:
    fused = reciprocal_rank_fusion({"bm25": ["doc-7"], "vec": ["doc-7"]})
    assert [(doc, ranks) for doc, _, ranks in fused] == [("doc-7", {"bm25": 1, "vec": 1})]
