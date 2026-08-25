"""The Cohere response must be a permutation of the batch it answers (#186).

`CohereReranker.rerank` reads two fields off each response row.
`relevance_score` was guarded, with a comment stating why: *"The Cohere API is
an external, uncontrolled source: a malformed or erroring response can hand back
a non-finite relevance_score."* `index` — the other field, off the same row, two
lines below — was fed straight into `batch[r.index]`, and it is the field that
decides *which document a score is attached to*.

The `Reranker` Protocol says `rerank` returns "candidates re-sorted by
relevance", i.e. a permutation of the input. Measured against three candidates,
eight response shapes broke that and five broke it silently:

    index = -1      ['D2', 'D1', 'D2']   D2 twice, D0 gone
    index = -3      ['D0', 'D1', 'D2']   looks perfect; the top score belongs
                                         to a document three positions away
    duplicate index ['D0', 'D0', 'D2']   D0 twice, D1 gone
    fewer results   ['D0']               two candidates silently dropped
    empty results   []                   the whole retrieval evaporates

Each row below is fixtured so that exactly one of the three checks is the one
that fires — a row that is short *and* has a bad index would pass this file
while proving only that the count check works.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_kit.reranker import Candidate, CohereReranker


class _Row:
    def __init__(self, index: Any, score: float) -> None:
        self.index = index
        self.relevance_score = score


class _Response:
    def __init__(self, rows: list[_Row]) -> None:
        self.results = rows


class _Client:
    """Stub whose `rerank` returns whatever `rows_for` builds for the batch."""

    def __init__(self, rows_for: Any) -> None:
        self.rows_for = rows_for
        self.calls = 0

    def rerank(self, *, model: str, query: str, documents: list[str], request_options: dict) -> Any:
        self.calls += 1
        return _Response(self.rows_for(documents))


def _candidates(n: int) -> list[Candidate]:
    return [Candidate(external_id=f"D{i}", text=f"text-{i}", metadata={}) for i in range(n)]


def _reranker(client: _Client, *, batch_size: int = 100) -> CohereReranker:
    """Build without `__init__` so the test needs neither the `cohere` extra nor
    an API key — the same construction the existing `test_reranker.py` uses."""
    rr = CohereReranker.__new__(CohereReranker)
    rr.client = client  # type: ignore[attr-defined]
    rr.model = "rerank-test"  # type: ignore[attr-defined]
    rr.batch_size = batch_size  # type: ignore[attr-defined]
    rr.timeout_s = 5.0  # type: ignore[attr-defined]
    return rr


def _wellformed(documents: list[str]) -> list[_Row]:
    """Descending scores, every index present exactly once."""
    return [_Row(i, 1.0 - 0.1 * i) for i in range(len(documents))]


# (label, rows_for, expected message fragment). Every row is full-length except
# the three that are explicitly about the count, so each case exercises the
# check it is named for.
_BAD: list[tuple[str, Any, str]] = [
    ("index = -1", lambda d: [_Row(-1, 0.9), _Row(1, 0.5), _Row(2, 0.1)], "index -1 for a batch"),
    (
        "index = -3, which silently wraps to 0",
        lambda d: [_Row(-3, 0.9), _Row(1, 0.5), _Row(2, 0.1)],
        "index -3 for a batch",
    ),
    (
        "index past the end",
        lambda d: [_Row(7, 0.9), _Row(1, 0.5), _Row(2, 0.1)],
        "index 7 for a batch",
    ),
    (
        "duplicate index",
        lambda d: [_Row(0, 0.9), _Row(0, 0.5), _Row(2, 0.1)],
        "more than once in one batch",
    ),
    ("float index", lambda d: [_Row(1.0, 0.9), _Row(1, 0.5), _Row(2, 0.1)], "non-integer index"),
    ("str index", lambda d: [_Row("1", 0.9), _Row(1, 0.5), _Row(2, 0.1)], "non-integer index"),
    ("None index", lambda d: [_Row(None, 0.9), _Row(1, 0.5), _Row(2, 0.1)], "non-integer index"),
    (
        "bool index (True indexes as 1)",
        lambda d: [_Row(True, 0.9), _Row(0, 0.5), _Row(2, 0.1)],
        "non-integer index",
    ),
    ("fewer results than the batch", lambda d: [_Row(0, 0.9)], "1 result(s) for a batch of 3"),
    ("empty results", lambda d: [], "0 result(s) for a batch of 3"),
    (
        "more results than the batch",
        lambda d: [_Row(i % len(d), 0.5) for i in range(6)],
        "6 result(s) for a batch of 3",
    ),
]


def test_the_table_is_not_vacuous() -> None:
    assert len(_BAD) >= 10
    # All three checks must be represented, or a row could be silently
    # rewritten to trip a different one and the coverage claim would rot.
    fragments = {frag for _, _, frag in _BAD}
    assert any("non-integer" in f for f in fragments)
    assert any("for a batch" in f and "non-integer" not in f for f in fragments)
    assert any("more than once" in f for f in fragments)
    assert any("result(s) for a batch" in f for f in fragments)


def test_a_wellformed_response_is_unchanged() -> None:
    """The control. Without this, a guard that rejected everything would pass
    every case above."""
    rr = _reranker(_Client(_wellformed))
    out = rr.rerank("query", _candidates(3))
    assert [r.external_id for r in out] == ["D0", "D1", "D2"]
    assert [r.rerank_rank for r in out] == [1, 2, 3]
    assert [round(r.rerank_score, 2) for r in out] == [1.0, 0.9, 0.8]


@pytest.mark.parametrize(("label", "rows_for", "fragment"), _BAD, ids=[r[0] for r in _BAD])
def test_malformed_responses_raise_valueerror(label: str, rows_for: Any, fragment: str) -> None:
    """`ValueError`, not `IndexError`/`TypeError`.

    Every other guard in this module — `k`, `length_penalty`, `batch_size`,
    `timeout_s`, `relevance_score` — raises `ValueError` naming the field, so a
    caller catching the documented failure mode has to catch these too.
    """
    rr = _reranker(_Client(rows_for))
    with pytest.raises(ValueError, match="Cohere rerank returned") as exc:
        rr.rerank("query", _candidates(3))
    assert fragment in str(exc.value), str(exc.value)


def test_negative_index_is_caught_rather_than_silently_resolving() -> None:
    """The row that motivates a range check rather than an upper-bound check.

    `batch[-3]` on a 3-element batch is `batch[0]` — no exception, three
    candidates in and three out, in order. Only the score attribution is wrong,
    which is invisible at the output shape.
    """
    rr = _reranker(_Client(lambda d: [_Row(-3, 0.9), _Row(1, 0.5), _Row(2, 0.1)]))
    with pytest.raises(ValueError, match="silently resolves to a different document"):
        rr.rerank("query", _candidates(3))


def test_a_short_response_does_not_silently_drop_candidates() -> None:
    """Same harm the `batch_size` guard already names on the operator-supplied
    road: "every candidate silently dropped ... no error"."""
    rr = _reranker(_Client(lambda d: []))
    with pytest.raises(ValueError, match="silently drops candidates"):
        rr.rerank("query", _candidates(3))


def test_the_contract_is_enforced_per_batch_not_per_call() -> None:
    """`rerank` chunks by `batch_size`; each request is independently a
    permutation of its own slice. A response that is valid for the whole call
    but wrong for its batch must still be rejected."""
    seen: list[int] = []

    def rows_for(documents: list[str]) -> list[_Row]:
        seen.append(len(documents))
        # Always answers as if the batch were the full 3 candidates, which is
        # correct for the first batch of 2 only by accident and wrong for it too.
        return [_Row(i, 1.0 - 0.1 * i) for i in range(3)]

    rr = _reranker(_Client(rows_for), batch_size=2)
    with pytest.raises(ValueError, match="3 result\\(s\\) for a batch of 2"):
        rr.rerank("query", _candidates(3))
    assert seen == [2], "should fail on the first batch, before issuing the second request"


def test_a_valid_multi_batch_call_still_works() -> None:
    """Control for the per-batch check: correct per-slice responses merge fine."""
    rr = _reranker(_Client(_wellformed), batch_size=2)
    out = rr.rerank("query", _candidates(5))
    assert sorted(r.external_id for r in out) == ["D0", "D1", "D2", "D3", "D4"]
    assert [r.rerank_rank for r in out] == [1, 2, 3, 4, 5]
