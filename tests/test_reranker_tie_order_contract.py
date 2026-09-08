"""Among equal scores, a reranker preserves input order (#207).

`reciprocal_rank_fusion` settled this class twice — #40 for caller method
order, #205 for the float artifact that stopped the tie-break ever firing — and
`#180` settled the SQL half. Nobody asked it of the reranker, which is the layer
between the two.

`CohereReranker.rerank` ended with `merged.sort(key=lambda pair: pair[0],
reverse=True)`: no tie-break, and — unlike its sibling backend — an insertion
order that is not the input order. `merged` is filled per batch in
`response.results` order, which the API returns sorted by relevance. So among
equal scores the output was decided by two things that are not properties of the
documents: whatever order the API happened to return the tied rows in, and which
batch each candidate landed in, i.e. `batch_size`.

Measured on the unfixed code with a client whose score depends only on the text
and which returns tied rows in reverse input order — nothing in the API contract
forbids that:

    input order: ['D1', 'D2', 'A', 'Z']       (D1 and D2 carry identical text)
      batch_size=  1: ['A', 'D1', 'D2', 'Z']
      batch_size=  2: ['A', 'D2', 'D1', 'Z']  <- same candidates, same scores
      batch_size=100: ['A', 'D2', 'D1', 'Z']

`batch_size` is documented as a request-size knob. `rerank_rank` flows into the
citation payload, so two runs over one corpus could cite a different chunk id
for the same claim with no visible cause — the scores a consumer would inspect
are identical.

**Ties are guaranteed, not coincidental.** `documents = [c.text for c in batch]`
is all the API sees, so two candidates carrying the same `text` get the same
score by construction. Chunk overlap, a passage indexed twice, and the "union of
two SQL paths" `fusion.py` names all produce that.

The contract is stated on the `Reranker` Protocol and run here against **every**
backend in the module, so a third one inherits it rather than re-deriving it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from rag_kit import reranker as reranker_mod
from rag_kit.reranker import Candidate, CohereReranker, LexicalOverlapReranker, ScoredCandidate

QUERY = "alpha bravo charlie"

#: Distinct texts and the score a text-only scorer gives each. `dup` appears
#: under two different external ids, which is the tie.
_CORPUS: list[tuple[str, str]] = [
    ("D1", "alpha bravo charlie delta"),
    ("D2", "alpha bravo charlie delta"),
    ("A", "alpha bravo charlie"),
    ("Z", "zulu"),
    ("D3", "alpha bravo charlie delta"),
]


#: The only ranking that satisfies the contract for `_CORPUS`: `Z` is the one
#: document with no query token, and the other four tie at 1.0 — so they come
#: back in input order. Spelled out rather than derived, because a derived
#: expectation would re-implement the rule it is meant to check.
EXPECTED_ORDER = ["D1", "D2", "A", "D3", "Z"]


def _candidates() -> list[Candidate]:
    return [Candidate(external_id=eid, text=text, metadata={}) for eid, text in _CORPUS]


# --- a client that is a pure function of the document text ----------------


class _TextOnlyClient:
    """Scores by token overlap with `QUERY`; identical text scores identically.

    `tie_order` decides how the fake API orders rows it scored equally —
    `"input"`, `"reverse"` or `"rotate"`. The real API makes no promise here,
    so the contract must hold for all three.
    """

    def __init__(self, tie_order: str = "reverse") -> None:
        self.tie_order = tie_order
        self.calls = 0

    @staticmethod
    def _score(document: str) -> float:
        q = set(QUERY.split())
        return len(q & set(document.split())) / len(q)

    def rerank(
        self, *, model: str, query: str, documents: list[str], request_options: dict[str, Any]
    ) -> Any:
        self.calls += 1
        rows = [
            SimpleNamespace(index=i, relevance_score=self._score(d))
            for i, d in enumerate(documents)
        ]
        if self.tie_order == "input":
            rows.sort(key=lambda r: (-r.relevance_score, r.index))
        elif self.tie_order == "reverse":
            rows.sort(key=lambda r: (-r.relevance_score, -r.index))
        elif self.tie_order == "rotate":
            rows.sort(key=lambda r: (-r.relevance_score, (r.index + 1) % max(len(rows), 1)))
        else:  # pragma: no cover - guarded by the parametrize list
            raise AssertionError(self.tie_order)
        return SimpleNamespace(results=rows)


def _cohere(client: Any, *, batch_size: int) -> CohereReranker:
    """Build the backend without the optional `cohere` extra, as the other
    reranker tests in this repo do."""
    rr = CohereReranker.__new__(CohereReranker)
    rr.client = client
    rr.model = "rerank-english-v3.0"
    rr.batch_size = batch_size
    rr.timeout_s = 30.0
    return rr


#: (label, callable taking the candidate list and returning the reranked list).
#: Every backend in `rag_kit.reranker` must appear here — see the discovery
#: test below, which fails if one is added and not wired in.
_BACKENDS: list[tuple[str, Callable[[list[Candidate]], list[ScoredCandidate]]]] = [
    ("LexicalOverlapReranker", lambda c: LexicalOverlapReranker().rerank(QUERY, c)),
    (
        "CohereReranker",
        lambda c: _cohere(_TextOnlyClient(), batch_size=2).rerank(QUERY, c),
    ),
]


def _tied_groups(rows: list[ScoredCandidate]) -> list[list[str]]:
    groups: dict[float, list[str]] = {}
    for row in rows:
        groups.setdefault(row.rerank_score, []).append(row.external_id)
    return [ids for ids in groups.values() if len(ids) > 1]


# --- the contract ---------------------------------------------------------


@pytest.mark.parametrize(("label", "rerank"), _BACKENDS, ids=[b[0] for b in _BACKENDS])
def test_equal_scores_keep_their_input_order(
    label: str, rerank: Callable[[list[Candidate]], list[ScoredCandidate]]
) -> None:
    candidates = _candidates()
    rows = rerank(candidates)

    tied = _tied_groups(rows)
    # Anti-vacuous, and it is the arm that matters: a corpus with no tie
    # satisfies every assertion below *against the unfixed code*, because the
    # defect only manifests where two documents score equally.
    assert tied, f"{label}: the corpus produced no tie, so this proves nothing"
    assert any(len(g) >= 3 for g in tied), f"{label}: need a group larger than a pair"

    input_positions = {c.external_id: i for i, c in enumerate(candidates)}
    for group in tied:
        assert group == sorted(group, key=lambda eid: input_positions[eid]), (
            f"{label}: tied group {group} is not in input order"
        )


def test_every_backend_in_the_module_is_covered() -> None:
    """Discovered from the module, not listed by hand.

    A fourth backend added without a row here would otherwise inherit nothing:
    the contract lives on the Protocol, and a Protocol does not execute.
    """
    implementations = {
        name
        for name, obj in vars(reranker_mod).items()
        if inspect.isclass(obj)
        and obj.__module__ == reranker_mod.__name__
        and hasattr(obj, "rerank")
        and name != "Reranker"
    }
    assert implementations, "discovery found no backends at all"
    assert implementations == {label for label, _ in _BACKENDS}, (
        f"backends not under the tie contract: {implementations - {b[0] for b in _BACKENDS}}"
    )


# --- the two axes the Cohere path was actually sensitive to ---------------


@pytest.mark.parametrize("tie_order", ["input", "reverse", "rotate"])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 100])
def test_the_cohere_ranking_is_independent_of_batch_size_and_api_tie_order(
    tie_order: str, batch_size: int
) -> None:
    """Swept, not asserted on one hand-built pair.

    `batch_size` decides which candidates share a request, and only candidates
    sharing a request can be reordered relative to each other by the API's tie
    choice — so a single batch size can agree with the fixed answer by accident.
    """
    rows = _cohere(_TextOnlyClient(tie_order), batch_size=batch_size).rerank(QUERY, _candidates())
    assert [r.external_id for r in rows] == EXPECTED_ORDER
    assert [r.rerank_rank for r in rows] == [1, 2, 3, 4, 5]


def test_the_batching_really_did_split_the_tied_group() -> None:
    """The sweep above is only meaningful if small batch sizes split the tie.

    If every candidate landed in one request at every batch size, the parametrize
    would be seven copies of one case.
    """
    client = _TextOnlyClient()
    _cohere(client, batch_size=2).rerank(QUERY, _candidates())
    assert client.calls == 3, "5 candidates at batch_size=2 must be three requests"

    single = _TextOnlyClient()
    _cohere(single, batch_size=100).rerank(QUERY, _candidates())
    assert single.calls == 1


def test_scores_and_non_tied_order_are_unchanged() -> None:
    """The tie-break must not move anything that was not tied."""
    rows = _cohere(_TextOnlyClient(), batch_size=3).rerank(QUERY, _candidates())
    assert [r.rerank_score for r in rows] == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert rows[-1].external_id == "Z"
    # And the reranker still re-sorts: `Z` was fourth in the input.
    assert [c.external_id for c in _candidates()][3] == "Z"


# --- the neighbours, built and run ----------------------------------------


def _sort_by_external_id(pairs: list[tuple[float, int, Candidate]]) -> list[str]:
    """Fusion's rule — tie-break on the document id — applied here.

    Deterministic, and wrong for this seam: the input is already a ranking
    (`Retriever.search`'s fused list) and a lexicographic rule discards it. It
    also disagrees with `LexicalOverlapReranker`, so the two backends would
    answer differently for the same tie.
    """
    return [
        c.external_id for _s, _p, c in sorted(pairs, key=lambda row: (-row[0], row[2].external_id))
    ]


def test_the_doc_id_tie_break_neighbour_reorders_the_tied_group() -> None:
    """It is deterministic, which is why it is tempting, and it is not this rule.

    The two backends score differently — `LexicalOverlapReranker` applies a
    length penalty, so `A` is not even tied there — so the comparison that
    means something is not "same final order" but "same treatment of a tied
    group". Both keep one in input order; the doc-id rule re-sorts it, throwing
    away the fused retrieval ranking the input carries.
    """
    candidates = _candidates()
    scores = {"D1": 1.0, "D2": 1.0, "D3": 1.0, "A": 1.0, "Z": 0.0}
    pairs = [(scores[c.external_id], i, c) for i, c in enumerate(candidates)]

    by_id = _sort_by_external_id(pairs)
    assert by_id == ["A", "D1", "D2", "D3", "Z"]
    assert by_id[:4] != EXPECTED_ORDER[:4], "the doc-id rule moves the tied group"

    shipped = [
        r.external_id for r in _cohere(_TextOnlyClient(), batch_size=2).rerank(QUERY, candidates)
    ]
    assert shipped == EXPECTED_ORDER

    # And the sibling backend keeps *its* tied group in input order too, which
    # is the shared property — not a shared final order.
    lexical = LexicalOverlapReranker().rerank(QUERY, candidates)
    assert _tied_groups(lexical) == [["D1", "D2", "D3"]]


def test_the_reverse_true_neighbour_ranks_the_last_tied_candidate_first() -> None:
    """`sort(key=lambda row: (row[0], row[1]), reverse=True)` reads as the same
    change and reverses the tie-break along with the score, which is the exact
    instability being fixed — restated with different arithmetic."""
    candidates = _candidates()
    scores = {"D1": 1.0, "D2": 1.0, "D3": 1.0, "A": 1.0, "Z": 0.0}
    pairs = [(scores[c.external_id], i, c) for i, c in enumerate(candidates)]

    reversed_key = [
        c.external_id for _s, _p, c in sorted(pairs, key=lambda row: (row[0], row[1]), reverse=True)
    ]
    assert reversed_key == ["D3", "A", "D2", "D1", "Z"]
    assert reversed_key != EXPECTED_ORDER
    # It is the tied group reversed, which is the instability with a sign on it.
    assert reversed_key[:4] == EXPECTED_ORDER[:4][::-1]


def test_the_sort_each_batch_neighbour_is_still_batch_size_dependent() -> None:
    """ "Sort each batch before merging" reads as making the merge deterministic.

    It normalizes *within* a request and leaves the cross-request order exactly
    where it was, so the answer still moves with `batch_size`.
    """

    def batched_presort(candidates: list[Candidate], *, batch_size: int) -> list[str]:
        scores = {"D1": 1.0, "D2": 1.0, "D3": 1.0, "A": 1.0, "Z": 0.0}
        merged: list[tuple[float, Candidate]] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            rows = [(scores[c.external_id], c) for c in batch]
            # the API's own arbitrary tie order, normalized per batch
            rows.sort(key=lambda pair: (-pair[0], pair[1].external_id))
            merged.extend(rows)
        merged.sort(key=lambda pair: pair[0], reverse=True)
        return [c.external_id for _s, c in merged]

    candidates = _candidates()
    assert batched_presort(candidates, batch_size=2) != batched_presort(candidates, batch_size=5)
