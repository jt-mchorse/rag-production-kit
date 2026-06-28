"""Tests for the reranking layer.

Three things to test hermetically:
1. The Reranker Protocol is satisfied by both shipped backends (LexicalOverlap +
   a stub Cohere backend driven by a fake client).
2. LexicalOverlapReranker produces deterministic, query-relevant ordering.
3. rerank_delta_ndcg gives sensible numbers on textbook examples.
"""

from __future__ import annotations

import math

import pytest

from rag_kit.reranker import (
    Candidate,
    LexicalOverlapReranker,
    Reranker,
    ScoredCandidate,
    rerank_delta_ndcg,
)


def _make_candidates(items: list[tuple[str, str]]) -> list[Candidate]:
    return [Candidate(external_id=ext_id, text=text, metadata={}) for ext_id, text in items]


# ----------------------------------------------------------------------
# LexicalOverlapReranker
# ----------------------------------------------------------------------


def test_lexical_overlap_promotes_query_matching_candidates():
    rr = LexicalOverlapReranker(length_penalty=0.0)
    candidates = _make_candidates(
        [
            ("a", "Pure cats and weather."),
            ("b", "Postgres tuning for production workloads."),
            ("c", "Cohere reranker performance benchmarks."),
        ]
    )
    out = rr.rerank("postgres tuning workload", candidates)
    assert [r.external_id for r in out[:1]] == ["b"]


def test_lexical_overlap_returns_proper_scoredcandidate_shape():
    rr = LexicalOverlapReranker(length_penalty=0.0)
    candidates = _make_candidates([("x", "hello world")])
    out = rr.rerank("hello", candidates)
    assert len(out) == 1
    assert isinstance(out[0], ScoredCandidate)
    assert out[0].external_id == "x"
    assert out[0].rerank_rank == 1
    assert out[0].rerank_score == pytest.approx(1.0)


def test_lexical_overlap_is_deterministic():
    rr = LexicalOverlapReranker(length_penalty=0.0)
    candidates = _make_candidates([("a", "one two"), ("b", "two three"), ("c", "three four")])
    a = rr.rerank("two three four", candidates)
    b = rr.rerank("two three four", candidates)
    assert [r.external_id for r in a] == [r.external_id for r in b]
    assert [r.rerank_score for r in a] == [r.rerank_score for r in b]


def test_lexical_overlap_with_no_query_tokens_preserves_input_order():
    rr = LexicalOverlapReranker()
    candidates = _make_candidates([("a", "x"), ("b", "y"), ("c", "z")])
    out = rr.rerank("???!!!", candidates)
    assert [r.external_id for r in out] == ["a", "b", "c"]
    for r in out:
        assert r.rerank_score == 0.0


def test_lexical_overlap_length_penalty_breaks_ties_toward_shorter():
    rr = LexicalOverlapReranker(length_penalty=0.001)
    candidates = _make_candidates(
        [
            ("long", "matching " + "x " * 100),  # same overlap as short
            ("short", "matching"),
        ]
    )
    out = rr.rerank("matching", candidates)
    # Shorter chunk wins on the length-penalty tiebreak.
    assert [r.external_id for r in out] == ["short", "long"]


def test_lexical_overlap_length_penalty_does_not_override_relevance():
    # #90: the length penalty must only break ties, never override a genuine
    # overlap difference. Pre-fix `penalty = length_penalty * len(text)` was
    # unbounded in char count, so a more-relevant LONG chunk (overlap 0.5) was
    # demoted below a less-relevant SHORT chunk (overlap 0.25) because the long
    # chunk's raw-length penalty dwarfed its higher overlap.
    rr = LexicalOverlapReranker(length_penalty=0.001)
    candidates = _make_candidates(
        [
            ("more_relevant_long", "refund policy " + "x " * 233),  # overlap 2/4, ~480 chars
            ("less_relevant_short", "refund yesterday only"),  # overlap 1/4, 21 chars
        ]
    )
    out = rr.rerank("refund policy window days", candidates)
    # The strictly-higher-overlap chunk must win despite being far longer.
    assert [r.external_id for r in out] == ["more_relevant_long", "less_relevant_short"]
    assert out[0].rerank_score > out[1].rerank_score


def test_lexical_overlap_penalty_stays_below_overlap_quantum():
    # The bounded length factor keeps the penalty in [0, length_penalty), which
    # is strictly below the smallest overlap step (1/len(q_tokens)). Even an
    # enormous chunk can't push the penalty past the coefficient.
    rr = LexicalOverlapReranker(length_penalty=0.001)
    out = rr.rerank("alpha", _make_candidates([("huge", "alpha " + "z " * 100000)]))
    # overlap is 1.0; penalty < 0.001, so the score stays just under 1.0.
    assert 0.999 < out[0].rerank_score < 1.0


def test_lexical_overlap_rejects_negative_length_penalty():
    with pytest.raises(ValueError, match=r"finite number >= 0.0"):
        LexicalOverlapReranker(length_penalty=-0.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_lexical_overlap_rejects_non_finite_length_penalty(bad):
    # Sign-only `length_penalty < 0` let NaN/Inf through (both comparisons are
    # False), poisoning every score → all-NaN sorts as a no-op so the relevant
    # chunk is silently not surfaced first. Mirror the repo's finiteness sweep
    # (fusion.k #38, telemetry.ModelPrice, CostRecord latency).
    with pytest.raises(ValueError, match=r"finite number >= 0.0"):
        LexicalOverlapReranker(length_penalty=bad)


def test_lexical_overlap_finite_penalty_still_ranks_relevant_first():
    # The clean path is unaffected: a finite penalty produces a sane, real
    # ordering (the regression a non-finite penalty silently caused).
    rr = LexicalOverlapReranker(length_penalty=0.001)
    cands = _make_candidates(
        [
            ("irrelevant", "the sky is blue and very far away today"),
            ("relevant", "python async pipelines tutorial"),
        ]
    )
    out = rr.rerank("python async pipelines", cands)
    assert out[0].external_id == "relevant"
    assert all(math.isfinite(c.rerank_score) for c in out)


def test_lexical_overlap_rejects_empty_query():
    rr = LexicalOverlapReranker()
    with pytest.raises(ValueError, match="non-empty"):
        rr.rerank("", _make_candidates([("a", "x")]))


def test_lexical_overlap_satisfies_reranker_protocol():
    # Structural typing: any object with .rerank(...) matching the signature
    # is accepted as Reranker. This compiles + runs only if the signature lines up.
    rr: Reranker = LexicalOverlapReranker()
    out = rr.rerank("a b c", _make_candidates([("x", "a b")]))
    assert out[0].external_id == "x"


# ----------------------------------------------------------------------
# CohereReranker (stubbed client)
# ----------------------------------------------------------------------


class _FakeCohereResultRow:
    def __init__(self, index: int, score: float) -> None:
        self.index = index
        self.relevance_score = score


class _FakeCohereResponse:
    def __init__(self, rows: list[_FakeCohereResultRow]) -> None:
        self.results = rows


class _FakeCohereClient:
    def __init__(self, score_for_text: dict[str, float]) -> None:
        self.scores = score_for_text
        self.calls: list[dict] = []

    def rerank(self, *, model: str, query: str, documents: list[str], request_options: dict):
        self.calls.append(
            {
                "model": model,
                "query": query,
                "documents": list(documents),
                "options": request_options,
            }
        )
        rows = [_FakeCohereResultRow(i, self.scores.get(d, 0.0)) for i, d in enumerate(documents)]
        return _FakeCohereResponse(rows)


def test_cohere_reranker_uses_relevance_scores_to_reorder(monkeypatch):
    from rag_kit import reranker as reranker_mod

    fake_client = _FakeCohereClient({"text-A": 0.1, "text-B": 0.9, "text-C": 0.5})
    rr = reranker_mod.CohereReranker.__new__(reranker_mod.CohereReranker)
    rr.client = fake_client  # type: ignore[attr-defined]
    rr.model = "rerank-test"
    rr.batch_size = 100
    rr.timeout_s = 5.0

    candidates = _make_candidates([("A", "text-A"), ("B", "text-B"), ("C", "text-C")])
    out = rr.rerank("query", candidates)

    assert [r.external_id for r in out] == ["B", "C", "A"]
    assert [r.rerank_rank for r in out] == [1, 2, 3]
    assert fake_client.calls[0]["model"] == "rerank-test"
    assert fake_client.calls[0]["options"] == {"timeout_in_seconds": 5.0}


def test_cohere_reranker_batches_large_inputs():
    from rag_kit import reranker as reranker_mod

    fake_client = _FakeCohereClient({f"t{i}": 1.0 - i / 10.0 for i in range(10)})
    rr = reranker_mod.CohereReranker.__new__(reranker_mod.CohereReranker)
    rr.client = fake_client
    rr.model = "rerank-test"
    rr.batch_size = 3
    rr.timeout_s = 5.0

    candidates = _make_candidates([(f"id{i}", f"t{i}") for i in range(10)])
    out = rr.rerank("query", candidates)

    # 10 documents, batch_size=3 → ceil(10/3) = 4 batches.
    assert len(fake_client.calls) == 4
    # Highest score (t0) should be first.
    assert out[0].external_id == "id0"
    assert len(out) == 10


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_cohere_reranker_rejects_non_finite_relevance_score(bad):
    # An external/uncontrolled API can return a non-finite relevance_score. Left
    # unguarded it poisons generator._top_score's max() and the refusal gate
    # `top < threshold` (False for NaN → answers when it should refuse, #86).
    # Sibling of the operator-threshold guard #78 on the same gate.
    from rag_kit import reranker as reranker_mod

    fake_client = _FakeCohereClient({"text-A": 0.1, "text-B": bad})
    rr = reranker_mod.CohereReranker.__new__(reranker_mod.CohereReranker)
    rr.client = fake_client
    rr.model = "rerank-test"
    rr.batch_size = 100
    rr.timeout_s = 5.0

    candidates = _make_candidates([("A", "text-A"), ("B", "text-B")])
    with pytest.raises(ValueError, match="non-finite relevance_score"):
        rr.rerank("query", candidates)


def test_cohere_reranker_accepts_finite_including_negative_score():
    # The guard must not reject a *present* finite score, including a legitimately
    # negative one (Cohere scores aren't constrained to [0, 1] across all models).
    from rag_kit import reranker as reranker_mod

    fake_client = _FakeCohereClient({"text-A": -0.5, "text-B": 0.0, "text-C": 0.9})
    rr = reranker_mod.CohereReranker.__new__(reranker_mod.CohereReranker)
    rr.client = fake_client
    rr.model = "rerank-test"
    rr.batch_size = 100
    rr.timeout_s = 5.0

    candidates = _make_candidates([("A", "text-A"), ("B", "text-B"), ("C", "text-C")])
    out = rr.rerank("query", candidates)
    assert [r.external_id for r in out] == ["C", "B", "A"]
    assert out[-1].rerank_score == -0.5


def test_cohere_reranker_empty_input_returns_empty_no_calls():
    from rag_kit import reranker as reranker_mod

    fake_client = _FakeCohereClient({})
    rr = reranker_mod.CohereReranker.__new__(reranker_mod.CohereReranker)
    rr.client = fake_client
    rr.model = "x"
    rr.batch_size = 100
    rr.timeout_s = 5.0
    out = rr.rerank("query", [])
    assert out == []
    assert fake_client.calls == []


# ----------------------------------------------------------------------
# rerank_delta_ndcg
# ----------------------------------------------------------------------


def test_rerank_delta_no_change_is_one():
    delta = rerank_delta_ndcg(["a", "b", "c", "d"], ["a", "b", "c", "d"], k=3)
    assert delta.ndcg_displacement == pytest.approx(1.0)
    assert delta.top_k_overlap == 3
    assert delta.top_k_size == 3


def test_rerank_delta_full_reverse_is_low():
    delta = rerank_delta_ndcg(["a", "b", "c", "d"], ["d", "c", "b", "a"], k=4)
    assert delta.ndcg_displacement < 1.0


def test_rerank_delta_top_k_overlap_under_total_reorder():
    delta = rerank_delta_ndcg(
        ["a", "b", "c", "d", "e"],
        ["c", "a", "b", "e", "d"],
        k=3,
    )
    # before top-3 = {a, b, c}, after top-3 = {c, a, b} → overlap 3.
    assert delta.top_k_overlap == 3


def test_rerank_delta_disjoint_top_k():
    delta = rerank_delta_ndcg(
        ["a", "b", "c", "d", "e"],
        ["x", "y", "z", "a", "b"],
        k=3,
    )
    assert delta.top_k_overlap == 0


def test_rerank_delta_handles_empty():
    delta = rerank_delta_ndcg([], [], k=5)
    assert delta.n_input == 0
    assert delta.ndcg_displacement == 1.0


def test_rerank_delta_rejects_zero_k():
    with pytest.raises(ValueError, match="positive"):
        rerank_delta_ndcg(["a"], ["a"], k=0)


# ----------------------------------------------------------------------
# #40 — rerank_delta_ndcg k validation extended to isinstance(int)+positive
# ----------------------------------------------------------------------


class TestRerankDeltaNdcgKValidation:
    @pytest.mark.parametrize("bad", [0, -1, -5])
    def test_rejects_non_positive(self, bad: int) -> None:
        from rag_kit.reranker import rerank_delta_ndcg

        with pytest.raises(ValueError, match=r"k must be a positive integer"):
            rerank_delta_ndcg(["a", "b"], ["b", "a"], k=bad)

    @pytest.mark.parametrize("bad", [True, False, 0.5, 1.5, "5", None])
    def test_rejects_non_int(self, bad) -> None:
        from rag_kit.reranker import rerank_delta_ndcg

        with pytest.raises(ValueError, match=r"k must be a positive integer"):
            rerank_delta_ndcg(["a", "b"], ["b", "a"], k=bad)  # type: ignore[arg-type]

    def test_accepts_one_minimum(self) -> None:
        from rag_kit.reranker import rerank_delta_ndcg

        delta = rerank_delta_ndcg(["a", "b"], ["b", "a"], k=1)
        # Just confirm the call succeeded with the minimum valid k.
        assert delta.top_k_size == 1


# ----------------------------------------------------------------------
# #98 — rerank_delta_ndcg rejects duplicate external_ids (the silently-wrong
# telemetry path: a repeated id pushed ndcg_displacement past its [0, 1] range)
# ----------------------------------------------------------------------


class TestRerankDeltaNdcgDuplicateIds:
    def test_rejects_duplicate_in_after(self) -> None:
        # Pre-fix this returned ~1.34 — the documented ceiling is 1.0.
        with pytest.raises(ValueError, match=r"after contains duplicate"):
            rerank_delta_ndcg(["a", "b", "c"], ["a", "a", "a"], k=3)

    def test_rejects_duplicate_in_before(self) -> None:
        with pytest.raises(ValueError, match=r"before contains duplicate"):
            rerank_delta_ndcg(["a", "a", "b"], ["a", "b", "c"], k=3)

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            (["a", "b", "c", "d"], ["a", "b", "c", "d"]),
            (["a", "b", "c", "d"], ["d", "c", "b", "a"]),
            (["a", "b", "c"], ["c", "a", "b"]),
            (["a", "b", "c", "d", "e"], ["e", "a", "d", "b", "c"]),
            (["a", "b", "c"], ["x", "y", "z"]),  # disjoint, still distinct
        ],
    )
    def test_distinct_id_inputs_stay_within_unit_interval(
        self, before: list[str], after: list[str]
    ) -> None:
        # The [0, 1] invariant the docstring/field comment promise, locked for
        # any distinct-id permutation (and the disjoint case).
        delta = rerank_delta_ndcg(before, after, k=len(before))
        assert 0.0 <= delta.ndcg_displacement <= 1.0
