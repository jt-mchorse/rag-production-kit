"""Reciprocal Rank Fusion math. No database required."""

from __future__ import annotations

import math

import pytest

from rag_kit import reciprocal_rank_fusion


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion({}) == []


def test_single_method_passes_through_order():
    fused = reciprocal_rank_fusion({"a": ["d1", "d2", "d3"]})
    assert [doc for doc, _, _ in fused] == ["d1", "d2", "d3"]


def test_doc_in_both_methods_outranks_doc_in_one():
    fused = reciprocal_rank_fusion(
        {
            "lexical": ["d1", "d2", "d3"],
            "dense": ["d2", "d4", "d5"],
        }
    )
    ranking = {doc: i for i, (doc, _, _) in enumerate(fused)}
    # d2 appears in both rankings, should fuse above d1 (only in lexical at rank 1)?
    # d1 rank-1 in lexical only: score = 1/61
    # d2 rank-2 in lexical + rank-1 in dense: score = 1/62 + 1/61
    # So d2 ranks above d1.
    assert ranking["d2"] < ranking["d1"]


def test_per_method_ranks_returned():
    fused = reciprocal_rank_fusion(
        {
            "lexical": ["d1", "d2"],
            "dense": ["d2", "d3"],
        }
    )
    by_doc = {doc: ranks for doc, _, ranks in fused}
    assert by_doc["d1"] == {"lexical": 1}
    assert by_doc["d2"] == {"lexical": 2, "dense": 1}
    assert by_doc["d3"] == {"dense": 2}


def test_score_formula_matches_paper():
    # Cormack et al.'s formula: score = sum over methods of 1 / (k + rank)
    fused = reciprocal_rank_fusion(
        {"lexical": ["d1"], "dense": ["d1"]},
        k=60,
    )
    assert math.isclose(fused[0][1], 1 / 61 + 1 / 61, abs_tol=1e-12)


def test_invalid_k_rejected():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({"a": ["d1"]}, k=0)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({"a": ["d1"]}, k=-1)


def test_higher_k_smooths_contribution():
    # The marginal score for a rank-1 hit shrinks as k grows.
    small_k = reciprocal_rank_fusion({"a": ["d1"]}, k=10)[0][1]
    large_k = reciprocal_rank_fusion({"a": ["d1"]}, k=1000)[0][1]
    assert small_k > large_k


def test_descending_order():
    fused = reciprocal_rank_fusion(
        {
            "lexical": ["d1", "d2", "d3", "d4"],
            "dense": ["d4", "d3", "d2", "d1"],
        }
    )
    scores = [s for _, s, _ in fused]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------------
# #40 — extend sign-only k > 0 to isinstance(int) + reject bool + positive
# ----------------------------------------------------------------------


class TestFusionKValidation:
    @pytest.mark.parametrize("bad", [0, -1, -60])
    def test_rejects_non_positive_int(self, bad: int) -> None:
        with pytest.raises(ValueError, match=r"k must be a positive integer"):
            reciprocal_rank_fusion({"a": ["d1"]}, k=bad)

    @pytest.mark.parametrize("bad", [True, False, 0.5, 1.5, 60.0, "60", None])
    def test_rejects_non_int(self, bad) -> None:
        # bool is an int subclass in Python — must be rejected explicitly because
        # True=1 silently shifts the RRF constant from 60 to 1, distorting scores.
        # Float 0.5 silently changes the score curve, 60.0 looks fine but is
        # contractually wrong (the constant is an integer per Cormack et al. 2009).
        with pytest.raises(ValueError, match=r"k must be a positive integer"):
            reciprocal_rank_fusion({"a": ["d1"]}, k=bad)  # type: ignore[arg-type]

    def test_accepts_one_minimum(self) -> None:
        out = reciprocal_rank_fusion({"a": ["d1"]}, k=1)
        assert out
        assert out[0][0] == "d1"
