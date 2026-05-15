"""HashEmbedder + Embedder protocol tests. No database required."""

from __future__ import annotations

import math

import pytest

from rag_kit import EMBEDDING_DIM, Embedder, HashEmbedder


def test_default_dim_matches_schema():
    e = HashEmbedder()
    v = e.embed("hello")
    assert len(v) == EMBEDDING_DIM == 64


def test_deterministic_same_input_same_vector():
    e = HashEmbedder()
    assert e.embed("the quick brown fox") == e.embed("the quick brown fox")


def test_distinct_inputs_distinct_vectors():
    e = HashEmbedder()
    assert e.embed("alpha") != e.embed("beta")


def test_unit_length():
    e = HashEmbedder()
    v = e.embed("any non-empty string")
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_floats_in_unit_range():
    e = HashEmbedder()
    v = e.embed("range check")
    assert all(-1.0 <= x <= 1.0 for x in v)


def test_custom_dim():
    e = HashEmbedder(dim=128)
    assert len(e.embed("x")) == 128


def test_invalid_dim_rejected():
    with pytest.raises(ValueError, match="positive"):
        HashEmbedder(dim=0)
    with pytest.raises(ValueError, match="positive"):
        HashEmbedder(dim=-32)
    with pytest.raises(ValueError, match="multiple of 8"):
        HashEmbedder(dim=10)


def test_hash_embedder_satisfies_protocol():
    e: Embedder = HashEmbedder()
    assert isinstance(e, Embedder)
