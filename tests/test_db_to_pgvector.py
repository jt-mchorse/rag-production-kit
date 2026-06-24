"""Tests for ``rag_kit.db.to_pgvector`` (#82).

``to_pgvector`` is the single chokepoint both the indexer write-path
(``Indexer.add_documents``) and the retriever query-path
(``Retriever._hybrid_search``) funnel BYO-``Embedder`` output through before
it reaches pgvector. A non-finite component (``NaN`` / ``±Inf``) from a
normalization divide-by-zero, an ``Inf`` overflow, or a NaN-poisoned model
output must be rejected loudly at this seam — unguarded it reaches pgvector
as the bare token ``nan``/``inf`` and either errors opaquely far from the
embedder or silently corrupts dense-channel ordering. Same seam-validation
posture as llm-cost-optimizer ``_validate_embedding`` (#88).
"""

from __future__ import annotations

import math

import pytest

from rag_kit.db import to_pgvector


@pytest.mark.parametrize(
    "bad",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "inf", "-inf"],
)
def test_to_pgvector_rejects_non_finite_component(bad: float):
    # A non-finite component anywhere in the vector must fail loud, naming
    # the index, rather than emitting `[..,nan,..]` for pgvector to choke on.
    with pytest.raises(ValueError, match="index 1 must be finite"):
        to_pgvector([0.1, bad, 0.3])


def test_to_pgvector_error_names_first_offending_index():
    # Index 0 is the first to violate; the message should point there.
    with pytest.raises(ValueError, match="index 0 must be finite"):
        to_pgvector([math.nan, 0.2])


def test_to_pgvector_accepts_all_finite_vector():
    # Regression guard: the finiteness check must not reject legitimate
    # vectors, and the formatted literal shape is unchanged.
    assert (
        to_pgvector([1.0, 2.0, 3.0])
        == "[" + ",".join(repr(float(v)) for v in (1.0, 2.0, 3.0)) + "]"
    )


def test_to_pgvector_accepts_negative_and_zero_components():
    # Finite negatives and zeros are legitimate embedding components.
    literal = to_pgvector([-0.5, 0.0, 0.5])
    assert literal.startswith("[")
    assert literal.endswith("]")
    assert "nan" not in literal
    assert "inf" not in literal


def test_to_pgvector_empty_vector_is_unchanged():
    # Empty-vector handling is out of scope for this guard (pgvector's own
    # dimension check surfaces it clearly); the finiteness check must not
    # change the existing empty-list behavior.
    assert to_pgvector([]) == "[]"
