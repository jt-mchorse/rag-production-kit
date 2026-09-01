"""Tests for ``rag_kit.db.to_pgvector`` (#82, #194).

``to_pgvector`` is the single chokepoint both the indexer write-path
(``Indexer.add_documents``) and the retriever query-path
(``Retriever._hybrid_search``) funnel BYO-``Embedder`` output through before
it reaches pgvector. A non-finite component (``NaN`` / ``±Inf``) from a
normalization divide-by-zero, an ``Inf`` overflow, or a NaN-poisoned model
output must be rejected loudly at this seam — unguarded it reaches pgvector
as the bare token ``nan``/``inf`` and either errors opaquely far from the
embedder or silently corrupts dense-channel ordering. Same seam-validation
posture as llm-cost-optimizer ``_validate_embedding`` (#88).

#194 added the other half: the guard checked every *component* and never how
many of them there were, while ``infra/postgres/init.sql`` pins the count at
``vector(64)``. Because a wrong width now fails first, the finiteness cases
below build full-width vectors — a three-element literal was only ever
incidental to what they test.
"""

from __future__ import annotations

import math

import pytest

from rag_kit import EMBEDDING_DIM
from rag_kit.db import to_pgvector


def _vec(*head: float) -> list[float]:
    """A width-correct vector whose leading components are *head*.

    Padded with zeros, which are legitimate components (asserted below), so
    the padding cannot itself trip either guard.
    """
    assert len(head) <= EMBEDDING_DIM
    return [*head] + [0.0] * (EMBEDDING_DIM - len(head))


# ---------------------------------------------------------------------------
# Finiteness (#82)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "inf", "-inf"],
)
def test_to_pgvector_rejects_non_finite_component(bad: float):
    # A non-finite component anywhere in the vector must fail loud, naming
    # the index, rather than emitting `[..,nan,..]` for pgvector to choke on.
    with pytest.raises(ValueError, match="index 1 must be finite"):
        to_pgvector(_vec(0.1, bad, 0.3))


def test_to_pgvector_error_names_first_offending_index():
    # Index 0 is the first to violate; the message should point there.
    with pytest.raises(ValueError, match="index 0 must be finite"):
        to_pgvector(_vec(math.nan, 0.2))


def test_to_pgvector_accepts_all_finite_vector():
    # Regression guard: the finiteness check must not reject legitimate
    # vectors, and the formatted literal shape is unchanged.
    vec = _vec(1.0, 2.0, 3.0)
    assert to_pgvector(vec) == "[" + ",".join(repr(float(v)) for v in vec) + "]"


def test_to_pgvector_accepts_negative_and_zero_components():
    # Finite negatives and zeros are legitimate embedding components.
    literal = to_pgvector(_vec(-0.5, 0.0, 0.5))
    assert literal.startswith("[")
    assert literal.endswith("]")
    assert "nan" not in literal
    assert "inf" not in literal


# ---------------------------------------------------------------------------
# Width (#194)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n",
    [0, 1, EMBEDDING_DIM - 1, EMBEDDING_DIM + 1, 768, 1024],
    ids=["empty", "one", "one-short", "one-long", "bge-768", "voyage-1024"],
)
def test_to_pgvector_rejects_wrong_width(n: int):
    """Every wrong width, including the real embedders' defaults.

    768 and 1024 are not exotic: they are what BGE, Cohere and Voyage return,
    and swapping in one of those is the documented intent of the `Embedder`
    seam. `EMBEDDING_DIM - 1` / `+ 1` pin the boundary — an off-by-one is
    exactly what a hand-written stub embedder produces.
    """
    with pytest.raises(ValueError, match="must have exactly 64 components"):
        to_pgvector([0.0] * n)


def test_to_pgvector_empty_vector_is_rejected():
    """Deliberate reversal of the previous contract.

    This assertion used to read ``to_pgvector([]) == "[]"``, justified by a
    comment claiming "pgvector's own dimension check surfaces it clearly".
    It does not surface clearly: it surfaces inside ``executemany`` or the
    dense-channel SQL, after the whole batch has been embedded — which is the
    entire argument the sibling finiteness guard already won. The literal
    ``"[]"`` was never a value any caller wanted.
    """
    with pytest.raises(ValueError, match="got 0"):
        to_pgvector([])


def test_wrong_width_error_names_both_widths():
    """The message has to say what was expected *and* what arrived.

    Naming only one leaves the operator guessing which side is wrong — the
    embedder they swapped in, or the schema they forgot to migrate.
    """
    with pytest.raises(ValueError, match="exactly 64") as exc:
        to_pgvector([0.0] * 1024)
    message = str(exc.value)
    assert "exactly 64" in message
    assert "got 1024" in message
    assert "EMBEDDING_DIM" in message
    assert "init.sql" in message


def test_width_is_checked_before_finiteness():
    """Order is deliberate and worth pinning.

    A 1024-wide vector full of NaN has two problems, and the width is the root
    cause — it says "you swapped the embedder", where the component index says
    "one number went bad". Reporting the deeper one first is the actionable
    choice, and it is the order the implementation happens to have, so it gets
    an assertion rather than staying an accident.
    """
    with pytest.raises(ValueError, match="must have exactly 64 components"):
        to_pgvector([math.nan] * 1024)


def test_expected_dim_override_is_honoured():
    """The parameter exists so a caller with a genuinely different width can
    pass it without mutating a module constant. There is deliberately no
    value that disables the check."""
    assert to_pgvector([1.0, 2.0], expected_dim=2) == "[1.0,2.0]"
    with pytest.raises(ValueError, match="exactly 2 components"):
        to_pgvector([1.0, 2.0, 3.0], expected_dim=2)


def test_default_expected_dim_is_the_exported_constant():
    """Not a literal 64 in ``db.py``.

    D-003 makes the width per-deployment and names ``EMBEDDING_DIM`` as the
    Python-side knob an operator edits alongside ``init.sql``. A hardcoded 64
    here would silently ignore that edit.
    """
    import inspect

    signature = inspect.signature(to_pgvector)
    assert signature.parameters["expected_dim"].default == EMBEDDING_DIM


def test_the_reference_embedder_passes_its_own_guard():
    """Anti-vacuous arm: the shipped `HashEmbedder` must satisfy the check.

    A guard that rejected the repo's own embedder would turn every test above
    green while breaking the quickstart.
    """
    from rag_kit import HashEmbedder

    literal = to_pgvector(HashEmbedder().embed("hello world"))
    assert literal.startswith("[")
    assert literal.endswith("]")
    assert literal.count(",") == EMBEDDING_DIM - 1
