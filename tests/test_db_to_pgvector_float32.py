"""`to_pgvector` refuses a component outside float32's range (#419).

#82 checked each component for finiteness as a float64, but `vector(64)` is
float32. A component past ~3.4e38 passed the guard and was refused by Postgres
("out of range for type vector") after `Indexer.add_documents` had embedded
the whole batch -- the late, opaque failure #82 and #194 exist to prevent.

Measured on PostgreSQL 17.6 + pgvector (`select '[v]'::vector(1)`):

    3.4028234663852886e+38  FLT_MAX            stored
    3.4028235677973366e+38  rounding midpoint  stored (as FLT_MAX)
    3.5e+38, 1e+39, -3.5e+38                   ERROR: out of range for type vector
    1e-50                                      stored as 0 (underflow: #420)

So only magnitudes strictly above the midpoint are refused, and the guard never
refuses a value pgvector would store.
"""

from __future__ import annotations

import math

import pytest

from rag_kit import EMBEDDING_DIM
from rag_kit.db import to_pgvector
from rag_kit.indexer import Document, Indexer

MIDPOINT = 3.4028235677973366e38
FLT_MAX = 3.4028234663852886e38


def _vec(*head: float) -> list[float]:
    return [*head] + [0.0] * (EMBEDDING_DIM - len(head))


@pytest.mark.parametrize(
    "bad",
    [
        math.nextafter(MIDPOINT, math.inf),
        -math.nextafter(MIDPOINT, math.inf),
        3.5e38,
        -3.5e38,
        1e39,
        1.7976931348623157e308,
        10**39,  # an int: float() takes it, pgvector does not
    ],
    ids=repr,
)
def test_a_component_past_float32_is_refused_naming_its_index(bad: float) -> None:
    with pytest.raises(ValueError, match=r"index 2 must fit in float32"):
        to_pgvector(_vec(0.5, -0.5, bad))


@pytest.mark.parametrize(
    "good", [FLT_MAX, -FLT_MAX, MIDPOINT, -MIDPOINT, 1e38, 1e-50, 1.4e-45, 0.0, -0.0], ids=repr
)
def test_a_component_pgvector_stores_is_accepted(good: float) -> None:
    """Control: FLT_MAX and the midpoint are stored by pgvector, so are tiny values."""
    assert to_pgvector(_vec(good)).startswith(f"[{good!r},")


def test_a_huge_int_is_still_refused_as_non_finite_first() -> None:
    """#396's path (an int past float range) keeps its own message."""
    with pytest.raises(ValueError, match="must be finite"):
        to_pgvector(_vec(10**400))


class _Embedder:
    def __init__(self, component: float) -> None:
        self.component = component
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return _vec(self.component)


class _NoSql:
    def cursor(self):  # pragma: no cover - must not run
        raise AssertionError("SQL was issued despite an out-of-range embedding")

    def transaction(self):  # pragma: no cover - must not run
        raise AssertionError("SQL was issued despite an out-of-range embedding")


def test_the_write_path_stops_on_the_first_document_before_any_sql() -> None:
    embedder = _Embedder(1e39)
    docs = [Document(external_id=f"d{i}", text=f"text {i}") for i in range(5)]
    with pytest.raises(ValueError, match="must fit in float32"):
        Indexer(_NoSql(), embedder).add_documents(docs)
    assert embedder.calls == 1
