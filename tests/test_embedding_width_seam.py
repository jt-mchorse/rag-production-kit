"""The embedding-width guard at both call sites, and the schema it defends (#194).

`to_pgvector`'s own unit tests live in `tests/test_db_to_pgvector.py`. This file
covers the two things those cannot:

1. **The seam.** A wrong-width `Embedder` has to fail at the *first* document,
   before the rest of the batch is embedded — that is the whole cost argument
   for checking in Python rather than letting pgvector object. And the query
   path has to be guarded too, where a width mismatch is a failed search rather
   than a failed insert.

2. **The coupling D-003 asserts but never enforced.** D-003 makes the width a
   per-deployment setting and says the operator "edits both the column
   dimension in `infra/postgres/init.sql` and the `EMBEDDING_DIM` constant".
   Nothing checked that those two agree, so an operator who edited one and not
   the other landed in exactly the failure #194 is about — with the Python
   guard now in place, in the *opposite* direction (a correct embedder rejected
   against a stale constant). An operator obligation with no check is a
   convention, not a contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag_kit import EMBEDDING_DIM, HashEmbedder
from rag_kit.indexer import Document, Indexer
from rag_kit.retriever import Retriever

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INIT_SQL = _REPO_ROOT / "infra" / "postgres" / "init.sql"

#: Anchored on the column *name* inside the DDL, not on a bare `vector(`.
#: The same file contains `to_tsvector('english', ...)` a few lines down, which
#: a naive scan matches — parse the declaration, don't grep the token.
_EMBEDDING_COLUMN = re.compile(r"^\s*embedding\s+vector\((\d+)\)", re.MULTILINE)


class _WidthEmbedder:
    """Counts its calls so a test can prove where the failure landed."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1] * self.width


class _RecordingCursor:
    """A cursor that fails the test if any SQL reaches it."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("SQL was issued despite a wrong-width embedding")

    def executemany(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("SQL was issued despite a wrong-width embedding")

    def fetchall(self):  # pragma: no cover - must not run
        raise AssertionError("SQL was issued despite a wrong-width embedding")


class _RecordingConn:
    def cursor(self):
        return _RecordingCursor()

    def commit(self):  # pragma: no cover - must not run
        raise AssertionError("commit reached despite a wrong-width embedding")


# ---------------------------------------------------------------------------
# The schema coupling
# ---------------------------------------------------------------------------


def test_init_sql_declares_exactly_one_embedding_column() -> None:
    """Anti-vacuous arm for the parse below.

    A regex that stopped matching — after a DDL reformat, say — would make the
    coupling test vacuously true, which is the failure mode a doc/schema lock
    is most prone to.
    """
    matches = _EMBEDDING_COLUMN.findall(_INIT_SQL.read_text(encoding="utf-8"))
    assert len(matches) == 1, (
        f"expected exactly one `embedding vector(N)` column declaration in "
        f"{_INIT_SQL.name}; found {matches}"
    )


def test_schema_width_and_embedding_dim_agree() -> None:
    """D-003's operator obligation, enforced.

    `EMBEDDING_DIM` is what `to_pgvector` validates against; `init.sql` is what
    Postgres enforces. If they drift, every insert fails — either in Python
    against a stale constant, or in Postgres against a stale column.
    """
    declared = int(_EMBEDDING_COLUMN.search(_INIT_SQL.read_text(encoding="utf-8")).group(1))
    assert declared == EMBEDDING_DIM, (
        f"infra/postgres/init.sql declares vector({declared}) but "
        f"rag_kit.EMBEDDING_DIM is {EMBEDDING_DIM}. D-003 asks for both to be "
        "changed together."
    )


def test_the_reference_embedder_matches_the_schema() -> None:
    """The third leg: the shipped `HashEmbedder` actually produces that width.

    Two constants agreeing with each other says nothing if the default
    embedder disagrees with both.
    """
    assert len(HashEmbedder().embed("anything")) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------


def test_wrong_width_fails_on_the_first_document_not_after_the_batch() -> None:
    """The cost argument, asserted rather than assumed.

    `add_documents` embeds each document as it builds its row, so a guard
    inside that loop stops after one embedding call. Without it the operator
    pays for all five — and against a metered API that is the difference
    between a typo and a bill.
    """
    embedder = _WidthEmbedder(width=1024)
    indexer = Indexer(_RecordingConn(), embedder)
    docs = [Document(external_id=f"d{i}", text=f"text {i}") for i in range(5)]

    with pytest.raises(ValueError, match="must have exactly 64 components"):
        indexer.add_documents(docs)

    assert embedder.calls == 1, (
        f"the whole batch was embedded before the guard fired ({embedder.calls} calls); "
        "the check must be inside the loop, not after it"
    )


def test_a_correct_width_embedder_still_reaches_the_sql() -> None:
    """Anti-vacuous arm: the guard must not reject the good case.

    Uses the recording connection's `executemany` refusal in reverse — a
    correct-width batch is expected to get that far, so the AssertionError it
    raises is the *success* signal here.
    """
    indexer = Indexer(_RecordingConn(), _WidthEmbedder(width=EMBEDDING_DIM))
    with pytest.raises(AssertionError, match="SQL was issued"):
        indexer.add_documents([Document(external_id="d0", text="text")])


# ---------------------------------------------------------------------------
# The query path
# ---------------------------------------------------------------------------


def test_query_path_rejects_a_wrong_width_query_embedding() -> None:
    """A width mismatch here is a failed search, not a failed insert.

    Same guard, different consequence: the dense channel's `<=>` operator
    errors on mismatched dimensions, so without this the operator sees a
    Postgres error mid-retrieval instead of a message naming the embedder.

    Nothing reaches SQL — which is only true because the embed-and-validate
    step was hoisted above the lexical channel. It used to sit between the two
    channels, so a wrong-width query ran the lexical query first and threw the
    rows away. #194's first criterion is "before any SQL is issued", and the
    write path met it for free while the query path did not.
    """
    retriever = Retriever(_RecordingConn(), _WidthEmbedder(width=768))
    with pytest.raises(ValueError, match="must have exactly 64 components"):
        retriever.search("a query")


def test_query_path_embeds_exactly_once_before_failing() -> None:
    """The hoist must not have duplicated the embedder call.

    Moving a line that both computes and validates is exactly the edit that
    leaves a stale copy behind; a second call would be an extra billed request
    per query on a metered embedder.
    """
    embedder = _WidthEmbedder(width=768)
    with pytest.raises(ValueError, match="exactly 64 components"):
        Retriever(_RecordingConn(), embedder).search("a query")
    assert embedder.calls == 1
