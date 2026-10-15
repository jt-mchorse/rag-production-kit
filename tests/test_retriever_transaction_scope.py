"""A search leaves the connection in the transaction state it found it (#346).

`Retriever._hybrid_search` ran its two SELECTs bare on the caller's
connection, and `rag_kit.db.connect()` leaves autocommit off, so after the
first `search()` the connection sat idle in transaction holding
`AccessShareLock` on `documents`:

* a second connection's `Indexer.clear()` (`TRUNCATE`) waited forever, and
* one statement the server rejected failed every later search on that
  connection with `InFailedSqlTransaction`.

The `pg` tests run against the real server (CI's pgvector service); the
unit test pins the scope's placement with a fake connection, so the
database-free job sees it too.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from rag_kit import Document, HashEmbedder, Indexer, Retriever
from rag_kit.db import connect

CORPUS = [
    ("doc-buffers", "Postgres shared_buffers tuning sets memory for caching."),
    ("doc-hnsw", "The HNSW index trades recall for speed."),
    ("doc-refund", "Our refund policy gives Pro customers 14 days."),
]

#: A query the server rejects: `plainto_tsquery` builds one AND node per word
#: and overflows `max_stack_depth` (2 MB by default) long before this.
#: Measured on PG 17: 10,000 words succeed, 20,000 fail.
_SERVER_REJECTED_QUERY = " ".join(["postgres"] * 100_000)


def _seed(conn: Any) -> None:
    indexer = Indexer(conn, HashEmbedder())
    indexer.clear()
    indexer.add_documents(Document(ext, text) for ext, text in CORPUS)


def _status(conn: Any) -> str:
    return conn.info.transaction_status.name


@pytest.mark.pg
def test_search_leaves_an_idle_connection_idle(pg_conn):
    _seed(pg_conn)
    assert _status(pg_conn) == "IDLE"
    results = Retriever(pg_conn, HashEmbedder()).search("shared_buffers", k=2)
    assert results
    assert results[0].external_id == "doc-buffers"
    assert _status(pg_conn) == "IDLE"


@pytest.mark.pg
def test_a_second_connections_clear_is_not_blocked_by_a_search(pg_conn):
    """The lock is released: `TRUNCATE` from another connection proceeds."""
    _seed(pg_conn)
    Retriever(pg_conn, HashEmbedder()).search("shared_buffers", k=2)
    other = connect(os.environ["DATABASE_URL"])
    try:
        with other.cursor() as cur:
            # Before #346 this TRUNCATE waited forever; bound it so a
            # regression is a failure, not a hung job.
            cur.execute("SET lock_timeout = '5s'")
        other.commit()
        Indexer(other, HashEmbedder()).clear()
        with other.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            (count,) = cur.fetchone()
        other.commit()
        assert count == 0
    finally:
        other.close()


@pytest.mark.pg
def test_a_query_the_server_rejects_does_not_poison_the_next_search(pg_conn):
    import psycopg

    _seed(pg_conn)
    retriever = Retriever(pg_conn, HashEmbedder())
    with pytest.raises(psycopg.Error):
        retriever.search(_SERVER_REJECTED_QUERY, k=2)
    assert _status(pg_conn) == "IDLE"
    results = retriever.search("shared_buffers", k=2)
    assert results[0].external_id == "doc-buffers"


@pytest.mark.pg
def test_a_callers_open_transaction_survives_search_success_and_failure(pg_conn):
    """Inside the caller's transaction the scope is a SAVEPOINT: a successful
    search does not commit the caller's work, a failed one does not abort it."""
    import psycopg

    _seed(pg_conn)
    retriever = Retriever(pg_conn, HashEmbedder())
    with pg_conn.cursor() as cur:
        cur.execute("UPDATE documents SET text = 'uncommitted' WHERE external_id = 'doc-hnsw'")
    assert _status(pg_conn) == "INTRANS"

    retriever.search("shared_buffers", k=2)
    assert _status(pg_conn) == "INTRANS"

    with pytest.raises(psycopg.Error):
        retriever.search(_SERVER_REJECTED_QUERY, k=2)
    assert _status(pg_conn) == "INTRANS"
    with pg_conn.cursor() as cur:
        cur.execute("SELECT text FROM documents WHERE external_id = 'doc-hnsw'")
        (text,) = cur.fetchone()
    assert text == "uncommitted"

    pg_conn.rollback()
    with pg_conn.cursor() as cur:
        cur.execute("SELECT text FROM documents WHERE external_id = 'doc-hnsw'")
        (text,) = cur.fetchone()
    pg_conn.commit()
    assert text == CORPUS[1][1]


# ----------------------------------------------------------------------
# Database-free: where the scope sits
# ----------------------------------------------------------------------


class _Scope:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __enter__(self) -> None:
        self._log.append("enter")

    def __exit__(self, *exc: object) -> None:
        self._log.append("exit")


class _Cursor:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._dense = False

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._dense = "<=>" in sql
        self._log.append("dense" if self._dense else "lexical")

    def fetchall(self) -> list[tuple[Any, ...]]:
        if self._dense:
            return [("doc-a", "alpha", {}, 0.0)]
        return [("doc-a", "alpha", {})]


class _TransactionalConn:
    def __init__(self) -> None:
        self.log: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.log)

    def transaction(self) -> _Scope:
        return _Scope(self.log)


class _LoggingEmbedder:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._inner = HashEmbedder()

    def embed(self, text: str) -> list[float]:
        self._log.append("embed")
        return self._inner.embed(text)


def test_both_channels_run_inside_one_transaction_scope_and_the_embed_outside():
    conn = _TransactionalConn()
    results = Retriever(conn, _LoggingEmbedder(conn.log)).search("alpha", k=1)
    assert [r.external_id for r in results] == ["doc-a"]
    assert conn.log == ["embed", "enter", "lexical", "dense", "exit"]


def test_a_connection_without_transaction_runs_unscoped():
    """The duck-typed fakes across the suite offer only `cursor()`."""

    class _CursorOnly:
        def __init__(self) -> None:
            self.log: list[str] = []

        def cursor(self) -> _Cursor:
            return _Cursor(self.log)

    conn = _CursorOnly()
    results = Retriever(conn, HashEmbedder()).search("alpha", k=1)
    assert [r.external_id for r in results] == ["doc-a"]
    assert conn.log == ["lexical", "dense"]
