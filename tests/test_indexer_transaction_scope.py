"""An indexer write joins the caller's transaction and survives its own failure (#348).

`Indexer.add_documents` and `Indexer.clear` ran their statement bare on the
caller's connection and then called `conn.commit()` unconditionally:

* psycopg forbids `commit()` inside a `conn.transaction()` block, so the
  wrapping `add_documents`' docstring suggested raised `ProgrammingError`
  and the block rolled the write back -- zero rows; and
* nothing rolled back a failed statement, so one server error left the
  connection `INERROR` and every later write and search on it failed with
  `InFailedSqlTransaction`.

The `pg` tests run against the real server (CI's pgvector service); the unit
tests pin the scope's commit rule with a fake connection, so the
database-free job sees it too.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from rag_kit import Document, HashEmbedder, Indexer, Retriever
from rag_kit.db import connect

#: A chunk the trigger refuses: 200,000 distinct words make a tsvector of
#: about 4 MB, over Postgres's 1 MiB limit (`ProgramLimitExceeded`, #326).
_OVERSIZED_TEXT = " ".join(f"w{i}x{i * 7919}" for i in range(200_000))


def _status(conn: Any) -> str:
    return conn.info.transaction_status.name


def _ids(conn: Any) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT external_id FROM documents ORDER BY external_id")
        out = [row[0] for row in cur.fetchall()]
    conn.rollback()
    return out


@pytest.fixture
def other_conn(pg_conn: Any) -> Any:
    conn = connect(os.environ["DATABASE_URL"])
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.pg
def test_add_documents_inside_a_caller_transaction_block_is_committed_by_the_block(
    pg_conn: Any, other_conn: Any
) -> None:
    indexer = Indexer(pg_conn, HashEmbedder())
    with pg_conn.transaction():
        assert indexer.add_documents([Document("doc-a", "alpha text")]) == 1
        # Still the caller's transaction: another session cannot see it yet.
        assert _ids(other_conn) == []
    assert _ids(other_conn) == ["doc-a"]
    assert _status(pg_conn) == "IDLE"


@pytest.mark.pg
def test_a_caller_block_that_fails_after_add_documents_rolls_the_write_back(
    pg_conn: Any, other_conn: Any
) -> None:
    indexer = Indexer(pg_conn, HashEmbedder())

    def index_then_abort() -> None:
        with pg_conn.transaction():
            indexer.add_documents([Document("doc-a", "alpha text")])
            raise RuntimeError("caller aborts")

    with pytest.raises(RuntimeError, match="caller aborts"):
        index_then_abort()
    assert _ids(other_conn) == []


@pytest.mark.pg
def test_clear_inside_a_caller_transaction_block(pg_conn: Any, other_conn: Any) -> None:
    indexer = Indexer(pg_conn, HashEmbedder())
    indexer.add_documents([Document("doc-a", "alpha text")])
    with pg_conn.transaction():
        indexer.clear()
        indexer.add_documents([Document("doc-b", "bravo text")])
    assert _ids(other_conn) == ["doc-b"]


@pytest.mark.pg
def test_add_documents_on_an_idle_connection_still_commits(pg_conn: Any, other_conn: Any) -> None:
    Indexer(pg_conn, HashEmbedder()).add_documents([Document("doc-a", "alpha text")])
    assert _status(pg_conn) == "IDLE"
    assert _ids(other_conn) == ["doc-a"]


@pytest.mark.pg
def test_a_server_rejected_batch_leaves_the_connection_usable(pg_conn: Any) -> None:
    embedder = HashEmbedder()
    indexer = Indexer(pg_conn, embedder)
    with pytest.raises(Exception, match="too long for tsvector"):
        indexer.add_documents([Document("doc-big", _OVERSIZED_TEXT)])
    assert _status(pg_conn) == "IDLE"
    assert indexer.add_documents([Document("doc-a", "alpha postgres text")]) == 1
    assert [r.external_id for r in Retriever(pg_conn, embedder).search("postgres")] == ["doc-a"]


@pytest.mark.pg
def test_a_failed_write_inside_a_caller_block_rolls_back_only_itself(
    pg_conn: Any, other_conn: Any
) -> None:
    indexer = Indexer(pg_conn, HashEmbedder())
    with pg_conn.transaction():
        indexer.add_documents([Document("doc-a", "alpha text")])
        with pytest.raises(Exception, match="too long for tsvector"):
            indexer.add_documents([Document("doc-big", _OVERSIZED_TEXT)])
        indexer.add_documents([Document("doc-b", "bravo text")])
    assert _ids(other_conn) == ["doc-a", "doc-b"]


@pytest.mark.pg
def test_clear_that_times_out_on_a_lock_can_be_retried(pg_conn: Any, other_conn: Any) -> None:
    indexer = Indexer(pg_conn, HashEmbedder())
    indexer.add_documents([Document("doc-a", "alpha text")])
    with pg_conn.cursor() as cur:
        cur.execute("SET lock_timeout = '300ms'")
    pg_conn.commit()
    try:
        with other_conn.cursor() as cur:  # holds AccessShareLock, idle in transaction
            cur.execute("SELECT 1 FROM documents LIMIT 1")
        with pytest.raises(Exception, match="lock timeout"):
            indexer.clear()
        assert _status(pg_conn) == "IDLE"
        other_conn.rollback()
        indexer.clear()
        assert _ids(other_conn) == []
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("RESET lock_timeout")
        pg_conn.commit()


# ---------------------------------------------------------------------------
# Database-free: the commit rule, pinned with a fake connection
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.log.append("execute")

    def executemany(self, sql: str, rows: Any) -> None:
        self.log.append("executemany")


class _Transaction:
    def __init__(self, conn: _TxConn) -> None:
        self.conn = conn

    def __enter__(self) -> None:
        self.conn.log.append("begin")
        self.conn._num_transactions += 1

    def __exit__(self, *exc: object) -> None:
        self.conn._num_transactions -= 1
        self.conn.log.append("end")


class _TxConn:
    def __init__(self, block_depth: int) -> None:
        self.log: list[str] = []
        self._num_transactions = block_depth

    def cursor(self) -> _Cursor:
        return _Cursor(self.log)

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def commit(self) -> None:
        self.log.append("commit")


@pytest.mark.parametrize("call", ["add_documents", "clear"])
def test_writes_run_in_a_transaction_scope_and_commit_outside_a_block(call: str) -> None:
    conn = _TxConn(block_depth=0)
    indexer = Indexer(conn, HashEmbedder())
    if call == "add_documents":
        indexer.add_documents([Document("doc-a", "alpha")])
    else:
        indexer.clear()
    statement = "executemany" if call == "add_documents" else "execute"
    assert conn.log == ["begin", statement, "end", "commit"]


@pytest.mark.parametrize("call", ["add_documents", "clear"])
def test_writes_inside_a_caller_block_leave_the_commit_to_the_caller(call: str) -> None:
    conn = _TxConn(block_depth=1)
    indexer = Indexer(conn, HashEmbedder())
    if call == "add_documents":
        indexer.add_documents([Document("doc-a", "alpha")])
    else:
        indexer.clear()
    assert "commit" not in conn.log
    assert conn.log[0] == "begin"
    assert conn.log[-1] == "end"
