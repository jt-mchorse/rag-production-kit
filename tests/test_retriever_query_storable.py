"""A query Postgres cannot take is refused before the rewriter or embedder runs (#366).

`Retriever.search` checked only that the query was non-empty. A NUL or a lone
surrogate failed in psycopg's parameter adaptation -- `DataError` /
`UnicodeEncodeError` -- after `rewriter.rewrite()` and `embedder.embed()` had
both run. Measured on `aa57aaf` against Postgres 17.6: one embed call, then
`psycopg.DataError: PostgreSQL text fields cannot contain NUL (0x00) bytes`.
#308 refuses the same characters in a `Document` before any embedding; this is
the query path's half.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_kit import HashEmbedder, Retriever, RewriteResult

_BAD = {
    "nul": "who wrote\x00 Macbeth",
    "high-surrogate": "who wrote \ud800Macbeth",
    "low-surrogate": "who wrote \udfffMacbeth",
}

# Neighbours of the surrogate block: both are ordinary code points.
_GOOD = {
    "plain": "who wrote Macbeth",
    "U+D7FF": "who wrote ퟿Macbeth",
    "U+E000": "who wrote Macbeth",
}


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._inner = HashEmbedder()

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._inner.embed(text)


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._conn.executed.append(sql)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return []


class _Conn:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)


class _CountingRewriter:
    def __init__(self, sub_queries: tuple[str, ...] | None = None) -> None:
        self.calls = 0
        self._sub_queries = sub_queries

    def rewrite(self, query: str) -> RewriteResult:
        self.calls += 1
        return RewriteResult(sub_queries=self._sub_queries or (query,), reasoning="test")


def _retriever() -> tuple[Retriever, _Conn, _CountingEmbedder]:
    conn = _Conn()
    embedder = _CountingEmbedder()
    return Retriever(conn, embedder), conn, embedder


@pytest.mark.parametrize("name", sorted(_BAD))
def test_bad_query_is_refused_before_embedding_or_sql(name: str) -> None:
    retriever, conn, embedder = _retriever()
    with pytest.raises(ValueError, match=r"^query contains .*cannot take as a text parameter"):
        retriever.search(_BAD[name], k=3)
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_BAD))
def test_bad_query_is_refused_before_the_rewriter(name: str) -> None:
    retriever, conn, embedder = _retriever()
    rewriter = _CountingRewriter()
    with pytest.raises(ValueError, match=r"^query contains "):
        retriever.search(_BAD[name], k=3, rewriter=rewriter)
    assert rewriter.calls == 0
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_BAD))
def test_bad_sub_query_is_refused_before_any_sub_query_is_embedded(name: str) -> None:
    # The multi-hop path embeds one sub-query at a time, so a bad SECOND one
    # used to cost the first one's embedding and SQL before failing.
    retriever, conn, embedder = _retriever()
    rewriter = _CountingRewriter(("who wrote Hamlet", _BAD[name]))
    with pytest.raises(ValueError, match=r"^rewriter sub_queries\[1\] contains "):
        retriever.search("who wrote Hamlet and Macbeth", k=3, rewriter=rewriter)
    assert embedder.calls == []
    assert conn.executed == []


def test_the_message_names_the_character_and_its_index() -> None:
    retriever, _, _ = _retriever()
    with pytest.raises(ValueError, match=r"a NUL \(\\x00\) at index 9"):
        retriever.search(_BAD["nul"], k=3)
    with pytest.raises(ValueError, match=r"a lone surrogate \(U\+D800\) at index 10"):
        retriever.search(_BAD["high-surrogate"], k=3)


@pytest.mark.parametrize("name", sorted(_GOOD))
def test_control_ordinary_query_still_runs_both_channels(name: str) -> None:
    retriever, conn, embedder = _retriever()
    assert retriever.search(_GOOD[name], k=3) == []
    assert embedder.calls == [_GOOD[name]]
    assert len(conn.executed) == 2


@pytest.mark.parametrize("name", sorted(_GOOD))
def test_control_ordinary_sub_queries_still_run(name: str) -> None:
    retriever, conn, embedder = _retriever()
    rewriter = _CountingRewriter(("who wrote Hamlet", _GOOD[name]))
    assert retriever.search("q", k=3, rewriter=rewriter) == []
    assert embedder.calls == ["who wrote Hamlet", _GOOD[name]]
    assert len(conn.executed) == 4
