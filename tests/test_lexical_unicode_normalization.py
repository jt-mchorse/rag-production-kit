"""The lexical channel matches a word whatever its Unicode normal form (#324).

Postgres FTS compares code points. `documents.tsv` came from the text as
written and the query went to `plainto_tsquery` as given, so a chunk storing
`café` decomposed (NFD: `e` + U+0301, as macOS file names and some PDF
extractors produce) was invisible to the same word typed composed (NFC), and
the reverse. The trigger now builds `tsv` from the NFC form and `Retriever`
sends the NFC form of the query, the normalisation `rag_kit.text.word_tokens`
already applies on the Python lexical paths (#285).
"""

from __future__ import annotations

import unicodedata
from typing import Any

import pytest

from rag_kit import Document, HashEmbedder, Indexer, Retriever
from rag_kit.db import to_pgvector


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


class _LogConn:
    """Records every (sql, params) and returns no rows."""

    def __init__(self) -> None:
        self.log: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self) -> _LogConn:
        return self

    def __enter__(self) -> _LogConn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.log.append((sql, params))

    def fetchall(self) -> list[Any]:
        return []


def _params(conn: _LogConn, marker: str) -> tuple[Any, ...]:
    return next(p for s, p in conn.log if marker in s.upper())


def test_the_lexical_query_is_sent_in_nfc() -> None:
    conn = _LogConn()
    query = _nfd("café crème")
    assert query != _nfc(query)
    Retriever(conn, HashEmbedder()).search(query, k=3)
    lex = _params(conn, "TS_RANK_CD")
    assert lex[:2] == (_nfc(query), _nfc(query))


def test_the_dense_query_is_embedded_as_given() -> None:
    conn = _LogConn()
    query = _nfd("café crème")
    Retriever(conn, HashEmbedder()).search(query, k=3)
    dense = _params(conn, "EMBEDDING <=>")
    assert dense[0] == to_pgvector(HashEmbedder().embed(query))


def _lexical_hits(conn: Any, query: str) -> list[str]:
    results = Retriever(conn, HashEmbedder()).search(query, k=5)
    return [r.external_id for r in results if "lexical" in r.ranks]


@pytest.mark.pg
@pytest.mark.parametrize(
    ("doc_form", "query_form"),
    [(_nfd, _nfc), (_nfc, _nfd)],
    ids=["nfd-chunk-nfc-query", "nfc-chunk-nfd-query"],
)
def test_a_chunk_is_found_whatever_the_normal_form(pg_conn, doc_form, query_form) -> None:
    text = doc_form("Le café propose une crème brûlée.")
    indexer = Indexer(pg_conn, HashEmbedder())
    indexer.clear()
    indexer.add_documents([Document("dessert", text), Document("other", "Shipping takes days.")])
    for word in ("café", "crème", "brûlée"):
        assert _lexical_hits(pg_conn, query_form(word)) == ["dessert"], word


@pytest.mark.pg
def test_the_stored_text_is_not_normalised(pg_conn) -> None:
    text = _nfd("Le café propose une crème brûlée.")
    indexer = Indexer(pg_conn, HashEmbedder())
    indexer.clear()
    indexer.add_documents([Document("dessert", text)])
    (hit,) = Retriever(pg_conn, HashEmbedder()).search(_nfc("café"), k=1)
    assert hit.text == text
