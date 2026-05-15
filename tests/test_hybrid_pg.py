"""End-to-end hybrid retrieval against a real Postgres + pgvector.

Skipped automatically when ``DATABASE_URL`` is unset. CI runs these
against the ``pgvector/pgvector:pg16`` service container; local
developers can run ``docker compose up -d`` and
``DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest -m pg``.
"""

from __future__ import annotations

import pytest

from rag_kit import Document, HashEmbedder, Indexer, Retriever

CORPUS = [
    # Lexical-favored: the query word "refund" appears verbatim.
    ("doc-refund", "Our refund policy gives Pro plan customers 14 days to request a return."),
    # Dense-favored: paraphrases the query without sharing keywords.
    ("doc-cancel", "If a subscriber wants their money back, the cancellation window is two weeks."),
    # Off-topic distractors.
    ("doc-shipping", "Standard shipping ships orders within three business days."),
    ("doc-support", "Support is available Monday through Friday from 9am to 5pm."),
    ("doc-loyalty", "Loyalty members receive a free month after twelve months of service."),
]


def _seed(conn):
    indexer = Indexer(conn, HashEmbedder())
    indexer.clear()
    n = indexer.add_documents(Document(ext, text) for ext, text in CORPUS)
    assert n == len(CORPUS)


@pytest.mark.pg
def test_indexer_writes_rows(pg_conn):
    _seed(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        (count,) = cur.fetchone()
    assert count == len(CORPUS)


@pytest.mark.pg
def test_indexer_is_idempotent_by_external_id(pg_conn):
    indexer = Indexer(pg_conn, HashEmbedder())
    indexer.clear()
    indexer.add_documents([Document("doc-x", "first version")])
    indexer.add_documents([Document("doc-x", "second version")])
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE external_id = 'doc-x'")
        (count,) = cur.fetchone()
        cur.execute("SELECT text FROM documents WHERE external_id = 'doc-x'")
        (text,) = cur.fetchone()
    assert count == 1
    assert text == "second version"


@pytest.mark.pg
def test_retriever_returns_top_k_with_scores(pg_conn):
    _seed(pg_conn)
    retr = Retriever(pg_conn, HashEmbedder())
    results = retr.search("refund window", k=3)
    assert 1 <= len(results) <= 3
    for r in results:
        assert r.external_id
        assert r.text
        assert r.fused_score > 0
        # Per-method ranks visible so callers can debug fusion.
        assert any(m in r.ranks for m in ("lexical", "dense"))


@pytest.mark.pg
def test_lexical_query_finds_keyword_match(pg_conn):
    _seed(pg_conn)
    retr = Retriever(pg_conn, HashEmbedder())
    results = retr.search("refund Pro plan", k=3)
    ids = [r.external_id for r in results]
    # The keyword-bearing doc must be in the top-3 of a lexical-friendly query.
    assert "doc-refund" in ids
    # And the lexical channel must be one of the channels that ranked it.
    refund = next(r for r in results if r.external_id == "doc-refund")
    assert "lexical" in refund.ranks


@pytest.mark.pg
def test_query_input_validated(pg_conn):
    _seed(pg_conn)
    retr = Retriever(pg_conn, HashEmbedder())
    with pytest.raises(ValueError, match="non-empty"):
        retr.search("", k=3)
    with pytest.raises(ValueError, match="positive"):
        retr.search("anything", k=0)
