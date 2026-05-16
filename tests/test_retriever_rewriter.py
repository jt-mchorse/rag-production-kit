"""Tests for `Retriever.search(rewriter=...)` integration (#3, D-014).

The single-query hybrid path uses pgvector and is exercised in
``test_hybrid_pg.py`` (gated on ``DATABASE_URL``). These tests stand up
a minimal fake psycopg-style connection so the rewriter wiring can be
unit-tested without a live database — they verify:

- `rewriter=None` is the existing behavior (no extra retrieval calls).
- A single-sub-query rewrite collapses to one hybrid search using the
  rewritten query.
- A multi-sub-query rewrite runs hybrid search once per sub-query and
  fuses the rankings via RRF across sub-queries.
- The reranker, when combined with the rewriter, scores against the
  original user query (not any single sub-query).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rag_kit import HashEmbedder, Retriever, RewriteResult
from rag_kit.reranker import LexicalOverlapReranker, ScoredCandidate

# ----------------------------------------------------------------------
# Fake psycopg-style connection
# ----------------------------------------------------------------------


@dataclass
class _Row:
    external_id: str
    text: str
    metadata: dict[str, Any]


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self._buf: list[tuple[str, str, dict[str, Any]]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc) -> None:
        self._buf = []

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        # Inspect SQL shape to decide which channel we're answering.
        upper = sql.upper()
        if "TS_RANK_CD" in upper or "PLAINTO_TSQUERY" in upper:
            query = params[0]
            limit = params[2]
            self._buf = self._conn.lexical_results(query, limit)
        elif "EMBEDDING <=>" in upper:
            qvec = params[0]
            limit = params[1]
            self._buf = self._conn.dense_results(qvec, limit)
        else:
            raise AssertionError(f"unexpected SQL: {sql}")
        self._conn.query_log.append((sql, params))

    def fetchall(self) -> list[tuple[str, str, dict[str, Any]]]:
        return list(self._buf)


class _FakeConn:
    """A minimal stand-in for a psycopg connection.

    `lexical_by_query` maps each (case-insensitive, lowercased) query
    string to a list of ``(external_id, text, metadata)`` rows that the
    "lexical channel" should return for that query. `dense_default` is
    returned by the dense channel for every query (the dense channel is
    less query-sensitive in this fake — sufficient for testing fusion
    behavior).
    """

    def __init__(
        self,
        *,
        lexical_by_query: dict[str, list[tuple[str, str, dict[str, Any]]]],
        dense_default: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        self._lexical_by_query = lexical_by_query
        self._dense_default = dense_default
        self.query_log: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def lexical_results(
        self, query: str, limit: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        rows = self._lexical_by_query.get(query.lower(), [])
        return rows[:limit]

    def dense_results(
        self, qvec: Any, limit: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        return list(self._dense_default[:limit])


# ----------------------------------------------------------------------
# Helper: stub Rewriter
# ----------------------------------------------------------------------


@dataclass
class _StubRewriter:
    """Returns a fixed RewriteResult regardless of input."""

    result: RewriteResult

    def rewrite(self, query: str) -> RewriteResult:
        return self.result


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_rewriter_none_keeps_single_query_path():
    """rewriter=None must not change the existing search behavior."""
    conn = _FakeConn(
        lexical_by_query={
            "refund window": [
                ("doc-refund", "Refund policy: 14 days for the Pro plan.", {}),
            ],
        },
        dense_default=[
            ("doc-cancel", "If you cancel, your money is returned within two weeks.", {}),
        ],
    )
    retriever = Retriever(conn, HashEmbedder())
    results = retriever.search("refund window", k=2)
    assert {r.external_id for r in results} == {"doc-refund", "doc-cancel"}
    # Each result carries the per-method ranks from the single-query path.
    for r in results:
        assert set(r.ranks.keys()) <= {"lexical", "dense"}
    # Exactly one lexical + one dense round-trip (no multi-hop expansion).
    sqls = [entry[0] for entry in conn.query_log]
    lexical_calls = [s for s in sqls if "PLAINTO_TSQUERY" in s.upper()]
    dense_calls = [s for s in sqls if "EMBEDDING <=>" in s.upper()]
    assert len(lexical_calls) == 1
    assert len(dense_calls) == 1


def test_single_sub_query_rewrite_collapses_to_single_hybrid_call():
    """A rewriter returning (q,) must call the hybrid path once with that q."""
    conn = _FakeConn(
        lexical_by_query={
            "rewritten query": [("doc-x", "x text", {})],
        },
        dense_default=[("doc-y", "y text", {})],
    )
    rewriter = _StubRewriter(
        RewriteResult(sub_queries=("rewritten query",), reasoning="no_decomposition")
    )
    retriever = Retriever(conn, HashEmbedder())
    results = retriever.search("original query", k=2, rewriter=rewriter)
    assert {r.external_id for r in results} == {"doc-x", "doc-y"}
    # Lexical channel was called with the rewritten query, not the original.
    lex_params = [
        entry[1] for entry in conn.query_log if "PLAINTO_TSQUERY" in entry[0].upper()
    ]
    assert lex_params, "lexical channel was not called"
    assert lex_params[0][0] == "rewritten query"


def test_multi_sub_query_rewrite_runs_hybrid_per_sub_query_and_fuses():
    """≥2 sub-queries → one hybrid search per sub-query, RRF-fused across them."""
    conn = _FakeConn(
        lexical_by_query={
            "who founded anthropic?": [
                ("doc-founders", "Anthropic was founded by Dario and Daniela Amodei.", {}),
            ],
            "where did they work before anthropic?": [
                ("doc-prior", "The founders previously worked at OpenAI.", {}),
            ],
        },
        dense_default=[],  # keep dense empty so the lexical channel dominates
    )
    rewriter = _StubRewriter(
        RewriteResult(
            sub_queries=("Who founded Anthropic?", "Where did they work before Anthropic?"),
            reasoning="multi_question_and_pattern",
        )
    )
    retriever = Retriever(conn, HashEmbedder())
    results = retriever.search("Who founded Anthropic and where did they work before?", k=2, rewriter=rewriter)
    ids = {r.external_id for r in results}
    assert ids == {"doc-founders", "doc-prior"}, ids
    # Per-method ranks dict now carries sub-query keys, not lexical/dense.
    for r in results:
        assert any(k.startswith("subquery_") for k in r.ranks), r.ranks
    # Two sub-queries → two lexical SQL calls.
    lexical_calls = [s for s in (entry[0] for entry in conn.query_log) if "PLAINTO_TSQUERY" in s.upper()]
    assert len(lexical_calls) == 2


def test_multi_hop_path_with_reranker_uses_original_query():
    """When rewriter expands and reranker is also given, the reranker
    must see the *original* user query (intent is the merged question),
    not any one sub-query.
    """
    conn = _FakeConn(
        lexical_by_query={
            "who founded anthropic?": [
                ("doc-founders", "Anthropic was founded by Dario and Daniela Amodei.", {}),
            ],
            "where did they work before?": [
                ("doc-prior", "The founders previously worked at OpenAI.", {}),
            ],
        },
        dense_default=[],
    )
    rewriter = _StubRewriter(
        RewriteResult(
            sub_queries=("Who founded Anthropic?", "Where did they work before?"),
            reasoning="multi_question_and_pattern",
        )
    )

    seen_rerank_queries: list[str] = []

    class _SpyReranker:
        def rerank(
            self,
            query: str,
            candidates,  # type: ignore[no-untyped-def]
        ):
            seen_rerank_queries.append(query)
            return [
                ScoredCandidate(
                    external_id=c.external_id,
                    text=c.text,
                    metadata=c.metadata,
                    rerank_score=1.0 - i * 0.1,
                    rerank_rank=i + 1,
                )
                for i, c in enumerate(candidates)
            ]

    retriever = Retriever(conn, HashEmbedder())
    original = "Who founded Anthropic and where did they work before?"
    results = retriever.search(
        original,
        k=2,
        rewriter=rewriter,
        reranker=_SpyReranker(),
    )
    assert seen_rerank_queries == [original]
    for r in results:
        assert r.rerank_rank is not None
        assert r.rerank_score is not None


def test_multi_hop_path_with_lexical_overlap_reranker_orders_top_to_original_query():
    """End-to-end: rewriter expands, then a real (dep-free) reranker
    re-scores against the original query. The doc that matches the
    original best should rank first."""
    conn = _FakeConn(
        lexical_by_query={
            "who founded openai?": [
                ("doc-openai-founders", "OpenAI was founded by Sam Altman and others.", {}),
            ],
            "where is anthropic headquartered?": [
                ("doc-hq", "Anthropic is headquartered in San Francisco.", {}),
            ],
        },
        dense_default=[],
    )
    rewriter = _StubRewriter(
        RewriteResult(
            sub_queries=("Who founded OpenAI?", "Where is Anthropic headquartered?"),
            reasoning="multi_question_and_pattern",
        )
    )
    retriever = Retriever(conn, HashEmbedder())
    # The original query mentions "Anthropic headquarters", so the
    # lexical-overlap reranker should put doc-hq on top.
    results = retriever.search(
        "What about Anthropic headquarters?",
        k=2,
        rewriter=rewriter,
        reranker=LexicalOverlapReranker(),
    )
    assert results[0].external_id == "doc-hq"


def test_rewriter_returning_empty_raises():
    rewriter = _StubRewriter(RewriteResult(sub_queries=(), reasoning="bad"))
    conn = _FakeConn(lexical_by_query={}, dense_default=[])
    retriever = Retriever(conn, HashEmbedder())
    with pytest.raises(ValueError, match="empty sub_queries"):
        retriever.search("anything", k=1, rewriter=rewriter)
