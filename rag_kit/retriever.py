"""Retriever: hybrid BM25 + dense ANN, fused with Reciprocal Rank Fusion.

The retriever runs two cheap top-k queries against Postgres — one over
the lexical (tsvector) channel using ``ts_rank_cd``, one over the dense
(pgvector HNSW) channel using cosine distance — then fuses them with RRF
(`rag_kit.fusion.reciprocal_rank_fusion`). The fused result surfaces
per-method ranks so consumers can see which channel pulled each doc up.

`search()` accepts an optional `reranker=...` to apply a cross-encoder
on top of the RRF output. When set, the reranker over-fetches by the
candidate multiplier (so it has more candidates than the final `k` to
choose from), then truncates back to `k`.

`search()` also accepts an optional `rewriter=...` (D-014) that
decomposes the query into 1..K sub-queries before retrieval. When the
rewriter returns ≥2 sub-queries, hybrid search runs per sub-query and
the resulting rankings are fused with RRF across sub-queries; the
per-method ranks dict on each returned result then carries the
per-sub-query ranks (`subquery_0`, `subquery_1`, ...) instead of
`lexical` / `dense`. Single-sub-query rewrites collapse to the existing
single-shot path with the rewritten query.
"""

from __future__ import annotations

import unicodedata
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from .db import to_pgvector
from .embedder import Embedder
from .fusion import DEFAULT_K, reciprocal_rank_fusion
from .indexer import _unstorable
from .reranker import Candidate, Reranker
from .rewriter import Rewriter


def _require_sendable(text: str, where: str) -> None:
    """Refuse a query Postgres cannot take as a text parameter (#366).

    A NUL or a lone surrogate failed in psycopg's parameter adaptation as a
    `DataError` / `UnicodeEncodeError` -- after the rewriter and the embedder
    had already been paid for. The predicate is #308's, which refuses the same
    characters in a `Document` before any embedding. A non-`str` is left to the
    code that already handles it; this check is about characters, not types.
    """
    if not isinstance(text, str):
        return
    reason = _unstorable(text)
    if reason is not None:
        raise ValueError(
            f"{where} contains {reason}, which Postgres cannot take as a text parameter"
        )


# How many candidates each channel pulls before fusion. Tuned to keep
# fusion meaningful (RRF needs distinct ranks) without overfetching.
_CANDIDATE_MULTIPLIER = 4


def _read_scope(conn: Any) -> AbstractContextManager[Any]:
    """The transaction scope a search's SELECTs run in (#346).

    `conn.transaction()` for a psycopg connection; no scope for a duck-typed
    connection that only offers `cursor()`, which is what the unit-test fakes
    are and which has no transaction to leave open.
    """
    transaction = getattr(conn, "transaction", None)
    return transaction() if callable(transaction) else nullcontext()


@dataclass
class RetrievalResult:
    """One returned chunk from a hybrid search.

    Carries the per-method ranks so the caller can see *why* the doc
    surfaced (e.g., "ranked 1 in dense but missing in lexical"). When a
    reranker is applied, `rerank_score` and `rerank_rank` are populated;
    otherwise they're `None`.
    """

    external_id: str
    text: str
    metadata: dict[str, Any]
    fused_score: float
    ranks: dict[str, int]
    rerank_score: float | None = None
    rerank_rank: int | None = None


class Retriever:
    """Run the hybrid retrieval against a live Postgres connection.

    Pass `reranker=...` to `search()` to apply a cross-encoder on top of
    the RRF result. The reranker is opt-in (D-007) so existing callers
    keep their hybrid-only behavior.

    Pass `rewriter=...` to `search()` to decompose the query into 1..K
    sub-queries before retrieval (D-014). The rewriter is opt-in for the
    same backwards-compatibility reason as the reranker; existing
    callers that omit it see the exact same single-query behavior.
    """

    def __init__(self, conn: Any, embedder: Embedder, *, k_rrf: int = DEFAULT_K) -> None:
        # `reciprocal_rank_fusion` re-validates `k > 0` defensively (it's part
        # of the public surface), but waiting for the first `search()` call to
        # raise points the stack trace at `fusion.py` and surfaces `k` rather
        # than the constructor-side `k_rrf` name — confusing for operators
        # mis-wiring the retriever. Hoist the failure to the construction site.
        if not isinstance(k_rrf, int) or isinstance(k_rrf, bool) or k_rrf <= 0:
            raise ValueError(f"k_rrf must be a positive integer, got {k_rrf!r}")
        self.conn = conn
        self.embedder = embedder
        self.k_rrf = k_rrf

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        reranker: Reranker | None = None,
        rewriter: Rewriter | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if not query:
            raise ValueError("query must be non-empty")
        # Before the rewriter and the embedder, both of which may be paid
        # calls: this query cannot reach the lexical channel either way (#366).
        _require_sendable(query, "query")

        if rewriter is None:
            return self._hybrid_search(query, k, reranker=reranker)

        rewrite_result = rewriter.rewrite(query)
        sub_queries = rewrite_result.sub_queries
        if not sub_queries:
            raise ValueError("rewriter returned empty sub_queries")
        # All of them before the first is embedded: the multi-hop path embeds
        # one sub-query at a time, so a bad second one cost the first (#366).
        for idx, sq in enumerate(sub_queries):
            _require_sendable(sq, f"rewriter sub_queries[{idx}]")
        if len(sub_queries) == 1:
            # Single-sub-query rewrites collapse to the existing single-shot path
            # using the (possibly normalized) rewritten query for *retrieval*, but
            # the reranker must score against the *original* user query — same
            # contract the multi-hop path follows (intent is the user's query, not
            # the reformulation). Without this, a normalizing/paraphrasing rewriter
            # silently re-orders results against the rewritten string.
            return self._hybrid_search(sub_queries[0], k, reranker=reranker, rerank_query=query)

        return self._multi_hop_search(
            original_query=query,
            sub_queries=sub_queries,
            k=k,
            reranker=reranker,
        )

    def rerank_pool(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """The candidates `search(query, k, reranker=r)` hands its reranker (#417).

        Channels `k * _CANDIDATE_MULTIPLIER` deep, fused, cut to that many, in
        fused order. `StreamingPipeline` reranks this pool so the SSE path and
        the library cite the same chunks: it used to call `search` with an
        already over-fetched `k`, and `_hybrid_search` over-fetched again, so
        each channel was read 16k deep instead of 4k and the pools differed.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if not query:
            raise ValueError("query must be non-empty")
        _require_sendable(query, "query")
        return self._hybrid_search(query, k, reranker=None, pool=True)

    # ------------------------------------------------------------------
    # Single-query hybrid path (the pre-D-014 behavior, factored out so
    # the multi-hop path can call it once per sub-query).
    # ------------------------------------------------------------------

    def _hybrid_search(
        self,
        query: str,
        k: int,
        *,
        reranker: Reranker | None,
        rerank_query: str | None = None,
        pool: bool = False,
    ) -> list[RetrievalResult]:
        # `pool=True` returns the whole pre-rerank pool (`rerank_pool`, #417).
        # `query` drives retrieval (lexical + dense channels). `rerank_query`
        # is what the reranker scores against; it defaults to `query` so the
        # `rewriter=None` and multi-hop-per-sub-query callers are unchanged,
        # and the single-sub-query rewrite path overrides it with the original
        # user query (see `search`).
        candidate_k = k * _CANDIDATE_MULTIPLIER

        # Embed and validate before either channel issues SQL (#194). The
        # dense literal isn't used until the second channel, but `to_pgvector`
        # is also the width/finiteness guard, and a query embedding of the
        # wrong width means this search cannot succeed either way — so running
        # the lexical query first only buys a round-trip whose results get
        # thrown away. Hoisting it makes "rejected before any SQL is issued"
        # true on the query path as well as the write path.
        qvec = to_pgvector(self.embedder.embed(query))

        # Both channels run in one `conn.transaction()` (#346). The SELECTs used
        # to run bare on the caller's connection, and `rag_kit.db.connect()`
        # leaves autocommit off, so the first search left the connection idle
        # in transaction holding `AccessShareLock` on `documents` until
        # something else on it committed: a second connection's
        # `Indexer.clear()` (`TRUNCATE`) waited forever, and one statement the
        # server rejected (a 20,000-word query overflows `plainto_tsquery`'s
        # stack) failed every later search with `InFailedSqlTransaction`.
        # psycopg's scope is `BEGIN ... COMMIT` on an idle connection and a
        # `SAVEPOINT` inside a transaction the caller already has open, rolled
        # back on error either way, so a search leaves the connection in the
        # state it found it. The embedding above stays outside it: no lock is
        # held while the embedder runs.
        with _read_scope(self.conn):
            # --- Lexical channel ------------------------------------------------
            # `ORDER BY <non-unique key> LIMIT n` leaves the choice among tied rows
            # undefined, and Postgres settles it by physical row order — so *which
            # documents come back* moved whenever rows moved on disk (#180). An
            # ordinary no-op `UPDATE ... SET text = text` was enough. Measured on
            # Postgres 16 with six documents sharing one term profile, all at
            # ts_rank_cd 0.1, LIMIT 3:
            #
            #   initial insert     -> doc-alpha,  doc-bravo,   doc-charlie
            #   one row rewritten  -> doc-bravo,  doc-charlie, doc-delta
            #   two more rewritten -> doc-delta,  doc-echo,    doc-foxtrot
            #
            # Complete membership turnover: the first and third sets share nothing.
            # For a RAG kit that is which chunks get retrieved, generated from, and
            # cited. Equal ranks are ordinary, not contrived — templated or
            # boilerplate documents share a term profile by construction.
            #
            # `#40` gave RRF a doc-id tiebreak, but on a different axis: it made the
            # fused output independent of the order *methods* are supplied in. It
            # cannot absorb an unstable input *ranking*, because a different input
            # rank changes the fused score itself, which that tiebreak never sees.
            # Feeding the three orderings above into `reciprocal_rank_fusion` gave
            # three different top-3s.
            #
            # The tiebreak is free here, verified with EXPLAIN: `ts_rank_cd` is a
            # computed expression no index can order by, so this query already
            # sorts, and `external_id` only extends the existing sort key. Same
            # plan, same Seq Scan.
            #
            # `COLLATE "C"` (#370): a bare `external_id ASC` sorts in the
            # database's collation, while the dense tiebreak below and RRF's
            # both sort in Python, by code point. On an en_US cluster (CI's
            # pgvector image is one) ids differing only in case or `-`/`_`
            # came back `doc_e, doc-a, Doc-B` here against `DOC-G, Doc-B,
            # doc-a` on C, and the fused top-1 changed with the locale. "C"
            # orders by byte, which for UTF-8 is code-point order -- what
            # Python's `sorted` uses. Still free: it changes the comparison,
            # not the plan.
            lexical_sql = """
            SELECT external_id, text, metadata
            FROM documents
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC,
                     external_id COLLATE "C" ASC
            LIMIT %s
            """
            # NFC, because the trigger builds `tsv` from the NFC form (#324): FTS
            # compares code points, so a decomposed `café` and a composed one were
            # different words. Lexical channel only -- the dense channel embeds the
            # query exactly as given, which is the embedder's business.
            lex_query = unicodedata.normalize("NFC", query)
            with self.conn.cursor() as cur:
                cur.execute(lexical_sql, (lex_query, lex_query, candidate_k))
                lex_rows = cur.fetchall()

            # --- Dense channel --------------------------------------------------
            # Same defect, deliberately a different shape (#180). Appending
            # `, external_id` to this ORDER BY would make it deterministic too, but
            # that is exactly the form that stops pgvector using an HNSW
            # index-ordered scan, turning an ANN lookup into a full scan plus sort.
            # So: select the distance, leave the ORDER BY alone so the index scan is
            # preserved, and impose the deterministic order on the returned rows
            # below.
            #
            # What that fixes and what it does not: the dense *ranking* becomes a
            # function of the data. Membership at the LIMIT boundary is still the
            # index's choice, so a tied row can still fall outside `candidate_k` —
            # a weaker guarantee than the lexical channel gets. That is the right
            # place to stop, because HNSW is an *approximate* index whose recall is
            # already not exact (the open question in vector-search-at-scale#71).
            # Making dense membership exact is a different problem from making it
            # deterministic.
            #
            # (`qvec` is computed at the top of this method, before the lexical
            # channel — see the note there.)
            dense_sql = """
            SELECT external_id, text, metadata, embedding <=> %s::vector AS dist
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            with self.conn.cursor() as cur:
                cur.execute(dense_sql, (qvec, qvec, candidate_k))
                dense_raw = cur.fetchall()
        # Sort on (distance, external_id): distance first so the channel's
        # ranking semantics are untouched, `external_id` only to settle exact
        # ties. Python's sort is stable, so rows the database already separated
        # keep their order.
        dense_rows = [
            (ext_id, text, meta)
            for ext_id, text, meta, _dist in sorted(dense_raw, key=lambda r: (r[3], r[0]))
        ]

        # Build a single row-by-external_id index so the fused result can
        # reconstruct full objects without going back to the database.
        by_id: dict[str, tuple[str, dict[str, Any]]] = {}
        for ext_id, text, meta in lex_rows:
            by_id[ext_id] = (text, meta)
        for ext_id, text, meta in dense_rows:
            by_id.setdefault(ext_id, (text, meta))

        rankings = {
            "lexical": [r[0] for r in lex_rows],
            "dense": [r[0] for r in dense_rows],
        }
        fused = reciprocal_rank_fusion(rankings, k=self.k_rrf)

        # Build pre-rerank results from the top `candidate_k` fused entries
        # so the reranker has more candidates than the final `k` to pick from.
        pre_rerank: list[RetrievalResult] = []
        for ext_id, score, ranks in fused[:candidate_k]:
            text, meta = by_id[ext_id]
            pre_rerank.append(
                RetrievalResult(
                    external_id=ext_id,
                    text=text,
                    metadata=meta or {},
                    fused_score=score,
                    ranks=ranks,
                )
            )

        if pool:
            return pre_rerank
        if reranker is None:
            return pre_rerank[:k]

        candidates = [
            Candidate(external_id=r.external_id, text=r.text, metadata=r.metadata)
            for r in pre_rerank
        ]
        scored = reranker.rerank(rerank_query if rerank_query is not None else query, candidates)

        # Re-attach the pre-rerank fused metadata to the reranked output.
        pre_by_id = {r.external_id: r for r in pre_rerank}
        out: list[RetrievalResult] = []
        for sc in scored[:k]:
            base = pre_by_id[sc.external_id]
            out.append(
                RetrievalResult(
                    external_id=sc.external_id,
                    text=sc.text,
                    metadata=sc.metadata,
                    fused_score=base.fused_score,
                    ranks=base.ranks,
                    rerank_score=sc.rerank_score,
                    rerank_rank=sc.rerank_rank,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Multi-hop path: hybrid search per sub-query, fuse rankings via RRF
    # across sub-queries. Optional rerank on top.
    # ------------------------------------------------------------------

    def _multi_hop_search(
        self,
        *,
        original_query: str,
        sub_queries: tuple[str, ...],
        k: int,
        reranker: Reranker | None,
    ) -> list[RetrievalResult]:
        # Over-fetch per sub-query so the cross-sub-query fusion has
        # plenty of candidates and the reranker sees more than `k`.
        per_sub_k = max(k * _CANDIDATE_MULTIPLIER, k * 2)
        rankings: dict[str, list[str]] = {}
        seen: dict[str, RetrievalResult] = {}

        for idx, sq in enumerate(sub_queries):
            sub_results = self._hybrid_search(sq, per_sub_k, reranker=None)
            rankings[f"subquery_{idx}"] = [r.external_id for r in sub_results]
            for r in sub_results:
                seen.setdefault(r.external_id, r)

        fused = reciprocal_rank_fusion(rankings, k=self.k_rrf)

        pre_rerank: list[RetrievalResult] = []
        for ext_id, score, ranks in fused[:per_sub_k]:
            base = seen[ext_id]
            pre_rerank.append(
                RetrievalResult(
                    external_id=base.external_id,
                    text=base.text,
                    metadata=base.metadata,
                    fused_score=score,
                    ranks=ranks,
                )
            )

        if reranker is None:
            return pre_rerank[:k]

        # The reranker should score against the original user query, not
        # any one sub-query — the user's intent is the merged question.
        candidates = [
            Candidate(external_id=r.external_id, text=r.text, metadata=r.metadata)
            for r in pre_rerank
        ]
        scored = reranker.rerank(original_query, candidates)
        pre_by_id = {r.external_id: r for r in pre_rerank}
        out: list[RetrievalResult] = []
        for sc in scored[:k]:
            base = pre_by_id[sc.external_id]
            out.append(
                RetrievalResult(
                    external_id=sc.external_id,
                    text=sc.text,
                    metadata=sc.metadata,
                    fused_score=base.fused_score,
                    ranks=base.ranks,
                    rerank_score=sc.rerank_score,
                    rerank_rank=sc.rerank_rank,
                )
            )
        return out
