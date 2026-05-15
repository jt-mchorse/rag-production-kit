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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .db import to_pgvector
from .embedder import Embedder
from .fusion import DEFAULT_K, reciprocal_rank_fusion
from .reranker import Candidate, Reranker

# How many candidates each channel pulls before fusion. Tuned to keep
# fusion meaningful (RRF needs distinct ranks) without overfetching.
_CANDIDATE_MULTIPLIER = 4


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
    """

    def __init__(self, conn: Any, embedder: Embedder, *, k_rrf: int = DEFAULT_K) -> None:
        self.conn = conn
        self.embedder = embedder
        self.k_rrf = k_rrf

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        reranker: Reranker | None = None,
    ) -> list[RetrievalResult]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not query:
            raise ValueError("query must be non-empty")

        candidate_k = k * _CANDIDATE_MULTIPLIER

        # --- Lexical channel ------------------------------------------------
        lexical_sql = """
        SELECT external_id, text, metadata
        FROM documents
        WHERE tsv @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(lexical_sql, (query, query, candidate_k))
            lex_rows = cur.fetchall()

        # --- Dense channel --------------------------------------------------
        qvec = to_pgvector(self.embedder.embed(query))
        dense_sql = """
        SELECT external_id, text, metadata
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(dense_sql, (qvec, candidate_k))
            dense_rows = cur.fetchall()

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

        if reranker is None:
            return pre_rerank[:k]

        candidates = [
            Candidate(external_id=r.external_id, text=r.text, metadata=r.metadata)
            for r in pre_rerank
        ]
        scored = reranker.rerank(query, candidates)

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
