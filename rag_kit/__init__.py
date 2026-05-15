"""rag-production-kit: hybrid retrieval (BM25 + pgvector) with RRF fusion + cross-encoder reranking.

Public surface:

    from rag_kit import (
        Document, HashEmbedder, Indexer, Retriever, reciprocal_rank_fusion,
        # Reranking (#2):
        LexicalOverlapReranker, CohereReranker, Candidate, ScoredCandidate,
        rerank_delta_ndcg,
    )

Layers shipped:
- #1: Schema in infra/postgres/init.sql (documents table with tsvector + vector).
- #1: Embedder protocol + HashEmbedder reference implementation.
- #1: Indexer.add_documents — persist text + tsvector + dense embedding.
- #1: Retriever.search — run FTS + ANN in parallel, fuse with RRF.
- #2: Reranker protocol + LexicalOverlapReranker (dep-free) + CohereReranker
      (production). Wired into Retriever.search(reranker=...).

Layers in later PRs:
- Citation enforcement and refusal on weak context (#4)
- Streaming intermediate events (#5)
- Cost telemetry (#6)
- Eval harness integration + Recall@5 measurement (#7)
"""

from .embedder import EMBEDDING_DIM, Embedder, HashEmbedder
from .fusion import reciprocal_rank_fusion
from .indexer import Document, Indexer
from .reranker import (
    Candidate,
    CohereReranker,
    LexicalOverlapReranker,
    RerankDelta,
    Reranker,
    ScoredCandidate,
    rerank_delta_ndcg,
)
from .retriever import RetrievalResult, Retriever

__all__ = [
    "EMBEDDING_DIM",
    # Embedder
    "Embedder",
    "HashEmbedder",
    # Indexing
    "Document",
    "Indexer",
    # Retrieval
    "RetrievalResult",
    "Retriever",
    "reciprocal_rank_fusion",
    # Reranking (#2)
    "Candidate",
    "CohereReranker",
    "LexicalOverlapReranker",
    "RerankDelta",
    "Reranker",
    "ScoredCandidate",
    "rerank_delta_ndcg",
]
