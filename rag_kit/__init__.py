"""rag-production-kit: hybrid retrieval (BM25 + pgvector) with RRF fusion.

Issue #1 surface:

    from rag_kit import Document, HashEmbedder, Indexer, Retriever, reciprocal_rank_fusion

Layers shipped here (#1):
- Schema in infra/postgres/init.sql (documents table with tsvector + vector).
- Embedder protocol + HashEmbedder reference implementation.
- Indexer.add_documents — persist text + tsvector + dense embedding.
- Retriever.search — run FTS + ANN in parallel, fuse with RRF.

Layers in later PRs:
- Cross-encoder reranking (#2)
- Citation enforcement and refusal on weak context (#4)
- Streaming intermediate events (#5)
- Eval harness integration + Recall@5 measurement (#7)
"""

from .embedder import EMBEDDING_DIM, Embedder, HashEmbedder
from .fusion import reciprocal_rank_fusion
from .indexer import Document, Indexer
from .retriever import RetrievalResult, Retriever

__all__ = [
    "EMBEDDING_DIM",
    "Document",
    "Embedder",
    "HashEmbedder",
    "Indexer",
    "RetrievalResult",
    "Retriever",
    "reciprocal_rank_fusion",
]
