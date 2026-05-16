"""rag-production-kit: hybrid retrieval (BM25 + pgvector) with RRF fusion + cross-encoder reranking + SSE streaming + cited generation.

Public surface:

    from rag_kit import (
        Document, HashEmbedder, Indexer, Retriever, reciprocal_rank_fusion,
        # Reranking (#2):
        LexicalOverlapReranker, CohereReranker, Candidate, ScoredCandidate,
        rerank_delta_ndcg,
        # Generation with citations (#4):
        TemplateGenerator, AnthropicGenerator, GeneratedAnswer, Refusal,
        Citation, enforce_citations,
        # Streaming (#5):
        StreamEvent, StreamingPipeline, PhaseTimings, TokenStream, to_sse,
    )

Layers shipped:
- #1: Schema in infra/postgres/init.sql (documents table with tsvector + vector).
- #1: Embedder protocol + HashEmbedder reference implementation.
- #1: Indexer.add_documents — persist text + tsvector + dense embedding.
- #1: Retriever.search — run FTS + ANN in parallel, fuse with RRF.
- #2: Reranker protocol + LexicalOverlapReranker (dep-free) + CohereReranker
      (production). Wired into Retriever.search(reranker=...).
- #4: Generator protocol + TemplateGenerator (dep-free) + AnthropicGenerator
      (production). Citation enforcement and weak-context refusal both
      via structured outputs (GeneratedAnswer | Refusal).
- #5: StreamingPipeline — sync-generator pipeline that emits typed phase
      events (retrieving / retrieved / reranking / reranked / generating /
      token / generated / done / error); `to_sse()` wire-formats for SSE.

Layers in later PRs:
- Cost telemetry (#6)
- Eval harness integration + faithfulness measurement (#7)
"""

from .embedder import EMBEDDING_DIM, Embedder, HashEmbedder
from .fusion import reciprocal_rank_fusion
from .generator import (
    AnthropicGenerator,
    Citation,
    CitationError,
    GeneratedAnswer,
    Generator,
    Refusal,
    TemplateGenerator,
    enforce_citations,
    split_sentences,
)
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
from .streaming import (
    EventType,
    PhaseTimings,
    RetrieverLike,
    StreamEvent,
    StreamingPipeline,
    TokenStream,
    to_sse,
)

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
    # Generation (#4)
    "AnthropicGenerator",
    "Citation",
    "CitationError",
    "GeneratedAnswer",
    "Generator",
    "Refusal",
    "TemplateGenerator",
    "enforce_citations",
    "split_sentences",
    # Streaming (#5)
    "EventType",
    "PhaseTimings",
    "RetrieverLike",
    "StreamEvent",
    "StreamingPipeline",
    "TokenStream",
    "to_sse",
]
