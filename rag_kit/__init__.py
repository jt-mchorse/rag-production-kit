"""rag-production-kit: hybrid retrieval (BM25 + pgvector) with RRF fusion + cross-encoder reranking + SSE streaming + cited generation + query rewriting + cost telemetry.

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
        # Query rewriting / decomposition (#3):
        TemplateRewriter, AnthropicRewriter, Rewriter, RewriteResult,
        # Cost telemetry (#6):
        CostRecord, ModelPrice, PriceTable, TelemetryStore, Aggregate,
    )

Layers shipped:
- #1: Schema in infra/postgres/init.sql (documents table with tsvector + vector).
- #1: Embedder protocol + HashEmbedder reference implementation.
- #1: Indexer.add_documents — persist text + tsvector + dense embedding.
- #1: Retriever.search — run FTS + ANN in parallel, fuse with RRF.
- #2: Reranker protocol + LexicalOverlapReranker (dep-free) + CohereReranker
      (production). Wired into Retriever.search(reranker=...).
- #3: Rewriter protocol + TemplateRewriter (dep-free) + AnthropicRewriter
      (production). Pre-retrieval decomposition into 1..K sub-queries.
      Wired into Retriever.search(rewriter=...); multi-sub-query rewrites
      RRF-fuse rankings across sub-queries.
- #4: Generator protocol + TemplateGenerator (dep-free) + AnthropicGenerator
      (production). Citation enforcement and weak-context refusal both
      via structured outputs (GeneratedAnswer | Refusal).
- #5: StreamingPipeline — sync-generator pipeline that emits typed phase
      events (retrieving / retrieved / reranking / reranked / generating /
      token / generated / done / error); `to_sse()` wire-formats for SSE.
- #6: Cost telemetry — CostRecord per request (tokens, USD, latency,
      per-phase timings), TelemetryStore (stdlib sqlite3, 24-hour window),
      PriceTable (operator-supplied, no defaults shipped — D-015),
      Aggregate (p50/p95/p99). Dashboard via
      `scripts/telemetry_dashboard.py`.

Layers in later PRs:
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
from .rewriter import (
    AnthropicRewriter,
    Rewriter,
    RewriteResult,
    TemplateRewriter,
)
from .streaming import (
    EventType,
    PhaseTimings,
    RetrieverLike,
    StreamEvent,
    StreamingPipeline,
    TokenStream,
    to_sse,
)
from .telemetry import (
    Aggregate,
    CostRecord,
    ModelPrice,
    PriceTable,
    TelemetryStore,
    UnknownModelError,
)
from .telemetry import (
    aggregate as aggregate_telemetry,
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
    # Rewriting (#3)
    "AnthropicRewriter",
    "RewriteResult",
    "Rewriter",
    "TemplateRewriter",
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
    # Cost telemetry (#6)
    "Aggregate",
    "CostRecord",
    "ModelPrice",
    "PriceTable",
    "TelemetryStore",
    "UnknownModelError",
    "aggregate_telemetry",
]
