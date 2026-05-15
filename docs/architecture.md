# Architecture

## Shipped (issues #1, #2)

The hybrid retrieval path plus the optional cross-encoder reranker.
One Postgres container, two indexes, one fusion step, one optional
rerank backend.

```mermaid
flowchart LR
    classDef shipped fill:#dcffe4,stroke:#22863a,color:#000
    classDef pending fill:#fff5b4,stroke:#c69400,color:#000

    Corpus["Source corpus<br/>(text + external_id)"]:::shipped --> Embed1["Embedder<br/>(protocol; HashEmbedder default)"]:::shipped
    Embed1 --> PG[(Postgres + pgvector<br/>documents.tsv (GIN)<br/>documents.embedding (HNSW))]:::shipped
    Corpus --> PG

    Query["Query text"]:::shipped --> Embed2["Embedder.embed(query)"]:::shipped
    Embed2 --> Dense["Dense channel<br/>embedding &lt;=&gt; query<br/>top-k"]:::shipped
    Query --> Lex["Lexical channel<br/>plainto_tsquery + ts_rank_cd<br/>top-k"]:::shipped
    PG --> Dense
    PG --> Lex

    Dense --> RRF["Reciprocal Rank Fusion<br/>(k=60 default)"]:::shipped
    Lex --> RRF
    RRF --> Rerank["Cross-encoder reranker<br/>(#2; opt-in)<br/>LexicalOverlap (default) | Cohere"]:::shipped
    Rerank --> Out["RetrievalResult[]<br/>fused_score + per-method ranks<br/>+ rerank_score + rerank_rank"]:::shipped

    Out --> Cite["Citation enforcement<br/>+ weak-context refusal (#4)"]:::pending
    Cite --> Stream["Streaming intermediate events<br/>(#5)"]:::pending
    Stream --> Answer["Answer + citations"]:::pending
```

## Reranking layer (this PR — issue #2)

The reranker is an opt-in step that runs after RRF fusion and re-orders
the top candidates by a higher-quality relevance signal. The contract
is intentionally narrow:

```python
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> list[ScoredCandidate]: ...
```

Two backends ship:

- **`LexicalOverlapReranker`** — dep-free fallback. Token-overlap
  heuristic with a small length-penalty tiebreaker. Not "good"; just
  deterministic and hermetic, so CI exercises the rerank flow without
  external services.
- **`CohereReranker`** — production binding, lazy-imports the `cohere`
  SDK (`pip install 'rag-production-kit[rerank-cohere]'`). Configurable
  model id (default `rerank-english-v3.0`), batch size, timeout. Reads
  `COHERE_API_KEY`.

`Retriever.search(query, k, reranker=...)` is backwards-compatible:
default behavior (no reranker passed) returns the RRF-fused top-k
unchanged. When a reranker is passed, the retriever over-fetches by
`_CANDIDATE_MULTIPLIER` so the reranker has more candidates than `k`
to choose from, then truncates to `k` after reranking. Each result's
`rerank_score` and `rerank_rank` are populated; the original
`fused_score` and per-method `ranks` are preserved for telemetry.

`rerank_delta_ndcg(before, after, k=...)` computes how much the
reranker actually moved the input. Use it to track rerank-delta over
time as model versions change.

## Schema (infra/postgres/init.sql)

A single corpus table with text, an auto-maintained `tsvector` (English
config in v0.1), and a `vector(64)` embedding column.

- **Lexical index:** GIN on `tsv`, populated by a trigger
  (`documents_tsv_trigger`) that runs on `INSERT` and on `UPDATE OF text`.
- **Dense index:** HNSW on `embedding` with `vector_cosine_ops`, default
  parameters `m=16, ef_construction=64`. Tuning lives in
  [vector-search-at-scale][vssa].

[vssa]: https://github.com/jt-mchorse/vector-search-at-scale

The embedding dimension is **64 in v0.1**, matched to the `HashEmbedder`
reference. A deployment swapping in a real embedder (Voyage, Cohere,
OpenAI, BGE, …) changes the `vector(N)` column dimension and the default
in `rag_kit.embedder.EMBEDDING_DIM` together. See **D-003**.

## Fusion (rag_kit/fusion.py)

```
score(doc) = Σ over methods m where doc appears:  1 / (k + rank_m(doc))
```

with k=60 by default (the value from the original RRF paper). The fused
result returns per-method ranks alongside the score so consumers can
debug *which channel* surfaced a doc.

## Pending

- Cross-encoder reranking (#2)
- Query rewriting and decomposition (#3)
- Citation enforcement, weak-context refusal (#4)
- Streaming intermediate events (#5)
- Cost telemetry (#6)
- Eval harness integration + Recall@5 (#7)
- Next.js demo frontend with inline citations (#8)
