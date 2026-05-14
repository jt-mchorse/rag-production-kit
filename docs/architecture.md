# Architecture

## Shipped (this PR — issue #1)

The hybrid retrieval path. One Postgres container, two indexes, one
fusion step.

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
    RRF --> Out["RetrievalResult[]<br/>fused_score + per-method ranks"]:::shipped

    Out --> Rerank["Cross-encoder reranking<br/>(#2)"]:::pending
    Rerank --> Cite["Citation enforcement<br/>+ weak-context refusal (#4)"]:::pending
    Cite --> Stream["Streaming intermediate events<br/>(#5)"]:::pending
    Stream --> Answer["Answer + citations"]:::pending
```

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
