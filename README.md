# rag-production-kit
> Production-grade RAG reference: hybrid retrieval (BM25 + pgvector), cross-encoder reranking, citation enforcement, streaming, cost telemetry, eval suite.

![CI](https://github.com/jt-mchorse/rag-production-kit/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## What this is

The most common failure mode of a "we want to do RAG" project isn't the
LLM — it's the retrieval. Dense embeddings miss exact-string queries
(account IDs, product codes, error messages), and lexical search misses
paraphrases. `rag-production-kit` is the reference implementation of the
retrieval, reranking, and answer-grounding patterns you actually need
when you ship to production — wired against Postgres + pgvector so
there's one container to bring up, not a fleet.

**Hybrid retrieval** is the foundation: documents land in a single
`documents` table with two indexes — GIN over a `tsvector` for lexical
(BM25-style) and HNSW over a `vector` column for dense — and the
retriever runs both per query, fusing with [Reciprocal Rank Fusion]
(Cormack et al., SIGIR 2009). The fused result surfaces per-method ranks
alongside the score so you can see whether a chunk came in lexically,
densely, or both — invaluable when you're tuning a corpus.

**Cross-encoder reranking** sits on top, opt-in via
`Retriever.search(reranker=...)`. Two backends ship: a dep-free
`LexicalOverlapReranker` (so CI exercises the rerank flow without
external services) and a `CohereReranker` (production binding behind
the `[rerank-cohere]` extra). Each result carries the reranker's
score and the new rank alongside the original `fused_score` and
per-method ranks; `rerank_delta_ndcg(before, after)` quantifies how
much the reranker actually moved things, for telemetry.

Everything beyond #1 + #2 is staged in follow-up issues: query
rewriting and decomposition ([#3]), citation enforcement with
weak-context refusal ([#4]), streaming intermediate events ([#5]), cost
telemetry ([#6]), and eval harness integration with a Recall@5 number
against a real corpus ([#7]). The eval harness lives in its own repo
([llm-eval-harness]) and is imported, not vendored.

[Reciprocal Rank Fusion]: https://dl.acm.org/doi/10.1145/1571941.1572114
[#2]: https://github.com/jt-mchorse/rag-production-kit/issues/2
[#3]: https://github.com/jt-mchorse/rag-production-kit/issues/3
[#4]: https://github.com/jt-mchorse/rag-production-kit/issues/4
[#5]: https://github.com/jt-mchorse/rag-production-kit/issues/5
[#6]: https://github.com/jt-mchorse/rag-production-kit/issues/6
[#7]: https://github.com/jt-mchorse/rag-production-kit/issues/7
[llm-eval-harness]: https://github.com/jt-mchorse/llm-eval-harness

## Architecture

See [`docs/architecture.md`](docs/architecture.md). The high-level
shape of the shipped layer:

```
                                  ┌─────────────────────────┐
                                  │  Postgres + pgvector    │
   ┌─────────────┐  embed text    │  documents(             │
   │   Indexer   │ ───────────▶   │    text, tsv (GIN),     │
   └─────────────┘                │    embedding (HNSW)     │
                                  │  )                      │
   ┌─────────────┐  query +       │                         │
   │  Retriever  │  embed query   │  · ts_rank_cd top-k     │
   └─────────────┘ ◀───────────── │  · embedding <=> top-k  │
          │   RRF fuse            └─────────────────────────┘
          ▼
   ┌─────────────────────────────────────────────────────────┐
   │  RetrievalResult[]                                      │
   │  (external_id, text, fused_score, ranks{lex, dense})    │
   └─────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# 1. Bring up Postgres + pgvector
docker compose up -d
# (waits for pg_isready; init.sql is mounted to /docker-entrypoint-initdb.d/)

# 2. Install the package
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

```python
from rag_kit import Document, HashEmbedder, Indexer, Retriever
from rag_kit.db import connect

embedder = HashEmbedder()  # swap for your real embedder
with connect() as conn:    # reads DATABASE_URL or falls back to the compose service
    indexer = Indexer(conn, embedder)
    indexer.add_documents([
        Document("doc-1", "Our refund policy gives Pro customers 14 days."),
        Document("doc-2", "If a subscriber wants their money back, the window is two weeks."),
        Document("doc-3", "Standard shipping ships within three business days."),
    ])

    retr = Retriever(conn, embedder)
    for r in retr.search("refund window", k=3):
        print(f"{r.external_id:10}  fused={r.fused_score:.4f}  ranks={r.ranks}")
        print(f"  {r.text}")
```

Run the test suite:

```bash
pytest -m "not pg"     # unit tests (embedder, fusion math) — no DB needed
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest -m pg   # integration
```

## Benchmarks / Results

*Recall@5 measurement is pending issue [#7] (eval-harness integration
against a real corpus + held-out query set). The fused-retrieval API is
shipped and tested in this PR; the empirical quality number comes after
the eval harness lands here.*

## Demo

*60-second demo pending — depends on the streaming response layer ([#5]).*

## Why these decisions

See [`MEMORY/core_decisions_human.md`](MEMORY/core_decisions_human.md).

## License

MIT
