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

**Citation enforcement and weak-context refusal** are the answer
layer. A `Generator` takes retrieved chunks and produces either a
`GeneratedAnswer` (every claim sentence cites a retrieved chunk via
inline `[cite:<external_id>]` markers, validated structurally) or a
`Refusal` with a machine-readable `reason`. The refusal path fires
*before* the LLM is called when retrieval is weak (top score below
the caller's `threshold`) and *after* generation if the model's output
can't be reconciled with the retrieved chunks. Two generators ship,
mirroring the reranker pattern: a dep-free `TemplateGenerator` for
hermetic CI, and an `AnthropicGenerator` behind the `[rag-anthropic]`
extra.

**Streaming intermediate events** turn the pipeline into a typed event
stream. `StreamingPipeline.run(query, k)` is a sync generator that
yields a `StreamEvent` at every phase boundary (`retrieving` /
`retrieved` / `reranking` / `reranked` / `generating` / `token` /
`generated` / `done`, plus `error` if anything throws). `to_sse(event)`
serializes each one as a Server-Sent Events frame for a browser
frontend; the demo under `demo/streaming/` is a single-file stdlib HTTP
server that wires the whole thing up — zero web-framework deps. The
generator phase is a `TokenStream` protocol seam: any callable that
yields strings plugs in, so the same pipeline works with the #4
generator, an Anthropic SDK stream, or an offline stub for eval runs.

Everything beyond #1 + #2 + #4 + #5 is staged in follow-up issues:
query rewriting and decomposition ([#3]), cost telemetry ([#6]), and
eval harness integration with faithfulness measurement against the
citation contract plus a Recall@5 number against a real corpus
([#7]). The eval harness lives in its own repo ([llm-eval-harness])
and is imported, not vendored.

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

### Generation with citations

```python
from rag_kit import GeneratedAnswer, Refusal, TemplateGenerator

# `retrieved` is the list returned by Retriever.search above.
gen = TemplateGenerator()
out = gen.generate("when do refunds expire?", retrieved, threshold=0.05)
if isinstance(out, GeneratedAnswer):
    print(out.text)                           # "...[cite:doc-1]. ...[cite:doc-2]."
    for c in out.citations:
        print(f"  cite={c.external_id}: {c.text}")
else:                                         # isinstance(out, Refusal)
    print(f"REFUSED: {out.reason} ({out.detail})")
```

Swap to the production generator with `AnthropicGenerator()` (requires
`pip install -e '.[rag-anthropic]'` and `ANTHROPIC_API_KEY`). Citation
enforcement is also exposed standalone as `enforce_citations(text, retrieved)`
so alternative generators (or a downstream eval harness; see #7) can reuse it.

Run the test suite:

```bash
pytest -m "not pg"     # unit tests (embedder, fusion math, streaming) — no DB needed
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest -m pg   # integration
```

### Streaming (issue #5)

Drop in your own retriever (or use `Retriever` against PG) and any
callable that yields tokens; everything else is the same:

```python
from rag_kit import LexicalOverlapReranker, StreamingPipeline, to_sse

pipe = StreamingPipeline(
    retriever,                                # anything with .search(query, k, *, reranker)
    reranker=LexicalOverlapReranker(),        # or CohereReranker, or None
    token_stream=my_token_callable,           # or None to skip generation
)

for event in pipe.run("how do I tune shared_buffers?", k=3):
    print(to_sse(event), end="")              # SSE wire format, or use event.type/payload directly
```

Try the included demo locally:

```bash
python -m demo.streaming.server   # in-memory corpus, no Postgres needed
open http://127.0.0.1:8765/
```

## Benchmarks / Results

The streaming pipeline carries **< 0.15 ms p95 overhead** end-to-end on
an in-memory corpus (1000 queries, M-series Mac, Python 3.14) — see
[`docs/benchmarks.md`](docs/benchmarks.md). Production end-to-end p50/p95
on Postgres is tracked under [#6] (cost telemetry & latency).

Three eval suites run on every PR against the synthetic `rag-qa-v0.1`
golden set (8 examples, 10-chunk in-memory corpus, dep-free
`TemplateGenerator`). First baseline (n=8, deterministic):

| suite          | mean score | reproducer                          |
| -------------- | ---------: | ----------------------------------- |
| faithfulness   |       1.00 | `python -m evals.run_eval`          |
| recall_at_5    |       1.00 | `python -m evals.run_eval`          |
| correctness    |       0.90 | `python -m evals.run_eval`          |

Real-LLM eval runs (Anthropic-backed `Generator`, real pgvector
retrieval) are operator-triggered with `ANTHROPIC_API_KEY` + `DATABASE_URL`
locally — the CI fixture path covers regressions in the deterministic
pipeline and keeps the workflow API-key-free.

## Demo

`demo/streaming/` — single-file stdlib server + HTML client that
renders the SSE stream live. 60-second video pending; the demo is
runnable today as documented above.

## Why these decisions

See [`MEMORY/core_decisions_human.md`](MEMORY/core_decisions_human.md).

## License

MIT
