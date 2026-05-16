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

**Query rewriting and decomposition** is the pre-retrieval seam,
opt-in via `Retriever.search(rewriter=...)`. A `Rewriter` takes the raw
user query and returns 1..K sub-queries plus a short reasoning string;
when the rewriter returns ≥2 sub-queries, the retriever runs hybrid
search per sub-query and fuses the rankings with RRF across sub-queries
(the per-method ranks dict on each returned result then carries
`subquery_0`, `subquery_1`, … instead of `lexical` / `dense`). Two
backends ship — a dep-free `TemplateRewriter` that handles common
multi-hop patterns (`Compare X with Y`, `A. Then B.`, multi-question
conjunctions like `Who is X and where did Y work?`) for hermetic CI,
and an `AnthropicRewriter` behind the existing `[rag-anthropic]` extra
that asks Claude to decompose the query and validates the JSON
response. The kwarg defaults to `None`, so every existing caller keeps
its exact single-query behavior.

**Cost telemetry** captures per-request token counts, USD cost, and
per-phase latency to a stdlib SQLite file. The `PriceTable` is
operator-supplied — **no default prices ship** (D-015), same posture
as the no-fabricated-benchmarks rule extended to pricing: querying a
model that isn't in the table raises `UnknownModelError` rather than
silently emitting `$0.00`. A dep-free local dashboard
(`scripts/telemetry_dashboard.py`, stdlib `http.server` + inline SVG)
renders the last 24 hours with p50 / p95 / p99 latency, total USD,
and the most recent 20 records — works air-gapped.

Everything beyond #1 + #2 + #3 + #4 + #5 + #6 is staged in follow-up
issues: eval harness integration with faithfulness measurement against
the citation contract plus a Recall@5 number against a real corpus
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

### Query rewriting / decomposition (issue #3)

Multi-hop questions (e.g., `Who founded Anthropic and where did they
work before?`) retrieve better when split into independent sub-queries.
Pass a `Rewriter` to `Retriever.search`; when it decomposes (≥2
sub-queries), the retriever runs hybrid search per sub-query and fuses
the rankings with RRF across sub-queries. When the rewriter chooses
not to decompose (1 sub-query), the call collapses back to the existing
single-shot path.

```python
from rag_kit import Retriever, TemplateRewriter

results = retr.search(
    "Who founded Anthropic and where did they work before?",
    k=5,
    rewriter=TemplateRewriter(),     # or AnthropicRewriter() in production
)
for r in results:
    # per-method ranks become per-sub-query ranks when the rewriter expanded
    print(r.external_id, r.ranks)    # e.g., {'subquery_0': 1, 'subquery_1': 2}
```

The `TemplateRewriter` is dep-free and deterministic — its decomposition
patterns are listed in [`rag_kit/rewriter.py`](rag_kit/rewriter.py). The
`AnthropicRewriter` asks Claude to return strict JSON
(`{"sub_queries": [...], "reasoning": "..."}`) and validates the
response; install with `pip install -e '.[rag-anthropic]'`.

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

### Cost telemetry (issue #6)

Per-request capture of token counts, USD cost, and per-phase latency,
persisted to a stdlib SQLite file (`telemetry.db`) and visualized via a
dep-free local HTTP dashboard.

```python
from rag_kit import CostRecord, ModelPrice, PriceTable, TelemetryStore

# Prices are operator-supplied — nothing ships in the table (D-015).
prices = PriceTable({
    "claude-opus-4-7":   ModelPrice(prompt_per_million=15.0, completion_per_million=75.0),
    "claude-sonnet-4-6": ModelPrice(prompt_per_million=3.0,  completion_per_million=15.0),
})

with TelemetryStore("./telemetry.db") as store:
    rec = CostRecord.build(
        ts=None,                                   # default time.time()
        query="how do I tune shared_buffers?",
        model="claude-opus-4-7",
        retrieved_count=5,
        prompt_tokens=1200,
        completion_tokens=180,
        total_latency_ms=420.0,
        per_phase_ms={"retrieving": 50.0, "reranking": 30.0, "generating": 340.0},
        price_table=prices,
    )
    store.record(rec)                              # → row in cost_records
    # 24-hour window with p50 / p95 / p99 latency, totals, and counts:
    from rag_kit import aggregate_telemetry
    agg = aggregate_telemetry(store.last_24h())
    print(f"n={agg.n} total=${agg.total_usd} p95={agg.latency_p95_ms:.0f}ms")
```

`CostRecord` is computed from a `PriceTable`; calling `cost()` for a
model that isn't configured raises `UnknownModelError` rather than
silently emitting `$0.00` — same posture as the no-fabricated-
benchmarks rule (D-013) extended to prices (D-015).

Launch the dashboard (single-page, inline-SVG chart, no external
assets) — works air-gapped:

```bash
python -m scripts.telemetry_dashboard --db ./telemetry.db --port 8766
# open http://127.0.0.1:8766/

# or seed deterministic synthetic records on a fresh DB:
python -m scripts.telemetry_dashboard --db ./telemetry.db --seed 60 --port 8766
```

Endpoints:

- `GET /` → HTML dashboard (24-hour window, 4 stat cards, latency chart,
  most recent 20 records).
- `GET /api/last_24h` → JSON snapshot of the same window for scripts /
  downstream tooling.

## Benchmarks / Results

The streaming pipeline carries **< 0.15 ms p95 overhead** end-to-end on
an in-memory corpus (1000 queries, M-series Mac, Python 3.14) — see
[`docs/benchmarks.md`](docs/benchmarks.md). Production end-to-end p50/p95
on Postgres is captured per-request by the cost-telemetry layer ([#6],
section above) — there is no fabricated headline number; the operator's
own SQLite store is the source of truth.

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

**Rewriter recall@k on a synthetic multi-hop fixture** (issue #3,
`scripts/bench_rewriter.py`, 18-chunk corpus, 8 multi-hop questions,
dep-free `TemplateRewriter` against the same in-memory token-overlap
retriever the eval suite uses). Real, reproducible numbers; not LLM-backed:

| k | mean recall (baseline) | mean recall (rewriter) | improvements / regressions |
|---|---:|---:|---:|
| 2 | 0.625 | 0.688 | 1 / 0 |
| 3 | 0.625 | 0.812 | 3 / 0 |
| 5 | 0.875 | 0.938 | 1 / 0 |

The gap is largest at k=3, where the candidate budget is tight relative
to the two distinct facts each multi-hop query needs. No query regresses
on this fixture; running it on your own corpus is two lines:
`python -m scripts.bench_rewriter --k 3 --output md`.
`AnthropicRewriter` numbers are pending an operator-triggered run (the
script wires it but the bench fixture above is `TemplateRewriter`-only
so CI stays API-key-free).

## Demo

`demo/streaming/` — single-file stdlib server + HTML client that
renders the SSE stream live. 60-second video pending; the demo is
runnable today as documented above.

## Why these decisions

See [`MEMORY/core_decisions_human.md`](MEMORY/core_decisions_human.md).

## License

MIT
