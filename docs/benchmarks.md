# Benchmarks

*All numbers in this file are real measurements with reproducible scripts. Never fabricated.*

## Status

Where each number lives, or why there is none. This table used to mark all
five rows **pending** against #2, #6 and #7 — all three closed — while the
README published the eval baselines (#233).

| Metric                                | Where it lives                                                                 | Value |
| ------------------------------------- | ------------------------------------------------------------------------------ | ----- |
| Recall@5 on the eval golden set       | `evals/baselines/recall_at_5.json` (n=8, synthetic `rag-qa-v0.1`, every PR)     | 1.00  |
| End-to-end answer faithfulness        | `evals/baselines/faithfulness.json` (n=8, synthetic `rag-qa-v0.1`, every PR)    | 1.00  |
| Answer correctness                    | `evals/baselines/correctness.json` (n=8, synthetic `rag-qa-v0.1`, every PR)     | 0.90  |
| Retrieval latency p50 / p95 / p99     | captured per request by the telemetry layer (#6); no headline number, by design | —     |
| Cost per request                      | captured per request by the telemetry layer (#6); no headline number, by design | —     |
| Pipeline overhead (streaming)         | [Streaming pipeline](#streaming-pipeline-5) below                               | < 0.15 ms p95 |
| Reranker quality lift over fused-only | [Reranker lift](#reranker-lift-234) below — lexical stand-in, synthetic fixture  | +0.125 recall@3 |

The three eval rows are the deterministic CI fixture — an 8-example synthetic
golden set over a 10-chunk in-memory corpus with the dep-free
`TemplateGenerator` — not a held-out benchmark on real data. Real-LLM,
real-pgvector runs are operator-triggered (see the README), and the latency
and cost rows are the operator's own telemetry store rather than a number
this file could honestly quote.

## Reranker lift (#234)

How much `LexicalOverlapReranker` — the dep-free stand-in the pipeline uses in
CI — improves recall@k over the retriever's own order. It scores query-token
*coverage* where the retriever scores overlap *density*, and that difference is
all the lift can come from. **This is the lift of that stand-in on synthetic
fixtures, not a claim about a cross-encoder**; a `CohereReranker` number needs
an API key and is not measured here.

Reproduce:

```bash
python -m scripts.bench_reranker --output md
```

<!-- bench-reranker:table:begin -->
Over-fetch 10 candidates from the in-memory token-overlap retriever, then take the top k in the retriever's order (**fused-only**) or after `LexicalOverlapReranker` (**reranked**).

| fixture | k | fused-only recall@k | reranked recall@k | Δ | improved / regressed |
| ------- | -: | ------------------: | ----------------: | -: | -------------------- |
| multi-hop | 1 | 0.438 | 0.500 | +0.062 | 1 / 0 |
| multi-hop | 3 | 0.625 | 0.750 | +0.125 | 2 / 0 |
| multi-hop | 5 | 0.875 | 0.875 | +0.000 | 0 / 0 |
| eval golden set | 1 | 1.000 | 1.000 | +0.000 | 0 / 0 |
| eval golden set | 3 | 1.000 | 1.000 | +0.000 | 0 / 0 |
| eval golden set | 5 | 1.000 | 1.000 | +0.000 | 0 / 0 |

- **multi-hop**: 18 chunks, 8 questions (scripts/bench_rewriter.py).
- **eval golden set**: 10 chunks, 8 questions (evals/dataset, rag-qa-v0.1). Fused-only recall is already 1.000 at every k, so this fixture cannot show an improvement; a regression would still show.
<!-- bench-reranker:table:end -->

## Streaming pipeline (#5)

Pure-pipeline overhead: how much time `StreamingPipeline` adds on top
of the components it composes. The benchmark uses an in-memory
retriever and the dep-free `LexicalOverlapReranker`, so the numbers
isolate **pipeline-side** cost (event allocation, dataclass
instantiation, generator yield) from Postgres-side cost. Production
end-to-end p50/p95 will be dominated by the retriever's DB roundtrip,
not by this overhead.

Reproduce:

```bash
python -m scripts.bench_streaming --n 1000 --k 3
```

| phase       | n    | p50 (ms) | p95 (ms) |
| ----------- | ---- | -------- | -------- |
| retrieving  | 1000 | 0.06     | 0.07     |
| reranking   | 1000 | 0.04     | 0.05     |
| generating  | 1000 | 0.01     | 0.01     |
| **total**   | 1000 | **0.11** | **0.14** |

Run: 2026-05-16, Apple Silicon (arm64), Python 3.14.0. Throughput
~8.5 k queries/s in the same configuration. End-to-end **production**
latency (against a real PG + Anthropic SDK) is captured per request by
the telemetry layer (#6) — these numbers are *only* the pipeline plumbing,
by design.
