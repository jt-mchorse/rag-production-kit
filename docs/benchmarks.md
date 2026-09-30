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
| Reranker quality lift over fused-only | **pending** [#234] — not measured anywhere yet                                  | —     |

[#234]: https://github.com/jt-mchorse/rag-production-kit/issues/234

The three eval rows are the deterministic CI fixture — an 8-example synthetic
golden set over a 10-chunk in-memory corpus with the dep-free
`TemplateGenerator` — not a held-out benchmark on real data. Real-LLM,
real-pgvector runs are operator-triggered (see the README), and the latency
and cost rows are the operator's own telemetry store rather than a number
this file could honestly quote.

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
