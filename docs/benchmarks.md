# Benchmarks

*All numbers in this file are real measurements with reproducible scripts. Never fabricated.*

## Status

| Metric                              | Status        | Tracking issue |
| ----------------------------------- | ------------- | -------------- |
| Recall@5 on a held-out query set    | **pending**   | [#7]           |
| Retrieval latency p50 / p95 / p99   | **pending**   | [#6]           |
| Reranker quality lift over fused-only | **pending** | [#2]           |
| End-to-end answer faithfulness      | **pending**   | [#7]           |
| Cost per request                    | **pending**   | [#6]           |

[#2]: https://github.com/jt-mchorse/rag-production-kit/issues/2
[#6]: https://github.com/jt-mchorse/rag-production-kit/issues/6
[#7]: https://github.com/jt-mchorse/rag-production-kit/issues/7

This PR ships the hybrid-retrieval API and the SQL schema it runs on.
The Recall@5 measurement depends on the eval-harness wiring (#7), which
in turn imports [`llm-eval-harness`][leh] to evaluate against a real
corpus + held-out query set. That work is filed and tracked; the number
will appear here, with a reproducer script, when it ships.

[leh]: https://github.com/jt-mchorse/llm-eval-harness

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
latency (against a real PG + Anthropic SDK) is tracked separately under
#6 — these numbers are *only* the pipeline plumbing, by design.
