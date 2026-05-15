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
