# Session History (human-readable)

Chronological log of work sessions. Most recent first below the divider.

---

## 2026-05-14 — Issue #1: hybrid retrieval (BM25 + pgvector with RRF)
**Duration:** ~70 min · **Branch:** `session/2026-05-14-1430-issue-01`

- Stood up the `rag_kit` package with the hybrid retrieval path:
  `infra/postgres/init.sql` (documents table with auto-maintained
  `tsvector` GIN index and `vector(64)` HNSW index),
  `docker-compose.yml` for one-command pgvector bring-up, and the
  Python layer (`Embedder` protocol + `HashEmbedder`, `Indexer`,
  `Retriever`, `reciprocal_rank_fusion`).
- 21 tests: 16 hermetic unit tests for the embedder + RRF math, 5
  integration tests against a real Postgres marked `pytest.mark.pg`.
  CI runs the integration tests against a GitHub Actions service
  container (`pgvector/pgvector:pg16`); unit job stays green on every
  machine.
- README replaced with real "What this is" + Quickstart that runs on
  a fresh clone (`docker compose up` + the 8-line indexer/retriever
  snippet). Benchmarks section says Recall@5 is pending issue #7
  rather than carrying a fabricated number.
- Three decisions recorded: dep discipline (D-002), per-deployment
  vector dim (D-003), RRF as fusion strategy with k=60 default (D-004).

**Why this work, this session:** Issue #1 is the foundation every later
layer (#2 reranking, #4 citation enforcement, #5 streaming, #7 eval)
will build on. The repo had been at bare scaffolding since bootstrap.

**Open questions / blockers:** Recall@5 number is intentionally deferred
to #7 (eval-harness integration). Stated honestly in the README and
the benchmarks doc rather than fabricated.

**Next session:** Issue #2 (cross-encoder reranking) — sits cleanly on
top of the `Retriever.search` output shipped here.

## 2026-05-15 — Issue #2: Cross-encoder reranking layer
**Duration:** ~60 min · **Branch:** `session/2026-05-15-1437-issue-02`

- Shipped `rag_kit/reranker.py`: `Reranker` Protocol with one method (D-005), `Candidate` + `ScoredCandidate` dataclasses, `LexicalOverlapReranker` (dep-free fallback, D-006), `CohereReranker` (production binding behind the new `[rerank-cohere]` extra, lazy-imported), and `rerank_delta_ndcg()` telemetry helper.
- Wired `Retriever.search(query, k, reranker=...)` as opt-in (D-007) so existing callers keep their hybrid-only behavior. When a reranker is passed, the retriever over-fetches by the candidate multiplier so the reranker has more candidates than `k` to choose from. `RetrievalResult` gained two optional fields (`rerank_score`, `rerank_rank`) populated only when a reranker was used.
- 18 new hermetic tests (8 LexicalOverlap + 3 Cohere stub + 6 nDCG telemetry + 1 protocol conformance) plus 2 new pg integration tests (`test_retriever_with_reranker_populates_rerank_fields`, `test_retriever_without_reranker_leaves_rerank_fields_none`). All passing locally.
- Updated `docs/architecture.md` and README to reflect the reranking layer; benchmarks honestly stay marked pending issue #7.

**Why this work, this session:** Reranking is the highest-leverage retrieval-quality lever after hybrid fusion, and the Protocol shape is the same single-method seam already adopted in `eval-harness` (Backend) and earlier in this repo (Embedder) — locking it now keeps the portfolio consistent.

**Open questions / blockers:** Acceptance criterion 3 ("Bench: recall@5 with/without reranker recorded") deferred to issue #7 (eval-harness integration) since the no-fabricated-benchmarks rule precludes guessing. Live `CohereReranker` calibration requires `COHERE_API_KEY` + budget the operator runs.

**Next session:** Issue #4 (citation enforcement and weak-context refusal) is the natural next step now that we have reranked candidates; it's the layer that turns "the right chunks" into "an answer that's grounded in them."
