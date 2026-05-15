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
