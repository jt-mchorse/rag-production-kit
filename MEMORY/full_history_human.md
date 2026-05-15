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

## 2026-05-15 — Issue #4: Citation enforcement and weak-context refusal
**Duration:** ~60 min · **Branch:** `session/2026-05-15-1915-issue-4`

- Shipped `rag_kit/generator.py`: `Generator` Protocol with one method (D-008), `Citation` / `GeneratedAnswer` / `Refusal` dataclasses, `TemplateGenerator` (dep-free, deterministic, hermetic-CI rationale matches D-006's `LexicalOverlapReranker`), and `AnthropicGenerator` (production binding behind the new `[rag-anthropic]` extra, lazy-imported with both real-SDK and dict-shaped content-block handling so unit tests can use simple fakes).
- Two distinct refusal paths (D-009): the `threshold` check fires *before* the LLM is called when retrieval is weak (`top_score < threshold`, using `rerank_score` if present else `fused_score`) and the citation-enforcement check fires *after* generation if the model's output cannot be reconciled with the retrieved chunks. Both return `Refusal` with a machine-readable `reason` (`insufficient_context` / `unparseable_output`).
- `enforce_citations(text, retrieved)` is exposed as a free function so alternative generators (or the downstream eval harness in #7) can reuse it. It splits text on sentence-terminal punctuation and rejects sentences without `[cite:...]` markers as well as dangling references to ids not in `retrieved`.
- 18 new hermetic tests (`test_generator.py`) covering: sentence split edge cases, citation enforcement happy path + missing marker + dangling id + empty text + dedup, `TemplateGenerator` happy path + threshold refusal + empty refusal + rerank-score preference + `max_chunks` limit + invalid constructor, `AnthropicGenerator` validated response + pre-LLM threshold refusal + citation-error to refusal + explicit REFUSE marker + dict-shaped content blocks.
- Updated `rag_kit/__init__.py` exports and the public-surface docstring; added a "Generation with citations" subsection to the README quickstart and a paragraph in "What this is" explaining the two-refusal-path design.
- 51/51 hermetic tests pass; 7 pg-integration tests skipped as before (no DATABASE_URL).

**Why this work, this session:** Reranked candidates without a generator with citation enforcement is half the story — production RAG either grounds every claim or refuses. The Protocol + extras shape matches the patterns this repo already uses for Embedder (D-002) and Reranker (D-005), so adoption cost is zero.

**Open questions / blockers:** Acceptance criterion 3 ("faithfulness measured via `llm-eval-harness` in CI") deferred to issue #7. The recorded D-008 contract (Generator Protocol with `[rag-anthropic]` extra) is the seam the eval harness will score against. Real Anthropic-API calibration requires `ANTHROPIC_API_KEY` + budget the operator runs.

**Next session:** Either issue #5 (streaming intermediate events — depends on a working generator, which now exists) or issue #6 (cost telemetry — also depends on the generator). Issue #4 stays open with the eval-harness checkbox pending #7.

## 2026-05-16 — Issue #5: Streaming intermediate events (SSE)
**Duration:** ~75 min · **Branch:** `session/2026-05-16-0309-issue-5`

- Shipped `rag_kit/streaming.py`: `StreamEvent` dataclass with a typed `EventType` literal, `StreamingPipeline` (sync generator wrapping retriever + optional reranker + optional `TokenStream`), `to_sse()` SSE wire-formatter, and `PhaseTimings` with dep-free linear-interp percentiles for benchmarking. Errors anywhere become a final `error` event rather than raising out, so an SSE consumer always sees a clean terminal frame.
- 23 hermetic tests covering event ordering across every combination of optional phases, the SSE wire format (event line + JSON data + Unicode roundtrip + unjsonifiable fallback), the error path on both retrieval and rerank failures, input validation, and `PhaseTimings` percentile math (NIST type-7 known values, empty-handling, edge-clamping). Full suite: 56 hermetic pass, 7 pg-integration skipped as before.
- `demo/streaming/` runs the whole thing without Postgres: `server.py` (stdlib `http.server`, no FastAPI) plus `index.html` + `app.js` (vanilla JS consuming the SSE stream via `fetch` + `TextDecoder` + frame parser, rendering each phase as a card with elapsed-ms badges). Smoke-tested locally — frames arrive as expected.
- `scripts/bench_streaming.py` drives N synthetic queries through the pipeline against an in-memory corpus and prints a p50/p95 table per phase. Real measured numbers (n=1000, M-series Mac, Python 3.14.0): retrieving p95 = 0.07 ms, reranking p95 = 0.05 ms, generating p95 = 0.01 ms, total p95 = 0.14 ms, ~8.5 k q/s. Written into `docs/benchmarks.md` with the date and host so they're reproducible and not fabricated.
- README "What this is" expands to cover the streaming layer; new Quickstart subsection shows the 6-line streaming snippet and how to run the demo. `docs/architecture.md` mermaid updated: `#5` moves to shipped, with a dedicated streaming-layer subsection below the diagram.
- Two decisions: D-010 (sync generator, not asyncio — retriever and reranker are sync; coloring `async` would force a propagation tax for no concurrency win at the pipeline layer) and D-011 (demo server is stdlib `http.server`, not FastAPI — keeps the base install dep-free per D-002).

**Why this work, this session:** Streaming was the next unblocked priority:high issue, independent of the issue-#4 generator work (the `TokenStream` Protocol is the seam where that generator plugs in). Also unblocks the 60-second demo deliverable, which previously cited #5 as the gating dep.

**Open questions / blockers:** None — the pipeline ships behind a Protocol seam, so the #4 generator drops in without code changes here. Production end-to-end p50/p95 (against real PG + an LLM SDK) is tracked under #6, not duplicated here.

**Next session:** Either jump to a different repo per the night-session multi-issue loop — likely `agent-orchestration-platform` (tied for most priority:high open) or `vector-search-at-scale` (in-flight draft PR #7).
