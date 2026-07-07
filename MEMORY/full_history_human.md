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

## 2026-05-16 — Issue #4: Resume PR #11 (rebase + ready)
**Duration:** ~35 min · **Branch:** `session/2026-05-15-1915-issue-4` (force-pushed)

- Rebased PR #11 onto current `main` (the streaming work #5 had landed in between). 3 commits replayed; 6 files conflicted and were resolved cleanly: `README.md` "What this is" now describes #1 + #2 + #4 + #5 together; `rag_kit/__init__.py` exports both the Generator (#4) and Streaming (#5) public surfaces; MEMORY files interleave D-008/D-009 (this PR) and D-010/D-011 (already on main) in chronological order with the placeholder "reserved-for-#11" comment removed.
- Local hermetic suite 74/74 pass, ruff lint and format clean. PR #11 CI is green on the rebased head across lint + unit (3.11/3.12) + integration-pg + memory-check (`mergeState: CLEAN`).
- Edited issue #4 body to move acceptance criterion 3 ("Faithfulness measured via llm-eval-harness in CI") into issue #7, where the cross-repo eval-harness wiring actually lives — the Generator Protocol + `enforce_citations()` contract shipped here is the seam #7 will score against.
- Marked PR #11 ready for review. The next scheduled session can squash-merge per D-004.

**Why this work, this session:** PR #11 was stuck `CONFLICTING` against `main` after the streaming PR landed, blocking both the generator from shipping and the eval-harness integration in #7 (which builds on the Generator). 30 minutes of conflict resolution unblocks two issues.

**Open questions / blockers:** None. PR #11 is ready for review. Issue #4 is 100% complete; issue #7 picks up the eval-harness wiring next.

**Next session:** Issue #7 in this repo (eval-harness integration) — sits cleanly on top of the now-ready Generator Protocol.

## 2026-05-16 — Issue #7: Eval harness integration (faithfulness, recall@k, correctness)
**Duration:** ~30 min · **Branch:** `session/2026-05-16-1517-issue-7`

- Wired `llm-eval-harness` into this repo via a new `[eval]` extra (pinned to commit `2398cc3`) and a fresh `.github/workflows/eval.yml` that runs on every PR. The workflow installs the extra, regenerates current eval fixtures via `python -m evals.run_eval`, diffs each suite against its committed baseline, and posts a single composite sticky PR comment with all three deltas (D-012).
- Three suites land. `faithfulness` reuses `enforce_citations` from #4: per-row 1.0 iff the generator produced a valid `GeneratedAnswer`, 0.0 on `Refusal` or citation failure. `recall_at_5` scores fraction of `provenance.gold_chunk_ids` present in the top-5 retrieved external_ids. `correctness` is deterministic content-token overlap with the expected output as a judge-rubric proxy (the real LLM-judge path is operator-triggered with `ANTHROPIC_API_KEY`).
- Hermetic by design. An in-memory token-overlap retriever stands in for pgvector so the workflow runs with zero external services. Mirrors the fixture pattern in `llm-eval-harness`'s own `eval.yml` — committed baselines, regenerated current, diff-and-comment. First baselines from this session: faithfulness 1.0, recall@5 1.0, correctness 0.90 against the 8-row synthetic `rag-qa-v0.1` dataset.
- Corpus chunks intentionally single-sentence (D-013) so `TemplateGenerator`'s one-cite-per-sentence shape satisfies `enforce_citations` without an LLM. The real-LLM eval path (operator-triggered) will switch to `AnthropicGenerator` against arbitrary-shape chunks.
- 13 new hermetic tests in `tests/test_eval_run.py` exercise the tokenizer, in-memory retriever, three per-suite scorers, JSON write/read, sticky-comment poster (create + patch + dry-run via fake `urlopen`), and a roundtrip over the committed baselines.
- README "Benchmarks / Results" section replaces the "pending issue #7" placeholder with real measured baseline numbers and the reproducer command.

**Why this work, this session:** Issue #7 was the third acceptance box on issue #4 (refactored into its own issue per the prior session) and the eval-harness gateway every other repo in the portfolio is supposed to import. Closing it ships the cross-repo wiring that proves the harness is real — both for this repo and for `llm-cost-optimizer` / `agent-orchestration-platform` (which copy this pattern).

**Open questions / blockers:** Real Anthropic-API runs require `ANTHROPIC_API_KEY` + budget the operator allocates; the workflow design supports a future `RAG_EVAL_BACKEND=anthropic` switch but doesn't wire it now to avoid an accidentally-billed CI run. The judge calibration data is committed but unused until the operator runs `eval-harness judge calibrate` against the same API key.

**Next session:** Either #3 (query rewriting / decomposition) or #6 (cost telemetry) in this repo — both build on the now-shipped Generator + eval suite.

## 2026-05-16 — Issue #3: Query rewriting / decomposition
**Duration:** ~60 min · **Branch:** `session/2026-05-16-1905-issue-3`

- Shipped a `Rewriter` Protocol (single `rewrite(query) -> RewriteResult`) plus two backends, mirroring D-005 / D-008. `TemplateRewriter` is dep-free and deterministic: rule-based decomposition over `Compare X with Y`, `A. Then B.`, and multi-question `and` splits (only fires when both halves look like questions, so "wine and cheese pairings" doesn't fragment). `AnthropicRewriter` lazy-imports the SDK via the existing `[rag-anthropic]` extra and validates strict JSON `{"sub_queries": [...], "reasoning": "..."}` — malformed output raises rather than silently passing through.
- Wired into `Retriever.search(query, k, *, rewriter=None)`. The kwarg defaults to `None` so existing callers are unchanged (parallel to D-007). When the rewriter expands to ≥2 sub-queries, hybrid search runs once per sub-query and the resulting rankings are fused with RRF across sub-queries; the per-method `ranks` dict on each `RetrievalResult` then carries `subquery_0`, `subquery_1`, … keys instead of `lexical` / `dense`. The reranker, when combined with the rewriter, scores against the *original* user query — the merged intent is the contract, not any one sub-query. The pre-D-014 single-query hybrid path is factored into `_hybrid_search` and called once per sub-query.
- Bench measured, real, reproducible: `scripts/bench_rewriter.py` over an 18-chunk synthetic multi-hop fixture (8 questions, each gold set is two facts from distinct topic groups). Mean recall@3 jumps from 0.625 baseline to 0.812 with `TemplateRewriter` (three improvements, zero regressions); recall@5 from 0.875 to 0.938 (one improvement, zero regressions). The gap is largest at k=3 where the candidate budget is tight relative to the two distinct facts each multi-hop query needs. Numbers published in the README under "Benchmarks / Results". `AnthropicRewriter` numbers are intentionally pending an operator-triggered API-key run.
- 29 new hermetic tests (16 rewriter unit, 13 retriever-integration via a fake psycopg connection). Full suite 116/116 hermetic + 7 pg-integration skipped. Ruff clean.

**Why this work, this session:** #3 was one of two unblocked priority:med issues after the eval-harness integration shipped (#7). Picked it over #6 (cost telemetry) because the Protocol pattern was already a load-bearing seam in three other layers (embedder, reranker, generator) — closing #3 with the same pattern keeps the public surface consistent and matches D-005/D-008 cleanly, where #6 would have introduced a new abstraction (telemetry sink) that has no established sibling.

**Open questions / blockers:** None. PR will go up for review per D-004; the next scheduled session can squash-merge.

**Next session:** Loop to a different portfolio repo per the multi-issue session prompt. Remaining work in this repo: #6 (cost telemetry), #8 (Next.js demo frontend).

## 2026-05-16 — Issue #6: Cost telemetry per request
**Duration:** ~50 min · **Branch:** `session/2026-05-16-1928-issue-6`

- Shipped `rag_kit/telemetry.py`: `CostRecord` per-request blob (timestamp UTC, query, model, retrieved_count, prompt/completion tokens, prompt/completion/total USD, total latency ms, per-phase ms dict); `PriceTable` operator-supplied model→price map; `TelemetryStore` thin stdlib `sqlite3` wrapper (single table, single index on ts) with `record/since/last_24h` and a context-manager; `aggregate` returning `Aggregate(n, totals, p50/p95/p99 latency)` using the same NIST type-7 linear-interp percentile math as `streaming.PhaseTimings`. No new runtime deps — D-002 preserved.
- D-015 recorded: `PriceTable` ships zero defaults; calling `cost()` for an unknown model raises `UnknownModelError` instead of silently emitting `$0.00`. Same posture as D-013's no-fabricated-corpus rule extended to no-fabricated-prices. Public list prices drift, and a silent zero would be a load-bearing bug in any dashboard built on top.
- `scripts/telemetry_dashboard.py`: single-page stdlib `http.server` + `ThreadingHTTPServer` dashboard reading the last 24h from the SQLite store. Renders an inline SVG line chart of per-request latency over time plus four stat cards (n requests, total USD, p50/p95 latency) and the 20 most recent rows. Works air-gapped — no Chart.js, no external CDN. `--seed N` flag fills a fresh DB with deterministic synthetic records (labeled `synthetic-N` in the query field) so the chart can be exercised before any real requests flow through. Two endpoints: `GET /` (HTML), `GET /api/last_24h` (JSON snapshot).
- 21 new hermetic tests in `tests/test_telemetry.py`: PriceTable math + missing-model + no-defaults + override semantics, CostRecord composition + monkey-patched-now + propagated UnknownModelError + negative-latency rejection, TelemetryStore roundtrip + since() filtering + chronological ordering + last_24h(now=...) windowing + context-manager, percentile NIST type-7 known values + edge cases + out-of-range rejection, aggregate empty + totals + percentile, and an interop test asserting `telemetry.percentile` and `streaming.PhaseTimings.percentile` return the same number on identical input. Full suite 108/108 hermetic + 7 pg-integration skipped. Ruff clean.
- README "Cost telemetry (#6)" section with the schema, the operator-supplied-prices pattern, and the dashboard commands. "What this is" gains a paragraph describing the layer plus the D-015 posture. The old "tracked under #6" pointer in "Benchmarks / Results" is replaced with the actual mechanism.

**Why this work, this session:** #6 was the second of two unblocked priority:med issues in this repo (after #3 query rewriting, shipped earlier in this same session in a separate branch). Cost telemetry was the v0.1 quality-bar item missing from "real numbers in the README" — without it the repo would have shipped without a way to *get* the production-end-to-end p50/p95 numbers the README was supposed to surface. The operator-supplied-prices posture (D-015) is the same shape as D-013's no-fabricated-corpus rule, so adopting it cost nothing new.

**Open questions / blockers:** None. PR will go up for review per D-004; the next scheduled session can squash-merge.

**Next session:** Loop to a different portfolio repo per the multi-issue session prompt — only #8 (Next.js demo frontend) remains in this repo and it's a bigger build. Alternatively, target `llm-cost-optimizer` or `embedding-model-shootout` next.

## 2026-05-18 — Issue #8: Next.js demo frontend with inline citations
**Duration:** ~55 min · **Branch:** `session/2026-05-18-issue-08` · **PR:** #16

- Shipped `demo/nextjs/`: a minimal Next.js 15 + React 19 App Router app that demonstrates the streaming-with-citations pattern. Stands alongside the existing `demo/streaming/` Python demo — same wire format (D-016), different layer. Self-contained: deterministic in-repo fixtures so a fresh clone runs `npm run dev` with no Postgres, no Anthropic key, no Python backend.
- The page layout: query box, pipeline phase pills (idle → active → done as SSE events arrive), streamed answer on the left with inline `[N]` citation chips, retrieved-chunks panel on the right. Hovering a chip highlights the matching chunk; clicking smooth-scrolls + flashes the highlight. The route handler emits the *same* events as the Python demo (retrieving / retrieved / reranking / reranked / generating / token / generated / done) so the protocol is the load-bearing artifact, not the UI framework.
- 13 hermetic tests: 6 on the corpus + streamer (retrieve determinism, zero-overlap fallback, citation 1..N invariant, reassembly), 7 on the route's SSE protocol shape. `?delay=0` strips per-token sleep so the suite runs in <300 ms. Production build is clean (`/` static 2.05 kB, `/api/stream` dynamic).
- D-016 records the "re-emit the same protocol, don't invent a new one" decision. Same posture as the existing D-005.

**Why this work, this session:** Last remaining med-priority issue across the portfolio and the largest scope (90-min estimate). Fits the night session's room for a bigger swing after six smaller shipped issues.

**Open questions / blockers:** As with the other Next-shaped PRs this night, the in-browser walkthrough wasn't performed on this branch — protocol contract + production build + unit tests cover the wiring, but the chip hover/scroll feel needs a human reviewer's eye. PR body is honest about this.

**Next session:** Loop continues against the low-priority backlog or wrap. With 7+ issues shipped tonight, wrap is the more honest call — the remaining low-priority items are 15-30 min ad-hoc work that doesn't move the build sequence.

## 2026-05-18 — Issue #17: Architecture doc covers all 8 shipped layers
**Duration:** ~25 min · **Branch:** `session/2026-05-18-1527-issue-17` · **PR:** [#18](https://github.com/jt-mchorse/rag-production-kit/pull/18) (ready)

- Rewrote `docs/architecture.md` from "one layer shipped" to "eight layers shipped": one top-of-doc integrated mermaid showing the offline index path (corpus → embedder → pgvector) and the online query path (query → rewriter → lex+dense → RRF → reranker → generator + citations → streaming + telemetry → SSE / Next.js demo), plus one per-layer section for #1 through #8.
- Each layer section has a prose statement of what it does, a mermaid of its data flow, the relevant D-NNN references (D-002 through D-016), and a "composes with" line linking it to neighbours. Mermaid labels containing parens are fully double-quoted — same lint applied to the cost-optimizer architecture doc last session.
- README Architecture section dropped its ASCII block for a one-paragraph summary that points at the rewritten doc. Bottom-of-doc "## Pending" section removed — every runtime layer in §2 has shipped.

**Why this work, this session:** Every original `priority:high` issue is closed; the recently shipped Next.js demo (#8) brings the repo to feature-complete per its §2 spec. The architecture diagram was the most visible remaining §1 quality-bar gap — there was a diagram, but it labeled #4 as pending and listed every layer after #2 in a stale "Pending" section. Filling that gap is the cleanest move toward v0.1.

**Open questions / blockers:** None — PR is ready for review.

**Next session:** Continue the multi-issue loop; next zero-open-issue repo in build sequence is embedding-model-shootout (per §8) but it has an in-flight PR with a lint blocker (PR #9) — either fix that or proceed to chunking-strategies-lab.

## 2026-05-18 — Issue #19: snapshot test for README rewriter recall@k table
**Duration:** ~20 min · **Branch:** `session/2026-05-18-1931-issue-19`

- Added `tests/test_rewriter_bench_snapshot.py` (4 tests). Parametrized over k ∈ {2, 3, 5}, each one locates the README's "Rewriter recall@k" table by its header signature, parses the row, and asserts every numeric cell matches what `scripts.bench_rewriter._run(k)` + `_summary(...)` produce. The row-count test guards against silently dropping or adding a k-row.
- The README has hard-coded recall numbers (`0.625`, `0.688`, `0.812`, `0.938`) for the synthetic multi-hop fixture. The bench is deterministic — `TemplateRewriter` dep-free, fixed in-process corpus + questions, the in-memory token-overlap retriever — but no test enforced that the README rows matched the live script output until now.
- Failure messages on every assertion name the per-k regen command and a `git diff README.md` hint.
- Verified the failure path by tampering one cell (k=3 rewriter mean `0.812` → `0.999`); the test fired with the regen hint visible.

**Why this work, this session:** Third snapshot test landed today across the portfolio (cost-optimizer for `docs/savings.{json,md}` + table, prompt-regression-suite for `docs/regression_demo.html`, now rag-kit for the README rewriter table). The handoff §10 commits the portfolio to "no fabricated benchmarks"; snapshot tests are the structural enforcement mechanism wherever a doc cites a measured number.

**Open questions / blockers:** None — PR ready for review.

**Next session:** The streaming-overhead p95 numbers in `docs/benchmarks.md` are wall-clock dependent and intentionally out of scope; a separate file should track real-LLM eval-suite numbers (faithfulness/recall_at_5/correctness) once they have a similar snapshot gap worth closing.

## 2026-05-19 — Issue #21: snapshot test for README eval mean-score table
**Duration:** ~25 min · **Branch:** `session/2026-05-19-1509-issue-21`

- Added `tests/test_eval_bench_snapshot.py` (4 tests). Parses the README's "First baseline" `| suite | mean score | reproducer |` table by header signature, then parametrized over `(faithfulness, recall_at_5, correctness)`: each test asserts the README's mean cell matches what `evals.run_eval.run_all_suites()` produces today, within `abs=5e-3` (two-decimal half-round). The row-count guard asserts the suite set is exactly the orchestrator's `SUITES`, so silent additions / drops fail loudly.
- A module-scoped fixture caches one `run_all_suites()` invocation across the three parametrized rows, keeping the test under 0.2s.
- Failure messages on every assertion name the regen command (`python -m evals.run_eval --write-baselines`) and a `git diff README.md` hint.
- Tamper-verified by editing the `correctness` cell `0.90` → `0.99` in the README; the test fired with the regen hint, then reverted to green.

**Why this work, this session:** Issue #19 explicitly flagged the eval mean-score table as the worth-following-up parallel gap, after the rewriter recall table was locked in the same pattern. The portfolio's "no fabricated benchmarks" rule (handoff §10) is enforced structurally by snapshot tests wherever a doc cites a measured number — this PR closes the second of two README benchmark tables in this repo.

**Open questions / blockers:** None — PR ready for review.

**Next session:** Both README benchmark tables are now locked; the streaming p95 numbers stay out of scope. Continue the multi-issue loop into the next zero-issue portfolio repo (embedding-model-shootout or chunking-strategies-lab per §8).

## 2026-05-20 — Issue #23: lock rag_kit public surface
**Duration:** ~35 min · **Branch:** `session/2026-05-20-0311-issue-23`

- Added `tests/test_public_surface.py` (5 test functions, 13 test items: 4 standalone + 9 parametrized submodule anchors) that locks the top-level `rag_kit` surface across five orthogonal axes: `__version__` is semver-ish, every `__all__` entry is bound and non-None, `__all__` agrees bidirectionally with the AST-parsed `from .X import …` block (filtered on `level >= 1` for relative imports — same adaptation `prompt-regression-suite#20` used), the README quickstart's quoted imports resolve, and one anchor per submodule survives.
- Companion change: added `__version__ = "0.0.1"` to `rag_kit/__init__.py`, mirroring `pyproject.toml`'s `[project] version`. No `importlib.reload` workaround was needed — this repo doesn't ship a pytest plugin via entry-points like `llm-eval-harness` does, so `rag_kit/__init__.py` is instrumented from the start (already at 100%).
- Tamper-verified three of five axes locally: a bad `__version__` fires test (1); dropping `"Document"` from `__all__` fires test (3) naming `{'Document'}`; alias-renaming `HashEmbedder as HashEmbedderV2` fires tests (2), (4), (5)[embedder] simultaneously plus (3) — full surface coverage working as designed.

**Why this work, this session:** Fourth strike of the portfolio-wide public-surface hygiene pattern, applied to the largest top-level surface in the portfolio (42 names, nine submodules). Explicitly named as one of the remaining five targets in the 2026-05-19 day-session report, and earliest in build sequence among them.

**Open questions / blockers:** None — PR ready for review.

**Next session:** Continue the night-session multi-issue loop into the next portable target: `embedding-model-shootout`, then `chunking-strategies-lab`, then `python-async-llm-pipelines`, then the Python example in `mcp-server-cookbook`.

## 2026-05-22 — README "What this is" section claimed #7 was staged but it shipped 6 days ago; #8 was unmentioned (#27)

**Duration:** ~30 min. **Issue:** [#27](https://github.com/jt-mchorse/rag-production-kit/issues/27). **PR:** [#28](https://github.com/jt-mchorse/rag-production-kit/pull/28).

The README's "What this is" section walked the reader through six bold sub-sections, one per shipped layer (#1 hybrid retrieval through #6 cost telemetry), and then closed with a paragraph claiming "Everything beyond #1 + #2 + #3 + #4 + #5 + #6 is staged in follow-up issues" — naming #7 (eval-harness integration) as pending and not mentioning #8 (Next.js demo) at all. Both shipped a week ago. The Architecture section three lines below already said "Eight runtime layers ship today" with #7 and #8 inclusive, so the same README contradicted itself top-to-bottom.

Replaced the stale paragraph with two new bold sub-sections, one for #7 and one for #8, of the same shape as the existing six. Added the missing `[#8]` link reference. Architecture, Quickstart, Benchmarks, Demo, and the bodies for layers #1–#6 are unchanged.

Lock-against-drift: `tests/test_readme_what_this_is_lists_shipped_layers.py` (+4 tests) extracts every column-0 bold opener (`**...**`) inside the What-this-is section and asserts the set matches a canonical eight-layer frozenset (catches both missing and extra layers symmetrically), hard-pins the stale `/staged\s+in\s+follow-up\s+issues/` phrase absent (whitespace-insensitive because the original wrapped at `follow-up\n`), and cross-checks the Architecture section's "Eight runtime layers ship today" line still parses cleanly and agrees on the count. Tamper-verified.

Issue #27 was filed in-session: after Phase A merged seven PRs and the cost-optimizer #25 PR closed, no actionable priority:high/med issues remained open across the portfolio (the seven open issues are all priority:low demo-capture work that needs a human screen recorder). This drift was caught while scanning rag-production-kit for an actionable bug; tenth post-v0.1 README-vs-code drift fix in the portfolio pattern.

**Open questions / blockers:** None. **Next session:** continue the multi-issue loop or fall through to another repo for similar drift hunting.

## 2026-05-23 — Architecture-doc drift lock (#29)

**Duration:** ~25 min. **Issue:** [#29](https://github.com/jt-mchorse/rag-production-kit/issues/29). **PR:** [#30](https://github.com/jt-mchorse/rag-production-kit/pull/30).

This repo is the **first in the portfolio with both coverage axes** — the doc annotates surfaces with both `(#NN)` issue references AND `D-NNN` decision references. The lock covers both axes simultaneously (compare: `llm-cost-optimizer` PR #28 D-NNN-only; `vector-search-at-scale` PR #22 #NN-only).

Four invariants pinned: path-token reachability with `OPERATOR_SUPPLIED_PATHS` allow-list (currently empty), closed-feature-issue coverage anchored to `KNOWN_SHIPPED_ISSUES = (1..8)`, active-decision coverage anchored to `MEMORY/core_decisions_ai.md` non-superseded entries from D-002, and banned-phrase absence. Tamper-verified all four. No `docs/architecture.md` changes — already in steady state.

**Why this work, this session:** Second of five sister issues in this night-session sweep across the Python half of the portfolio.

**Open questions / blockers:** none.

**Next session:** continue the sweep across `chunking-strategies-lab`, `python-async-llm-pipelines`, `agent-orchestration-platform`.

## 2026-05-23 — 60-second demo capture script (#25, AC3 of 3)

**Duration:** ~25 min. **Issue:** [#25](https://github.com/jt-mchorse/rag-production-kit/issues/25). **PR:** [#31](https://github.com/jt-mchorse/rag-production-kit/pull/31).

Fourth issue in the day-session multi-issue loop, after `llm-eval-harness#33`, `llm-cost-optimizer#29`, and `prompt-regression-suite#28`. Three stages map to the two demo surfaces from #25's spec plus a hermetic in-process preview:

- **STAGE 1 (auto, hermetic).** `scripts/capture_demo.py` composes `StreamingPipeline` directly with the stubs `demo.streaming.server` already ships — `FakeRetriever`, `SlowReranker`, `_stub_token_stream`, `_CORPUS`. Re-using rather than redefining means a future change to the demo's corpus or the FakeRetriever's `sleep_ms` flows automatically through to the recorded phase sequence — the two surfaces can't drift apart. Each event prints as an SSE frame matching what `curl -N` against the live server shows over the wire.

- **STAGE 2 (operator-action).** Cheat-sheet for the actual SSE server the README cites: `python -m demo.streaming.server` then `curl -N 'http://localhost:8765/stream?q=postgres+tuning'`. `--launch-server` subprocess-spawns the server and curls for one-key recording sessions; off by default because the server is long-running and can't run hermetically in CI.

- **STAGE 3 (operator-action).** Cheat-sheet for `cd demo/nextjs && npm run dev`, the `http://localhost:3000` URL, and a three-step click checklist (search input + phase pills tick through retrieve → rerank → generate → citation chip hover highlights matching chunk → retrieved-chunks panel collapse/expand). The Next.js dev server is **never** auto-spawned even with `--launch-server` — npm/node startup cost makes that a poor recording experience; the operator launches it themselves before hitting record.

`tests/test_capture_demo_smoke.py` adds four tests under the same hermetic contract as `tests/test_streaming.py`. The first test asserts STAGE 1 emits the exact phase sequence the streaming-test contract defines (`retrieving` first, full phase set present, `done` last, at least one `token` event between `generating` and `generated`).

**Why this work, this session:** Fourth loop iteration. Build-sequence position 4. After this PR the portfolio's demo-script coverage is 6 of 7 (six repos with `capture_demo` script committed; one remaining — `mcp-server-cookbook#16`). The pattern across the four PRs in this loop is stable enough that the cookbook PR will be a straight transcription.

**Open questions / blockers:** AC1 + AC2 are operator-only (screen recorder + README embed). PR is ready for review on AC3 standalone — issue #25 stays open until JT records.

**Next session:** `mcp-server-cookbook` #16 to close the AC3-coverage loop across the portfolio.

## 2026-05-24 — Issue #32: `--suite` filter on `evals/run_eval.py`

**Duration:** ~25 min. **Issue:** [#32](https://github.com/jt-mchorse/rag-production-kit/issues/32). **Branch:** `session/2026-05-24-0332-issue-32`.

`evals/run_eval.py` always wrote all three suite JSONs (`faithfulness`, `recall_at_5`, `correctness`), and `--write-baselines` clobbered all three. There was no way to iteratively update one baseline without `git restore`-ing the unrelated two by hand. Parallel to `llm-eval-harness`'s own `--suite` filter on the runner — same dev-iteration use case, different repo.

`--suite faithfulness|recall_at_5|correctness` now filters what `write_runs` sees and what `_post_composite_comment` renders. `run_all_suites()` deliberately keeps computing all three — scoring is one pass over the dataset, so the savings is in disk writes (and the surprise-baseline-clobber path), not compute. The composite-comment shape stays stable for any CI subscriber: unselected suites render as `_(skipped via --suite filter)_` instead of dropping out of the table entirely. Validation is manual rather than `argparse choices=` so an unknown value can exit 2 with the inventory printed on stderr (matches `llm-eval-harness`'s `--tags` UX).

Four new tests cover the filter on the default current-dir path, the `--write-baselines` path (the workflow the filter was added for), the unknown-suite stderr-inventory exit-2 path, and a regression guard that no-`--suite` still writes all three.

**Why this work, this session:** Fourth issue in the night-session multi-issue loop, after CLI surface fixes in `llm-eval-harness` #34 (`diff --format markdown`), `llm-cost-optimizer` #30 (`--dry/--no-dry` parity), and `prompt-regression-suite` #29 (`run --format html --out`). The pattern this run keeps surfacing: every repo had at least one CLI parity gap that was easy to find by reading the source once.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the loop to build-sequence #5 (`embedding-model-shootout`). Most of the same shape (Python repo with bench script) and likely a similar parity gap.

## 2026-05-24 — Issue #34: Retriever validates k_rrf > 0 at construction
**Duration:** ~20 min · **Branch:** `session/2026-05-24-issue-34`

- `Retriever.__init__` at `rag_kit/retriever.py:72` accepted `k_rrf: int = DEFAULT_K` without validation. The value flowed to `reciprocal_rank_fusion(rankings, k=self.k_rrf)` which raises at call time — but the operator's stack trace pointed at `fusion.py:38` and named `k`, not the constructor-side `k_rrf`. A `Retriever(k_rrf=0)` typo wasn't caught until the first `search()` call, with an error message that required chasing a variable rename.
- Added a single `if k_rrf <= 0: raise ValueError(f"k_rrf must be positive, got {k_rrf}")` before the assignment. Inline comment documents the defense-in-depth trade: `reciprocal_rank_fusion` keeps its call-time guard because the function is also part of the public surface; programmatic callers passing `k=0` directly still get caught there.
- Eight new tests in `tests/test_retriever_rewriter.py` under a `#34` block: zero raises with `k_rrf` in the message; parametrized negative sweep over `-1, -60, -1000`; default constructor regression pin; parametrized positive acceptance over `1, 60, 120` (boundary + default-equivalent + above-default). Reused the existing `_FakeConn` fixture so the tests stay hermetic (no `DATABASE_URL` dependency).

**Why this work, this session:** Brings `Retriever` in line with the rest of the `rag_kit` surface — `HashEmbedder.__init__` validates `dim > 0` and `dim % 8 == 0`; `TemplateGenerator.__init__` validates `max_chunks > 0`; `Document.__post_init__` validates `external_id` and `text`. `k_rrf` was the one constructor numeric parameter flowing through to a call-time-only guard. Sister to today's `llm-cost-optimizer` #32 and `llm-eval-harness` #38 and `prompt-regression-suite` #33 — four repos in a row, same family.

**Open questions / blockers:** none — PR ready for review.

**Next session:** With four iterations behind and limited time left in the 180-minute cap, check the clock before deciding whether to start a fifth. Build sequence #5 (`embedding-model-shootout`) and #6 (`chunking-strategies-lab`) are the next reasonable targets if there's time.

## 2026-05-25 — Issue #36: ModelPrice validates non-negative rates in __post_init__
**Duration:** ~15 min · **Branch:** `session/2026-05-24-issue-36`

- `ModelPrice` at `rag_kit/telemetry.py:45` is a frozen dataclass accepting `prompt_per_million` and `completion_per_million` floats. `cost()` at line 52 already validated token counts non-negative — but the per-million **rates** themselves were not validated. A negative rate flowed through `prompt_tokens * rate / 1_000_000` and silently inverted the sign of `CostRecord.total_usd` downstream. D-015 explicitly calls silent-zero "a load-bearing bug in cost dashboards"; this extends the same posture to silent-negative.
- Added `__post_init__` raising `ValueError(f"{field} must be >= 0.0; got {value}")` for either rate `< 0.0`. Comment in source documents the D-015 anchor. `frozen=True` only blocks reassignment, not the initial set, so `__post_init__` works cleanly on a frozen dataclass.
- Seven new cases in `tests/test_telemetry.py` under a `#36` block: parametrized over (field × bad-value, 4 cases); inclusive-zero accepted (1 case); `PriceTable.add(..., -X, Y)` wrap-through validates through the realistic operator-supplies-bad-config path (2 cases). Full suite 196/196 + 7 skipped (`DATABASE_URL` not set locally).

**Why this work, this session:** Direct mirror of `llm-cost-optimizer` PR #35 (`ModelPricing.__post_init__`) shipped earlier in this same day session. The two cost-aware repos in the portfolio now defend their dashboards consistently. Third Phase B+C target after `llm-eval-harness` #40 (drift thresholds) and `llm-cost-optimizer` #34 (pricing rates).

**Open questions / blockers:** none — PR ready for review.

**Next session:** Time remaining in the 180-min cap permits another iteration. Build sequence #5 (`embedding-model-shootout`) or #6 (`chunking-strategies-lab`) are the natural next pickups; both have public-surface numeric parameters worth scanning.

## 2026-05-25 — Issue #38: ModelPrice and CostRecord finiteness guards
**Duration:** ~20 min · **Branch:** `session/2026-05-24-issue-38`

- Two existing sign-only range checks in telemetry let `NaN`/`+/-Infinity` through. `ModelPrice.__post_init__` (`telemetry.py:60-61`) accepted NaN rates that propagated through `ModelPrice.cost` → `CostRecord.total_usd = NaN` → `aggregate` sums NaN across the window → cost dashboard renders "NaN" silently. Same harm shape as D-015's silent-zero, one arithmetic layer downstream. `CostRecord.build.total_latency_ms` (`telemetry.py:154-155`) accepted NaN latency that propagated through `percentile(values, q)` where the sort over NaN is implementation-defined → p95/p99 silently wrong.
- Tightened both to finiteness using `math.isfinite`. Error messages updated from "must be >= 0.0" / "must be non-negative" to "must be a finite number >= 0.0" / "must be a finite non-negative number" so callers can grep the new contract. Two pre-existing tests pinning the old strings updated in place.
- 9 new parametrized tests in `tests/test_telemetry.py` under a `#38` block: rejection per field over `[NaN, +Infinity, -Infinity]` for both ModelPrice fields and `total_latency_ms`. Test count 212 (was 203). Ruff + format clean.

**Why this work, this session:** Eighth Phase B+C target in the 360-min night session. Second PR in rag-production-kit tonight; the first was via the Phase A fixup-merge of #37 (`ModelPrice` negative-rate `__post_init__`). That covered sign; this covers NaN/Infinity, completing the D-015 silent-zero → silent-negative → silent-NaN/Infinity arc on the telemetry surface.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the loop. Remaining unvisited-tonight-for-second-iteration: `embedding-model-shootout`, `chunking-strategies-lab`, `vector-search-at-scale`, `python-async-llm-pipelines`. Each had a fixup-merge today but no Phase B+C finiteness sweep.

## 2026-05-25 — Issue #40: retrieval-fusion k → isinstance(int) + reject bool sweep
**Duration:** ~30 min · **Branch:** `session/2026-05-25-1545-issue-40`

- Five sign-only `k <= 0` checks at retrieval-fusion public boundaries accepted `bool` (True/False are `int` subclasses in Python) and `float` (0.5 silently truncated in SQL LIMIT bind; 60.0 looked fine but is contractually wrong). Telemetry side landed in #38; this PR closes the retrieval side.
- Sites tightened, all to `"k must be a positive integer, got {k!r}"` shape: `fusion.reciprocal_rank_fusion` k (True silently shifted the RRF constant from 60 to 1, distorting the `1/(k+rank)` score curve), `Retriever.__init__` k_rrf, `Retriever.search` k (float k=2.5 propagated into `LIMIT 2.5`, surfacing as an opaque psycopg2 error far from the call site), `rerank_delta_ndcg` k, `StreamingPipeline.run` k.
- Deferred to a follow-up issue if needed: `generator.max_chunks`, `embedder.dim`, and `streaming.PhaseTimings.percentile` p — independent failure modes (not retrieval-fusion math).
- Pre-existing tests pinning the old `"must be positive"` string updated in `test_retriever_rewriter.py` (two k_rrf sites) and `test_streaming.py` (run k). `test_fusion.py`'s existing loose match `"positive"` continued to work unchanged.
- 45 new parametrized tests across four test files (250 total, was 205); ruff clean.

**Why this work, this session:** Third Phase B+C target in the 180-min day session. After fixing the README test-count drift in `mcp-server-cookbook` and filing+closing `llm-cost-optimizer#38` (BatchRequest/Result/Job __post_init__), I noticed via grep that the rag-production-kit retrieval surface still had sign-only int checks that the recent telemetry-side sweep #38 hadn't reached. The pattern from `llm-eval-harness#42 runs.py limit` validation transfers cleanly.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the loop. If `generator.max_chunks` / `embedder.dim` / `streaming.PhaseTimings.percentile` p turn out to matter, file as a separate follow-up issue with its own session plan. Don't bundle.

## 2026-05-25 — Issue #42: Close #41's deferred validation gaps (generator, embedder, streaming)
**Duration:** ~30 min · **Branch:** `session/2026-05-25-1700-issue-42`

- `Generator.max_chunks` and `HashEmbedder.dim`: replaced sign-only `<= 0` with the portfolio positive-int contract (`not isinstance(int) or isinstance(bool) or <= 0`), matching `runs.list_runs.limit` (`llm-eval-harness#42`) and #41's retrieval-side k validators. For `dim`, the multiple-of-8 check now runs after the type contract, so `dim=True` no longer surfaces "must be a multiple of 8" (wrong error message for the underlying type bug).
- `PhaseTimings.percentile.p`: narrower fix — reject `NaN`, non-numeric types, and `bool`. Preserved the documented clamp contract for out-of-range finite values (`-5` → values[0], `110` → values[-1]) plus `inf`/`-inf`, per the explicit "match numpy's well-behaved default" intent pinned by `test_phase_timings_percentile_clamps_edges`. The real failure mode was NaN slipping both clamp branches and reaching `int(NaN)` deep in interpolation.
- 63 new parametrize tests in `tests/test_deferred_validation_sweep.py`. Pre-existing `test_invalid_max_chunks_rejected` updated to the new error-message shape. Test suite 250 → 308 (non-pg). Ruff clean.

**Why this work, this session:** Second Phase B+C target in today's 180-min DAY session. PR #41 explicitly listed these three sites as "Deferred to a follow-up if needed — independent failure modes, not retrieval-fusion math." Closing the deferred list keeps the contract uniform across the repo's construction sites.

**Open questions / blockers:** none — PR ready for review.

**Next session:** `embedding-model-shootout#34`'s deferred list (`hash_embedder.dim/ngram`, `synthesize_queries n/min/max`) is the natural next target — explicit deferred-list entry, same pattern, build-sequence position #5.

## 2026-05-26 — Issue #44: Atomic `write_runs` closes the Python atomicity arc
**Duration:** ~20 min · **Branch:** `session/2026-05-26-1528-issue-44`

- `evals/run_eval.py::write_runs` wrote the three per-suite eval JSONs (faithfulness, recall_at_5, correctness) via `path.write_text(...)` in a loop. The eval GitHub Action's composite sticky PR comment (`_post_composite_comment`, run_eval.py:301) parses all three; a SIGINT mid-eval leaves a half-written suite JSON that either crashes the comment-posting step or silently posts a corrupt comment.
- Added `rag_kit/io_utils.py` with `atomic_write_text(path, text)` — natural home keeping the public surface tight, module can grow other IO utilities. Same shape as the helpers in `llm-eval-harness#48`, `llm-cost-optimizer#42`, and `prompt-regression-suite#39` filed and merged earlier today.
- Routed `write_runs` through it; dropped the orphaned `out_dir.mkdir(...)` (helper auto-creates parent dirs).
- 8 new tests in `tests/test_atomic_write.py`: six unit invariants on the helper plus two integration tests on `write_runs`. The load-bearing integration test uses a **selective monkeypatch** that raises only on the *second* of the three `os.replace` calls, proving four invariants in one test: (a) the first file is fully replaced atomically with valid suite JSON, (b) the second file's pre-existing stale content is bitwise preserved (helper never touched destination before rename), (c) the third file is never reached, (d) no `.tmp` leftovers in `out_dir`. Full suite 308 → 316 (7 hybrid_pg skips pre-existing). Lint + format green.

**Why this work, this session:** Fourth Phase B+C target in today's 180-min DAY session, completing the portfolio-wide atomicity arc across all four Python repos with artifact-emitting CLI or script chains. The TypeScript repos (`agent-orchestration-platform`, `mcp-server-cookbook`, `nextjs-streaming-ai-patterns`, `ai-app-integration-tests`) also write artifacts but with different ergonomics — those would be a separate arc if pursued. Cross-file inconsistent-state harm (`out_dir` mixing fresh + stale suite JSONs after partial-success) is documented in the issue body but explicitly out of scope here.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Atomicity arc closed for Python. Four consecutive same-shape PRs in one day session has made the helper shape a documented portfolio pattern with four independent concrete instances. Next sessions can pivot to a different harm class — TypeScript repos for atomicity (different ergonomics), or a fresh harm class entirely (concurrency safety, signal handling, error path quality, cross-file invariants).

## 2026-05-26 — Issue #46: README decision-range upper-bound lock
**Duration:** ~7 min · **Branch:** `session/2026-05-26-2326-issue-46`

- Added `tests/test_readme_decision_range.py` with the active-decision-range upper-bound invariant.
- Replaced README placeholder `D-NNN` with explicit `D-002…D-016` bound.

**Why this work, this session:** Propagation 4 of 10 of the cross-portfolio drift class authored in chunking-strategies-lab.

**Open questions / blockers:** none.

**Next session:** Continue propagation to embedding-model-shootout.

## 2026-05-27 — Issue #48: CONTRIBUTING.md cadence-wording propagation
**Duration:** ~3 min · **PR:** #49

- Replaced pre-D-008 `~60-minute session cap` line with D-008 (180/360 min, multi-issue loop) and D-004 (Phase A PR auto-merge) wording, matching the bootstrap template post-portfolio-ops#3.

**Why this work, this session:** Iteration in the autonomous NIGHT session propagation arc for portfolio-ops#3.

**Open questions / blockers:** none.

**Next session:** continue portfolio propagation.

## 2026-06-01 — Issue #50: `Aggregate.to_dict` + `TelemetryStore.dump_aggregate_json` (observability parity)
**Duration:** ~30 min · **Branch:** `session/2026-06-01-1944-issue-50`

- Added `Aggregate.to_dict()` to `rag_kit/telemetry.py` returning the seven fields as a stable JSON dict. Field set is locked exhaustively via `dataclasses.fields` so a future addition without a serializer update is caught loudly. The pattern mirrors `CacheTelemetry.to_dict` and `CacheStats.to_dict` from llm-cost-optimizer's #50 and #52.
- Added `TelemetryStore.dump_aggregate_json(path, *, since_ts=None)` writing the current rolling aggregate via the package-level `rag_kit.io_utils.atomic_write_text` (the same helper #44 wired for the eval-action sticky comment, so the atomic-write story is now portfolio-wide). When `since_ts` is omitted the writer defaults to the last-24h window the dashboard already renders, so a cron-driven observability sink gets sensible behavior with no arguments. On-disk shape is sorted-keys JSON + `indent=2` + trailing newline — byte-shape parity with the cost-optimizer's `dump_aggregate_json` and `dump_stats_json`, so a single log-parsing config consumes all portfolio observability artifacts.
- 10 new tests in `tests/test_telemetry_dump.py` mirror the cost-optimizer matrix: field-set lock via `dataclasses.fields`, JSON round-trip, empty-aggregate zero-state (`latency_p*_ms=0.0` not NaN — locked explicitly), on-disk shape with sorted-keys check, parent-dir auto-create (from `atomic_write_text`), atomic overwrite with no tempfile leftovers, `since_ts` window correctness (a record below the floor is excluded), default-window correctness (a 25-hour-old record is excluded with no `since_ts` passed), zero-state canary writer, plus a regression check that the existing `aggregate` function export is unaffected.
- README cost-telemetry paragraph extended with one sentence on the new observability shape citing #50. `docs/architecture.md` layer-6 invariants section gains a parallel paragraph naming the parity with the cost-optimizer's #50/#52. `tests/test_architecture_doc.py::KNOWN_SHIPPED_ISSUES` extends to `(..., 8, 50)`; the hard-pin assertion updated to match.

**Why this work, this session:** Iteration 4 of today's DAY session. Iterations 1, 2, 3 closed `llm-eval-harness#58`, `prompt-regression-suite#49`, `llm-cost-optimizer#52`. The natural Phase B follow-on was the rag-kit Aggregate dataclass, which had the same gap — runtime state, no `to_dict`, no atomic-write companion. With this PR, three repos and four state objects share one observability shape.

**Open questions / blockers:** none — full pytest pass (Postgres-dependent tests skipped as expected), ruff check + format clean, live smoke shows the on-disk JSON has the expected shape and the `since_ts` window actually filters.

**Next session:** with the observability arc landed across two repos and three state objects, the next clean parity target is `agent-orchestration-platform`'s trace store — same shape question (does its rollup expose a `to_dict` + atomic writer?). Out of scope here; would be one more clean iteration if this pattern continues to be the right unit of work.

## 2026-06-17 — Issue #52: Workflow YAML-parseability lock
**Duration:** ~12 min · **Branch:** `session/2026-06-17-1912-issue-52`

Added `tests/test_workflows_yaml_parseable.py` and pulled `pyyaml>=6.0`
into `[project.optional-dependencies].dev`. The test parametrizes
`yaml.safe_load` plus a non-empty `jobs:` assertion over every `*.yml`
under `.github/workflows/` — today that's `ci.yml` and `eval.yml`, so
5 tests total (1 smoke + 2 parse + 2 jobs).

**Why this work, this session:** Same justification as `llm-eval-harness#60`
— `portfolio-ops#27` closed a 21-day silent CI outage from one
unquoted colon-space in a `run:` value, and the followup explicitly
calls for propagating the lock to all 12 portfolio repos. This is the
second hop. `rag-production-kit`'s workflows are YAML-safe today (they
use the `run: |` block-scalar form) — the lock makes that *cannot*
drift.

**Open questions / blockers:** none — full `pytest` (332 → 337) +
`ruff` clean locally; PR #53 open and waiting for CI.

**Next session:** continue propagation to the remaining 10 portfolio
repos.

## 2026-06-18 — Issue #54: timeout-minutes guard + lock test
**Duration:** ~25 min · **Branch:** `session/2026-06-18-0313-issue-54`

- Added `timeout-minutes` to every job in `ci.yml` (`lint`, `unit`,
  `memory-check` at 15; `integration-pg` at 20 — pg container start
  plus pgvector setup plus the full `pytest -m pg` suite is the
  longest-running job in the repo and deserves headroom) and `eval.yml`
  (`eval-suite` at 15, mirroring llm-eval-harness's eval workflow).
- Added `tests/test_workflows_timeout_minutes.py` — 16 new tests: 1
  smoke + 5 jobs × 3 parametrized invariants (`timeout-minutes` is
  present, is an int (not bool/str), is in policy band `[1, 30]`). Each
  invariant fails as its own line so a regression names the offending
  job exactly, not a single rolled-up summary.

**Why this work, this session:** GitHub Actions defaults to 360 min/job
when `timeout-minutes` is unset, so a hung job (network stall during
`pip install`, infinite test loop, pgvector health-check loop) burns
the full 6-hour ceiling before the runner kills it. `llm-eval-harness`
PR #63 shipped the canonical first hop and the portfolio-ops audit (#36)
added a `--check missing-timeout` fingerprint that surfaces every
unprotected repo weekly. This PR is the propagation hop for
`rag-production-kit`; the audit will drop this repo from its findings
once #54 merges.

**Open questions / blockers:** none. Test count 337 → 353. Full pytest
clean; ruff check + ruff format --check clean; 7 pg integration tests
remain skipped because no local Postgres is configured (unchanged).

**Next session:** continue propagation across the remaining 8 unprotected
repos. Priority-tier order per D-009: chunking-strategies-lab,
nextjs-streaming-ai-patterns next; then build-sequence: embedding-model-shootout,
vector-search-at-scale, python-async-llm-pipelines, agent-orchestration-platform,
mcp-server-cookbook, ai-app-integration-tests, plus portfolio-ops itself.

## 2026-06-18 — Issue #56: concurrency guard + lock test
**Duration:** ~12 min · **Branch:** `session/2026-06-18-1522-issue-56`

- Added top-level `concurrency:` to `ci.yml` (`ci-${{ github.ref }}`)
  and `eval.yml` (`eval-${{ github.ref }}`, distinct groups so they run
  concurrently on the same ref).
- Copied `tests/test_workflows_concurrency.py` from llm-eval-harness;
  docstring origin updated to this repo's #56.

**Why this work, this session:** third per-repo hop in the
concurrency-lock propagation arc (after llm-eval-harness #64 and
llm-cost-optimizer #60). Audit fingerprint shipped in portfolio-ops #41
surfaces every workflow missing the lock; after this PR merges, this
repo drops off that finding set.

**Open questions / blockers:** none. Test count 348 → 355.

**Next session:** continue propagation to remaining 9 unprotected repos.

## 2026-06-19 — Issue #58: PhaseTimings observability parity
**Duration:** ~25 min · **Branch:** `session/2026-06-19-0334-issue-58`

- Added `to_dict()` to `rag_kit.streaming.PhaseTimings` as the canonical
  alias for `summary()`. Both methods coexist; `summary()` keeps its
  semantic name and no caller breaks.
- Added `dump_summary_json(path)` routing through
  `rag_kit/io_utils.atomic_write_text` with byte-shape parity to
  `TelemetryStore.dump_aggregate_json` (sorted keys, `indent=2`,
  trailing newline).
- 6 new tests mirroring the canonical six shapes.

**Why this work, this session:** closes the last runtime-aggregate-state
observability gap in this repo. After this PR, both runtime aggregate-
state classes (`Aggregate`, `PhaseTimings`) expose the same
observability shape — same `to_dict()` name, same atomic-write helper,
same on-disk byte shape. Sibling of the llm-cost-optimizer runtime trio
shipped this hour.

**Open questions / blockers:** none. 355 → 361 pytest passes. PR #59
open and ready.

**Next session:** consider plumbing `--out PATH` into `bench_streaming.py`
to call `dump_summary_json` directly so benchmark capture is hermetic.

## 2026-06-19 — Issue #60: `bench_streaming.py --out PATH` for hermetic JSON capture
**Duration:** ~25 min · **Branch:** `session/2026-06-19-issue-60`

- Added `--out PATH` argparse arg to `scripts/bench_streaming.py::main`.
  When set, the bench runs as usual, prints the stdout table as usual,
  then calls `timings.dump_summary_json(args.out)` to atomic-write the
  per-phase summary as JSON. `--out` is a sink, not a replacement —
  stdout behavior is unchanged so existing local-runner workflows
  don't drift.
- New `tests/test_bench_streaming.py` (10 tests) exercises the real
  CLI via `subprocess.run([sys.executable, "-m",
  "scripts.bench_streaming", ...])`: per-phase shape on disk, stdout
  unsuppressed when `--out` is set, no file created without `--out`,
  parent-dir auto-create, atomic-overwrite shape (assert by brace
  count + payload-parses-cleanly + no `.tmp` leftovers, since
  wall-clock p50/p95 prevent byte-identity assertions across runs),
  `--help` references the writer, parametrized phase-key coverage.
- Initial `byte_identity_across_runs` assertion failed on first
  pytest run — wall-clock measurements always vary. Pivoted to
  shape-based assertions so the test is hermetic against the
  bench's own nondeterminism while still catching real append /
  partial-write regressions.

**Why this work, this session:** closes the explicit "Next session"
follow-up from #58. After this PR the runtime-aggregate-state
observability arc in this repo is complete — both classes
(`Aggregate`, `PhaseTimings`) have `to_dict` + `dump_*_json` writers,
and the bench script that drives `PhaseTimings` exposes the writer to
operators as `--out`. Continues the same observability-sink-parity
arc that landed earlier today across 6 repos (`validate --out` x5 +
PhaseTimings#58).

**Open questions / blockers:** none. 361 → 371 pytest passes (7
DB-dependent skips unchanged). PR #61 open and ready.

**Next session:** the runtime-aggregate-state observability arc in
this repo is genuinely saturated. Pivot to another priority-tier
repo's followup hint — `llm-cost-optimizer` has a "plumb RouterStats
into the savings dashboard" hint; `llm-eval-harness` has a
`drift --output` normalization decision pending.

## 2026-06-19 — Issue #60: PR #61 lint fix-up to unblock merge
**Duration:** ~10 min · **Branch:** `session/2026-06-19-issue-60` (existing PR branch)

- Phase A PR-review pass caught a single ruff PT018 failure in PR #61 (a previous iteration of the same multi-issue day session). The remaining 5 of 6 CI jobs (unit ×2, integration-pg, eval-suite, memory-check) were already green; only `lint` blocked merge.
- Split the compound `assert body2.startswith("{") and body2.endswith("}\n")` at `tests/test_bench_streaming.py:141` into two independent assertions. Pushed the one-line fix to the existing branch rather than opening a new session branch — the change is pure lint repair with zero behavioral diff.
- All 6 jobs green post-push. PR #61 squash-merged into main; branch deleted; issue #60 auto-closed via `Closes #60` in the PR body.

**Why this work, this session:** Phase A review-and-merge pass surfaced the blocker; the fix was a 1-line lint repair, so it was faster to push the fix directly to PR #61 and complete the merge than to open a separate fix-up PR. Continues the multi-issue, multi-repo day-session loop.

**Open questions / blockers:** none.

**Next session:** continue the day-session loop on the next priority-tier repo with actionable unblocked work (`llm-cost-optimizer` #66 follow-on, or a new substantive issue elsewhere).

## 2026-06-22 — Issue #63: guard PhaseTimings.record against non-finite/negative ms
**Duration:** ~20 min · **Branch:** `session/2026-06-22-0405-issue-63`

- Found by reading `telemetry.py` then `streaming.py` side by side: `CostRecord.build` rejects a non-finite/negative `total_latency_ms` (a #38 finiteness guard, with a comment explaining that a NaN poisons `percentile` because sorting a list containing NaN is undefined), but the streaming layer's own ingestion point — `PhaseTimings.record` — had no such guard. A NaN/Inf/negative `ms` was appended silently and read back as a silently-wrong p50/p95/p99.
- Demonstrated the bug empirically (recording a NaN made `percentile("retrieving", 50)` return `nan`), then added the matching `not math.isfinite(ms) or ms < 0` guard so a corrupt timing fails loud at the record site. `bool` is intentionally not rejected — the telemetry sibling accepts it (bool-is-int) and exact parity is the point of the change.
- 12 new cases in the canonical `test_deferred_validation_sweep.py` (which already held the `PhaseTimings.percentile.p` finiteness tests). Suite 371 → 383, ruff clean. PR #64 ready.

**Why this work, this session:** the portfolio is saturated; this was a real silent-corruption bug in the cost-telemetry/streaming p50/p95/p99 path, found as a concrete asymmetry against an existing sibling guard — not a synthetic fill. I again declined the alternative `from_dict` reader for `Aggregate`/`PhaseTimings` `to_dict` for the same reason as in llm-cost-optimizer: nothing hand-rolls typed reconstruction from those artifacts.

**Open questions / blockers:** none.

**Next session:** `AnthropicGenerator` default model is `claude-opus-4-7` (no date suffix) while the eval-harness judge default pins a dated id (`claude-haiku-4-5-20251001`); worth a consistency pass on default model pinning across the portfolio if a future session needs filler here.

## 2026-06-22 — Issue #65: fusion — dedupe a doc duplicated within one method's ranking
**Duration:** ~25 min · **Branch:** `session/2026-06-22-1106-issue-65`

- Found during Phase A code-reading: `reciprocal_rank_fusion` assumed each method ranks each doc at most once. A duplicate `doc_id` within one method's list double-added the `1/(k+rank)` term (corrupting the fused score) and overwrote the recorded per-method rank with the worse (last) one. Real inputs produce this — a union of two SQL paths, a hybrid surfacing one row via two routes, or an upstream dedup bug — and the result was a silently mis-ordered list with no error.
- Fix: per-method `seen` set; count only the first (best-rank) occurrence per method, matching Cormack et al. (2009)'s "one term per ranker per doc". Cross-method fusion is unchanged.
- 3 new tests (score-once-at-best-rank, records-first-rank, cross-method-ordering-preserved). Suite 383 → 386, 7 pg tests skipped locally. Ruff clean. PR #66 ready.

**Why this work, this session:** the portfolio is saturated (only binary demo-capture tasks open). This was a real correctness bug in the repo's core hybrid-retrieval math, untested before now — strictly higher value than a synthetic fill.

**Open questions / blockers:** none.

**Next session:** `PhaseTimings.to_dict()` + `Aggregate.to_dict()` shipped without symmetric `from_json`/`from_dict` readers (flagged in eval-harness session memory as the third hop of the from_json propagation chain). Verify a real consumer needs the reader before filling it — otherwise it's API-completeness, not a bug. The equal-fused-score tie-break ordering is also undocumented (deterministic insertion order today).

## 2026-06-22 — Issue #67: enforce_citations — false refusal on a claim-less fragment
**Duration:** ~25 min · **Branch:** `session/2026-06-22-1943-issue-67`

- Found via a Phase A Explore-subagent sweep over the RAG core (fusion/retriever/reranker/generator/streaming/telemetry); rag-production-kit picked as a priority-tier repo under the D-009 loop bias after mcp-server-cookbook#54 was skipped (a `decision-revisit` security-guard change deliberately gated for JT, D-007 fall-through). RRF in `fusion.py` was verified correct (1-indexed rank, per-method dedup, validated k). The real bug: `split_sentences` only dropped whitespace-only fragments, so a stray terminator that splits off a punctuation-only fragment (a lone `.`) survived as a "sentence"; `enforce_citations` then demanded a `[cite:...]` marker on it and refused an otherwise fully-cited answer — a false refusal in the D-009 citation guard.
- Fix: drop fragments with no alphanumeric content. A real claim requires a word/number and digits stay alphanumeric, so a bare-number fragment remains enforced — this can't mask an uncited claim. The "contained no sentences" guard and the missing-marker / dangling-id / whitespace behavior are preserved.
- 4 tests (claim-less fragment dropped, number fragment kept, fully-cited answer with stray terminator accepted, all-claim-less text still refused). The 3 behavior-change tests fail pre-fix. Suite 386 → 390, ruff clean. PR #68 ready.

**Why this work, this session:** the repo had zero open issues; a dogfood sweep of a priority-tier repo surfaced a real false-refusal bug in a headline RAG feature (citation enforcement) — higher value than no work.

**Open questions / blockers:** none for this issue. Separately: the naive sentence split mis-handles abbreviations (`U.S.`, `e.g.`) on a real Anthropic generator path — deliberately deferred as a larger design question (the docstring documents the simple split as a generator-cooperation contract).

**Next session:** citation enforcement now tolerates claim-less fragments. The abbreviation-aware-splitting question is the remaining lead if a future session wants to harden the real-generator path.

## 2026-06-22 — Issue #69: generator — _top_score clamped negative rerank scores to 0.0
**Duration:** ~20 min · **Branch:** `session/2026-06-22-2339-issue-69`

- Found via a Phase A dogfood Explore agent over the RAG core, then verified end-to-end myself. `_top_score` seeded `best = 0.0` and only updated on `score > best`, so an all-negative score set returned `0.0` instead of the true (negative) maximum — contradicting its "Return the maximum confidence score" docstring. Negative scores are reachable with the shipped `LexicalOverlapReranker` (`overlap − length_penalty·len(text)`); a real 200-char zero-overlap chunk scores `-1.0`. `fused_score` (RRF) is always positive, so the bug bites only when rerank scores are present and all negative.
- Impact: (1) the `Refusal.top_score` was reported as `0.0` instead of the real negative value (misleading telemetry under the default 0.02 threshold), and (2) at a non-positive `threshold` the `0.0` clamp made `top < threshold` False, so the kit generated an answer from chunks it should have refused as insufficient context (a decision flip).
- Fix: return `max(...)` over `rerank_score` (or `fused_score`) across chunks, preserving the empty-list → 0.0 behavior. 3 regression tests (true negative max; refusal reports real negative top_score; `threshold=0.0` all-negative refuses). All three fail pre-fix. Suite 390 → 393, ruff clean. PR #70 ready.

**Why this work, this session:** rag-production-kit is a priority-tier repo with no open issues; a dogfood sweep surfaced a real correctness defect in a headline feature (the weak-context refusal gate) — strictly higher value than synthetic filler.

**Open questions / blockers:** none.

**Next session:** no specific `generator.py` lead remains; the refusal gate now reports and compares the true max. (The earlier abbreviation-aware-splitting lead from #67 is still open as a larger design question.)

## 2026-06-23 — Issue #71: TemplateGenerator falsely refused multi-sentence chunks
**Duration:** ~25 min · **Branch:** `session/2026-06-23-0349-issue-71`

- Fixed a false-refusal bug in `TemplateGenerator`. It emitted one `[cite:id]` marker per chunk by wrapping the whole chunk text in a single template sentence. Since the indexer stores whole-document prose, a retrieved chunk is routinely multi-sentence; `enforce_citations`' `split_sentences` then fragmented the output and left every sentence but the last uncited, yielding a false `unparseable_output` refusal. The in-repo corpus only avoided it because every chunk was hand-authored as a single period-less sentence.
- Now emits one cited sentence per sentence inside each chunk (same splitter the validator uses); each chunk still yields one deduped Citation. Corrected the false `# pragma: no cover` note. Added a multi-sentence-chunk test. Red pre-fix, green post-fix. Suite 393 → 394 (7 Postgres-gated skips), ruff clean.

**Why this work, this session:** found by a second-pass deep read in the night session's Phase A dogfood wave (the first pass on this repo came back clean). A real correctness bug on the shipped `TemplateGenerator` path used by `evals/run_eval.py`.

**Open questions / blockers:** none.

**Next session:** `TemplateRewriter` sub-query dedup (redundant-but-correct) left out of scope.

## 2026-06-23 — Issue #73: single-sub-query rewrite reranked against the rewritten query, not the original
**Duration:** ~25 min · **Branch:** `session/2026-06-23-1910-issue-73`

- A Phase A dogfood sweep of the retriever path found that `Retriever.search(rewriter=..., reranker=...)` reranked against the wrong query when the rewriter returns a single sub-query (the common no-decomposition case). `_hybrid_search` uses its `query` arg for both retrieval and reranking, and the single-sub-query branch passed the rewritten query — so the reranker scored candidates against the reformulated string instead of the user's original query.
- The multi-hop path already reranks against the original query (with a comment and a locking test); this makes the single-sub-query path honor the same contract. Added a `rerank_query` param to `_hybrid_search` (defaults to `query`, so other callers are untouched) and a spy-reranker regression test. Suite 395 passed / 7 pg-skipped, ruff clean.

**Why this work, this session:** the portfolio's only open `priority:high` issues elsewhere were operator-blocked or `decision-revisit` security work; a parallel dogfood pass over three priority-tier repos surfaced this real, well-grounded correctness gap on a documented public seam.

**Open questions / blockers:** none.

**Next session:** none specific to this issue.

## 2026-06-24 — Issue #75: LexicalOverlapReranker.length_penalty had a sign-only guard (NaN/Inf slipped through)
**Duration:** ~20 min · **Branch:** `session/2026-06-23-2345-issue-75`

- A Phase A dogfood code-read of the reranker path found that `LexicalOverlapReranker.__init__` validated `length_penalty` with a sign-only `length_penalty < 0` check. `NaN < 0` and `inf < 0` are both `False`, so a non-finite penalty was accepted and then poisoned every candidate's score (`overlap - length_penalty * len(text)` → NaN/-Inf). All-NaN scores sort as a no-op (NaN comparisons are false), so the relevant chunk was silently *not* surfaced first — a plausible-looking but wrong ranking with no error.
- Reproduced: `LexicalOverlapReranker(length_penalty=NaN)` constructs, and reranking ranks the irrelevant doc first (scores `[nan, nan]`) vs a finite penalty ranking the relevant doc first (`0.969` vs `-0.039`). `+inf` accepted too.
- Fix: widened the guard to `not math.isfinite(length_penalty) or length_penalty < 0`, mirroring the repo's finiteness sweep on `fusion.k` (#38), `telemetry.ModelPrice`, and `CostRecord` latency. Updated the existing negative-penalty test's message match and added NaN/+Inf/-Inf rejection tests plus a finite-penalty-ranks-relevant-first regression test. Red pre-fix / green post-fix. Suite 399 passed / 7 pg-skipped, ruff clean.

**Why this work, this session:** 4th issue of a multi-issue DAY run; after three core fixes (`llm-eval-harness` #85, `llm-cost-optimizer` #83, `agent-orchestration-platform` #53) the priority-tier data/loader paths were saturated, and this was the one reranker knob still on a sign-only check — a genuine deviation from the repo's own documented finiteness sweep, not a fabricated find.

**Open questions / blockers:** none.

**Next session:** `CohereReranker`'s `batch_size` / `timeout` are a separate path, not audited this session — a possible follow-up only if a concrete gap surfaces.

---
## 2026-06-24 — Issue #78: non-finite threshold silently bypassed the generator refusal gate
**Duration:** ~24 min · **Branch:** `session/2026-06-24-0335-issue-78`

- Both `TemplateGenerator.generate` and `AnthropicGenerator.generate` used `threshold` in the gate `if top < threshold:` without validating finiteness. A NaN (or -Inf) threshold made the comparison False, bypassing the gate and answering from chunks that should be refused; +Inf forced an unconditional refusal. No diagnostic.
- Added a shared `_validate_threshold` helper rejecting non-finite thresholds with a descriptive ValueError, wired into both generators. Finite negative thresholds stay valid (`_top_score` can be negative, #69). Added the missing `import math`.
- 5 new tests (parametrized NaN/±Inf, finite-negative still answers, Anthropic rejects NaN before client call). Red via `git stash`, green after. Suite 399 → 404, ruff clean.

**Why this work, this session:** rag-production-kit was the next priority-tier repo by build-sequence tie-break; the reranker/retriever/fusion/telemetry paths were saturated, so a dogfood sweep surfaced the generator's unguarded refusal threshold.

**Open questions / blockers:** none. Process note: I started editing before cutting the session branch this round — caught it at the red-check (stash showed "WIP on main"), moved the changes onto the branch, and filed the issue + plan before committing. Watch the branch-first ordering next time.

**Next session:** embedder.py / indexer.py / streaming.py remain the dogfood frontier in this repo.

---
## 2026-06-24 — Issue #80: non-finite latency corrupted the p50 metric and emitted invalid JSON
**Duration:** ~25 min · **Branch:** `session/2026-06-24-1532-issue-80`

- The public `telemetry.percentile(values, q)` validated empty input and the `q` range but not the finiteness of `values`. A non-finite `total_latency_ms` flowed into `aggregate()`, where `sorted()` put the NaN in an implementation-defined slot, so `latency_p50_ms` came back `nan` (position-dependent) and `dump_aggregate_json` serialized the bare `NaN` token — invalid JSON a strict log-tailer rejects whole. `CostRecord.build` already guards latency (#38), but direct dataclass construction (no `__post_init__`) bypasses it and `percentile` is exported on its own.
- Added a finiteness guard on `values` in `percentile()` (raise after the empty-check), matching the q-range guard and the `PhaseTimings.record`/`CostRecord.build` posture. 5 tests, red-without / green-with, full suite 409 pass / 7 Postgres-skip, ruff clean.

**Why this work, this session:** found via a Phase A dogfood sweep (targeting the less-recently-touched fusion/telemetry/streaming modules) and reproduced end-to-end; rag-production-kit was next in build sequence among the priority tier (D-009) this run.

**Open questions / blockers:** none.

**Next session:** a `CostRecord.__post_init__` finiteness guard (the data-boundary fix for direct construction) is a narrower follow-up; the `bool`-`q` coercion is a low-impact runner-up (q only ever passed internally as 0.5/0.95/0.99).

---
## 2026-06-24 — Issue #82: NaN/Inf embedding components reached pgvector unvalidated
**Duration:** ~20 min · **Branch:** `session/2026-06-24-1915-issue-82`

- `db.to_pgvector()` formatted an embedding vector with `repr(float(v))` and did no finiteness check. Both embedding entry points funnel BYO-`Embedder` output through it — the indexer write-path (`add_documents`) and the retriever query-path (`_hybrid_search`) — so a non-finite component (normalization divide-by-zero, `Inf` overflow, or a NaN-poisoned model) reached pgvector as the bare token `nan`/`inf`, surfacing either as an opaque `NaN not allowed in vector` error far from the embedder seam or as silent dense-channel ordering corruption on a tolerant build.
- Validated finiteness in `to_pgvector` itself (the single chokepoint both paths share), raising a `ValueError` that names the offending index. Added the first dedicated `to_pgvector` tests (5). Full suite green locally (pg-integration skips without a live DB), ruff clean.

**Why this work, this session:** the #80 session close explicitly named embedder/indexer as the next dogfood frontier; this is the direct rag-kit sibling of the `llm-cost-optimizer` `_validate_embedding` fix (#88) that was merged in this same run's Phase A review pass. rag-production-kit is in the D-009 priority tier.

**Open questions / blockers:** none.

**Next session:** empty-vector / dimension-mismatch validation at `to_pgvector` is a possible narrow follow-up, but pgvector's own dimension check already surfaces it clearly; streaming.py remains on the dogfood frontier.

---
## 2026-06-25 — Issue #84: deterministic RRF tie-breaking
**Duration:** ~20 min · **Branch:** `session/2026-06-25-1921-issue-84`

- `reciprocal_rank_fusion` sorted by fused score alone, so tied docs fell back to `scores` dict insertion order — which depends on the incidental order methods appear in the `rankings` mapping and the order doc ids appear within each list. RRF ties are common (`1/(k+rank)` sums collide for any symmetric rank configuration), so the same rankings could produce a different top-k purely from how the caller ordered the methods. For a production retriever feeding top-k into an LLM, that's a reproducibility bug.
- Added a doc-id-ascending secondary sort key (`key=lambda row: (-row[1], row[0])`). Same class as the chunking-strategies-lab cosine-tie fix (#69) merged earlier this run. 4 red-green tests (permutation-invariance and within-method-order fail without the fix; the other two pass coincidentally because their tie data already aligns with insertion order). Full suite 420 passed / 7 skipped, ruff clean.

**Why this work, this session:** rag-production-kit was the next priority-tier repo in build sequence this multi-issue day session and had zero open issues; dogfooding the fusion math surfaced a real, reachable determinism bug in the same tie-break class the portfolio just fixed elsewhere.

**Open questions / blockers:** none.

**Next session:** rag-production-kit has no open issues again; dogfood another core module (reranker/retriever/streaming) for the next edge-case fix.

## 2026-06-26 — Issue #86: guard CohereReranker relevance_score finiteness
**Duration:** ~20 min · **Branch:** session/2026-06-26-0327-issue-86

- `CohereReranker.rerank` used the Cohere API's `relevance_score` with no finiteness check. The API is an external, uncontrolled source — a malformed/erroring response can return NaN/Inf, which flows into `rerank_score` → `generator._top_score`'s `max()` → the refusal gate `top < threshold`. `NaN < threshold` is `False`, so the generator answers from chunks it should have refused.
- `_validate_threshold` (#78) already guards the operator-supplied-threshold half of this gate; this guards the API-supplied-score half, matching the #80/#82 external-value finiteness guards. `LexicalOverlapReranker` scores are finite by construction (#75), so only the Cohere path was exposed.
- Reject at the merge seam with a `ValueError` naming the source; 4 tests via the existing `_FakeCohereClient` scaffold. Full suite + ruff green.

**Why this work, this session:** Surfaced by a Phase A dogfood Explore agent and hand-verified before filing; closes the last unguarded operand of the refusal gate.

**Open questions / blockers:** none. A parallel dogfood flagged `vector-search-at-scale`'s `BenchmarkResult` lacking a `__post_init__` finiteness guard (low reachability — `ingest_seconds==0` basically never occurs with `perf_counter`); deferred as a possible low-pri follow-up.

**Next session:** rag-production-kit refusal gate is now fully guarded on both operands.

## 2026-06-26 — Issue #88: citation gate tolerates incidental whitespace in [cite:...] markers
**Duration:** ~30 min · **Branch:** `session/2026-06-26-2006-issue-88`

- `enforce_citations` matched `[cite:<id>]` and looked the captured id up without stripping. The LLM (`AnthropicGenerator`) path routinely emits incidental spaces — `[cite: doc1]`, `[cite:doc1 ]` — so the padded id mismatched the real `external_id`, read as a *dangling* citation, and a fully-grounded, correctly-cited answer was thrown away as `unparseable_output`. That's the worst RAG failure mode: a right answer surfaced as a refusal over a cosmetic quirk.
- Fixed by stripping the captured id before the `allowed` lookup and before keying the `seen` dedup map. Strictly more lenient and can't create a false-accept: a genuinely-unknown id still misses `allowed`, and corpus external_ids never carry leading/trailing whitespace. 7 new tests (4 parametrized whitespace markers, dedupe-across-variants, padded-unknown-still-dangling). Suite 423 → 430, ruff clean.

**Why this work, this session:** third repo of a multi-issue DAY run; rag-production-kit is priority-tier and ~15h stale. With no open backlog, a Phase A code-read of the headline citation-enforcement feature surfaced a real false-refusal path on the production LLM output, not just the hermetic template path.

**Open questions / blockers:** none.

**Next session:** citation-id matching now tolerates the common formatting variance; internal-whitespace ids (`[cite:doc 1]`) remain strict by design (a real internal space is a distinct external_id).

## 2026-06-26 — Issue #90: Reranker length penalty overriding relevance
**Duration:** ~25 min · **Branch:** `session/2026-06-26-2330-issue-90`

- `LexicalOverlapReranker` scored candidates as `overlap - length_penalty * len(c.text)`. `overlap` is bounded in `[0, 1]`, but the penalty grew unbounded with raw character count, so at realistic chunk sizes (hundreds of chars) it routinely exceeded the gap between distinct overlap levels — a less-relevant short chunk outranked a more-relevant long one. That contradicts the class's documented "the penalty is only a tie-breaker" contract. Reproduced: for a 4-token query, a chunk with overlap 0.5 (~480 chars) was demoted below a chunk with overlap 0.25 (21 chars).
- Fixed by bounding the length factor to `[0, 1)` via `len/(len+1)`, so `penalty ∈ [0, length_penalty)` — below the smallest overlap quantum for a tiny coefficient. `length_penalty=0.0` still yields `score == overlap`, and equal-overlap ties still break toward shorter (the factor is monotonic in length), but a genuine overlap difference can no longer be flipped. 2 regression tests; suite 430 → 432, ruff clean.

**Why this work, this session:** third issue of a multi-issue DAY run. After llm-eval-harness #105 and llm-cost-optimizer #98 I rotated to the next priority-tier repo in build sequence (rag-production-kit), which had no open backlog, so I dogfooded it with an Explore agent and filed #90 from a reproduced finding. The existing tie-break test only used equal-overlap chunks, so it never caught a penalty that overrides a real overlap difference.

**Open questions / blockers:** none.

**Next session:** three dogfood runners-up remain unfiled — `split_sentences` fractures abbreviations (false `unparseable_output` refusal), `_CITE_PATTERN` rejects external_ids containing `]`, and `PhaseTimings.summary` reports only p50/p95 while `telemetry.Aggregate` adds p99. File individually if a session needs small work here.

## 2026-06-27 — Issue #92: rewriter leaks the "Then" connective before punctuation
**Duration:** ~20 min · **Branch:** `session/2026-06-27-0324-issue-92`

- `_split_then` splits a sequential query on `_THEN_SPLIT` (lookahead `then\b` — a *word boundary*), but the per-part cleanup only stripped the connective when it was followed by a literal space (`startswith("then ")`). So `Then, ...`, `Then; ...`, `Then- ...` all split correctly but **leaked** the connective into the sub-query, contradicting the docstring and polluting the downstream BM25/dense retrieval query. Reproduced on main: `_split_then("Do A. Then, do B.")` → `['Do A.', 'Then, do B.']`.
- Fixed with `_THEN_PREFIX = re.compile(r"^then\b\W*", re.IGNORECASE)` and `s = _THEN_PREFIX.sub("", s)`, so the cleanup mirrors exactly what the split fires on. `then\b` leaves content words like "thence" untouched (no boundary), consistent with the split. Added 3 regression tests (end-to-end punctuated connectives, direct `_split_then` leak cases, and a "thence" no-false-split guard). Suite 431 → 435, ruff clean.

**Why this work, this session:** second issue of a multi-issue NIGHT run. A parallel dogfood agent surfaced this as a "borderline lead" it declined to promote (calling it heuristic-quality), but it is a genuine docstring-vs-behavior contract mismatch with a clean, minimal fix — worth closing.

**Open questions / blockers:** none.

**Next session:** the `then` connective cleanup now matches its split semantics; broader connective vocabulary ("next", "after that", "finally") remains intentionally out of scope.

## 2026-06-27 — Issue #94: Rewriter 'and' split emitted a malformed double terminator
**Duration:** ~15 min · **Branch:** `session/2026-06-27-1923-issue-94`

- The multi-question `and` decomposition re-appended `?` only when a conjunct didn't already end in `?`, so a question-like conjunct ending in `.` or `!` (e.g. "What is the price! and is it worth it?") got a `?` stacked on top → `What is the price!?`, breaking the documented well-formed-question contract.
- Fixed with `s.rstrip("?.!") + "?"` (idempotent on a lone `?`; leaves a decimal like `3.5` untouched). Added two lock tests — both `.`/`!` conjuncts and a decimal-not-mangled guard; the double-terminator test fails on the pre-fix code. Same class as the just-merged #93 `Then`-connective fix.

**Why this work, this session:** second issue of a multi-issue DAY run; surfaced alongside llm-cost-optimizer #102 by the same Phase A priority-tier dogfood sweep.

**Open questions / blockers:** none. Severity is low (trailing punctuation is ignored downstream by `plainto_tsquery`/embedding).

**Next session:** continue the multi-issue loop if time remains.

## 2026-06-28 — Issue #96: `_THEN_SPLIT` was not case-insensitive despite its docstring
**Duration:** ~20 min · **Branch:** `session/2026-06-28-1931-issue-96`

- `_THEN_SPLIT` drives the sequential-step rewrite (`"X. Then Y."` → two sub-queries) and its docstring claims a case-insensitive `"Then "` match, but the pattern enumerated `Then|then` literally with no `re.IGNORECASE`. All-caps / mixed-case connectives (`"THEN"`, `"ThEn"`) silently returned `no_decomposition`. The tell was an internal inconsistency: every sibling pattern (`_THEN_PREFIX`, `_COMPARE_RE`, `_AND_SPLIT_RE`) already had the flag — only the split omitted it. Reproduced firsthand before filing.
- Fixed by `re.compile(r"(?<=[.!?])\s+(?=then\b)", re.IGNORECASE)` and dropping the redundant alternation. The downstream `_THEN_PREFIX` strip is already case-insensitive, so any casing is cleaned once the split fires; the `then\b` boundary is unchanged so `"thence"` still doesn't false-split (#92 preserved). Added end-to-end + unit regression tests; suite 437 → 439 passed (7 Postgres skips), ruff clean.

**Why this work, this session:** third substantive issue of a multi-issue DAY run, rotating to a fresh priority-tier repo each iteration (after #116 in llm-eval-harness and #104 in llm-cost-optimizer) to avoid same-repo append-only MEMORY conflicts. rag-production-kit had zero open issues, so a Phase A dogfood sweep surfaced this. Two weaker dogfood findings deferred (generator cite-marker-after-terminator false refusal; missing `CohereReranker.batch_size` guard).

**Open questions / blockers:** none.

**Next session:** continue the loop if time remains.

## 2026-06-28 — Issue #98: `rerank_delta_ndcg` reported displacement > 1.0 on duplicate ids
**Duration:** ~20 min · **Branch:** `session/2026-06-28-2317-issue-98`

- A duplicated `external_id` in a ranking pushed `ndcg_displacement` past its documented `1.0` ceiling (1.34 for a tripled top id): `rel` is keyed by id and the ideal is `dcg(before_list)` over the *distinct* `before` order, so a repeated id in `after` re-adds its full relevance into the actual DCG while the ideal stays put — silently-wrong telemetry that would read as "the reranker improved beyond the input ideal," which is impossible. A duplicate in `before` instead double-counts the ideal.
- Fixed with a distinct-id guard that raises `ValueError` (matching the module's existing fail-loud seams on `k`, `length_penalty`, and the Cohere non-finite score), since a valid ranking is a set of distinct ids and a duplicate signals a caller bug worth surfacing. 7 tests added (dup-in-after / dup-in-before raise; a parametrized [0,1] invariant lock over distinct-id permutations). Suite 439 → 446, ruff check + format clean.

**Why this work, this session:** first substantive issue of a multi-issue DAY run. Phase A merged 5 clean PRs across 5 repos and the audit was clean; the only priority-tier stale repo (nextjs-streaming-ai-patterns) had nothing but an interactive JT-bound demo-capture issue, so I fell through to the saturated-portfolio dogfood→issue→PR pattern on priority-tier repos.

**Open questions / blockers:** none.

**Next session:** continue the loop — rotate to another priority-tier repo. Two deferred findings from session #96 remain candidates: generator citation-marker-after-terminator false refusal (med), and CohereReranker batch_size missing positive-int guard (low).

## 2026-06-29 — Issue #100: recall_at_5 docstring said binary, but it's fractional recall@k
**Duration:** ~14 min · **Branch:** `session/2026-06-29-0330-issue-new`

- The `evals/run_eval.py` module docstring described `recall_at_5` as binary — "per-row 1.0 iff every gold chunk id appears in the top-5; 0.0 otherwise" — but `_score_recall_at_5` computes fractional recall@k (`len(hits)/len(gold_ids)`). A half-hit scores 0.5, the intended and already-tested behavior (`test_score_recall_at_5_partial_credit`). The binary wording also contradicted the conventional recall@k definition and the sibling `correctness` suite's correct "fraction" framing.
- Rewrote the docstring bullet to describe fractional recall@k (incl. the empty-gold → 0.0 edge) and added an inline comment at the score site cross-referencing the locking test. `faithfulness` (genuinely binary) and `correctness` (already "fraction") were correct and untouched. No behavior change.

**Why this work, this session:** third issue of the night run. After the priority-tier and 36h-stale repos were exhausted/blocked, a parallel audit subagent swept rag-production-kit, found no logic bug (mature, 445 passing), and surfaced this single doc-contract drift — same class as python-async #68.

**Open questions / blockers:** none.

**Next session:** eval-metric docstrings now match the shipped fractional/binary semantics; recall@k partial-credit remains locked by its test.

## 2026-06-29 — Issue #102: CohereReranker construction args unguarded — `batch_size<=0` silently dropped all candidates
**Duration:** ~25 min · **Branch:** `session/2026-06-29-2325-issue-102`

- `CohereReranker.__init__` chunks candidates via `range(0, n, self.batch_size)`, so a non-positive `batch_size` makes that range empty and `rerank()` silently returns `[]` — every candidate dropped, the Cohere API never called, no error. The `batch_size or DEFAULT` idiom additionally swallowed an explicit `0` into the default `100`, masking operator error. Both were unreachable to guard because the validation sat *after* `import cohere`, so they only bit when the optional extra was installed (production). This is the twice-deferred #96/#98 finding ("batch_size missing positive-int guard … candidate for a future loop").
- Reproduced the silent loss firsthand without the cohere extra, via the module's own `__new__`-bypass test pattern: `batch_size=-1` → `rerank` returned `[]` (len 0), `client.rerank` calls = 0, 5 candidates in / 0 out / no error.
- Fixed by moving construction-arg validation to the top of `__init__`, before `import cohere`: reject non-`int`/`bool`/`<=0` `batch_size` (mirroring the `k` guard) and non-finite/`<=0` `timeout_s` (mirroring `length_penalty`), with `None` still meaning "use the default" (explicit `is None` check replaces the `or` idiom so `0` is rejected, not masked). A bad value now fails loud with a clear `ValueError` at construction even without the extra installed. 19 lock tests (bad `batch_size` ×7, bad `timeout_s` ×7, valid/None fall-through-to-`ImportError` ×5), hermetic because the guard precedes the import; rejection tests confirmed failing on pre-fix code. Suite 446 → 465, ruff clean.

**Why this work, this session:** second substantive issue of a multi-issue DAY run (after `llm-eval-harness` #122). Rotated to priority-tier `rag-production-kit`; a dogfood hunter found the repo otherwise clean but surfaced this real silent-wrong sibling-gap it couldn't reproduce itself (no cohere extra) — already a tracked #96/#98 deferred item.

**Open questions / blockers:** the generator citation-marker-after-terminator false-refusal finding (#96/#98) is still deferred — contract-ambiguous, needs a JT call on the marker-placement contract.

**Next session:** continue the loop on another repo to avoid same-repo append-only MEMORY conflicts; portfolio remains saturated.

## 2026-06-30 — Issue #104: ModelPrice.cost token guard was sign-only — NaN tokens → invalid JSON in the dashboard aggregate
**Duration:** ~20 min · **Branch:** `session/2026-06-30-0324-issue-104`

- `ModelPrice.cost` (`telemetry.py:71`) guarded token counts sign-only (`prompt_tokens < 0 or completion_tokens < 0`). A `NaN` slipped through (`NaN < 0` is `False`), and `float`/`bool` slipped through (no type check). The `NaN` then propagated `NaN * rate / 1_000_000` → `prompt_usd` → `CostRecord.total_usd` → `aggregate().total_usd`, and `dump_aggregate_json` serialized it as the bare token `NaN` — invalid JSON that a strict log-tailer rejects whole. The rate fields were already finiteness-tightened (#38) and percentile (#80), but this token-count seam was missed; `CostRecord.build` only finite-checks latency and delegates tokens to `cost()`, so there was no upstream backstop.
- Fixed by tightening the guard to non-negative `int` (rejecting `NaN/inf/float/bool`), mirroring the module's `dim`/`max_chunks`/`k` int contracts and keeping the `"non-negative"` substring so the existing negative-token test still matches. Guarding in `cost()` covers both direct callers and the `build` path.
- Parametrized lock test (`nan/inf/-inf/1.0/True` × prompt/completion) plus a valid-int over-rejection guard, confirmed failing pre-fix via `git stash`. Suite 474 → 476, ruff clean (split a compound assert per PT018).

**Why this work, this session:** fourth issue of a NIGHT multi-issue run; a dogfood hunter surfaced this in priority-tier `rag-production-kit`, reproduced firsthand before acting. Extends the #38/#80 finiteness sweep to its last seam.

**Open questions / blockers:** none — ready for review.

**Next session:** continue the loop.

## 2026-06-30 — Issue #106: `to_sse` emitted `NaN`/`Infinity` (invalid JSON), breaking the browser EventSource frame
**Duration:** ~20 min · **Branch:** `session/2026-06-30-1530-issue-106`

- `to_sse` (`streaming.py:342`) serialized payloads with `json.dumps(..., default=str)`. `default=str` rescues non-serializable *objects* but never floats, so the C encoder's default `allow_nan=True` emitted the bare tokens `NaN`/`Infinity`/`-Infinity` — invalid JSON. A browser's `EventSource` runs `JSON.parse` on the `data:` line and rejects the whole frame, silently breaking the stream. Reachable via free-form caller `metadata` (flows verbatim through `_chunk_to_event`) or a reranker's `rerank_score`. Reproduced firsthand.
- Fixed with a small recursive `_json_safe(obj)` that maps non-finite floats to `None` (walking dict/list/tuple) applied before `json.dumps` — parity with JS `JSON.stringify(NaN)`/`JSON.stringify(Infinity)` (both → `null`), keeping the documented "keep streaming alive, don't raise" `to_sse` contract. Scope is `to_sse` only; non-serializable objects still fall through to `default=str`.
- +8 tests (nested non-finite in metadata, top-level score + list item, `_json_safe` unit, finite over-rejection guard). Inverse safety net: reverting only the `to_sse` dumps line (helper kept) fails the metadata/score cases pre-fix. Suite 476 → 484 (7 PG-skipped), ruff clean.

**Why this work, this session:** third issue of a DAY multi-issue run; `rag-production-kit` had **zero open issues**, so dogfood-and-file (Phase B step 5). Read `streaming.py` myself while an Explore hunter scanned the other modules in parallel — it independently confirmed the `to_sse` finding and surfaced a sibling (filed as **#108**: `CostRecord.build` guards `total_latency_ms` but not `per_phase_ms` values → invalid JSON in the telemetry store). Left #108 for a future session to avoid a same-repo append-only MEMORY conflict with this PR. Picked rag-production-kit per the priority-tier build sequence after llm-eval-harness (#126) and llm-cost-optimizer (#114), both shipped earlier this run.

**Open questions / blockers:** none — ready for review.

**Next session:** continue the loop on another repo (chunking-strategies-lab next in the priority-tier build sequence); #108 awaits #107 merging first.

## 2026-06-30 — Issue #108: CostRecord.build guarded total_latency_ms but not per_phase_ms values → invalid JSON in the telemetry store
**Duration:** ~20 min · **Branch:** `session/2026-06-30-1933-issue-108`

- `CostRecord.build` (`rag_kit/telemetry.py`) validated `total_latency_ms` for finiteness (#38) but never checked the `per_phase_ms` *values*. A non-finite phase value reached `TelemetryStore.record`, which persists the map with `json.dumps(... allow_nan=True)` — writing the bare token `NaN`/`Infinity` (invalid JSON). `since()` then catches the `JSONDecodeError` on read-back and swallows the whole row's phases to `{}`, silently dropping the data. Same "no invalid JSON in a serialization sink" class as the `to_sse` fix (#106/#107), but the correct chokepoint here is ingestion — exactly where `total_latency_ms` is already guarded.
- Fixed by extending the latency contract to each per-phase value: reject NaN/±Inf, negative, non-numeric, and `bool` (an `int` subclass), raising `ValueError` that names the offending phase, placed right after the existing latency guard. +8 tests: parametrized NaN/+Inf/-Inf, a negative value, and non-numeric `str`/`None`/`bool` — each naming the phase and confirmed failing pre-fix (stash-and-rerun); plus a strict-JSON lock that a finite map persists with no `NaN`/`Infinity` tokens (raw SQLite column read through `json.loads(..., parse_constant=raise)`). Suite 484 → 492 (7 PG-skipped), ruff clean.

**Why this work, this session:** second issue of a DAY multi-issue run (after closing nextjs-streaming-ai-patterns #70). #108 was the followup filed during the #106 dogfood; it became actionable once #107 merged in this run's Phase A. Picked per the rule-3 tie-break (priority-tier, med-priority) after llm-cost-optimizer #97 fell through as a `decision-revisit` one-way blocker (D-007) needing JT.

**Open questions / blockers:** none — ready for review.

**Next session:** the read-path `since()` silent `JSONDecodeError`→`{}` swallow is now unreachable from clean ingestion but remains a latent read-side gap if a row is corrupted out-of-band — a candidate followup. Continue the loop.

## 2026-07-01 — Issue #110: split_sentences split on abbreviations, falsely refusing fully-cited answers
**Duration:** ~30 min · **Branch:** `session/2026-07-01-1915-issue-110`

- The citation enforcer's sentence splitter (`split_sentences`) treated every period-then-space as a sentence boundary, so any common abbreviation — `Dr.`, `Mr.`, `U.S.`, `e.g.`, `Inc.`, `vs.` — stranded a claim-less leading fragment (`"Dr."`). That fragment has alphanumeric content, survived the claim-less-fragment filter, and then `enforce_citations` demanded its own `[cite:...]` marker; the marker sits on the *real* sentence, so a fully-grounded, correctly-cited LLM answer was falsely refused as `unparseable_output`. Reproduced firsthand before filing (`"Dr. Smith discovered penicillin [cite:A]."` → `['Dr.', 'Smith discovered penicillin [cite:A].']`).
- Fixed with an abbreviation-aware merge pass: after the regex split, a fragment is re-joined with the next when it ends in a curated abbreviation or a single-letter initial. Deliberately lenient and documented — a genuine sentence ending in `Dr.`/`U.S.` is vanishingly rare, and the rare over-merge is far cheaper than refusing every answer that mentions an abbreviation. Also cleans up `TemplateGenerator`'s per-chunk rendering. +10 tests (each abbreviation family, single initial, multi-abbreviation, an over-merge/numeric-marker guard, and an end-to-end no-false-refusal regression); suite 502 pass / 7 skip, ruff clean. Inverse safety net confirmed via `git stash`.

**Why this work, this session:** first issue of the DAY run. `rag-production-kit` was the stalest priority-tier repo (20h) and earliest in the build sequence; with zero open issues, two parallel dogfood hunters drove the work. This bug was the higher-impact of two real finds — it defeats the repo's headline citation-enforcement guarantee on ordinary prose.

**Open questions / blockers:** none — PR #112 ready for review.

**Next session:** #111 (priority:low) tracks the second, lower-impact find — non-ASCII sentence terminators (`？！。؟`) leaving a doubled terminator in `TemplateRewriter`'s "and"-split. Continue the loop.

## 2026-07-02 — Issue #111: non-ASCII sentence terminators left a doubled terminator
**Duration:** ~20 min · **Branch:** `session/2026-07-02-0311-issue-111`

- `TemplateRewriter`'s "and"-split re-appends a canonical `?` to each conjunct after stripping the trailing terminator, but the strip set (`rstrip("?.!")`) was ASCII-only. A conjunct ending in a full-width `？`/`！`, ideographic `。`, or Arabic `؟` (common CJK/Arabic IME input) kept its terminator and got a `?` stacked on top — e.g. `"where did they work？?"`. Fixed with a module-level `_TERMINATORS = "?.!？！؟。"` used at both strip sites, extending the #94 well-formed-question contract to non-ASCII enders.
- Reproduced firsthand before and after (all four enders); +5 tests (4-param non-ASCII well-formedness + an internal-char-untouched guard). Inverse safety net: the 4 parametrized tests fail pre-fix. Suite 502 → 507, ruff + format clean.

**Why this work, this session:** first issue of the NIGHT run. #111 was the only actionable, non-blocked, non-demo open issue across all 12 repos — filed as a followup by the prior session's dogfood hunt. The portfolio is deeply saturated, so a pre-reproduced tracked bug was the highest-value pick.

**Open questions / blockers:** none — PR ready for review.

**Next session:** continue the loop.

## 2026-07-02 — Issue #114: bench_streaming.py had no --k/--n validation (traceback on --k 0, silent empty run on --n 0)
**Duration:** ~20 min · **Branch:** `session/2026-07-02-2348-bench-streaming-argval`

- `scripts/bench_streaming.py::main` had no CLI input validation, unlike its sibling `scripts/bench_rewriter.py` (which already does `parser.error("--k must be positive")`). So `--k 0` surfaced `StreamingPipeline.run`'s `ValueError` as a **raw traceback** (it raises before the pipeline's own try/except, so it escaped `main()`), and `--n 0`/negative **silently exited 0** with an all-zero four-phase table that reads as a real run. Reproduced both firsthand.
- **Fix:** added `if args.n <= 0: ap.error("--n must be positive")` and the same for `--k`, right after `parse_args()` — mirroring the sibling. `ap.error(...)` prints usage + message to stderr and exits 2. +5 subprocess regression tests (4 parametrized zero/negative `--k`/`--n` → exit 2, clean message, no traceback, no table; 1 smallest-valid `--n 1 --k 1` still runs); all 4 nonpositive tests fail pre-fix. Suite 507 → 512, ruff + format clean.

**Why this work, this session:** third issue of a DAY run. After shipping llm-cost-optimizer #120 and mcp-server-cookbook #78, the portfolio was comprehensively freshly-hunted (5 clean dogfood hunts this run + prior-run hunts cover all 12 repos). Both earlier bugs this session lived in *peripheral tooling*, so I ran a peripheral-focused hunt on rag-production-kit (prior run hunted its core clean 5h ago); this `--k`/`--n` parity gap was the one reproducible peripheral defect. A consistency/robustness fix, not a wrong-output-on-valid-input bug — shipped because two sibling bench scripts handling the same flag differently (one clean error, one traceback) is a real operator-facing defect with a clear in-repo precedent.

**Open questions / blockers:** none — ready for review.

**Next session:** continue the loop. Portfolio remains saturated; peripheral/tooling surfaces are where the remaining small defects live.

## 2026-07-03 — Issue #116: bench_rewriter._print_md left the query cell's `|` unescaped (GFM table corruption)
**Duration:** ~20 min · **Branch:** `session/2026-07-03-0329-issue-116`

- `_print_md` (`scripts/bench_rewriter.py`) interpolated the free-form `r.query` into a GFM table cell wrapped in backticks but **without escaping `|`**. Backticks don't protect a literal pipe — GFM splits table cells on unescaped pipes before parsing inline-code spans — so a query with a pipe injected a spurious column (5 cells vs. the 4-column header). Reproduced firsthand (`compare cats | dogs` → 5 cells).
- **Fix:** `query = r.query.replace("|", "\\|")` before the f-string, mirroring the four sibling emitters already fixed (`comment._row_to_md` #130, `calibration.render_report` #134, `aggregate_markdown` #79, `run_matrix._render_summary` #100). +2 regression tests (piped query → exactly 4 GFM cells; pipe-free output unchanged). Latent today — the 8 shipped queries are pipe-free, so the README snapshot is byte-identical. Suite 512 → 514.

**Why this work, this session:** third issue of a NIGHT run. After five dogfood core-hunts (llm-eval-harness hit → #138; chunking, vector-search, agent-orchestration, python-async all clean), I ran a targeted direct grep for the recurring GFM pipe-escaping class across every table emitter in all 12 repos. Four were already escaped (#130/#134/#79/#100); this `_print_md` emitter was the one miss.

**Open questions / blockers:** none — ready for review.

**Next session:** continue the loop. Portfolio is deeply saturated; remaining open work is JT-blocked decision-revisits and operator-verification demos.

## 2026-07-03 — Issue #118: symbol-resolution doc-lock (propagates portfolio-ops #55) (~20 min)

**What got done.** Second per-repo propagation of the portfolio-ops #55 symbol-resolution lock (after llm-eval-harness #140). Added `test_doc_symbol_refs_resolve` to `tests/test_architecture_doc.py`: fully-qualified `rag_kit.<module>.<symbol>` refs (`io_utils.atomic_write_text`) resolved via importlib+`hasattr` (the #71 shape), and multi-word CamelCase public types (`HashEmbedder`, `CohereReranker`, `UnknownModelError`, …) checked against the `rag_kit` public surface. Firsthand-verified that the doc's `RunResult` token is **eval-harness's** type (evals-pipeline section), not rag_kit drift — rag_kit ships `RetrievalResult`/`RewriteResult`. Added a hard-pinned `EXTERNAL_SYMBOLS = ("RunResult",)` allowlist, a hard-pin test, and a shadow test that fails if an allow-listed name later appears in the rag_kit surface. Inverse-verified both drift styles are flagged and `RunResult` stays exempt. Suite 514 → 517, ruff clean.

**Why this work, this session:** fourth issue of the DAY run. After llm-eval-harness #140 shipped the first propagation, continued the systemic #55 effort to a second priority-tier repo. The key lesson: this propagation is genuinely per-repo — emb-shootout #71 is fully-qualified, llm-eval #140 is bare-submodule + bare-CamelCase, rag #118 is fully-qualified + CamelCase-with-external-allowlist. TS repos will need an exported-name check, not importlib.

**Open questions / blockers:** none — ready for review.

**Next session:** continue #55 propagation (llm-cost-optimizer, chunking, prompt-regression-suite, vector-search-at-scale, agent-orchestration-platform, mcp-server-cookbook; TS: nextjs, ai-app). Remaining non-propagation work stays JT-blocked.

## 2026-07-06 — Issue #120: CI gap for the Next.js demo
**Duration:** ~20 min · **Branch:** `session/2026-07-06-1515-issue-120`

- The Next.js demo (`demo/nextjs`) ships a Vitest suite (`corpus.test.ts`, `stream-route.test.ts` — 13 tests exercising corpus retrieval and the SSE `/api/stream` protocol), but `ci.yml` defined only Python jobs (ruff + pytest). No `npm`/`node` step ran anywhere, so a demo regression would pass CI silently.
- Added a `nextjs-demo` job: `setup-node@v4` (node 20, matching `engines.node >=20`), npm cache on the lockfile, `npm ci`, `npm run typecheck`, `npm test`. Excluded `next lint` (no ESLint config → interactive, not CI-safe) and `next build` (out of scope).
- Verified locally (`typecheck` exit 0, `npm test` 13 passed) before pushing.

**Why this work, this session:** Found via a cross-repo CI-coverage lens sweep after the portfolio's static issue queue was exhausted (all remaining issues JT-gated or headless-demo). Same enforcement-gap class as mcp-server-cookbook#90; both were multi-language repos with a second stack not wired into CI.

**Open questions / blockers:** none.

**Next session:** The CI-coverage lens is now swept across all 12 repos — only these 2 gaps existed (both multi-stack repos); the other 10 single-stack repos are clean. Don't re-sweep. Remaining open issues are JT-gated decision-revisits or headless demo captures.

## 2026-07-07 — Issue #122: Dashboard omits the documented p99 latency tile
**Duration:** ~20 min · **Branch:** `session/2026-07-07-1522-issue-122`

- The served telemetry dashboard rendered only p50 + p95 tiles, but `README.md:80` and `docs/architecture.md:234` both promise p50/p95/p99. `Aggregate` already computes `latency_p99_ms`; the render just dropped it. Reproduced firsthand on clean main (`p99 in html: False`).
- Added a p99 tile next to p95 (no fabricated number — p99 is computed) and widened the CSS grid `repeat(4,1fr)` → `repeat(5,1fr)`. New lock test pins p50/p95/p99 + the 5-column grid to the rendered HTML. Full suite 524 passed / 7 skipped; ruff clean.

**Why this work, this session:** Doc-vs-code drift on the *served* dashboard HTML — a sub-lens the earlier run-shipped-example sweep (stdout reproduce) missed. Found by a parallel dogfood hunt on the non-core scripts surface, verified firsthand.

**Open questions / blockers:** none. Note: the hunt agent also regenerated `evals/current/*.json`; those were dropped, not committed (benchmark-integrity, no unilateral fixture regen).

**Next session:** bench_streaming / bench_rewriter / capture_demo / run_eval CLI all audited clean; the dashboard p99 omission was the only real non-core finding.

## 2026-07-07 — Issue #124: demo-client AbortController unmount-teardown leak
**Duration:** ~30 min · **Branch:** `session/2026-07-07-2328-issue-124` · **PR:** #125

- The Next.js demo's `demo-client.tsx` wired an `AbortController` (signal → `fetch`, stored in `abortRef`, nulled in `finally`) but **never called `.abort()`** and imported no `useEffect` — the controller was vestigial and there was no unmount cleanup. Since a browser `fetch` isn't auto-aborted on component unmount, navigating away mid-stream left the `while (true) { reader.read() }` loop doing `setState` on a detached component and held the connection open until the server finished. Fixed by adding `useEffect(() => () => abortRef.current?.abort(), [])`; this also ends server work early (aborting closes the connection → the route's next `enqueue` throws → `finally { controller.close() }`), even though the route has no explicit `cancel()` handler. Added a source-level cleanup lock test (node/vitest idiom, no jsdom) that discovers every AbortController-owning component and asserts unmount teardown. Demo suite 13 → 16, `tsc --noEmit` clean.
- Same bug class as `nextjs-streaming-ai-patterns#78` shipped earlier this run; found by pattern-matching that hit to rag's demo frontend.

**Why this work, this session:** the static issue queue is globally exhausted, so work came from fresh-lens dogfood hunts. Nine hunts ran this run across nine lens families (percentile, TTL, retry, vsas param/concurrency, python-async TaskGroup, aop HITL/trace, ai-app mock-replay, mcp security, retrieval metrics); eight came back honestly empty — the backend/logic surface is decisively saturated. The single productive axis was **frontend streaming lifecycle**, which yielded both issues this run (#78 and #124). When backend hunts go empty, pivot to React unmount-cleanup/streaming-lifecycle lenses on the two Next.js frontends.

**Open questions / blockers:** none — ready for review.

**Next in this session's loop:** the AbortController-unmount lens is now swept on both frontend surfaces (nextjs 4 clients, rag demo 1 client). Continue only if a genuinely fresh lens surfaces; otherwise stop cleanly within the DAY 2–4 target (2 shipped).
