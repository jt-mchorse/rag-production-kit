# Core Decisions

Strategic decisions for this repo, with reasoning. Append-only — superseded decisions are marked, not removed.

## D-001 — Scope locked to portfolio handoff §2 (2026-05-10)
**Decision:** Scope of this repo is fixed by the portfolio handoff document, section 2.

**Why:** The handoff spec was deliberated; ad-hoc scope expansion within a session is the failure mode this prevents.

**Alternatives considered:** None — this is a baseline.

**Reversibility:** Expensive. Scope changes require a deliberate revisit and a new decision entry.

**Related issues:** —

## D-002 — Only required runtime dep is psycopg (2026-05-14)
**Decision:** The `rag-production-kit` Python package requires only `psycopg[binary]` at runtime. The Anthropic SDK, the eval harness (`llm-eval-harness`), web frontends, and any concrete embedder are all opt-in or BYOD via the `Embedder` protocol.

**Why:** This repo is meant to be the reference *patterns* — the actual model/embedder choices are deployment decisions, not library decisions. Pulling Anthropic into the package as a hard dep would lock users out of OpenAI-compatible deployments, and pulling the eval harness in would create a hard cycle with a sibling repo. Postgres+pgvector is the one thing the package definitionally talks to, so psycopg is the one allowed runtime dep.

**Alternatives considered:**
- Bundle the Anthropic SDK as a required dep — rejected; downstream consumers of the retrieval API have no reason to install it.
- Vendor `llm-eval-harness` as a subpackage — rejected; the eval harness lives in its own repo for a reason (it's imported by multiple portfolio repos).

**Reversibility:** Cheap. If the surface grows enough to need richer abstractions, this is a `dependencies` list edit, not an architectural change.

**Related issues:** #1, #2, #7.

## D-003 — Dense vector dimensionality is per-deployment, default 64 (2026-05-14)
**Decision:** The embedding column in the schema is declared `vector(64)` in v0.1, matched to the `HashEmbedder` reference. When a deployment swaps in a real embedder (Voyage, Cohere, OpenAI, BGE), it edits both the column dimension in `infra/postgres/init.sql` and the `EMBEDDING_DIM` constant in `rag_kit/embedder.py`. The package does not auto-detect the embedder's dimension because that hides schema migrations behind library code.

**Why:** Embedding-model choice is the single biggest tuning lever in any RAG stack, and hardcoding a dimension would lie to users about the cost of switching. Forcing the column dim to be explicit makes the migration visible — the moment a user changes embedders, the schema has to change too, and they think about reindexing intentionally.

**Alternatives considered:**
- Hardcoded 1024-dim column — rejected; mismatched with the `HashEmbedder` reference, and locks the default to a value many real embedders don't use.
- Store embeddings as JSONB with no typed column — rejected; no HNSW index, no dense ANN at production speed.

**Reversibility:** Cheap. The column is declared in one place; a migration is one `ALTER TABLE` + reindex.

**Related issues:** #1, #2.

## D-004 — Reciprocal Rank Fusion as the fusion strategy, k=60 default (2026-05-14)
**Decision:** The hybrid retriever fuses its lexical and dense candidate lists with Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009), using k=60 as the default smoothing constant. The fused result exposes per-method ranks alongside the fused score so consumers can see which channel surfaced each doc.

**Why:** RRF is the strong, parameter-free baseline that most production hybrid-retrieval stacks land on. It doesn't require score normalization across heterogeneous rankers (the BM25 and cosine-distance score distributions are not directly comparable). The per-method ranks in the return shape are non-obvious and load-bearing: without them, the only way to debug "why did this chunk appear in the top-5?" is by re-running the underlying queries in isolation.

**Alternatives considered:**
- Weighted score fusion (e.g., `0.7 * dense_score + 0.3 * lex_score`) — rejected; requires normalization, requires per-corpus tuning, less robust than RRF in published comparisons.
- Condorcet-style pairwise voting — rejected; same paper that introduced RRF showed it underperforms RRF empirically.
- Dense-only or lexical-only — rejected; the entire point of the repo is the patterns that production stacks actually use, and pure single-channel retrieval is the failure mode we're addressing.

**Reversibility:** Cheap. The fusion function is one module (`rag_kit/fusion.py`); swapping it is a localized refactor.

**Related issues:** #1.

## D-005 — Reranker is a single-method `Reranker` Protocol (2026-05-15)
**Decision:** `rag_kit.reranker.Reranker` is a `typing.Protocol` with one method, `rerank(query, candidates) -> list[ScoredCandidate]`. Backends conform structurally; no inheritance.

**Why:** Same shape as the `Embedder` Protocol (D-002 in this repo) and the `Backend` Protocol in `llm-eval-harness` (D-004 there). The portfolio is using the single-method Protocol pattern as the standard test-substitution seam — recognizable across repos, minimal ceremony per backend, easy to swap providers without touching call sites.

**Alternatives considered:**
- Hard-coded Cohere client inside `Retriever` — rejected: would force every test that touches retrieval to mock a specific SDK and would lock the production stack to one vendor.
- Abstract base class — rejected: same redundancy concern as in `llm-eval-harness`; one method doesn't need an ABC.
- sklearn-style `BaseEstimator`-with-`fit/predict` — rejected: reranking has no fit step, and the verbose interface adds noise without benefit.

**Reversibility:** Cheap. Adding optional methods to the Protocol is backward-compatible; renaming or removing requires a migration but the Protocol shape is small.

**Related issues:** #2, #4

## D-006 — `LexicalOverlapReranker` ships as the dep-free fallback (2026-05-15)
**Decision:** A `LexicalOverlapReranker` ships in the base install, dependency-free. It's a token-overlap heuristic with a small length-penalty tiebreaker. CI and library consumers exercise the rerank flow against this backend without an API key. Production retrieval quality is one BYO backend away (`CohereReranker` or your own).

**Why:** Without a local fallback, every test that touches reranking needs a mocked SDK or recorded fixtures, and the integration test in `tests/test_hybrid_pg.py` couldn't exercise the end-to-end retriever-with-reranker path against the existing pgvector service container. With the fallback, the rerank flow is tested as a normal hermetic integration test; the fallback's *score quality* is intentionally not the point — its existence is.

**Alternatives considered:**
- Require the `[rerank-cohere]` extra for any reranking — rejected: forces an external service into the basic test path, hurts library reusability.
- Ship no local fallback — rejected: tests would have to mock the Cohere SDK, which is exactly the kind of test-mock divergence the project tries to avoid.

**Reversibility:** Cheap. The fallback is a single class in `rag_kit/reranker.py`; replacing it is a one-file change.

**Related issues:** #2

## D-007 — `Retriever.search(reranker=...)` is opt-in; default behavior unchanged (2026-05-15)
**Decision:** The reranker is a keyword argument on `Retriever.search()`, defaulting to `None`. When `None`, the retriever returns the RRF-fused top-k unchanged (existing behavior). When a `Reranker` is passed, the retriever over-fetches by the candidate multiplier so the reranker has more to choose from, then truncates back to `k` after reranking.

**Why:** Backwards compatibility for the callers that already use the hybrid retrieval and don't want the cost (latency + dollars) of a reranker round-trip on every query. Putting the reranker on the constructor would force a binding-time choice that's actually a per-call concern (e.g., production-quality reranker for user-facing queries, no reranker for internal eval runs).

**Alternatives considered:**
- Reranker required (always run) — rejected: forces every consumer to pick a reranker even for paths that don't need one.
- Reranker on the constructor — rejected: turns a per-call concern into a per-instance concern; would force callers to maintain two `Retriever` instances if they want to reroute different queries through different backends.

**Reversibility:** Cheap. The kwarg can grow defaults or be promoted to required without changing call shapes that already pass it.

**Related issues:** #2

## D-008 — Generator is a Protocol with a dep-free TemplateGenerator default and an AnthropicGenerator behind an extra (2026-05-15)
**Decision:** `rag_kit.generator.Generator` is a `Protocol` with a single method `generate(query, retrieved, *, threshold) -> GeneratedAnswer | Refusal`. `TemplateGenerator` (dep-free) ships in the base install for hermetic CI; `AnthropicGenerator` lives behind the new `[rag-anthropic]` extra and lazy-imports the `anthropic` SDK.

**Why:** Same shape as the `Embedder` (D-002) and `Reranker` (D-005) seams in this package — one method, deterministic input, deterministic output shape, backends swap without changing call sites. The dep-free default means CI exercises the full citation/refusal flow without an API key (same rationale as `LexicalOverlapReranker` in D-006). A LangChain-style chain or a single concrete class with branching would have hidden the protocol contract and made the swappable-backends story muddier.

**Alternatives considered:**
- Hard-coded Anthropic client — rejected; forces every consumer to install the SDK even if they ship a different generator.
- One concrete `Generator` class with internal branching — rejected; hides the seam and grows ugly as more backends land.
- A LangChain-style chain — rejected; pulls in a heavy dep tree for what is one well-shaped function.

**Reversibility:** Cheap. The Protocol can grow keyword args (streaming, tool-use) without breaking existing implementers.

**Related issues:** #4, #7

## D-009 — Refusal happens pre-LLM on weak retrieval and post-LLM on invalid citations (2026-05-15)
**Decision:** Two distinct refusal paths. The threshold check fires before the LLM is called: if `max(rerank_score if present else fused_score) < threshold`, the generator returns a `Refusal(reason="insufficient_context", ...)` without making an API call. The citation-enforcement check fires after the LLM returns: if `enforce_citations(text, retrieved)` raises, the generator returns a `Refusal(reason="unparseable_output", ...)`.

**Why:** These are two different failure modes that deserve two different signals. Weak retrieval is a corpus problem (you don't have the answer in your index); invalid citations are a model problem (the LLM ignored instructions or the context is misaligned). A single post-LLM refusal would conflate them and waste tokens on the cases where retrieval is clearly insufficient. Asking the LLM to refuse itself is more permissive than we want — the threshold makes the refusal decision auditable and reproducible.

**Alternatives considered:**
- Single post-LLM refusal — rejected; pays for tokens we know are wasted.
- LLM-self-refusal only (no threshold) — rejected; not auditable, not reproducible, ignores that retrieval scores are an existing signal.

**Reversibility:** Cheap. The threshold is a per-call kwarg; the two reasons are part of `Refusal.reason` and additive.

**Related issues:** #4, #7

## D-010 — Streaming pipeline is a sync generator, not asyncio (2026-05-16)
**Decision:** `StreamingPipeline.run(query, k)` is a synchronous generator that yields `StreamEvent`s. Retrieval, reranking, and token streaming all run on the same sync thread. Async-IO is layered only at the HTTP boundary when the SSE wire frames need to be written (the demo server uses `http.server`, which is sync; production deployments wrap the same generator in FastAPI's `StreamingResponse` if they want ASGI).

**Why:** The retriever (`rag_kit/retriever.py`) and reranker (`rag_kit/reranker.py`) are both sync, and Postgres calls via `psycopg` are sync. Coloring the pipeline `async` would force every existing call site into `async def` without unlocking any real concurrency — there's nothing to interleave at the pipeline layer because each phase blocks on the next phase's input. The wins of `async` show up at the *server* layer (handling many concurrent SSE clients), and that's handled in the HTTP adapter, not in the pipeline.

**Alternatives considered:**
- `async` throughout — rejected: requires `psycopg` to be in async mode (different driver), forces all downstream consumers to be async-colored.
- Callback style (`emit_callable(event)`) — rejected: less Pythonic, harder to compose with `for event in pipe.run()` patterns, harder to test.
- Separate async streaming module sharing nothing with the sync one — rejected: double maintenance for no benefit; we have no need today.

**Reversibility:** Cheap. The pipeline is one file (~290 lines). When async retrieval ships (if ever), an `AsyncStreamingPipeline` can be added without touching the sync one.

**Related issues:** #5

## D-011 — Demo HTTP server is stdlib `http.server`, not FastAPI (2026-05-16)
**Decision:** The `demo/streaming/` server uses Python's stdlib `http.server.ThreadingHTTPServer` to expose SSE. The base install of `rag-production-kit` does not depend on FastAPI, Starlette, or Uvicorn. A FastAPI adapter is documented as a one-liner (`StreamingResponse(to_sse(e) for e in pipe.run(q))`) and is the recommended production deployment.

**Why:** D-002 commits the base install to `psycopg` as the only required runtime dep. Pulling FastAPI in would make the package install a multi-megabyte dependency for the 90% of consumers who use the streaming pipeline programmatically (e.g., in tests, in benchmarks, in their own server) and never touch this demo. The stdlib server proves the SSE wire format works against a real browser client without imposing that cost.

**Alternatives considered:**
- FastAPI as a required dep — rejected: violates D-002, bloats the package for non-demo consumers.
- FastAPI behind a `[demo]` extra with the demo inside the extra — rejected: more moving parts than the demo is worth, still introduces a non-trivial dep on `starlette` and `uvicorn` once installed.
- Starlette minimal app — rejected: same dep-bloat concern; not as obviously dep-free as `http.server`.

**Reversibility:** Cheap. The demo server is a single file. Replacing it with FastAPI (or anything else) is a swap-out, not a refactor; the `StreamingPipeline` and `to_sse()` it consumes are unchanged.

**Related issues:** #5

## D-012 — Eval orchestrator writes one RunResult JSON per suite; composite PR comment via direct GitHub API (2026-05-16)
**Decision:** `evals/run_eval.py` runs each of the three suites (`faithfulness`, `recall_at_5`, `correctness`) and writes one `RunResult`-shape JSON per suite under `evals/current/`. The PR comment is composed by `run_eval.py` itself — three suite-deltas rendered via `eval-harness diff-json --format markdown` and combined into a single comment behind a repo-specific marker (`<!-- rag-production-kit:eval-sticky -->`) — posted via direct `urllib`/GitHub API calls.

**Why:** `eval-harness comment` uses a single hardcoded marker (`<!-- eval-harness:sticky-comment -->`) by design. Calling it three times in one workflow would have each call clobber the previous one. We want one visible signal per PR with all three metrics, not three stickies fighting over the same comment slot. A repo-specific marker also keeps this comment from colliding with the harness's own demo sticky in repos that use both.

**Alternatives considered:**
- Three separate comments, one per suite, each with its own marker — rejected; clutters the PR.
- One suite combining the three metrics into a single composite mean — rejected; loses the per-metric signal that's the entire point of the eval ("did faithfulness regress separately from correctness?").
- Patch `llm-eval-harness` to support a marker argument — viable but out of scope for this issue; would land in the harness repo separately.

**Reversibility:** Cheap. The composite poster is ~40 lines of stdlib `urllib`; switching to a marker-arg upstream is a one-line edit when that ships.

**Related issues:** #7

## D-013 — Eval corpus is single-sentence chunks so `TemplateGenerator`'s one-cite-per-sentence shape satisfies `enforce_citations` (2026-05-16)
**Decision:** `evals/dataset/corpus_v1.jsonl` chunks are each one sentence. `TemplateGenerator` emits one `[cite:<id>].` per retrieved chunk; `enforce_citations` splits the generator's output on sentence terminals and requires a `[cite:...]` in each. Multi-sentence chunks would mean the generator's single appended cite covers multiple split sentences, the first of which would fail enforcement.

**Why:** The eval suite has to run hermetically in CI without an LLM. `TemplateGenerator` is the dep-free generator (D-008) that ships with this repo; if its output can't be made to satisfy `enforce_citations` against the eval corpus, the faithfulness suite is permanently stuck at 0.0 even when the pipeline is working perfectly. The fix is to make the corpus shape match what `TemplateGenerator` can emit. The real production path uses `AnthropicGenerator` against arbitrary-shape chunks; the real-LLM eval runs (operator-triggered) will switch to it with `ANTHROPIC_API_KEY`.

**Alternatives considered:**
- Multi-sentence chunks plus a smarter eval-only generator — rejected; introduces a second `TemplateGenerator` variant for eval that drifts from the one consumers actually run in production.
- Paragraph chunks with a relaxed citation rule (any sentence cites for the whole paragraph) — rejected; would water down the `enforce_citations` contract the rest of the repo depends on.
- Require `ANTHROPIC_API_KEY` in CI to use the real generator — rejected; the eval workflow has to be hermetic so it runs on every PR without secrets.

**Reversibility:** Cheap. The corpus is one JSONL file; reshaping it is a session's work. The dataset and metric definitions are unchanged.

**Related issues:** #7
