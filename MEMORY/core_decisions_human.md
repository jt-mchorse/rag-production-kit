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

## D-014 — Rewriter is a single-method Protocol with `TemplateRewriter` default + `AnthropicRewriter` extra, opt-in on `Retriever.search` (2026-05-16)
**Decision:** Pre-retrieval query rewriting / decomposition follows the same swappable seam as the embedder (D-002), reranker (D-005), and generator (D-008): a `Rewriter` Protocol with a single `rewrite(query) -> RewriteResult` method, a dep-free `TemplateRewriter` reference, and an `AnthropicRewriter` lazy-imported behind the *existing* `[rag-anthropic]` extra (no new extra needed). `Retriever.search` accepts `rewriter=...` as a kwarg defaulting to `None`, so every existing caller keeps its exact single-query behavior — parallel to D-007.

**Why:** Three reasons stack. (1) The protocol pattern is now the load-bearing seam in three of the four pluggable layers in this repo; making the rewriter follow it keeps the public surface consistent and lets a third-party `Rewriter` drop in the same way a third-party `Generator` does today. (2) `TemplateRewriter` carries the same hermetic-CI rationale as D-006 and D-013 — the full rewrite-and-retrieve flow has to be exercisable without an API key so the eval workflow stays free, and rule-based decomposition over a handful of common multi-hop patterns is enough to demonstrate the seam end-to-end. (3) Reusing `[rag-anthropic]` (already pulled in by `AnthropicGenerator`) avoids an extras-list sprawl problem: anyone using the production generator already has the SDK installed.

**Alternatives considered:**
- Hardcoded `AnthropicRewriter` with no seam — rejected; locks the repo to one provider, breaks the consistency with the three other pluggable layers, and the CI workflow would either need an API key or a separate code path.
- Pipeline step outside `Retriever.search` (caller decomposes, then calls search once per sub-query) — rejected; pushes RRF-across-sub-queries fusion onto every consumer, and `StreamingPipeline` would lose the rewriter as a typed phase.
- Abstract base class instead of `Protocol` — rejected; inconsistent with D-005 and D-008, and the structural typing of `Protocol` is exactly what lets test stubs and third-party rewriters drop in without inheritance.

**Reversibility:** Cheap. The protocol is one method; concrete rewriters are one class each; the retriever's multi-hop branch is one method (`_multi_hop_search`) that can be inlined or refactored without touching consumers.

**Related issues:** #3

## D-015 — Cost-telemetry `PriceTable` ships no defaults; operator supplies prices; unknown model raises (2026-05-16)
**Decision:** The `PriceTable` constructor in `rag_kit.telemetry` accepts an explicit mapping of model id → `ModelPrice` (or starts empty). No defaults ship with the repo. Calling `cost(model, ...)` for a model that isn't in the table raises `UnknownModelError` (subclass of `KeyError`). The eval / dashboard / unit tests supply their own fixture prices (clearly labeled synthetic).

**Why:** Same posture as D-013 (no fabricated benchmarks) extended to pricing. Public list prices for Claude, GPT, Cohere, etc. change frequently and vary by region, tier, and negotiated contract — the repo can't be the source of truth for a downstream deployment's actual numbers. Worse, a silent `$0.00` fallback would be a load-bearing bug: the cost dashboard would underreport, the operator would not notice until the bill arrived, and the table would have to be rolled back. Failing loud on unconfigured models means the dashboard is honest at all times; the operator is forced to make the price decision explicit at integration time, where they have the necessary context.

**Alternatives considered:**
- Ship Anthropic public list prices as defaults — rejected; prices drift, the repo would have to track them, and downstream deployments often have different contract numbers. The default would become a stale lie that the operator can't easily see.
- Ship zero defaults with a warning — rejected; warnings get suppressed in production logs and a `$0.00` total looks like real data on a dashboard.
- Require prices at `TelemetryStore.__init__` so a store can't be opened without them — rejected; couples storage and pricing concerns. A record can be inserted with zero tokens (e.g., for purely-retrieval calls) and not need a price entry; coupling them would force a price table for use cases that don't need one.

**Reversibility:** Cheap. `PriceTable` is one class; defaults could be added later as a separate `PriceTable.from_anthropic_list_prices(as_of_date=...)` constructor without changing existing callers.

**Related issues:** #6

## D-016 — Next.js demo re-emits the same SSE protocol as the Python demo, not a new wire format (2026-05-18)

**Decision:** The new `demo/nextjs/` frontend speaks the *same* SSE event protocol as `demo/streaming/server.py`. Frame shape `event: <type>\ndata: { payload, elapsed_ms }`. Event types: `retrieving`, `retrieved`, `reranking`, `reranked`, `generating`, `token`, `generated`, `done`. The TypeScript route handler (`demo/nextjs/app/api/stream/route.ts`) emits these events with the same names and the same payload keys the Python `to_sse()` helper produces.

**Why:** The protocol is the artifact this repo teaches — phases visible to the user as the model retrieves, reranks, then generates. Two implementations of the same protocol (Python stdlib server + TypeScript Next.js route) is more pedagogically useful than two protocols in one repo. An operator reading either demo learns the same wire format; an operator who wants to swap one backend for the other gets a one-protocol migration, not a re-architecture. The decision also makes the JS-tier demo a *real* port, not a parallel reinvention: the Python demo proves the protocol works under stdlib transport, the Next demo proves it works under Next 15's route handlers; both render against the same client mental model.

Same posture as D-005 (which records the rag-kit's `StreamingPipeline` as a single SSE protocol across phases) and the broader portfolio pattern of "one protocol across layers" seen in `nextjs-streaming-ai-patterns` D-005 and D-006.

**Alternatives considered:**
- Proxy the Next route to the running Python server — rejected: doubles the "fresh-clone runs" requirement (`docker compose up` + `python -m demo.streaming.server` + `npm run dev`). The demo's load-bearing posture is *runs on a fresh clone, no infra*.
- Invent a new JSON-streaming wire format more native to the Next idiom (e.g., `application/x-ndjson`) — rejected: silos the cookbook's teaching across two protocols. The repo would need two parallel explanations of "how phase events flow", one per backend.
- Serialize the whole answer at once on the server and animate client-side — rejected: drops the streaming pattern this repo is *about*. Real RAG UX wants the per-token feel; faking it with a client-side typewriter trades the protocol's teaching for a styling trick.

**Reversibility:** Cheap. The route handler is one file (~100 lines). Swapping it for a proxy or a different wire format would touch only `app/api/stream/route.ts` and the client's `handleFrame` parser.

**Related issues:** #8

## D-017 — the SSE wire serializer replaces, it does not reject (2026-08-26)

**Decision.** `to_sse` is *total*: for any input it returns a string that
`json.loads` accepts and that `.encode("utf-8")` accepts. `_json_safe` gets
there by replacing what it cannot represent — non-finite floats become `null`
at the key position as well as the value position, a key type `json.dumps`
would reject becomes a string, a coerced key collision resolves to a single
name, a cycle is named rather than raised, and text with no UTF-8 encoding is
replaced with U+FFFD.

**Why.** `to_sse` runs outside every error handler that could soften a failure.
`StreamingPipeline.run` wraps its generator body in
`except Exception -> yield StreamEvent("error", ...)`, but `to_sse` is called
*after* each event is yielded. And `demo/streaming/server.py` calls
`to_sse(event).encode("utf-8")` outside its `try`, which guards only
`wfile.write` and only for `BrokenPipeError` — with `send_response(200)` and the
headers already sent. So a raise there is not an error the operator sees; it is
a truncated `text/event-stream` with no `error` frame and no `done` frame, which
is byte-indistinguishable from a network drop. Against that, one replacement
character in one `metadata` field is a clear improvement.

**And it is deliberately the opposite of `llm-eval-harness#215`,** which rejects
an unencodable input outright at its dataset seam. The difference is the
contract, not a disagreement. That seam writes a file that has to be faithful,
and there is no faithful spelling of a lone surrogate to write, so refusing the
input is the only honest option. This seam's documented contract is "stream
alive, don't raise". Recording it so a later session does not "harmonise" the
two into a single rule and break one of them.

**Alternatives considered.** (1) Reject, matching `llm-eval-harness#215` —
rejected for the reason above. (2) Flip `ensure_ascii` to `True`, which would
make every frame pure ASCII and therefore always encodable — rejected because it
changes the bytes of every non-ASCII frame in the repo to fix a rare case, and
the existing comment in `tests/test_streaming.py` records `ensure_ascii=False`
as a deliberate compactness choice. (3) Catch at the demo server's write seam
only — rejected: it leaves the library's own documented guarantee false, and
`to_sse` is exported in `__all__` for callers who have no such server. (4) Let
`json.dumps` raise — that is the status quo being fixed.

**Reversibility.** Cheap. One helper, one chokepoint, and the whole "before"
behaviour is a measured variant table in `tests/test_sse_frame_totality.py`.

---

## D-018 — The reranker seam's tie rule is "preserve input order", stated on the Protocol
**Date:** 2026-09-08

**Decision.** Among equal scores, a `Reranker` returns candidates in their input
order. The rule lives on the `Reranker` Protocol docstring and is tested against
every backend in the module through one contract test that discovers the
backends rather than listing them. It is deliberately *not* `fusion.py`'s
tie-break by document id.

**Why.** `CohereReranker.rerank` ended with `merged.sort(key=lambda pair:
pair[0], reverse=True)` — no tie-break — and, unlike its sibling backend, its
insertion order is not the input order. `merged` is filled per batch in
`response.results` order, which the API returns sorted by relevance. So among
equal scores the output was decided by two things that are not properties of the
documents: whatever order the API happened to return the tied rows in, and which
batch each candidate landed in, i.e. `batch_size` — a knob documented purely as
a request-size limit. `rerank_rank` flows into the citation payload, so two runs
over one corpus could cite a different chunk id for the same claim while the
scores a consumer would inspect stay identical.

Ties here are guaranteed rather than coincidental: `documents = [c.text for c in
batch]` is all the API sees, so two candidates carrying the same text score
identically by construction. Chunk overlap, a passage indexed twice, and the
"union of two SQL paths" `fusion.py`'s own comment names all produce that.

**Why input order rather than doc id.** The input to a reranker is already a
ranking — `Retriever.search`'s fused list — and it carries signal a
lexicographic rule would discard. RRF has no incoming order to inherit, which is
why the two seams answer differently. And `LexicalOverlapReranker` already
satisfied input order via its stable sort and said so in a comment; making that
the seam's rule promotes an existing property rather than changing a working
backend.

**Alternatives considered.**
- Fusion's doc-id tie-break — rejected: discards the fused ranking, and would
  have moved `LexicalOverlapReranker` for no reason.
- Sorting each batch before merging — rejected, and built and run: it
  normalizes *within* a request and leaves the cross-request order exactly where
  it was, so the answer still moves with `batch_size`.
- `reverse=True` while carrying the position — rejected: it reverses the
  tie-break along with the score and ranks the *last* tied candidate first.
- Documenting that ties are undefined — rejected: that is the status quo with a
  sentence on it, and the status quo puts a different chunk id on a citation.

**Reversibility:** Cheap. One sort key and a Protocol docstring; the contract
test is the thing that would need rewriting, and it is one file.

**Related issues:** #207, #205, #180, #40

---

## D-019 — `RerankDelta` reports both set differences, and the symmetric one is a field because a search said so

**Date:** 2026-09-22 · **Reversibility:** cheap · **Issue:** #218 (follow-up to
#217 / #215)

**Decision.** `RerankDelta` gains two defaulted counts: `n_foreign` (ids in
`after` that `before` never held) and `n_dropped` (ids in `before` missing from
`after`). Both are reported, never raised. `len(after)` does not become a third
field; it is exactly derivable.

**Why.** #217 made the relevance scale a property of `before` alone, so
`["a","b","c"] → ["a","b","c","x"]` reports `1.0`. That is the *correct* answer
to the question the metric asks — "how much did the reranker move the input
ordering?" — and the answer is "not at all". The gap #218 reported is that
nothing in the dataclass revealed `after` holding an id `before` never had. A
foreign id contributes `rel = 0.0`, `n_input` counts `before`, and `top_k_size`
is capped by both lists, so a reranker emitting a thousand invented candidates
published a perfect telemetry row.

**The part worth recording is how AC3 went.** The issue asked whether the
symmetric case — a truncating reranker — gets its own count or is deliberately
left out. I had the answer drafted: *no symmetric count, because truncation is
never invisible, only conflated with reordering; every truncation strictly
lowers the displacement.* Before writing it down I ran the search. Over all 304
ordered subsets of a five-id `before` at `k=3` there are **zero** collision
classes spanning different output lengths — which is a true fact about that
corpus and the wrong corpus to conclude from. At seven ids and `k=5` there are
**360 classes in which a truncating and a non-truncating output agree on all
four fields to the last bit**. The smallest:

```
before = a b c d e f g                                  (k=5)
after₁ = a b c d f g      # dropped `e`
after₂ = c b a d f g e    # kept all seven, reordered the head
both  -> n_input=7, top_k_overlap=4, top_k_size=5,
         ndcg_displacement=0.9374720354963293           # d1 == d2 is True
```

So truncation is invisible in exactly the sense foreign ids are; it just needs
a larger input to demonstrate. `top_k_size = min(k, n_input, len(after))` is the
only field that can reveal a short `after`, and it stops being able to the
moment `len(after) >= k` — which is the ordinary operating region of a top-N
reranker, not an edge case.

**The two counts are not the same kind of signal**, and the dataclass comment
says so, because a dashboard author should not alarm on both. `n_foreign > 0`
always violates the invariant this module states in #215's guard — "a reranker
permutes the ids it was given". `n_dropped > 0` is routine: a top-N reranker
returns fewer ids than it received, by design. A count is a fact, not an
accusation; what differs is what the fact means.

**Reported, never raised.** #215 deliberately kept `["a","b","c"] →
["x","y","z"]` reporting `0.0` as the contrast row that made the empty-`before`
`1.0` look wrong, and turning foreign ids into a `ValueError` would reverse that
decision. `rerank_delta_ndcg` is telemetry: a reranker misbehaving in production
is exactly when a caller wants a number rather than an exception. Same posture
as `n_uncomparable` / `n_*_off_support` in llm-eval-harness (D-017, D-023,
D-024).

**No `n_output` field, and the reason is the default.** `len(after)` is exactly
`n_input - n_dropped + n_foreign` — the duplicate guard means every list's
length equals its set cardinality — and that identity is pinned across seven
shapes. An `n_output` field would have no honest default: `0` there means "the
reranker returned nothing", a real and alarming value standing in for an
unmeasured one, which is the fabricated-extreme default `llm-cost-optimizer`
D-018/D-019 ruled out. Both anomaly counts default to `0` truthfully, and that
is what makes AC4 — the four original fields still constructing on their own —
satisfiable at all.

**Counted over set membership of the whole ranking**, not `len()` and not the
top-k slice. `["a","b","c"] → ["a","b","x"]` drops one id and invents another at
equal length, so a length difference reports `0` in both directions while two
things went wrong; and `k` is a knob the *caller* chooses, so a sliced count
would make one reranker's behaviour report differently to two dashboards
watching the same stream.

**An existing test had named this issue as its own follow-up.**
`test_a_longer_after_does_not_change_the_other_telemetry_fields` asserted
whole-dataclass equality under the docstring "they are the fields a reader might
expect to reveal a longer `after`, and they do not — which is the follow-up's
premise." #218 is that follow-up, so the assertion had to move. Updated, not
deleted: it now makes the same claim over the four fields it was written about,
and additionally pins that `n_foreign` is what ends the blindness.

**Alternatives considered:**
- Raise on foreign ids — rejected; reverses #215's contrast row, and this is
  telemetry.
- Only `n_foreign`, recording the symmetric case as deliberately omitted —
  rejected; built and run, six arms red, and the 360-class search is what
  falsified the argument I was about to write down.
- An `n_output` field instead of `n_dropped` — rejected; no honest default, and
  the quantity is already derivable.
- Counts from a length difference — rejected; built and run, three arms red.
- Counts sliced to the top-k window — rejected; built and run, six arms red.

**Related issues:** #218, #217, #215

## D-020 — `started_at` is a measurement, not a literal

**Date:** 2026-09-23 · **Issues:** #221 (#7 is where the literal was born) ·
**Reversibility:** cheap

**Decision.** `evals/run_eval.py` stamps the real UTC start time into
`started_at`, with a caller override so tests can pin it, and `write_runs`
resolves that stamp once and shares it across all three suites. The frozen
`"2026-05-16T00:00:00Z"` literal is gone.

**Why.** The field had been that literal, unconditionally, since the file was
created — no parameter, no comment, no consumer inside this repo. The obvious
defence is determinism, and it does not survive the record's own contents:
`git_sha` is read from `git rev-parse HEAD` and `run_id` is
`sha256(suite|git_sha)`, so both move on every commit. The file was never
byte-stable. What freezing only the timestamp produced is a record that
contradicts itself — a run asserted to have started 2026-05-16 against a commit
that did not exist until months later. Running the documented command today
moved `git_sha` from `e40188cf` to `f4c6e8bd`, moved `run_id` with it, and left
the timestamp exactly where it was.

The module's own docstring says the run shape "matches
`eval_harness.runner.RunResult`". Upstream resolves the same field as
`started_at or utc_now_iso()`, with `started_at` a keyword argument documented
as "caller-overridable so tests can pin them". This was the one field where the
declared parity was false, and it was false in the direction that matters: no
real time by default, and no way to supply one.

Downstream it is not inert. In the exact `eval-harness` commit the `[eval]`
extra pins, `latest_run_id_for_suite` is `ORDER BY started_at DESC LIMIT 1` with
no tie-break, over an indexed `NOT NULL` column. Ingesting this repo's artifacts
makes every run tie on the sort key, so "the most recent run" is decided by
SQLite's scan order rather than by the data, and that function's own docstring
reasoning — "the ISO-8601 format is lexicographically sortable so a string
compare suffices" — becomes a no-op.

**The `Z` form is load-bearing.** `_utc_now_iso` uses
`strftime("%Y-%m-%dT%H:%M:%SZ")` and not `datetime.isoformat()`, which renders
`+00:00`. Because the store compares this column as a string, a `+00:00` stamp
is a correct timestamp that sorts wrongly against the existing `Z` rows. Built
and run as a neighbour: two arms red.

**One run is one stamp.** `write_runs` resolves the stamp once. Letting each
suite stamp itself would, at this field's one-second resolution, usually agree
anyway — and occasionally split a single run's three artifacts across a second
boundary. The first draft of the arm guarding this called the real clock and the
neighbour that drops the threading *passed it*; it is rebuilt around a clock
that advances one second per read, which separates them every time.

**Alternatives considered.**
- *Keep the literal and document it as determinism.* Rejected: the record is not
  deterministic, so the rationale is false on its own terms.
- *Stamp a real time with no override.* Rejected: it swaps one untestable value
  for another, and upstream already shows the signature that does not.
- *`datetime.isoformat()`.* Rejected, built and run: 2 arms red on the sort form.
- *Let each suite stamp itself.* Rejected, built and run.
- *Regenerate the committed baselines.* Rejected: `evals/baselines/` and
  `evals/current/` are records of runs that did happen around that date, so
  their timestamps are approximately true. This changes the writer, not the
  history — and CI regenerates `current/` in the runner without committing it,
  so nothing churns.

---

## D-021 — the refusal detail renders the pair so the ordering it asserts stays readable

**Date.** 2026-09-25 · **Issue.** #225 · **Reversibility.** cheap

**Decision.** `Refusal.detail` renders `top_score` and `threshold` through
`rag_kit.comparison.render_comparison`, which widens from four decimal places
only while the two values render identically, always returns both sides at the
same precision, and falls back to `repr` when no width in its budget separates
them.

**Why.** The refusal gate is `top < threshold`, compared at full float
precision. The sentence explaining that decision was two `.4f` fields, so a
near miss published `top_score=0.8500 below threshold=0.8500` — a sentence
that contradicts itself, on the response path of the demo product, at exactly
the margin where a caller asking "why did this query refuse?" reads it most
carefully. Measured through the public API before any edit: three different
margins all rendered that same string, while the control (`0.5` against
`0.85`) rendered correctly.

Nothing in the suite could go red over this. The *verdict* is correct in every
colliding case, so no assertion about refuse-or-answer can fire; the defect
existed only in the prose. The data was never wrong either — `Refusal.top_score`
and `.used_threshold` carried the full floats throughout, the same split
`embedding-model-shootout#149` found between a correct aggregate and a
collapsed table.

**What does not transfer from the siblings.** This is the third spelling of a
class `prompt-regression-suite` (D-012) and `llm-eval-harness` (D-026) already
fixed, and two things about theirs are wrong here.

`places` is a **required** keyword argument rather than a default. Centralising
inline formatters onto a helper with a hardcoded width silently re-renders
every call site that disagreed with it — the regression `llm-eval-harness#252`
shipped — and this module renders at four places where both siblings render at
three.

More interesting: both siblings cap the widening loop at 17 places and argue
that always separates two distinct values, because their operands live near
magnitude 1 (a cosine in `[-1, 1]`, a threshold in `(0, 1]`). Neither bound
holds here. `_top_score` is documented negative-capable (#69) and
`_validate_threshold` accepts any finite float on purpose. Measured: at a
magnitude of `1e-5` — an unremarkable fused score — `math.nextafter(1e-5, inf)`
is a distinct double that still renders identically at 17 places and needs 25.
So the `repr` fallback is load-bearing here rather than a subnormal-scale
formality, and its arm pins it at `1e-5` rather than at subnormal scale. The
argument was rewritten rather than inherited.

**The same-precision half is the one that is easy to miss, and the measurement
says why.** Built and ran the "widen only the score side" neighbour: the
*ordering* arm goes red on 2 of 5 rows, while the *structural* same-number-of-
decimal-places arm goes red on all 5. Three rows stay green under the ordering
arm because `top` is the side carrying the long expansion — which it is
whenever the threshold is a round configured number, and `_DEFAULT_THRESHOLD`
is `0.02`.

The second row that *does* catch it catches it for a reason nobody predicted.
`negative-top-score-#69-region` goes red not because the threshold carries the
expansion, but because rounding the *narrow* side away from zero flips the
comparison: `-1234.567890123` is greater than the rounded `-1234.5679`. That
is a second way the mismatch lies, and a table built only from positive scores
would never have surfaced it. Put a negative row in every comparison-rendering
table.

**Alternatives considered.** Each of these was built and run, not reasoned
about.
- *A wider fixed width (`.8f`).* Rejected: 4 collapse cells still red, and it
  breaks the ordinary-refusal byte-identity arm (2 red). `.4f` here was already
  wider than the `.3f` that collided in `prs#175`.
- *Widen only the side that needs it.* Rejected: 10 red on the structural arm.
- *Round the comparison to match the display.* Rejected: it makes the gate less
  precise to make the message consistent. Five repos have now rejected this.
- *A defaulted `places` inside the helper.* Rejected: at 3 it republishes the
  ordinary refusal as `0.500` (2 red) — the `leh#252` regression exactly.
- *No `repr` fallback, just cap at 17.* Rejected: 1 red at `1e-5`.
- *A bare `range(places, MAX + 1)`.* Rejected: 1 red — a caller width past the
  ceiling falls straight to `repr`, silently discarding the width it asked for.
- *Share one formatter across the three repos.* Rejected: separate
  distributions with no dependency between them. The duplication is the honest
  trade, and it is written down here so it does not read as accidental.

**Population, discovered rather than listed.** The two backends carried
byte-identical f-strings and a third backend is the obvious next change to
`generator.py`, so an AST arm rejects *any* f-string in that module carrying
two fixed-precision interpolations — cross-checked against the set of classes
defining `generate` (minus the `Protocol`), so the rule cannot end up walking
an empty corpus and the test module's backend list cannot drift behind it.
