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
