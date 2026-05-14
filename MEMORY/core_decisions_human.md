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
