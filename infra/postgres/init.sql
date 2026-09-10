-- rag-production-kit / Postgres schema for the hybrid retrieval path (#1).
--
-- This file is executed on container start by the docker-compose service
-- (volume-mounted to /docker-entrypoint-initdb.d/). It's also the single
-- source of truth the Python indexer/retriever assume.
--
-- Dense vector dimensionality is configured PER DEPLOYMENT by editing two
-- places together (D-003): the `vector(N)` column below, and the
-- `EMBEDDING_DIM` constant in `rag_kit/embedder.py`. The default 64 matches
-- `HashEmbedder`; a real embedder (e.g. 1024-d for voyage-3) means changing
-- both.
--
-- Two places and not one on purpose: the package deliberately does not
-- auto-detect the embedder's dimension, because that would hide a schema
-- migration behind library code. `to_pgvector` rejects a wrong-width
-- embedding at the Python seam before any SQL is issued, and
-- `tests/test_embedding_width_seam.py` asserts the two halves agree, so the
-- obligation is a contract rather than a convention.
--
-- This comment used to say the width was "parameterized through a settings
-- table". There is no settings table, and there never was — the mechanism
-- described here is the one `docs/architecture.md`, `rag_kit/db.py`'s error
-- message and D-003 have all described all along (#209).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- The single corpus table for v0.1. One row per indexed chunk.
-- Production layouts will eventually grow per-corpus tables and metadata
-- columns; this is the minimal shape the hybrid retriever needs.
CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    -- NOT NULL as well as UNIQUE (#209). Postgres UNIQUE permits MULTIPLE
    -- NULLs, so without this the guarantee `rag_kit/indexer.py` states --
    -- "It's UNIQUE in the schema, so re-indexing the same chunk overwrites
    -- cleanly" -- did not hold for a NULL id: `ON CONFLICT (external_id) DO
    -- UPDATE` never fires for one, and each re-index inserts another row.
    -- `Document.__post_init__` makes that unreachable from Python (#182);
    -- direct SQL is an ordinary corpus-loader path and this file is
    -- documented as a source of truth in its own right. `CREATE TABLE IF NOT
    -- EXISTS` means a running deployment is untouched by this, which is
    -- acceptable only because migrations are not a concern in v0 (see the
    -- header) -- an existing corpus keeps the nullable column until it is
    -- recreated.
    external_id     TEXT NOT NULL UNIQUE,         -- caller-supplied stable id
    text            TEXT NOT NULL,
    tsv             tsvector,                     -- BM25-style lexical channel
    -- Width is set in two places; see the header. `rag_kit/embedder.py`'s
    -- `EMBEDDING_DIM` is the other one.
    embedding       vector(64) NOT NULL,          -- dense channel
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lexical index (BM25-flavored via Postgres FTS + GIN).
CREATE INDEX IF NOT EXISTS documents_tsv_idx
    ON documents USING GIN (tsv);

-- Dense index (HNSW for low-latency ANN; cosine distance matches what the
-- retriever computes). HNSW parameters here are sensible defaults — the
-- vector-search-at-scale repo is where the parameter-sweep study lives.
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Maintain `tsv` automatically on insert/update. English config is a v0.1
-- default — a real deployment configures the dictionary per language.
CREATE OR REPLACE FUNCTION documents_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_tsv_update ON documents;
CREATE TRIGGER documents_tsv_update
    BEFORE INSERT OR UPDATE OF text ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_tsv_trigger();
