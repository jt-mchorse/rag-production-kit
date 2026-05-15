-- rag-production-kit / Postgres schema for the hybrid retrieval path (#1).
--
-- This file is executed on container start by the docker-compose service
-- (volume-mounted to /docker-entrypoint-initdb.d/). It's also the single
-- source of truth the Python indexer/retriever assume.
--
-- Dense vector dimensionality is parameterized through a settings table so
-- the same schema works for HashEmbedder (default 64-d) and any real
-- embedder a user plugs in (e.g., 1024-d for voyage-3). See D-003.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- The single corpus table for v0.1. One row per indexed chunk.
-- Production layouts will eventually grow per-corpus tables and metadata
-- columns; this is the minimal shape the hybrid retriever needs.
CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    external_id     TEXT UNIQUE,                  -- caller-supplied stable id
    text            TEXT NOT NULL,
    tsv             tsvector,                     -- BM25-style lexical channel
    embedding       vector(64) NOT NULL,          -- dense channel; dim documented in pyproject + README
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
