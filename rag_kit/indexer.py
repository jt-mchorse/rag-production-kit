"""Indexer: persist documents + their dense embedding + tsvector into Postgres.

The tsvector itself is computed by a database trigger (see init.sql) so
the Python side only writes raw text and the dense embedding; this keeps
the FTS configuration colocated with the schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .db import Jsonb, to_pgvector
from .embedder import Embedder


@dataclass
class Document:
    """One indexable chunk.

    ``external_id`` is the caller-supplied stable identifier (filename +
    chunk index, hash, etc.). It's UNIQUE in the schema, so re-indexing
    the same chunk overwrites cleanly.
    """

    external_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.external_id:
            raise ValueError("Document.external_id must be non-empty")
        if not self.text:
            raise ValueError("Document.text must be non-empty")


class Indexer:
    """Writes documents to the corpus table.

    Holds an open psycopg connection for the duration of the indexing run;
    the caller manages the lifetime (typically a context manager).
    """

    def __init__(self, conn: Any, embedder: Embedder) -> None:
        self.conn = conn
        self.embedder = embedder

    def add_documents(self, docs: Iterable[Document]) -> int:
        """Insert (or upsert by external_id) the given documents.

        Returns the number of rows written. Embeddings are computed in the
        caller's process — batching is the embedder's concern. Each row
        commits inside one transaction; the caller can wrap the call in
        ``with conn.transaction(): ...`` for stricter atomicity.
        """
        docs_list: Sequence[Document] = list(docs)
        if not docs_list:
            return 0

        rows = []
        for d in docs_list:
            vec = self.embedder.embed(d.text)
            rows.append(
                (
                    d.external_id,
                    d.text,
                    to_pgvector(vec),
                    Jsonb(d.metadata),
                )
            )

        sql = """
        INSERT INTO documents (external_id, text, embedding, metadata)
        VALUES (%s, %s, %s::vector, %s)
        ON CONFLICT (external_id) DO UPDATE
            SET text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """
        with self.conn.cursor() as cur:
            cur.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def clear(self) -> None:
        """Truncate the corpus. Useful in test setup; never call in prod."""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
        self.conn.commit()
