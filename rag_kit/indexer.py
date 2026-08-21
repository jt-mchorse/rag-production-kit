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

    It must be a non-empty string with no leading or trailing whitespace and
    no ``]``. Both rules exist so the id can be cited back: answers reference
    chunks with a ``[cite:<external_id>]`` marker, and ``enforce_citations``
    strips that marker before lookup. Internal characters are unconstrained --
    spaces, ``[``, newlines, dots and non-ASCII all round-trip -- so ordinary
    path-shaped ids like ``docs/guide.md#3`` are fine (#182).
    """

    external_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.external_id, str):
            raise ValueError(
                f"Document.external_id must be a string; got {type(self.external_id).__name__}"
            )
        # `not self.external_id` alone let through several ids that the citation
        # reader cannot resolve back to this document (#182). `enforce_citations`
        # strips a `[cite: ...]` marker before lookup -- it has to, because the
        # Anthropic path routinely emits padded markers -- and the comment
        # justifying that strip asserts "corpus external_ids never carry
        # leading/trailing whitespace". Nothing enforced the "never". This does.
        #
        # Measured before this guard, indexing a document and then validating a
        # *correct* citation to it:
        #
        #   ' doc1'   ACCEPTS -> REFUSES: dangling citation 'doc1'
        #   'doc1 '   ACCEPTS -> REFUSES: dangling citation 'doc1'
        #   'doc1\n'  ACCEPTS -> REFUSES: dangling citation 'doc1'
        #   '\tdoc1'  ACCEPTS -> REFUSES: dangling citation 'doc1'
        #   '   '     ACCEPTS -> REFUSES: dangling citation ''
        #   'doc]1'   ACCEPTS -> REFUSES: dangling citation 'doc'
        #
        # A fully-grounded answer refused as `unparseable_output`. Worse, with
        # both 'doc1' and ' doc1' in one corpus -- two distinct rows under the
        # schema's UNIQUE(external_id) -- a citation to ' doc1' resolved to
        # 'doc1' and rendered *that* chunk's text as the source. A citation
        # pointing at a chunk the claim did not come from is the one failure a
        # citation-enforcing RAG kit must not have.
        #
        # Two distinct causes, deliberately kept as two rules with two messages:
        # whitespace is defeated by the reader's `.strip()`, `]` by
        # `_CITE_PATTERN` (`\[cite:([^\]]+)\]` stops at the first `]`, so
        # `[cite:doc]1]` captures `doc`).
        #
        # Scoped to exactly those two. Internal characters round-trip fine and
        # are left alone -- 'doc 1', 'doc[1', 'doc\n1', 'doc.1' and 'docé1' all
        # resolve correctly, and external_id is documented as caller-supplied
        # (filename + chunk index, hash), so a charset allowlist here would
        # reject legitimate corpora to fix a problem they don't have.
        if self.external_id != self.external_id.strip():
            raise ValueError(
                "Document.external_id must not have leading or trailing whitespace "
                f"(got {self.external_id!r}); the citation reader strips markers "
                "before lookup, so a padded id cannot be cited back"
            )
        if not self.external_id:
            # After the strip check, this also covers the whitespace-only case:
            # `not '   '` is False, so the original guard passed it through.
            raise ValueError("Document.external_id must be non-empty")
        if "]" in self.external_id:
            raise ValueError(
                "Document.external_id must not contain ']' "
                f"(got {self.external_id!r}); the [cite:...] marker grammar "
                "terminates at the first ']', so such an id cannot be cited back"
            )
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
