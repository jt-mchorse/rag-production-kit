"""Postgres connection helpers.

We register the `vector` type with psycopg's adapter system once at
connection time so embedding columns round-trip through Python lists
without per-call coercion. Keep this module tiny on purpose — the
indexer/retriever own the SQL.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError as e:  # pragma: no cover
    raise ImportError("psycopg is required. Install with: pip install rag-production-kit") from e


DEFAULT_DATABASE_URL = "postgresql://rag:rag@localhost:5432/rag"


def connect(url: str | None = None) -> psycopg.Connection[Any]:
    """Open a psycopg connection.

    Resolves the connection string from (in order) the caller-supplied
    ``url``, ``DATABASE_URL`` env var, then the default that matches the
    ``docker-compose.yml`` service.
    """
    dsn = url or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL
    return psycopg.connect(dsn)


def to_pgvector(vec: list[float]) -> str:
    """Format a Python list as a pgvector literal: '[1,2,3]'.

    Using a string adapter keeps us out of psycopg's type-registration
    flow for the `vector` Postgres type — that hook moves between pgvector
    versions, and the string form is stable.
    """
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


__all__ = ["DEFAULT_DATABASE_URL", "Jsonb", "connect", "to_pgvector"]
