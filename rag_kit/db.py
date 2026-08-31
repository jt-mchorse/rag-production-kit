"""Postgres connection helpers.

We register the `vector` type with psycopg's adapter system once at
connection time so embedding columns round-trip through Python lists
without per-call coercion. Keep this module tiny on purpose — the
indexer/retriever own the SQL.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .embedder import EMBEDDING_DIM

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


def to_pgvector(vec: list[float], *, expected_dim: int = EMBEDDING_DIM) -> str:
    """Format a Python list as a pgvector literal: '[1,2,3]'.

    Using a string adapter keeps us out of psycopg's type-registration
    flow for the `vector` Postgres type — that hook moves between pgvector
    versions, and the string form is stable.

    Rejects a non-finite component (`NaN` / `±Inf`) at this seam (#82).
    Both embedding entry points funnel BYO-`Embedder` output through here —
    `Indexer.add_documents` on the write path and `Retriever._hybrid_search`
    on the query path — and a normalization divide-by-zero, an `Inf`
    overflow, or a NaN-poisoned model output can hand back a non-finite
    component. Unguarded, `repr(float(nan))` emits the bare token `nan`, the
    literal reaches pgvector, and it either errors opaquely far from the
    embedder seam (`ERROR: NaN not allowed in vector`, surfaced mid-
    `executemany` / mid dense-SQL) or — on a tolerant build, or for `Inf` —
    makes every `<=>` cosine-distance comparison undefined and silently
    corrupts dense-channel ordering. Fail loud here naming the offending
    index, the same seam-validation posture as the sibling embedding guard
    in llm-cost-optimizer (#88) and the finiteness sweep already applied to
    `telemetry.percentile` (#80), `PhaseTimings.record` (#63), generator
    `threshold` (#79), and reranker `length_penalty` (#76).

    Also rejects a wrong *width* (#194). That guard validated every component
    and never how many there were, while the schema pins the count —
    `infra/postgres/init.sql` declares `embedding vector(64) NOT NULL`. The
    `Embedder` Protocol is `embed(text) -> list[float]` with no dimension in
    it, and swapping in a real embedder is the documented intent of the seam;
    Voyage, Cohere and BGE default to 1024, 1024 and 768, so a wrong width is
    the *expected* first mistake rather than an exotic one. It fails in the
    same two places as a NaN and just as opaquely — and worse on the write
    path, because `Indexer.add_documents` builds each row as it embeds, so
    without this the operator pays every embedding call in the batch before
    Postgres objects. An empty vector is rejected by the same check rather
    than emitting the literal `"[]"`.

    `expected_dim` defaults to `EMBEDDING_DIM`, not to a literal `64`. D-003
    makes the width a per-deployment setting and states that the operator
    "edits both the column dimension in `infra/postgres/init.sql` and the
    `EMBEDDING_DIM` constant … The package does not auto-detect the embedder's
    dimension because that hides schema migrations behind library code." So
    reading the constant is the only option D-003 leaves open — neither a
    `dim` attribute on the Protocol nor a probe of the live column type is
    available — and `tests/test_schema_embedding_dim.py` is the check that the
    two halves of that operator obligation actually agree. There is
    deliberately no "skip the check" value: an escape hatch here re-opens
    exactly this bug.

    Raises:
        ValueError: when `vec` has the wrong number of components, or when any
            component is non-finite.
    """
    if len(vec) != expected_dim:
        raise ValueError(
            f"embedding must have exactly {expected_dim} components "
            f"(the width `infra/postgres/init.sql` pins with `vector({expected_dim})`, "
            f"exported as rag_kit.EMBEDDING_DIM); got {len(vec)}. "
            "A BYO Embedder returning a different width would otherwise fail inside "
            "executemany / the dense-channel SQL with an opaque pgvector dimension "
            "error, after the whole batch had already been embedded. If this "
            "deployment really does use a different width, D-003 asks you to change "
            "init.sql and EMBEDDING_DIM together."
        )
    out: list[str] = []
    for i, v in enumerate(vec):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(
                f"embedding component at index {i} must be finite; got {v!r} — "
                "a NaN/Inf value is rejected by pgvector and would otherwise surface "
                "as an opaque error far from the embedder seam (or silently corrupt "
                "dense-channel ordering)"
            )
        out.append(repr(f))
    return "[" + ",".join(out) + "]"


__all__ = ["DEFAULT_DATABASE_URL", "Jsonb", "connect", "to_pgvector"]
