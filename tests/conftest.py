"""Postgres-aware test fixtures.

Tests marked ``@pytest.mark.pg`` get a fresh, isolated database state via
the ``pg_conn`` fixture. The fixture is skipped (not failed) when no live
Postgres is reachable on the configured DSN, so the unit-test job stays
green on every machine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_SQL = REPO_ROOT / "infra" / "postgres" / "init.sql"


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script on ``;`` boundaries, respecting ``$$...$$``.

    psycopg3 executes one statement per ``execute()`` call, and the init
    script contains a PL/pgSQL function defined inside a dollar-quoted
    block — a naive split on ``;`` would cut the function body in half.
    """
    out: list[str] = []
    buf: list[str] = []
    in_dollar = False
    for line in sql.splitlines():
        if re.search(r"\$\$", line):
            # toggle once per `$$` occurrence on the line
            for _ in re.findall(r"\$\$", line):
                in_dollar = not in_dollar
        buf.append(line)
        if not in_dollar and line.rstrip().endswith(";"):
            stmt = "\n".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
    tail = "\n".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _maybe_connect() -> Any | None:
    """Try to open a connection; return None if unreachable."""
    try:
        from rag_kit.db import connect
    except ImportError:
        return None
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    try:
        return connect(dsn)
    except Exception:
        return None


@pytest.fixture(scope="session")
def _maybe_pg_conn():
    """Session-scoped probe: opens once, skips downstream if unavailable."""
    conn = _maybe_connect()
    if conn is None:
        yield None
        return
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def pg_conn(_maybe_pg_conn):
    """Per-test fixture: reset schema, return a live connection."""
    if _maybe_pg_conn is None:
        pytest.skip("DATABASE_URL not set or Postgres unreachable")
    if not INIT_SQL.exists():  # pragma: no cover
        pytest.skip(f"missing schema file: {INIT_SQL}")
    sql = INIT_SQL.read_text(encoding="utf-8")
    with _maybe_pg_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS documents CASCADE")
        for stmt in _split_sql_statements(sql):
            cur.execute(stmt)
    _maybe_pg_conn.commit()
    return _maybe_pg_conn
