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


#: A dollar-quote delimiter: ``$$`` or ``$tag$`` where *tag* is an identifier.
#: Postgres allows the tag to be empty (``$$``) or any identifier, and a body
#: is closed only by its own tag (#211).
_DOLLAR_TAG = re.compile(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$")


def _code_before_comment(line: str) -> str:
    """The part of *line* before a ``--`` comment.

    A statement ends at a ``;`` in CODE, never at one inside a comment. The
    splitter below respected ``$$...$$`` and nothing else, so a comment whose
    last character happened to be ``;`` cut the enclosing statement in half and
    psycopg reported ``syntax error at end of input`` pointing at the comment
    text (#209). A schema file is mostly prose, and prose has semicolons.

    ``--`` inside a single-quoted string is deliberately NOT modelled. This
    schema has no such literal, a real one would need a SQL lexer, and a helper
    that pretends to lex is worse than one whose limits are written down —
    ``test_conftest_sql_splitter.py`` has a named row for it so the choice is
    visible rather than assumed. A ``/* ... */`` block comment is declared the
    same way and for the same reason (#211).

    Called on *every* line since #211, including lines inside a dollar-quoted
    body. #209 made the splitter comment-aware through a ternary --
    ``_code_before_comment(line) if not in_dollar else line`` -- and only the
    ``not in_dollar`` arm got it, so a ``--`` comment inside a PL/pgSQL body
    had its text scanned for delimiters. A comment mentioning the ``$$`` the
    body itself is quoted with closed the block and flushed mid-function: the
    same defect #209 fixed, on the other branch of the same expression.
    """
    marker = line.find("--")
    return line if marker == -1 else line[:marker]


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script on ``;`` boundaries in code, respecting dollar quotes.

    psycopg3 executes one statement per ``execute()`` call, and the init
    script contains a PL/pgSQL function defined inside a dollar-quoted
    block — a naive split on ``;`` would cut the function body in half.

    Two things end a statement's line and only one of them was modelled: a
    ``;`` in code, and — wrongly — a ``;`` at the end of a ``--`` comment. The
    docstring named the one hazard it respected and was silent on the other,
    which is how a schema comment ending in ``(#182);`` broke the
    ``DATABASE_URL``-gated job and nothing else (#209).

    #211 closed two more, both of them in the delimiter scan rather than in the
    ``;`` test:

    * ``--`` comments are stripped on **both** arms now. #209's fix went in as
      ``_code_before_comment(line) if not in_dollar else line`` and only the
      first arm got it, so a comment *inside* a body had its text read as code
      — a comment mentioning ``$$`` closed the block and flushed mid-function.
      Measured: 3 statements where there is 1.
    * The **tagged** form ``$tag$ ... $tag$`` is recognised, and the close must
      carry the same tag as the open. ``AS $func$`` and ``AS $BODY$`` are the
      conventional PL/pgSQL spellings and were not dollar-quoted at all as far
      as this scan was concerned, so every ``;`` in the body split. Measured: 4
      statements where there is 1.

      Requiring the tag to match is the whole content of the second half. A
      scan that toggles on any ``$...$`` treats a bare ``$$`` inside a
      ``$func$`` body as a delimiter, when it is body text — the wrong fix is
      green on the tagged rows and re-breaks the untagged one.

    Still a line-oriented helper, deliberately, and its limits are declared in
    ``_code_before_comment`` and pinned as named rows with reachability
    assertions in ``test_conftest_sql_splitter.py``. A helper that pretends to
    lex SQL is worse than one whose limits are written down.
    """
    out: list[str] = []
    buf: list[str] = []
    # The open delimiter's full text (``$$`` or ``$tag$``) while inside a body,
    # else None. Holding the *text* and not a bool is what makes the close
    # have to match the open.
    open_tag: str | None = None

    def _flush() -> None:
        stmt = "\n".join(buf).strip()
        # A chunk with no code is a comment block, not a statement. Postgres
        # accepts an empty command without complaint, so emitting one is
        # harmless AND invisible -- which is exactly what makes it worth
        # dropping: a caller iterating "statements" should never be handed
        # something that executes nothing, because that is also what a split
        # gone wrong looks like.
        if stmt and any(
            line.strip() and not line.strip().startswith("--") for line in stmt.splitlines()
        ):
            out.append(stmt)

    for line in sql.splitlines():
        # Both arms (#211). Inside a body a `--` comment is still a comment.
        code = _code_before_comment(line)
        for tag in _DOLLAR_TAG.findall(code):
            if open_tag is None:
                open_tag = tag
            elif tag == open_tag:
                open_tag = None
            # else: a different tag inside an open body is body text, not a
            # delimiter. `$$` inside a `$func$` body is the case that makes
            # this an `elif` and not an `else`.
        buf.append(line)
        if open_tag is None and code.rstrip().endswith(";"):
            _flush()
            buf = []
    _flush()
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
