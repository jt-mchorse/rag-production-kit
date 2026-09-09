"""`_split_sql_statements` splits on `;` in CODE, not in comments (#209).

The helper feeds `infra/postgres/init.sql` to psycopg one statement at a time,
and its docstring named exactly one hazard it respected — a `;` inside a
`$$...$$` PL/pgSQL body. It was silent on the other one, and a schema comment
whose last character was `;` cut the enclosing `CREATE TABLE` in half:

    psycopg.errors.SyntaxError: syntax error at end of input
    LINE 11: ...nt.__post_init__` makes that unreachable from Python (#182);

A schema file is mostly prose, and prose has semicolons. This broke only the
`DATABASE_URL`-gated `integration-pg` job — the helper had no test of its own,
so its failure mode was invisible to every other job and to a local run without
Docker.

These tests are hermetic: the splitter is a pure function over text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import _split_sql_statements  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_SQL = REPO_ROOT / "infra" / "postgres" / "init.sql"


def _code_lines(statement: str) -> list[str]:
    return [
        line for line in statement.splitlines() if line.strip() and not line.strip().startswith("--")
    ]


def test_a_semicolon_inside_a_comment_does_not_end_a_statement() -> None:
    """The exact shape that broke `integration-pg`."""
    sql = "\n".join(
        [
            "CREATE TABLE t (",
            "    -- a comment that ends in a semicolon (#182);",
            "    id INTEGER",
            ");",
        ]
    )
    statements = _split_sql_statements(sql)
    assert len(statements) == 1, f"split into {len(statements)}: {statements}"
    assert "CREATE TABLE t (" in statements[0]
    assert statements[0].rstrip().endswith(");")


def test_a_trailing_comment_after_code_still_ends_the_statement() -> None:
    """The mirror. A `;` in code ends the statement even when a comment trails
    it — otherwise the fix for one direction breaks the other.
    """
    sql = "CREATE TABLE t (id INTEGER);  -- a trailing note\nCREATE INDEX i ON t (id);"
    statements = _split_sql_statements(sql)
    assert len(statements) == 2, statements


def test_a_dollar_quoted_body_is_still_one_statement() -> None:
    """The hazard the docstring already named, kept as a control: the comment
    fix must not disturb it.
    """
    sql = "\n".join(
        [
            "CREATE OR REPLACE FUNCTION f() RETURNS trigger AS $$",
            "BEGIN",
            "    NEW.tsv := to_tsvector('english', coalesce(NEW.text, ''));",
            "    RETURN NEW;",
            "END",
            "$$ LANGUAGE plpgsql;",
        ]
    )
    statements = _split_sql_statements(sql)
    assert len(statements) == 1, f"the function body was cut: {statements}"


def test_a_semicolon_in_a_comment_inside_a_dollar_body_is_also_safe() -> None:
    """Both hazards at once — the row that a fix for either one alone misses."""
    sql = "\n".join(
        [
            "CREATE OR REPLACE FUNCTION f() RETURNS trigger AS $$",
            "BEGIN",
            "    -- a note; with a semicolon",
            "    RETURN NEW;",
            "END",
            "$$ LANGUAGE plpgsql;",
        ]
    )
    assert len(_split_sql_statements(sql)) == 1


def test_the_real_schema_splits_into_its_statements() -> None:
    """Against the file on disk, so a future comment cannot break the gated job
    without breaking this one first.

    Anchored on the statement kinds rather than a bare count: a count alone
    would be satisfied by a split that produced the right number of wrong
    fragments.
    """
    statements = _split_sql_statements(INIT_SQL.read_text(encoding="utf-8"))
    firsts = [_code_lines(s)[0].strip() for s in statements if _code_lines(s)]
    assert any(f.startswith("CREATE TABLE IF NOT EXISTS documents") for f in firsts)
    assert sum(f.startswith("CREATE EXTENSION") for f in firsts) == 2
    assert sum(f.startswith("CREATE INDEX") for f in firsts) == 2
    assert any(f.startswith("CREATE OR REPLACE FUNCTION") for f in firsts)
    assert any(f.startswith("CREATE TRIGGER") for f in firsts)
    assert any(f.startswith("DROP TRIGGER") for f in firsts)


def test_every_emitted_statement_is_balanced() -> None:
    """The property the broken split violated, stated directly rather than as a
    count: a fragment that cut a `CREATE TABLE` in half has unbalanced
    parentheses, which is what "syntax error at end of input" means.

    Parentheses are counted over code only — the comments in this schema
    legitimately contain unmatched ones, e.g. "(#182)" and "(see the header)".
    """
    for statement in _split_sql_statements(INIT_SQL.read_text(encoding="utf-8")):
        code = "\n".join(_code_lines(statement))
        assert code.count("(") == code.count(")"), f"unbalanced fragment:\n{statement}"


def test_the_schema_really_does_contain_a_comment_ending_in_a_semicolon() -> None:
    """Anti-vacuous, and the thing that makes the rows above about this repo
    rather than about a hypothetical.

    If the schema's comments were ever rewritten to avoid semicolons, the tests
    above would still pass while covering nothing on the real file — so assert
    the hazardous shape is actually present.
    """
    comment_lines = [
        line.strip()
        for line in INIT_SQL.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("--")
    ]
    assert any(line.rstrip().endswith(";") for line in comment_lines), (
        "no schema comment ends in ';' any more; this file's rows no longer "
        "exercise the shape that broke integration-pg"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "-- only a comment\n",
        "-- only a comment ending in a semicolon;\n",
    ],
    ids=["empty", "comment-only", "comment-only-with-semicolon"],
)
def test_a_script_with_no_statements_emits_nothing_executable(sql: str) -> None:
    """A comment-only chunk must not be handed to `execute()` as if it were a
    statement — psycopg accepts it, but an empty command is not what the caller
    means and it hides a split that went wrong.
    """
    for statement in _split_sql_statements(sql):
        assert _code_lines(statement), f"emitted a comment-only statement: {statement!r}"


def test_the_string_literal_limit_is_declared_not_pretended() -> None:
    """`--` inside a single-quoted string is deliberately not modelled.

    Named here so the limit is visible rather than discovered. This schema has
    no such literal (asserted), a real one would need a SQL lexer, and a helper
    that pretends to lex is worse than one whose limits are written down.
    """
    sql = "SELECT '-- not a comment;' AS s;"
    statements = _split_sql_statements(sql)
    # The known-wrong answer, pinned: the splitter treats the `--` as a comment
    # and so never sees the `;`, leaving one unterminated fragment.
    assert len(statements) == 1
    assert INIT_SQL.read_text(encoding="utf-8").count("'--") == 0, (
        "the schema gained a string literal containing '--'; the splitter's "
        "declared limit is now reachable and needs a real lexer"
    )
