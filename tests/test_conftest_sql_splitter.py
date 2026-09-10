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
        line
        for line in statement.splitlines()
        if line.strip() and not line.strip().startswith("--")
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


# ---------------------------------------------------------------------------
# #211 — the delimiter scan, not the `;` test
#
# #209's fix went in as a ternary:
#
#     code = _code_before_comment(line) if not in_dollar else line
#
# and the comment-awareness landed on ONE arm of it. Inside a dollar-quoted
# body the raw line was used and the `$$` scan ran against it, so a `--`
# comment inside a PL/pgSQL body had its text read as code. That is the same
# defect #209 fixed, on the other branch of the same expression.
#
# Measured on the parent commit:
#
#     $$ inside a -- comment INSIDE a dollar body   -> 3 statements
#     tagged dollar quote $func$                    -> 4 statements
#     block comment whose line ends in ';'          -> 2 statements
#     control: plain $$ body                        -> 1 statement
#     control: -- comment ending in ';'             -> 1 statement
#
# The first is reachable in `init.sql` today: it has exactly one dollar-quoted
# function and it is spelled `$$`, so a comment inside that body documenting
# the delimiter it is quoted with closes the block. The second becomes
# reachable the moment a second function is written the conventional way.
# ---------------------------------------------------------------------------


def test_a_dollar_delimiter_named_inside_a_comment_does_not_close_the_body() -> None:
    """The #211 shape, and the mirror of the #209 shape one branch over.

    A comment that documents the delimiter the body is quoted with is the
    ordinary way this gets written, which is what makes it reachable rather
    than adversarial.
    """
    sql = "\n".join(
        [
            "CREATE FUNCTION f() RETURNS void AS $$",
            "BEGIN",
            "  -- the $$ delimiter is written like this",
            "  RAISE NOTICE 'x';",
            "END;",
            "$$ LANGUAGE plpgsql;",
        ]
    )
    statements = _split_sql_statements(sql)
    assert len(statements) == 1, f"body was cut into {len(statements)}: {statements}"
    # Not just the count: the whole body has to be in there. A split that
    # happened to rejoin would satisfy a count assertion.
    assert "RAISE NOTICE 'x';" in statements[0]
    assert statements[0].rstrip().endswith("$$ LANGUAGE plpgsql;")


@pytest.mark.parametrize(
    ("label", "tag"),
    [
        ("lowercase identifier", "$func$"),
        ("uppercase identifier", "$BODY$"),
        ("underscore-led", "$_x$"),
        ("digits after the first char", "$b2$"),
        ("untagged, the control", "$$"),
    ],
)
def test_a_tagged_dollar_body_is_one_statement(label: str, tag: str) -> None:
    """`AS $func$ ... $func$` is the conventional PL/pgSQL spelling.

    The scan matched `\\$\\$` only, so a tagged body was not dollar-quoted at
    all as far as it was concerned and every `;` in it split.
    """
    sql = "\n".join(
        [
            f"CREATE FUNCTION g() RETURNS void AS {tag}",
            "BEGIN",
            "  RAISE NOTICE 'a';",
            "  RAISE NOTICE 'b';",
            "END;",
            f"{tag} LANGUAGE plpgsql;",
        ]
    )
    statements = _split_sql_statements(sql)
    assert len(statements) == 1, f"{label}: body cut into {len(statements)}"
    assert "RAISE NOTICE 'b';" in statements[0], label


def test_a_bare_dollar_dollar_inside_a_tagged_body_is_body_text() -> None:
    """The separating row for the tagged half.

    This is the whole reason the close has to carry the *same* tag as the open.
    A scan that toggles on any `$...$` reads this `$$` as a delimiter, closes
    the body early, and splits on the next `;` — so it is green on the tagged
    rows above and red here. Tagging exists in Postgres precisely so a body can
    contain `$$`.
    """
    sql = "\n".join(
        [
            "CREATE FUNCTION g() RETURNS void AS $func$",
            "BEGIN",
            "  RAISE NOTICE 'uses $$ here';",
            "  RETURN;",
            "END;",
            "$func$ LANGUAGE plpgsql;",
        ]
    )
    statements = _split_sql_statements(sql)
    assert len(statements) == 1, f"body cut into {len(statements)}: {statements}"
    assert "RETURN;" in statements[0]


def test_the_schema_really_does_contain_a_dollar_quoted_body() -> None:
    """Anti-vacuous arm for the two rows above.

    They are hermetic by design, but the reason they matter is that `init.sql`
    has a dollar-quoted function *right now*. If it ever loses one, these rows
    stop covering anything reachable and someone should be told rather than
    left with a green suite.
    """
    text = INIT_SQL.read_text(encoding="utf-8")
    assert "$$" in text, "init.sql no longer contains a dollar-quoted body"
    assert "LANGUAGE plpgsql" in text
    # And the body is spelled untagged, which is what makes the comment shape
    # in the first test reachable in this file today.
    assert "AS $$" in text


def test_the_block_comment_limit_is_declared_not_pretended() -> None:
    """`/* ... */` is deliberately not modelled, like `--` in a string literal.

    Same reasoning, same shape of test: pin the known-wrong answer, then assert
    the limit is unreachable in the real schema so the declaration cannot
    quietly become a live bug.
    """
    sql = "\n".join(["CREATE TABLE t (", "  /* note ends here;", "  */", "  id INT", ");"])
    statements = _split_sql_statements(sql)
    # The known-wrong answer, pinned: the first line of the block comment ends
    # in `;`, which the line-oriented scan reads as a statement boundary.
    assert len(statements) == 2
    assert "/*" not in INIT_SQL.read_text(encoding="utf-8"), (
        "the schema gained a /* ... */ block comment; the splitter's declared "
        "limit is now reachable and needs a real lexer"
    )


def test_the_real_schema_statement_count_is_unchanged_by_the_widening() -> None:
    """A delimiter scan that got wider could merge statements that should split.

    The two widenings only ever *keep a body together*, never break one apart,
    so the real file must produce exactly what it produced before. Pinned
    against the count the `integration-pg` job has been executing.
    """
    statements = _split_sql_statements(INIT_SQL.read_text(encoding="utf-8"))
    # 8, measured on the parent commit and again after the change. The number
    # is written down rather than guessed: I guessed 11 first and the test
    # said 8, which is the only reason this assertion is worth anything.
    assert len(statements) == 8, (
        f"init.sql now splits into {len(statements)} statements; if that is "
        "intended, update this number and check integration-pg"
    )
    # And the function is one of them, whole.
    bodies = [s for s in statements if "LANGUAGE plpgsql" in s]
    assert len(bodies) == 1
    assert "to_tsvector" in bodies[0]
    assert "RETURN NEW;" in bodies[0]
