"""`init.sql`'s prose points at places that exist and actually say it (#209).

`tests/test_embedding_width_seam.py` (#194) already parses this file and pins
the *number* — that `vector(N)` and `EMBEDDING_DIM` agree. Its docstring says
"An operator obligation with no check is a convention, not a contract." The
obligation it enforces was stated correctly in `docs/architecture.md` and
**incorrectly in the file the test reads**:

    -- line 7:  "Dense vector dimensionality is parameterized through a
    --           settings table"          <- there is no settings table
    -- line 22: "dim documented in pyproject + README"
    --                                    <- neither mentions the dimension

Measured before the fix::

    grep -rn "settings table" rag_kit/ infra/ docs/ tests/ scripts/ README.md
      (no matches)
    grep -in "EMBEDDING_DIM|vector(64)|dimension" README.md
      (no matches)
    grep -in "dim|vector" pyproject.toml
      (nothing about the embedding width)

A lock on the value and none on the instructions. `init.sql` is the file an
operator opens *first* when changing a column width, its own header calls it
"the single source of truth the Python indexer/retriever assume", and the one
place that would have corrected them — `rag_kit/db.py`'s runtime error — does
not speak until after they have already got it wrong.

The checks here are **derived**, not a list of the two strings that were wrong:

- every backticked path the file names resolves on disk;
- every backticked ``UPPER_SNAKE`` symbol it names is importable from the
  package;
- and the sharp one: for every file the prose claims documents the width, that
  file must actually contain a width reference. That is the arm that catches
  "documented in pyproject + README", and it catches the next such pointer
  without anyone editing this test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import rag_kit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INIT_SQL = _REPO_ROOT / "infra" / "postgres" / "init.sql"
_ARCHITECTURE = _REPO_ROOT / "docs" / "architecture.md"
_DB_PY = _REPO_ROOT / "rag_kit" / "db.py"

#: Anything inside single backticks in the file's comments.
_BACKTICKED = re.compile(r"`([^`\n]+)`")

#: A repo-relative path: has a `/` and a file extension this repo uses.
_PATH_LIKE = re.compile(r"^[\w./-]+\.(?:py|sql|md|toml|json|yml|yaml)$")

#: A module-level constant: all caps with underscores, no dots or parens.
_CONST_LIKE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def _init_sql_text() -> str:
    return _INIT_SQL.read_text(encoding="utf-8")


def _backticked_tokens() -> list[str]:
    """Backticked tokens in the file's *comments*.

    Comments only: the DDL itself contains no backticks today, but a future
    quoted identifier would be code rather than a pointer at an operator, and
    this file's whole subject is what the prose tells an operator to go and
    edit.
    """
    return [m.group(1).strip() for c in _sql_comments() for m in _BACKTICKED.finditer(c)]


def test_the_file_actually_uses_backticks() -> None:
    """Anti-vacuous for every discovered check below: a file with no backticked
    tokens would satisfy all of them while checking nothing.
    """
    assert len(_backticked_tokens()) >= 8


def test_every_path_the_prose_names_resolves() -> None:
    paths = [t for t in _backticked_tokens() if _PATH_LIKE.match(t)]
    assert paths, "no path-shaped tokens found; the classifier stopped matching"
    missing = [p for p in paths if not (_REPO_ROOT / p).exists()]
    assert not missing, (
        f"infra/postgres/init.sql points an operator at files that do not exist: {missing}"
    )


def test_every_constant_the_prose_names_is_importable() -> None:
    consts = [t for t in _backticked_tokens() if _CONST_LIKE.match(t)]
    assert consts, "no constant-shaped tokens found; the classifier stopped matching"
    missing = [c for c in consts if not hasattr(rag_kit, c)]
    assert not missing, (
        f"infra/postgres/init.sql names constants that are not on the package surface: {missing}"
    )


def test_the_classifier_is_not_skipping_everything() -> None:
    """The two checks above only mean something if the classifier recognises a
    reasonable share of what the file writes. If a future edit switched to a
    different citation style, both would pass by matching nothing — which is
    the shape of the bug this file exists for, one level up.
    """
    tokens = _backticked_tokens()
    recognised = [t for t in tokens if _PATH_LIKE.match(t) or _CONST_LIKE.match(t)]
    assert len(recognised) >= 4, (
        f"only {len(recognised)} of {len(tokens)} backticked tokens were classified; "
        "the path/constant patterns have stopped matching this file's style"
    )


# --- the sharp one --------------------------------------------------------

#: A comment that is part of the width discussion *by vocabulary*.
#:
#: Two rewrites, both found by falsifying rather than by reading. First I keyed
#: on a verb — `(documented|configured|...) (in|by)` — and it matched none of the
#: corrected prose, because "configured PER DEPLOYMENT by editing" does not put
#: the preposition where the pattern expected. Then I keyed on the subject and it
#: still missed the ACTUAL bug, whose wording is the abbreviation: "dim
#: documented in pyproject + README". `dim` is now in the vocabulary, but the
#: lesson is that a lexical detector is a guess about how the next writer will
#: phrase things — which is why the structural half below exists and is the one
#: that carries this file's original defect.
_WIDTH_TOPIC = re.compile(r"EMBEDDING_DIM|vector\(|dimension|\bdim\b|\bwidth\b", re.IGNORECASE)

#: The embedding column's own declaration. Its trailing comment is part of the
#: width discussion BY CONSTRUCTION — it is the line the width is declared on —
#: so it is included regardless of what words it happens to use. That is where
#: "dim documented in pyproject + README" lived for four months.
_EMBEDDING_DECLARATION = re.compile(r"^\s*embedding\s+vector\(", re.MULTILINE)

#: What a "width reference" looks like in any of this repo's file types.
_WIDTH_REFERENCE = re.compile(r"EMBEDDING_DIM|vector\(\s*\d+\s*\)|vector\(\s*N\s*\)", re.IGNORECASE)

#: File-ish names the prose might use, with or without backticks or a path.
_FILE_MENTION = re.compile(r"[\w./-]+\.(?:py|sql|md|toml|json|yml|yaml)|\bpyproject\b|\bREADME\b")

_FILENAME_ALIASES = {"pyproject": "pyproject.toml", "README": "README.md"}


def _sql_comments() -> list[str]:
    """The comment text of every line, whether the comment is the whole line or
    trails a declaration.

    Trailing comments are the half this originally missed, and they are the half
    the bug was in: the stale "dim documented in pyproject + README" pointer sat
    after `embedding vector(64) NOT NULL,` on the column line, not on a `--`
    line of its own. A first version of this scan read only lines *starting*
    with `--`, so re-introducing the exact pointer #209 is about left it green.
    Falsifying each probe separately is what surfaced that.
    """
    comments: list[str] = []
    for line in _init_sql_text().splitlines():
        marker = line.find("--")
        if marker != -1:
            comments.append(line[marker:])
    return comments


def _embedding_declaration_comment() -> str:
    """The trailing comment on the `embedding vector(N)` line, or ``""``.

    Structural, not lexical. The width is declared on this line, so whatever it
    says about the width is in scope whether or not it uses a word a regex
    happens to know — and the pointer #209 is about said "dim", which the
    vocabulary above did not have until falsifying the probe showed it.
    """
    for line in _init_sql_text().splitlines():
        if _EMBEDDING_DECLARATION.match(line):
            marker = line.find("--")
            return line[marker:] if marker != -1 else ""
    return ""


def _width_discussion_lines() -> list[str]:
    """Every comment in `init.sql` that talks about the vector width.

    The union of the structural half (the declaration's own comment) and the
    lexical half (anything else using the vocabulary). Union rather than either
    alone: the header block is not structurally locatable, and the declaration's
    comment is not lexically reliable.
    """
    found = [c for c in _sql_comments() if _WIDTH_TOPIC.search(c)]
    declaration = _embedding_declaration_comment()
    if declaration and declaration not in found:
        found.append(declaration)
    return found


def test_the_embedding_declaration_is_locatable() -> None:
    """Anti-vacuous for the structural half. If the column were renamed or
    reformatted, `_embedding_declaration_comment` would silently return ``""``
    and the scan would quietly fall back to the lexical half alone — which is
    the half that missed this file's original defect.
    """
    assert _EMBEDDING_DECLARATION.search(_init_sql_text()), (
        "could not locate the `embedding vector(N)` declaration; the structural "
        "half of the width-discussion scan is dead"
    )


def _files_named_in_the_width_discussion() -> list[str]:
    """Repo-relative paths the width discussion points an operator at."""
    named: list[str] = []
    for line in _width_discussion_lines():
        for name in _FILE_MENTION.findall(line):
            resolved = _FILENAME_ALIASES.get(name, name)
            if resolved not in named:
                named.append(resolved)
    return named


def test_the_width_discussion_is_findable() -> None:
    """Anti-vacuous for both checks below. If `init.sql` stopped discussing the
    width in comments at all, they would pass by scanning nothing — and an
    operator opening the schema first would be told nothing, which is a worse
    version of the bug this file is about.
    """
    assert len(_width_discussion_lines()) >= 3


def test_every_file_the_width_discussion_names_actually_carries_the_width() -> None:
    """The arm that catches "dim documented in pyproject + README".

    Derived, not a denylist of the two names that were wrong: any future
    pointer at a file that says nothing about the width fails here without
    anyone editing this test. Measured before #209 — `pyproject.toml` mentions
    the width nowhere and `README.md` mentions it nowhere.
    """
    named = _files_named_in_the_width_discussion()
    assert named, "the width discussion names no files at all; see the test below"
    for target in named:
        path = _REPO_ROOT / target
        assert path.exists(), (
            f"init.sql's width discussion points an operator at {target}, which does not exist"
        )
        assert _WIDTH_REFERENCE.search(path.read_text(encoding="utf-8")), (
            f"init.sql's width discussion points an operator at {target}, but that "
            "file contains no EMBEDDING_DIM or vector(N) reference"
        )


def test_the_width_discussion_names_the_other_half_of_the_obligation() -> None:
    """And it must name a real file, not a mechanism.

    This is what a re-introduced "parameterized through a settings table" fails:
    that sentence names no file, so the second half of D-003's two-place
    obligation goes unstated and the check above has nothing to verify. The two
    arms are complementary — one rejects a pointer at the wrong file, this one
    rejects a pointer at no file.
    """
    named = _files_named_in_the_width_discussion()
    assert any(t.endswith("embedder.py") for t in named), (
        "init.sql's width discussion must name `rag_kit/embedder.py`, the other "
        f"half of D-003's two-place obligation; it named {named}"
    )


# --- the three sources must agree -----------------------------------------


@pytest.mark.parametrize(
    "source",
    ["infra/postgres/init.sql", "docs/architecture.md", "rag_kit/db.py"],
)
def test_all_three_sources_name_both_halves_of_the_obligation(source: str) -> None:
    """D-003's obligation is a *pair* of places, and three files describe it:
    this schema, the architecture doc, and the runtime error `to_pgvector`
    raises. Before #209 the schema described a third mechanism instead.

    Asserted per-source so the failure names which one drifted, rather than a
    single set-equality whose message says only "they disagree".
    """
    text = (_REPO_ROOT / source).read_text(encoding="utf-8")
    assert "EMBEDDING_DIM" in text, f"{source} does not name the EMBEDDING_DIM half"
    assert "init.sql" in text or source.endswith("init.sql"), (
        f"{source} does not name the init.sql half"
    )


def test_external_id_is_not_nullable() -> None:
    """`indexer.py` states the guarantee: "It's UNIQUE in the schema, so
    re-indexing the same chunk overwrites cleanly."

    Postgres UNIQUE permits multiple NULLs, so that held only for a non-NULL id
    — and `ON CONFLICT (external_id) DO UPDATE` never fires for a NULL one, so
    each re-index would insert another row instead of overwriting. Asserted
    here rather than in the `DATABASE_URL`-gated suite, because the claim is
    about the DDL and this way it runs on every push.
    """
    declaration = re.search(r"^\s*external_id\s+([^,\n]+)", _init_sql_text(), re.MULTILINE)
    assert declaration is not None, "could not find the external_id column declaration"
    constraints = declaration.group(1).upper()
    assert "NOT NULL" in constraints, (
        f"external_id is declared `{declaration.group(1).strip()}`; UNIQUE alone "
        "permits multiple NULLs, which defeats the ON CONFLICT upsert"
    )
    assert "UNIQUE" in constraints
