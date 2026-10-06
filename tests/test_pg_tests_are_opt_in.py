"""The pg integration tests never touch an ambient database unasked (#270).

A bare `pytest` ran them whenever DATABASE_URL was set, and their fixture ran
`DROP TABLE IF EXISTS documents CASCADE` in whatever database that named --
measured against a scratch Postgres, another app's two-row `documents` table
became rag's seeded schema. Two layers now: `-m "not pg"` in addopts, and the
fixture refuses a `documents` table that is not rag's.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import _refuse_foreign_documents_table

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PG_TEST = "tests/test_hybrid_pg.py"


def _selected(*args: str) -> int:
    """How many tests pytest would run, from its own summary line.

    Not by counting ids: this pytest's `--collect-only -q` prints `path: count`
    and `-v` cancels against addopts' `-q`, so an id filter is empty whatever
    is collected -- a probe that cannot tell selected from deselected.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "DATABASE_URL": "postgresql://x@127.0.0.1:1/nope"},
    ).stdout
    if re.search(r"no tests collected", out):
        return 0
    m = re.search(r"(\d+)(?:/\d+)? tests? collected", out)  # "7 tests" or "5/7 tests"
    assert m, out
    return int(m.group(1))


def test_a_bare_run_does_not_select_the_pg_tests() -> None:
    assert _selected(_PG_TEST) == 0


def test_an_explicit_dash_m_pg_still_selects_them() -> None:
    assert _selected(_PG_TEST, "-m", "pg") > 0  # what CI's pg job and the README run


class _Cursor:
    def __init__(self, columns: list[str]) -> None:
        self._columns = columns
        self._last = ""

    def execute(self, sql: str) -> None:
        self._last = sql

    def fetchall(self) -> list[tuple[str]]:
        return [(c,) for c in self._columns]

    def fetchone(self) -> tuple[str]:
        return ("otherapp",)


@pytest.mark.parametrize(
    ("columns", "refused"),
    [
        ([], False),  # no table yet
        (["id", "external_id", "text", "tsv", "embedding", "metadata", "created_at"], False),
        (["id", "title", "body"], True),  # another app's table
        (["id", "external_id"], True),  # a partial lookalike
    ],
)
def test_the_fixture_refuses_a_documents_table_that_is_not_rags(
    columns: list[str], refused: bool
) -> None:
    msg = _refuse_foreign_documents_table(_Cursor(columns))
    assert (msg is not None) is refused
    if refused:
        assert "otherapp" in msg
        assert "refusing to DROP" in msg
