"""No install hint resolves this project from a package index (#338).

This repo is installed from a checkout (`pip install -e '.[extra]'`); it is not
published to PyPI. A hint of the form `pip install 'DIST[extra]'` sends the
reader to the index instead, where `rag-production-kit` is unclaimed (404):
the hint fails today, and installs whoever registers the name tomorrow.
Following such a hint means installing whatever an unrelated publisher put
under that name, in the environment that holds the operator's API keys.

The distribution name is read from `pyproject.toml`, and every tracked text
file outside `MEMORY/` (an append-only log that may quote the old hint) is
scanned, so a hint added anywhere later fails here too.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]

# Everything after `pip install` on the same line, up to prose that follows
# the command ("... from the <repo> checkout"), which may name the repo.
_INSTALL = re.compile(r"\bpip3? install\b(.*?)(?:\s+(?:from|in)\s.*)?$")
# The distribution as a bare requirement: not part of a path or URL
# (`/jt-mchorse/NAME`, `git+https://...`) and not a longer name.
_FROM_INDEX = re.compile(rf"(?<![\w./+:@-]){re.escape(NAME)}(?![\w.-])", re.IGNORECASE)


def _tracked_text_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for rel in out:
        if rel.startswith("MEMORY/") or rel == Path(__file__).relative_to(ROOT).as_posix():
            continue
        path = ROOT / rel
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def _install_hints():
    for rel, text in _tracked_text_files():
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in _INSTALL.finditer(line):
                yield rel, lineno, m.group(1)


def test_no_install_hint_resolves_this_project_from_an_index() -> None:
    offenders = [
        f"{rel}:{lineno}: pip install{args}"
        for rel, lineno, args in _install_hints()
        if _FROM_INDEX.search(args)
    ]
    assert not offenders, (
        f"{NAME!r} is not published by this project; on an index it resolves to "
        "someone else's package or nothing. Use `pip install -e '.[extra]'` from "
        "the checkout:\n" + "\n".join(offenders)
    )


def test_the_scan_sees_the_repos_real_install_hints() -> None:
    """Control: an empty scan (a bad regex, `git ls-files` returning nothing)
    would make the test above pass vacuously."""
    hints = [args for _, _, args in _install_hints()]
    assert any("-e" in args and "." in args for args in hints), hints[:10]


def test_the_pattern_catches_the_shapes_it_exists_for() -> None:
    def from_index(rest: str) -> bool:
        m = _INSTALL.search("pip install" + rest)
        return m is not None and bool(_FROM_INDEX.search(m.group(1)))

    for bad in (f" '{NAME}[x]'", f" {NAME}", f' "{NAME}[x,y]"', f" -U {NAME}"):
        assert from_index(bad), bad
    for ok in (
        " -e '.[x]'",
        f" git+https://github.com/jt-mchorse/{NAME}",
        f" -e ../{NAME}",
        f" {NAME}-extras",
        f" -e . from the {NAME} checkout",
    ):
        assert not from_index(ok), ok
