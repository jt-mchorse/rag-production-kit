"""`.env.example` lists exactly the environment variables the code reads (#241).

The file listed `DATABASE_URL` and nothing else, while the code reads three
more: `ANTHROPIC_API_KEY` (generator, rewriter), `COHERE_API_KEY` (the Cohere
reranker, documented nowhere) and `GITHUB_TOKEN` (the eval runner's PR
comment). portfolio-ops#80's survey marked this repo done because the file
existed. The set is derived from source, so a new read fails here until it is
listed, and a listed variable nobody reads fails too.

Scope: every tracked `.py` except the hermetic unit tests (`tests/test_*.py`).
`tests/conftest.py` is in scope; it reads `DATABASE_URL` to gate the pg suite.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Every read here is explicit, so nothing is read only inside an SDK.
SDK_IMPLICIT: frozenset[str] = frozenset()

_READ_PATTERNS = (
    re.compile(r"""os\.environ\.get\(\s*["']([A-Z][A-Z0-9_]*)["']"""),
    re.compile(r"""os\.environ\[\s*["']([A-Z][A-Z0-9_]*)["']\s*\]"""),
    re.compile(r"""os\.getenv\(\s*["']([A-Z][A-Z0-9_]*)["']"""),
    re.compile(r"""["']([A-Z][A-Z0-9_]*)["']\s+in\s+os\.environ"""),
)


def _in_scope(rel: str) -> bool:
    parts = Path(rel).parts
    return not (len(parts) == 2 and parts[0] == "tests" and parts[1].startswith("test_"))


def names_read(text: str) -> set[str]:
    return {m for pattern in _READ_PATTERNS for m in pattern.findall(text)}


def _names_read_by_repo() -> set[str]:
    files = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    names: set[str] = set()
    for rel in files:
        if _in_scope(rel):
            names |= names_read((REPO_ROOT / rel).read_text(encoding="utf-8"))
    return names | SDK_IMPLICIT | {_token_env_default()}


def _token_env_default() -> str:
    """The one dynamic read: `os.environ.get(args.token_env)` in the eval runner.

    Resolved through the flag's default as written in source, so renaming the
    default moves the name this test requires.
    """
    text = (REPO_ROOT / "evals" / "run_eval.py").read_text(encoding="utf-8")
    m = re.search(r"""["']--token-env["'],\s*default=["']([A-Z][A-Z0-9_]*)["']""", text)
    assert m, "evals/run_eval.py no longer declares a --token-env default"
    assert "os.environ.get(args.token_env)" in text, "the dynamic read moved; update this resolver"
    return m.group(1)


def _names_listed() -> dict[str, str]:
    listed: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
        if m:
            listed[m.group(1)] = m.group(2)
    return listed


def test_the_reader_sees_every_spelling() -> None:
    text = (
        'os.environ.get("A_1")\nos.environ["B"]\nos.getenv( "C" )\n'
        "if 'D' in os.environ: pass\nos.environ.get(name)\n"
    )
    assert names_read(text) == {"A_1", "B", "C", "D"}


def test_the_scan_found_every_layers_reads() -> None:
    # A floor on the population, so a scope bug cannot make the arms below
    # compare two empty sets.
    assert {"DATABASE_URL", "ANTHROPIC_API_KEY", "COHERE_API_KEY"} <= _names_read_by_repo()


def test_the_dynamic_read_resolves_to_github_token() -> None:
    assert _token_env_default() == "GITHUB_TOKEN"


def test_every_variable_read_is_listed() -> None:
    assert ENV_EXAMPLE.is_file(), ".env.example is missing (handoff §10)"
    missing = sorted(_names_read_by_repo() - _names_listed().keys())
    assert not missing, f".env.example does not list {missing}, which the code reads"


def test_every_variable_listed_is_read() -> None:
    extra = sorted(_names_listed().keys() - _names_read_by_repo())
    assert not extra, f".env.example lists {extra}, which nothing reads"


def test_the_secrets_are_placeholders() -> None:
    listed = _names_listed()
    for name in ("ANTHROPIC_API_KEY", "COHERE_API_KEY", "GITHUB_TOKEN"):
        assert "your-" in listed[name], f"{name}={listed[name]!r} does not look like a placeholder"
