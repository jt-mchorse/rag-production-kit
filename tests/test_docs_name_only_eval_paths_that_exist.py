"""Docs describe only the eval and bench paths the code has (#414).

The README, `docs/benchmarks.md`, `evals/run_eval.py` and `eval.yml` described
an operator-triggered real-LLM / pgvector eval run selected by
`RAG_EVAL_BACKEND` (as `pgvector` in one comment, `anthropic` in the other).
Nothing read that variable, so an operator who set the keys still got the
hermetic fixture. The README also said `scripts/bench_rewriter.py` "wires"
`AnthropicRewriter` and runs "on your own corpus", and the script takes
neither a rewriter nor a corpus.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts import bench_rewriter

ROOT = Path(__file__).resolve().parent.parent


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", *patterns], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def _is_read(name: str) -> bool:
    reader = re.compile(rf"""(environ(\.get)?\s*[\[(]\s*|getenv\s*\(\s*)["']{name}["']""")
    return any(reader.search(p.read_text(encoding="utf-8")) for p in _tracked("*.py"))


def test_a_backend_switch_the_docs_name_is_read_by_some_code() -> None:
    """Any `RAG_EVAL_*` variable named outside MEMORY and tests must have a reader.

    `evals/__init__.py` named a third spelling, `RAG_EVAL_GENERATOR=anthropic`,
    that no issue listed; this arm found it. Tests are excluded because they
    name variables in order to check them (this file does).
    """
    texts = [
        p
        for p in _tracked("*.md", "*.py", "*.yml", "*.yaml", "*.toml", ".env.example")
        if "MEMORY" not in p.parts and "tests" not in p.parts
    ]
    named = {
        m.group(0)
        for p in texts
        for m in re.finditer(r"\bRAG_EVAL_[A-Z_]+\b", p.read_text(encoding="utf-8"))
    }
    unread = sorted(n for n in named if not _is_read(n))
    assert unread == [], f"named in docs/comments but read by no code: {unread}"


def test_the_switch_reader_regex_sees_a_real_read() -> None:
    """Control: the reader check is not vacuous (DATABASE_URL is read by db code)."""
    assert _is_read("DATABASE_URL")
    assert not _is_read("RAG_EVAL_BACKEND_NEVER_DEFINED")


def _bench_rewriter_options(capsys: pytest.CaptureFixture[str]) -> set[str]:
    with pytest.raises(SystemExit):
        bench_rewriter.main(["--help"])
    return set(re.findall(r"(--[a-z][a-z-]*)", capsys.readouterr().out)) - {"--help"}


def test_bench_rewriter_still_takes_no_rewriter_or_corpus(capsys) -> None:
    """The README's wording below is true only while this holds; adding an
    option means rewording the README, and this arm says so."""
    assert _bench_rewriter_options(capsys) == {"--k", "--output"}


@pytest.mark.parametrize(
    "claim",
    [
        r"operator-triggered with `ANTHROPIC_API_KEY`",
        r"script wires it",
        r"on your own corpus",
        r"RAG_EVAL_BACKEND",
        r"RAG_EVAL_GENERATOR",
    ],
)
def test_docs_do_not_claim_a_path_that_does_not_exist(claim: str) -> None:
    for name in ("README.md", "docs/benchmarks.md", "evals/__init__.py", "evals/run_eval.py"):
        text = " ".join((ROOT / name).read_text(encoding="utf-8").split())
        assert re.search(claim, text) is None, f"{name} claims {claim!r}"


def test_readme_says_the_real_llm_numbers_are_pending() -> None:
    text = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert "There is no real-LLM eval path yet." in text
    assert "an `AnthropicRewriter` number is pending an implementation (#414)" in text
