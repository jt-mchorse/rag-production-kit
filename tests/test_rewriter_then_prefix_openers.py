"""The Then-prefix strip keeps the opener that starts the next step (#332).

`_THEN_PREFIX` was `^then\\b\\W*`. `\\W` is every non-word character, so besides
the separator after the connective (`Then,`, #92) it took the opening quote,
bracket, emphasis marker, backtick, `¿` or `「` of the step's own content, and
left its closer unbalanced: `Then "Dune" ...` -> `Dune" ...`. #320/#322 keep
closers; this keeps their openers.
"""

from __future__ import annotations

import pytest

from rag_kit.rewriter import TemplateRewriter, _split_then


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (
            'Then "Attention Is All You Need" -- who wrote it?',
            '"Attention Is All You Need" -- who wrote it?',
        ),
        ("Then (optionally) build the HNSW index.", "(optionally) build the HNSW index."),
        ("Then **rerank** the results.", "**rerank** the results."),
        ("Then [the reranker] scores them.", "[the reranker] scores them."),
        ("Then 「東京」 search.", "「東京」 search."),
        ("Then ¿qué pasa con el índice?", "¿qué pasa con el índice?"),
        ("Then `pip install rag-kit` again.", "`pip install rag-kit` again."),
        ("Then .NET clients query it.", ".NET clients query it."),
        ("Then: (a) index the docs.", "(a) index the docs."),
        ("Then, “Dune” again.", "“Dune” again."),
    ],
)
def test_the_opener_of_the_next_step_is_kept(step: str, expected: str) -> None:
    out = TemplateRewriter().rewrite(f"Index the docs. {step}")
    assert out.reasoning == "sequential_then_pattern"
    assert out.sub_queries == ("Index the docs.", expected)


@pytest.mark.parametrize(
    "connective",
    ["Then", "Then,", "Then;", "Then:", "Then-", "Then —", "Then –", "Then...", "Then…", "Then ,"],
)
def test_separator_punctuation_still_goes_with_the_connective(connective: str) -> None:
    assert _split_then(f"Do A. {connective} do B.") == ["Do A.", "do B."]


def test_an_attached_dash_still_goes_with_the_connective() -> None:
    # #92's own repro.
    assert _split_then("Do A. then-do B.") == ["Do A.", "do B."]
