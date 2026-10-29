"""TemplateRewriter never keeps a blank step, and a format character splits like a space (#427).

#425 (#424) made `RewriteResult` refuse a blank sub-query. `TemplateRewriter`'s
`_norm` collapsed whitespace but kept a format character (category Cf) beside a
space or at an end, and `_split_then` filtered on truthiness. Measured on
`3cb15e8`, the `_BLANK` table through the 25 templates of `_SHAPES` below:

    "What is RAG. Then \\u200b"     ValueError: sub_queries[1] must be non-empty
        (12 cases: then trailing / "Then," trailing / 3-step trailing, on the
        zwsp, bom, word-joiner and mixed rows; the whitespace rows fell back to
        no_decomposition)
    "Compare RAG with \\u200b"      ('What is RAG?', 'What is \\u200b?')
    "What is RAG?\\u200b and how does BM25 work"   'What is RAG?\\u200b?'
    "\\u200bCompare RAG with BM25"  no_decomposition (twin: compare_pattern)

Of the 225 cases, 12 raised and 88 more differed from their whitespace twin,
the same template with `" "` in the blank slot. `_norm` now treats a blank run
that holds whitespace or touches an end as a separator, by `is_blank`'s rule,
so every case below equals its twin.
"""

from __future__ import annotations

import pytest

from rag_kit import TemplateRewriter
from rag_kit.text import is_blank
from tests.test_blank_query import _BLANK
from tests.test_non_str_query_type_check import _retriever

# Template -> (reasoning, sub_queries) of the whitespace twin. `{B}` is the
# blank slot: a whole step, a whole entity or conjunct, or the padding at
# either edge of one.
_SHAPES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # Then-split
    "then/trailing": ("What is RAG. Then {B}", "no_decomposition", ("What is RAG. Then",)),
    "then/trailing-comma": ("What is RAG. Then, {B}", "no_decomposition", ("What is RAG. Then,",)),
    "then/3-trailing": (
        "What is RAG. Then what is BM25. Then {B}",
        "sequential_then_pattern",
        ("What is RAG.", "what is BM25."),
    ),
    "then/middle": (
        "What is RAG. Then {B} Then what is BM25",
        "sequential_then_pattern",
        ("What is RAG.", "Then what is BM25"),
    ),
    "then/end-of-step": (
        "What is RAG.{B} Then what is BM25",
        "sequential_then_pattern",
        ("What is RAG.", "what is BM25"),
    ),
    "then/before-then": (
        "What is RAG. {B}Then what is BM25",
        "sequential_then_pattern",
        ("What is RAG.", "what is BM25"),
    ),
    "then/after-then": (
        "What is RAG. Then {B}what is BM25",
        "sequential_then_pattern",
        ("What is RAG.", "what is BM25"),
    ),
    "then/leading": (
        "{B}What is RAG. Then what is BM25",
        "sequential_then_pattern",
        ("What is RAG.", "what is BM25"),
    ),
    "then/trailing-pad": (
        "What is RAG. Then what is BM25{B}",
        "sequential_then_pattern",
        ("What is RAG.", "what is BM25"),
    ),
    # Compare
    "compare/a": ("Compare {B} with BM25", "no_decomposition", ("Compare with BM25",)),
    "compare/b": ("Compare RAG with {B}", "no_decomposition", ("Compare RAG with",)),
    "compare/leading": (
        "{B}Compare RAG with BM25",
        "compare_pattern",
        ("What is RAG?", "What is BM25?"),
    ),
    "compare/a-leading-pad": (
        "Compare {B}RAG with BM25",
        "compare_pattern",
        ("What is RAG?", "What is BM25?"),
    ),
    "compare/a-trailing-pad": (
        "Compare RAG{B} with BM25",
        "compare_pattern",
        ("What is RAG?", "What is BM25?"),
    ),
    "compare/b-trailing-pad": (
        "Compare RAG with BM25{B}",
        "compare_pattern",
        ("What is RAG?", "What is BM25?"),
    ),
    # "and" split
    "and/trailing": ("What is RAG and {B}", "no_decomposition", ("What is RAG and",)),
    "and/comma-trailing": ("What is RAG, and {B}", "no_decomposition", ("What is RAG, and",)),
    "and/leading": ("{B} and what is BM25", "no_decomposition", ("and what is BM25",)),
    "and/middle": (
        "What is RAG and {B} and how does BM25 work",
        "no_decomposition",
        ("What is RAG and and how does BM25 work",),
    ),
    "and/first-trailing-pad": (
        "What is RAG{B} and how does BM25 work",
        "multi_question_and_pattern",
        ("What is RAG?", "how does BM25 work?"),
    ),
    "and/second-leading-pad": (
        "What is RAG and {B}how does BM25 work",
        "multi_question_and_pattern",
        ("What is RAG?", "how does BM25 work?"),
    ),
    "and/after-question-mark": (
        "What is RAG?{B} and how does BM25 work",
        "multi_question_and_pattern",
        ("What is RAG?", "how does BM25 work?"),
    ),
    "and/after-final-question-mark": (
        "What is RAG and how does BM25 work?{B}",
        "multi_question_and_pattern",
        ("What is RAG?", "how does BM25 work?"),
    ),
    # No decomposition
    "none/leading": ("{B}What is RAG", "no_decomposition", ("What is RAG",)),
    "none/trailing": ("What is RAG{B}", "no_decomposition", ("What is RAG",)),
}


@pytest.mark.parametrize("blank", sorted(_BLANK))
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_a_blank_in_any_split_position_rewrites_as_its_whitespace_twin(
    shape: str, blank: str
) -> None:
    template, reasoning, sub_queries = _SHAPES[shape]
    out = TemplateRewriter().rewrite(template.format(B=_BLANK[blank]))
    assert not any(is_blank(sq) for sq in out.sub_queries)
    assert (out.reasoning, out.sub_queries) == (reasoning, sub_queries)


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_control_the_whitespace_twin_is_what_the_table_pins(shape: str) -> None:
    # Pins the expected column against the plain-space input, so the arm above
    # compares with what the rewriter really does for a space.
    template, reasoning, sub_queries = _SHAPES[shape]
    out = TemplateRewriter().rewrite(template.format(B=" "))
    assert (out.reasoning, out.sub_queries) == (reasoning, sub_queries)


@pytest.mark.parametrize("blank", ["zwsp", "bom", "word-joiner", "mixed"])
def test_search_with_the_template_rewriter_runs_for_a_zero_width_then_step(blank: str) -> None:
    # The issue's path: `Retriever.search` returned [] on d29bfde and raised on 3cb15e8.
    retriever, conn, embedder = _retriever()
    query = "What is RAG. Then " + _BLANK[blank]
    assert retriever.search(query, k=3, rewriter=TemplateRewriter()) == []
    assert embedder.calls == ["What is RAG. Then"]
    assert len(conn.executed) == 2


@pytest.mark.parametrize(
    "query",
    [
        # ZWNJ inside a Persian word, a soft hyphen, and a ZWJ emoji sequence: a
        # format-only run between two characters is part of the word.
        "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645",
        "co\u00adoperate",
        "\U0001f469\u200d\U0001f4bb tools",
    ],
)
def test_control_a_format_character_inside_a_word_is_kept(query: str) -> None:
    out = TemplateRewriter().rewrite(query)
    assert out.sub_queries == (query,)


def test_control_a_format_character_inside_a_then_step_is_kept() -> None:
    out = TemplateRewriter().rewrite("What is co\u00adoperation. Then what is BM25")
    assert out.sub_queries == ("What is co\u00adoperation.", "what is BM25")
