"""The rewriter's Then-split reads closers as the generator's splitter does (#320).

`_THEN_SPLIT` claimed "the identical closing-punct class" as
`generator._SENTENCE_SPLIT`, but kept a six-character class and at most one
closer after #262 widened the generator to 26 closers in runs. A first step
ending `.»`, `.」`, `.）`, `.**` or `.")` before "Then" did not decompose.
"""

from __future__ import annotations

import pytest

from rag_kit.generator import split_sentences
from rag_kit.rewriter import TemplateRewriter
from rag_kit.text import SENTENCE_CLOSERS

_REWRITER = TemplateRewriter()


@pytest.mark.parametrize("closers", list(SENTENCE_CLOSERS) + ['")', "**", "」）", "»"])
def test_a_step_ending_terminator_and_closers_decomposes(closers: str) -> None:
    query = f"Find the revenue.{closers} Then compute the growth?"
    result = _REWRITER.rewrite(query)
    assert result.reasoning == "sequential_then_pattern", (closers, result)
    assert result.sub_queries == (f"Find the revenue.{closers}", "compute the growth?")


@pytest.mark.parametrize("closers", ["", '"', '")', "**", "」"])
def test_the_generator_splits_the_same_text_at_the_same_place(closers: str) -> None:
    text = f"Find the revenue.{closers} Then compute the growth."
    assert split_sentences(text)[0] == f"Find the revenue.{closers}"
    assert _REWRITER.rewrite(text).sub_queries[0] == f"Find the revenue.{closers}"


@pytest.mark.parametrize(
    "query",
    [
        "What is the revenue then?",
        "Compare these, then decide which wins",
        "Thence the river runs.",
    ],
)
def test_no_sentence_end_no_split(query: str) -> None:
    assert _REWRITER.rewrite(query).reasoning != "sequential_then_pattern"
