"""The rewriter's strip sites see a terminator behind a closer (#322).

`_split_question_and` and `_split_compare` promise one `?` per sub-query, and
both did it with `rstrip(_TERMINATORS) + "?"`, which only sees a terminator
in the last position. A closer after the terminator (`"Dune?"`, `fusion?)`,
`**RAG?**`) hid it, and a second terminator was stacked after the closer.
#320 gave the Then-split the generator's closers. These two sites were not
on its list.
"""

from __future__ import annotations

import random

import pytest

from rag_kit.rewriter import TemplateRewriter
from rag_kit.text import SENTENCE_CLOSERS, SENTENCE_TERMINATORS


def _subs(query: str) -> tuple[str, ...]:
    return TemplateRewriter().rewrite(query).sub_queries


def _trailing_run(s: str) -> str:
    end = len(s)
    while end and s[end - 1] in SENTENCE_TERMINATORS + SENTENCE_CLOSERS:
        end -= 1
    return s[end:]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            'Who wrote "Dune?" and when was it published?',
            ('Who wrote "Dune"?', "when was it published?"),
        ),
        (
            "What is RRF (reciprocal rank fusion?) and how does HNSW work?",
            ("What is RRF (reciprocal rank fusion)?", "how does HNSW work?"),
        ),
        ("What is **RAG?** and how does it work?", ("What is **RAG**?", "how does it work?")),
        ("What is pgvector.) and how is it used?", ("What is pgvector)?", "how is it used?")),
        ('Compare "Python." with "Java."', ('What is "Python"?', 'What is "Java"?')),
        ("compare 「東京。」 with 「大阪。」", ("What is 「東京」?", "What is 「大阪」?")),
        ("Compare **Python!** with **Java?**", ("What is **Python**?", "What is **Java**?")),
    ],
)
def test_a_terminator_behind_a_closer_is_not_doubled(query: str, expected: tuple[str, ...]) -> None:
    assert _subs(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # A closer with no terminator in front: unchanged, `?` after it.
        ('Compare "Python" with "Java"', ('What is "Python"?', 'What is "Java"?')),
        ("What is __init__ and how is it called?", ("What is __init__?", "how is it called?")),
        # The bare-terminator contract of #94/#111/#132/#150, unchanged.
        ("Compare Python with Java.", ("What is Python?", "What is Java?")),
        ("Who is X？ and where did Y work…", ("Who is X?", "where did Y work?")),
        # A decimal is internal punctuation, not part of the trailing run.
        ("Compare Python 3.5 with Java 21!", ("What is Python 3.5?", "What is Java 21?")),
    ],
)
def test_the_existing_shapes_are_unchanged(query: str, expected: tuple[str, ...]) -> None:
    assert _subs(query) == expected


def test_no_sub_query_keeps_a_terminator_in_its_trailing_run() -> None:
    # A seeded sweep over terminator and closer runs at both split sites. The
    # checks in tests/test_rewriter.py look at `sub[-2]`, which is a closer in
    # every one of these, so they could not see the doubled terminator.
    rng = random.Random(322)
    pool = SENTENCE_TERMINATORS + SENTENCE_CLOSERS
    bad: list[tuple[str, str]] = []
    for _ in range(2_000):
        punct = rng.choice(SENTENCE_TERMINATORS) + "".join(
            rng.choice(pool) for _ in range(rng.randint(0, 3))
        )
        punct += rng.choice(SENTENCE_CLOSERS)
        for query in (
            f"What is alpha{punct} and how does beta work?",
            f"Compare alpha{punct} with beta{punct}",
        ):
            subs = _subs(query)
            assert len(subs) == 2, (query, subs)
            for sub in subs:
                run = _trailing_run(sub)
                if not sub.endswith("?") or any(ch in SENTENCE_TERMINATORS for ch in run[:-1]):
                    bad.append((query, sub))
    assert not bad, bad[:5]
