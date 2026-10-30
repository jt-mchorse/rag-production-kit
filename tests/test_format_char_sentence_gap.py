"""A format character beside a sentence end reads as nothing (#431).

`text.is_blank` (#424) counts a format character (category Cf) as blank, and
#430 taught `rewriter._norm` to read one beside a space as part of the space.
`generator._SENTENCE_SPLIT` still needed the gap after a terminator and its
closers to START with whitespace, so a Cf between the `.` and the space hid the
boundary. Measured on f9f433e:

    enforce_citations("Paris is in Germany.\\u200b The sky is blue [cite:a].")
        -> accepted, citing `a` (twin without U+200B: CitationError)
    TemplateRewriter().rewrite('What is RAG.\\u200b"  Then what is BM25')
        -> no_decomposition (twin: sequential_then_pattern)

Every case below is a template with an `{F}` slot. Its twin is the same template
with the slot empty, and each case must give the twin's result once its format
characters are removed: the same sentences, the same answer, the same rewrite.
"""

from __future__ import annotations

import sys
import unicodedata

import pytest

from rag_kit import TemplateRewriter
from rag_kit.generator import (
    CitationError,
    GeneratedAnswer,
    TemplateGenerator,
    enforce_citations,
    split_sentences,
)
from rag_kit.retriever import RetrievalResult

# U+E0001 (LANGUAGE TAG) is outside the BMP and the usual hand-written lists, so
# it shows the class comes from the catalog.
_FORMAT: dict[str, str] = {
    "zwsp": "​",
    "bom": "﻿",
    "word-joiner": "⁠",
    "soft-hyphen": "­",
    "language-tag": "\U000e0001",
    "mixed": "​﻿⁠",
}


def _drop_format(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _chunk(external_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        external_id=external_id, text=text, metadata={}, fused_score=1.0, ranks={}
    )


_RETRIEVED = [_chunk("a", "The sky is blue.")]

# Template -> whether the twin (`{F}` empty) is accepted by `enforce_citations`.
_ANSWERS: dict[str, tuple[str, bool]] = {
    "before-space": ("Paris is in Germany.{F} The sky is blue [cite:a].", False),
    "before-two-spaces": ("Paris is in Germany.{F}  The sky is blue [cite:a].", False),
    "after-space": ("Paris is in Germany. {F}The sky is blue [cite:a].", False),
    "both-sides": ("Paris is in Germany.{F} {F}The sky is blue [cite:a].", False),
    "before-newline": ("Paris is in Germany.{F}\nThe sky is blue [cite:a].", False),
    "bang": ("Paris is in Germany!{F} The sky is blue [cite:a].", False),
    "ellipsis": ("Paris is in Germany…{F} The sky is blue [cite:a].", False),
    "before-closer": ('Paris is "in Germany.{F}" The sky is blue [cite:a].', False),
    "after-closer": ('Paris is "in Germany."{F} The sky is blue [cite:a].', False),
    "bold": ("Paris is **in Germany.{F}** The sky is blue [cite:a].", False),
    "after-marker": ("The sky is blue.[cite:a]{F} Paris is in Germany.", False),
    "cjk": ("巴黎在德国。{F}天空是蓝色的 [cite:a]。", False),
    "devanagari": ("पेरिस जर्मनी में है।{F} The sky is blue [cite:a].", False),
    # The twin is one sentence, and so is the case. Only the last row was not
    # already so: `。\u200b」` split where `。」` does not, a false refusal.
    "control-no-gap": ("Paris is in Germany.{F}The sky is blue [cite:a].", True),
    "control-abbreviation": ("Dr.{F} Smith says the sky is blue [cite:a].", True),
    "control-numbered-item": ("1.{F} The sky is blue [cite:a].", True),
    "control-decimal": ("The sky is 3.{F}5 times bluer [cite:a].", True),
    "cjk-closer-is-not-a-gap": ("他说「巴黎在德国。{F}」天空是蓝色的 [cite:a]。", True),
}


def _verdict(text: str) -> bool:
    try:
        enforce_citations(text, _RETRIEVED)
    except CitationError:
        return False
    return True


@pytest.mark.parametrize("fmt", sorted(_FORMAT))
@pytest.mark.parametrize("shape", sorted(_ANSWERS))
def test_a_format_char_at_a_sentence_end_splits_as_its_twin(shape: str, fmt: str) -> None:
    template, accepted = _ANSWERS[shape]
    text = template.format(F=_FORMAT[fmt])
    twin = template.format(F="")
    assert [_drop_format(s) for s in split_sentences(text)] == split_sentences(twin)
    assert _verdict(text) is accepted


@pytest.mark.parametrize("shape", sorted(_ANSWERS))
def test_control_the_twin_verdict_is_what_the_table_pins(shape: str) -> None:
    template, accepted = _ANSWERS[shape]
    assert _verdict(template.format(F="")) is accepted


# Chunk texts for the `TemplateGenerator` round trip: the template writes one
# cited sentence per sentence the splitter reads, then validates the answer with
# the same splitter, so a writer that kept a format character where the reader
# now cuts would refuse a grounded chunk.
_CHUNKS: dict[str, str] = {
    "two-sentences": "The sky is blue.{F} Grass is green.",
    "after-space": "The sky is blue. {F}Grass is green.",
    "trailing": "The sky is blue.{F}",
    "closer": 'He said "go.{F}" Then he left.',
    "after-closer": 'He said "go."{F} Then he left.',
    "abbreviation": "Dr.{F} Smith wrote it.",
    "cjk": "天空是蓝的。{F}草是绿的。",
    "cjk-closer": "他说「天空是蓝的。{F}」草是绿的。",
}


@pytest.mark.parametrize("fmt", sorted(_FORMAT))
@pytest.mark.parametrize("shape", sorted(_CHUNKS))
def test_template_generator_round_trip_matches_its_twin(shape: str, fmt: str) -> None:
    chunk = _CHUNKS[shape]
    out = TemplateGenerator().generate("sky", [_chunk("a", chunk.format(F=_FORMAT[fmt]))])
    twin = TemplateGenerator().generate("sky", [_chunk("a", chunk.format(F=""))])
    assert isinstance(out, GeneratedAnswer), out
    assert isinstance(twin, GeneratedAnswer), twin
    assert _drop_format(out.text) == twin.text


# Template -> (reasoning, sub_queries) of the twin.
_QUERIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "then/before-closer": (
        'What is RAG.{F}"  Then what is BM25',
        "sequential_then_pattern",
        ('What is RAG."', "what is BM25"),
    ),
    "then/after-closer": (
        'What is RAG."{F} Then what is BM25',
        "sequential_then_pattern",
        ('What is RAG."', "what is BM25"),
    ),
    "then/question-closer": (
        "What is **RAG?{F}** Then what is BM25",
        "sequential_then_pattern",
        ("What is **RAG?**", "what is BM25"),
    ),
    "compare/cjk-unspaced": (
        "Compare Postgres with MySQL？{F}Which one is cheaper to host?",
        "no_decomposition",
        ("Compare Postgres with MySQL？Which one is cheaper to host?",),
    ),
    "compare/closer-spaced": (
        "Compare Postgres with MySQL？{F}」 Which one is cheaper to host?",
        "no_decomposition",
        ("Compare Postgres with MySQL？」 Which one is cheaper to host?",),
    ),
    "compare/cjk-closer": (
        "Compare 「Postgres。{F}」 with MySQL",
        "compare_pattern",
        ("What is 「Postgres」?", "What is MySQL?"),
    ),
    "and/question-title": (
        'When was "Who Framed Roger Rabbit?{F}" released and who directed it?',
        "multi_question_and_pattern",
        ('When was "Who Framed Roger Rabbit?" released?', "who directed it?"),
    ),
    "and/bare-question": (
        "What is RAG?{F} what is BM25 and how does dense retrieval work?",
        "no_decomposition",
        ("What is RAG? what is BM25 and how does dense retrieval work?",),
    ),
    "and/bang-closer": (
        'What is "RAG!{F}" What is BM25 and how does dense retrieval work?',
        "no_decomposition",
        ('What is "RAG!" What is BM25 and how does dense retrieval work?',),
    ),
    "and/trailing-closer": (
        'What is "RAG?{F}" and how does BM25 work',
        "multi_question_and_pattern",
        ('What is "RAG"?', "how does BM25 work?"),
    ),
}


@pytest.mark.parametrize("fmt", sorted(_FORMAT))
@pytest.mark.parametrize("shape", sorted(_QUERIES))
def test_a_format_char_at_a_sentence_end_rewrites_as_its_twin(shape: str, fmt: str) -> None:
    template, reasoning, sub_queries = _QUERIES[shape]
    out = TemplateRewriter().rewrite(template.format(F=_FORMAT[fmt]))
    assert out.reasoning == reasoning
    assert tuple(_drop_format(sq) for sq in out.sub_queries) == sub_queries


@pytest.mark.parametrize("shape", sorted(_QUERIES))
def test_control_the_rewrite_twin_is_what_the_table_pins(shape: str) -> None:
    template, reasoning, sub_queries = _QUERIES[shape]
    out = TemplateRewriter().rewrite(template.format(F=""))
    assert (out.reasoning, out.sub_queries) == (reasoning, sub_queries)


def test_format_characters_is_the_catalog_cf_set() -> None:
    from rag_kit.text import FORMAT_CHARACTERS

    expected = {
        chr(cp) for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)) == "Cf"
    }
    assert set(FORMAT_CHARACTERS) == expected
    assert {"​", "﻿", "⁠", "­", "\U000e0001"} <= expected
