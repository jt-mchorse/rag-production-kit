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

import re
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


# --- A blank at a line start reads as its twin (#435) -----------------------
#
# The splitter's gap took a format character or NBSP after `1.` (#433), so the
# `_ORDERED_MARKER` skip dropped that cut as a numbered item while
# `_STRUCTURAL_BREAK`, wanting `[ \t]` after the marker, read no list item:
# neither cut, and on a2c0736 `"Paris is in Germany\n1.​ The sky is blue
# [cite:a]."` was accepted where its twin is refused. The line-start reads now
# share one rule: a format character reads as nothing, and any whitespace other
# than a newline reads as a space. The twin applies exactly that, so every case
# below must get its twin's verdict, and a chunk its twin's template answer.
#
# Searched, not hand-picked: each line-start construct (ordered `1.` `1)` `12.`,
# bullets `-` `*` `+`, headings `#` `##`, a quote `>`, a blank line, a `>`-only
# line, a `:` lead-in, a heading lead) with the blank slot in every position it
# can take (after the marker, after the marker's space, before it, behind a `>`,
# inside a two-character marker), times every blank class in `_LINE_BLANKS`,
# times a space or a newline in front of the construct. No exclusion remained.

_LINE_BLANKS: dict[str, str] = {
    "space": " ",
    "tab": "\t",
    "nbsp": " ",
    "ideographic-space": "　",
    "zwsp": "​",
    "bom": "﻿",
    "word-joiner": "⁠",
    "soft-hyphen": "­",
    "language-tag": "\U000e0001",
    "zwsp-nbsp": "​ ",
}
_CONTEXTS = {"space": " ", "newline": "\n"}


def _line_twin(text: str) -> str:
    """`text` read the way the line-start rule reads it: Cf dropped, blanks as spaces."""
    return re.sub(r"[^\S\n]", " ", _drop_format(text))


def _line_marker_slots() -> dict[str, str]:
    slots: dict[str, str] = {}
    for marker in ("1.", "1)", "12.", "-", "*", "+", "#", "##", ">"):
        slots[f"{marker}/gap"] = marker + "{B}"
        slots[f"{marker}/gap-then-space"] = marker + "{B} "
        slots[f"{marker}/space-then-blank"] = marker + " {B}"
        slots[f"{marker}/indent"] = "{B}" + marker + " "
        slots[f"{marker}/quoted"] = ">{B}" + marker + " "
        if marker in ("12.", "##"):
            slots[f"{marker}/inside"] = marker[0] + "{B}" + marker[1:] + " "
    return slots


_LINE_ANSWERS: dict[str, str] = {}
for _name, _slot in _line_marker_slots().items():
    _LINE_ANSWERS[_name] = "Paris is in Germany{C}" + _slot + "The sky is blue [cite:a]."
    _LINE_ANSWERS[_name + "/terminated"] = (
        "Paris is in Germany.{C}" + _slot + "The sky is blue [cite:a]."
    )
_LINE_ANSWERS.update(
    {
        "blank-line": "Paris is in Germany{C}{B}\nThe sky is blue [cite:a].",
        "blank-line-two-blanks": "Paris is in Germany{C}{B}{B}\nThe sky is blue [cite:a].",
        "quote-blank-line": "> Paris is in Germany{C}>{B}\n> The sky is blue [cite:a].",
        "list-items": "- Paris is in Germany{C}-{B}The sky is blue [cite:a].",
        "lead-in": "Key settings:{B}{C}- The sky is blue [cite:a].",
        "lead-in-bold": "**Key settings:{B}**{C}- The sky is blue [cite:a].",
        "heading-lead": "##{B}Settings{C}- The sky is blue [cite:a].",
        "heading-lead-spaced": "##{B} Settings{C}- The sky is blue [cite:a].",
        "heading-lead-indented": "{B}## Settings{C}- The sky is blue [cite:a].",
    }
)


def _line_cases(template: str) -> list[str]:
    return [
        template.format(C=ctx, B=blank)
        for ctx in _CONTEXTS.values()
        for blank in _LINE_BLANKS.values()
    ]


@pytest.mark.parametrize("shape", sorted(_LINE_ANSWERS))
def test_a_blank_at_a_line_start_gets_its_twin_verdict(shape: str) -> None:
    mismatches = [
        (text, _verdict(text))
        for text in _line_cases(_LINE_ANSWERS[shape])
        if _verdict(text) is not _verdict(_line_twin(text))
    ]
    assert mismatches == []


# The twins the table compares against, pinned so a reader that lost the break
# (or the lead-in) everywhere cannot pass by agreeing with itself.
_LINE_TWIN_VERDICTS: dict[str, bool] = {
    "Paris is in Germany\n1. The sky is blue [cite:a].": False,
    "Paris is in Germany\n1) The sky is blue [cite:a].": False,
    "Paris is in Germany\n- The sky is blue [cite:a].": False,
    "Paris is in Germany\n## The sky is blue [cite:a].": False,
    "Paris is in Germany\n> 1. The sky is blue [cite:a].": False,
    "Paris is in Germany\n \nThe sky is blue [cite:a].": False,
    "> Paris is in Germany\n> \n> The sky is blue [cite:a].": False,
    "Paris is in Germany\n1.The sky is blue [cite:a].": True,
    "Paris is in Germany 1. The sky is blue [cite:a].": False,
    "Paris is in Germany 1) The sky is blue [cite:a].": True,
    "Key settings: \n- The sky is blue [cite:a].": True,
    "## Settings\n- The sky is blue [cite:a].": True,
}


@pytest.mark.parametrize("text", sorted(_LINE_TWIN_VERDICTS))
def test_control_the_line_twin_verdict_is_what_the_table_pins(text: str) -> None:
    assert _verdict(text) is _LINE_TWIN_VERDICTS[text]


# The issue's own rows (#435), each against its twin's pinned verdict.
@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("Paris is in Germany\n1.​ The sky is blue [cite:a].", False),
        ("Paris is in Germany\n1. The sky is blue [cite:a].", False),
        ("Paris is in Germany\n1)　The sky is blue [cite:a].", False),
        ("Paris is in Germany\n-﻿ The sky is blue [cite:a].", False),
        ("Key settings:​\n- The sky is blue [cite:a].", True),
        ("##​ Settings\n- The sky is blue [cite:a].", True),
    ],
)
def test_the_issue_rows_get_their_twin_verdict(text: str, accepted: bool) -> None:
    assert _verdict(text) is accepted


# The writer reads the same line starts (#399): `TemplateGenerator` keeps a
# sentence's block prefix in front of the template text and starts it on its
# own line. A prefix it read differently from the validator was a refused
# chunk, or an answer that differed from its twin's.
_LINE_CHUNKS: dict[str, str] = {
    name: "Intro text here.{C}" + slot + "The sky is blue."
    for name, slot in _line_marker_slots().items()
}
_LINE_CHUNKS.update(
    {
        "blank-line": "The sky is blue.{C}{B}\nGrass is green.",
        "heading-lead": "##{B} Settings{C}- The sky is blue.",
        "lead-in": "Key settings:{B}{C}- The sky is blue.",
    }
)


def _answer(chunk: str) -> str:
    out = TemplateGenerator().generate("sky", [_chunk("a", chunk)])
    return _line_twin(out.text) if isinstance(out, GeneratedAnswer) else repr(out)


@pytest.mark.parametrize("shape", sorted(_LINE_CHUNKS))
def test_template_generator_line_start_round_trip_matches_its_twin(shape: str) -> None:
    mismatches = [
        chunk
        for chunk in _line_cases(_LINE_CHUNKS[shape])
        if _answer(chunk) != _answer(_line_twin(chunk))
    ]
    assert mismatches == []
    # Not vacuous: the twin itself is a grounded, accepted answer.
    twin = TemplateGenerator().generate(
        "sky", [_chunk("a", _line_twin(_line_cases(_LINE_CHUNKS[shape])[0]))]
    )
    assert isinstance(twin, GeneratedAnswer), twin
