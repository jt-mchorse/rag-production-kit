"""A markdown line or paragraph break is a sentence boundary (#385).

`split_sentences` split only after a terminator. So a bullet or paragraph with no
closing punctuation merged into the next sentence, and an uncited claim rode that
sentence's [cite:...] marker. Measured on main (a2a3187) through
`AnthropicGenerator` with `stop_reason="end_turn"` and one chunk `pg-tuning`, both
rows in `_BYPASSES` came back as a `GeneratedAnswer` with `len(split_sentences) == 1`.

A blank line, a list item, and a heading are now boundaries. A bare single newline,
a heading line, and a `:` lead-in are not, so a cited list under a lead-in and a
hard-wrapped cited sentence are judged exactly as before. A fully cited numbered
list used to be refused on its `'1.'` marker fragment. It is now accepted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_kit.generator import (
    AnthropicGenerator,
    CitationError,
    GeneratedAnswer,
    Refusal,
    enforce_citations,
    split_sentences,
)
from rag_kit.retriever import RetrievalResult

CHUNK = RetrievalResult(
    external_id="pg-tuning",
    text="shared_buffers sets the shared cache.",
    metadata={},
    fused_score=0.9,
    ranks={"dense": 1},
)
C = "[cite:pg-tuning]"


def _generate(text: str) -> GeneratedAnswer | Refusal:
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn"
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: reply))
    return AnthropicGenerator(client=client).generate("q", [CHUNK], threshold=0.0)


UNCITED = "work_mem is unlimited by default and should never be changed"

# Each break shape with an uncited claim in it. All were accepted on a2a3187.
_BYPASSES = {
    "issue-bullets": f"Key Postgres settings:\n- shared_buffers sets the shared cache {C}\n- {UNCITED}\n"
    f"- effective_cache_size is a planner hint {C}",
    "issue-paragraphs": f"Postgres caps work_mem at 64 MB on every platform\n\nshared_buffers sets the shared cache {C}.",
    "dash": f"- shared_buffers sets the shared cache {C}\n- {UNCITED}\n- effective_cache_size is a hint {C}",
    "star": f"* shared_buffers sets the shared cache {C}\n* {UNCITED}\n* effective_cache_size is a hint {C}",
    "plus": f"+ shared_buffers sets the shared cache {C}\n+ {UNCITED}\n+ effective_cache_size is a hint {C}",
    "numbered-dot": f"1. shared_buffers sets the shared cache {C}\n2. {UNCITED}\n3. effective_cache_size is a hint {C}",
    "numbered-paren": f"1) shared_buffers sets the shared cache {C}\n2) {UNCITED}\n3) effective_cache_size is a hint {C}",
    "indented-nested": f"- shared_buffers sets the shared cache {C}\n  - {UNCITED}\n  - effective_cache_size is a hint {C}",
    "uncited-last-bullet": f"- shared_buffers sets the shared cache {C}\n- {UNCITED}",
    "blank-line-with-spaces": f"{UNCITED}\n   \nshared_buffers sets the shared cache {C}.",
    "heading-then-uncited": f"shared_buffers sets the shared cache {C}.\n## Defaults\n{UNCITED}\n\neffective_cache_size is a hint {C}.",
    "before-heading": f"{UNCITED}\n# Summary\nshared_buffers sets the shared cache {C}.",
    # The abbreviation merge pass must not re-join across a break.
    "abbreviation-before-break": f"Use a tool, e.g.\n\n- shared_buffers sets the shared cache {C}",
}


# The refused sentence must be the uncited claim itself. On main a numbered list
# was refused on its `'1.'` fragment, so "it was refused" alone would pass there.
_REFUSED_SENTENCE_STARTS = {
    "issue-paragraphs": "Postgres caps work_mem",
    "abbreviation-before-break": "Use a tool, e.g.",
}


@pytest.mark.parametrize("name", list(_BYPASSES))
def test_an_uncited_bullet_or_paragraph_is_refused(name: str) -> None:
    out = _generate(_BYPASSES[name])
    assert isinstance(out, Refusal), split_sentences(_BYPASSES[name])
    assert out.reason == "unparseable_output"
    refused = out.detail.split("no [cite:...] marker: ", 1)[1]
    head = _REFUSED_SENTENCE_STARTS.get(name, UNCITED)
    assert head in refused, out.detail
    assert "[cite:" not in refused, out.detail


# Fully cited structures, accepted before and after. These are the controls: the
# lead-in and heading rows are the false refusal that #385's fix had to avoid.
_CITED = {
    "bullets": f"- shared_buffers sets the shared cache {C}\n- effective_cache_size is a hint {C}",
    "lead-in": f"Key Postgres settings:\n- shared_buffers sets the shared cache {C}\n- effective_cache_size is a hint {C}",
    "lead-in-blank-line": f"Key Postgres settings:\n\n- shared_buffers sets the shared cache {C}",
    "heading": f"## Settings\n\nshared_buffers sets the shared cache {C}.",
    "heading-then-list": f"# Settings\n- shared_buffers sets the shared cache {C}\n- effective_cache_size is a hint {C}",
    "paragraphs": f"shared_buffers sets the shared cache {C}.\n\neffective_cache_size is a hint {C}.",
    "hard-wrapped": f"shared_buffers sets the shared\ncache for every backend {C}.",
    "hard-wrapped-bullet": f"- shared_buffers sets the shared\n  cache for every backend {C}\n- effective_cache_size is a hint {C}",
    "bold-not-a-bullet": f"shared_buffers sets the\n**shared** cache {C}.",
}

# Accepted only after the fix: on main the `1.` of the first item was cut off as
# a claim-less but digit-bearing fragment and refused the whole answer.
_NUMBERED = {
    "numbered-dot": f"1. shared_buffers sets the shared cache {C}\n2. effective_cache_size is a hint {C}",
    "numbered-lead-in": f"Settings:\n1. shared_buffers sets the shared cache {C}\n10. effective_cache_size is a hint {C}",
}


@pytest.mark.parametrize("text", list(_CITED.values()), ids=list(_CITED))
def test_a_fully_cited_structure_is_accepted(text: str) -> None:
    out = _generate(text)
    assert isinstance(out, GeneratedAnswer), out
    assert [c.external_id for c in out.citations] == ["pg-tuning"]


@pytest.mark.parametrize("text", list(_NUMBERED.values()), ids=list(_NUMBERED))
def test_a_fully_cited_numbered_list_is_accepted(text: str) -> None:
    out = _generate(text)
    assert isinstance(out, GeneratedAnswer), out


def test_each_bullet_is_its_own_sentence() -> None:
    text = f"Key Postgres settings:\n- a {C}\n- b {C}\n\nc {C}."
    assert split_sentences(text) == [f"Key Postgres settings:\n- a {C}", f"- b {C}", f"\nc {C}."]


def test_a_single_newline_inside_a_sentence_is_not_a_boundary() -> None:
    assert split_sentences("one claim that wraps\nonto a second line [cite:a].") == [
        "one claim that wraps\nonto a second line [cite:a]."
    ]


def test_a_break_inside_a_marker_does_not_split_it() -> None:
    # The #256 rule (never cut inside a marker) holds for the new boundaries.
    rr = RetrievalResult(
        external_id="a\n\n- b", text="x", metadata={}, fused_score=0.9, ranks={"dense": 1}
    )
    assert len(enforce_citations("claim [cite:a\n\n- b].", [rr])) == 1


def test_an_uncited_paragraph_names_itself_in_the_refusal() -> None:
    with pytest.raises(CitationError, match="Postgres caps work_mem"):
        enforce_citations(_BYPASSES["issue-paragraphs"], [CHUNK])
