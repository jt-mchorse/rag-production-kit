"""A CRLF blank line and a blockquote paragraph break are boundaries too (#391).

#385 made a blank line, a list item and a heading sentence boundaries, but only
as `\\n` with spaces or tabs in between, and only at the start of a line. So the
same blank line written `\\r\\n\\r\\n` (or `\\r\\r`), and a `>`-only line inside a
blockquote, still let an uncited paragraph ride the next one's [cite:...]
marker. Measured on main (1ff10c8) through `AnthropicGenerator` with
`stop_reason="end_turn"` and one chunk `pg-tuning`: every row in `_BYPASSES`
came back as a `GeneratedAnswer`, with `len(split_sentences) == 1`. A CRLF
list item or heading already split there, and is kept as a control.

The controls are fully cited CRLF and blockquote answers, including a lead-in
and a numbered list inside a quote (the two #385 exemptions), and hard-wrapped
sentences, which must stay one sentence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_kit.generator import (
    AnthropicGenerator,
    GeneratedAnswer,
    Refusal,
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
UNCITED = "work_mem is unlimited by default and should never be changed"
CITED = f"shared_buffers sets the shared cache {C}."


def _generate(text: str) -> GeneratedAnswer | Refusal:
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn"
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: reply))
    return AnthropicGenerator(client=client).generate("q", [CHUNK], threshold=0.0)


# An uncited first paragraph in each shape. All were accepted on 1ff10c8.
_BYPASSES = {
    "crlf-blank-line": f"{UNCITED}\r\n\r\n{CITED}",
    "cr-blank-line": f"{UNCITED}\r\r{CITED}",
    "crlf-blank-line-with-spaces": f"{UNCITED}\r\n  \r\n{CITED}",
    "quote-blank-line": f"> {UNCITED}\n>\n> {CITED}",
    "quote-blank-line-with-space": f"> {UNCITED}\n> \n> {CITED}",
    "nested-quote-blank-line": f"> > {UNCITED}\n> >\n> > {CITED}",
    "quote-list-item": f"> {UNCITED}\n> - {CITED}",
    "quote-bullets": f"> - {CITED}\n> - {UNCITED}\n> - effective_cache_size is a hint {C}",
    "quote-heading": f"> {UNCITED}\n> ## Summary\n> {CITED}",
    "crlf-quote-blank-line": f"> {UNCITED}\r\n>\r\n> {CITED}",
}


# Controls: a CRLF list item or heading was already a boundary on 1ff10c8, since
# its `\n` is still followed by the marker. They must stay refused.
_ALREADY_REFUSED = {
    "crlf-bullets": f"- {CITED}\r\n- {UNCITED}\r\n- effective_cache_size is a hint {C}",
    "crlf-heading": f"{UNCITED}\r\n# Summary\r\n{CITED}",
}


@pytest.mark.parametrize("name", list(_BYPASSES | _ALREADY_REFUSED))
def test_an_uncited_paragraph_is_refused(name: str) -> None:
    text = (_BYPASSES | _ALREADY_REFUSED)[name]
    out = _generate(text)
    assert isinstance(out, Refusal), split_sentences(text)
    assert out.reason == "unparseable_output"
    refused = out.detail.split("no [cite:...] marker: ", 1)[1]
    assert UNCITED in refused, out.detail
    assert "[cite:" not in refused, out.detail


# Fully cited answers in the same shapes: accepted before the fix and after it.
_CITED = {
    "crlf-paragraphs": f"{CITED}\r\n\r\neffective_cache_size is a hint {C}.",
    "crlf-lead-in": f"Key settings:\r\n- {CITED}\r\n- effective_cache_size is a hint {C}",
    "crlf-numbered": f"1. {CITED}\r\n2. effective_cache_size is a hint {C}",
    "crlf-heading": f"## Settings\r\n\r\n{CITED}",
    "quote-paragraphs": f"> {CITED}\n>\n> effective_cache_size is a hint {C}.",
    "quote-lead-in": f"> Key settings:\n>\n> - {CITED}\n> - effective_cache_size is a hint {C}",
    "quote-heading": f"> ## Settings\n>\n> {CITED}",
}

# Accepted only after the fix. The `1.` of `> 1. item` was not seen as a list
# marker behind the `>`, so it was cut off as a digit-bearing fragment and
# refused a fully cited numbered list inside a quote.
_CITED_AFTER_FIX = {
    "quote-numbered": f"> 1. {CITED}\n> 2. effective_cache_size is a hint {C}",
}


@pytest.mark.parametrize(
    "text", list((_CITED | _CITED_AFTER_FIX).values()), ids=list(_CITED | _CITED_AFTER_FIX)
)
def test_a_fully_cited_answer_is_accepted(text: str) -> None:
    out = _generate(text)
    assert isinstance(out, GeneratedAnswer), out
    assert [c.external_id for c in out.citations] == ["pg-tuning"]


@pytest.mark.parametrize(
    "text",
    [
        f"> shared_buffers sets the shared\n> cache for every backend {C}.",
        f"shared_buffers sets the shared\r\ncache for every backend {C}.",
        f"> shared_buffers sets the shared\r\n> cache for every backend {C}.",
    ],
    ids=["quote", "crlf", "crlf-quote"],
)
def test_a_hard_wrapped_sentence_stays_one_sentence(text: str) -> None:
    assert split_sentences(text) == [text]


def test_parts_are_cut_from_the_original_text() -> None:
    # Boundaries are found on a copy with `\r` read as `\n`; the parts are not.
    assert split_sentences(f"{UNCITED}\r\n\r\n{CITED}") == [f"{UNCITED}\r", f"\r\n{CITED}"]
