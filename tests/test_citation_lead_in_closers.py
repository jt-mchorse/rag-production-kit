"""A `:` lead-in followed by markdown closers, or a fullwidth `：`, is a lead-in (#400).

#385 kept a line ending in `:` attached to the cited list under it, but only when
the `:` was the line's LAST character. `**Key settings:**`, `*Key settings:*`,
`__Key settings:__` and `主要设置：` were each cut off as their own uncited
sentence, so a fully cited list under them was refused. Measured on main
(c5a014b) through `AnthropicGenerator` with `stop_reason="end_turn"` and one chunk
`pg-tuning`: every row in `_LEADS` came back as a `Refusal` ("sentence has no
[cite:...] marker"), with a blank line and with a single newline; `Key settings:`
was accepted.

The controls: a plain `Key settings:` is still accepted, a colon mid-line is still
not a lead-in, and an uncited bullet under a closer lead-in is still refused.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_kit.generator import AnthropicGenerator, GeneratedAnswer, Refusal, split_sentences
from rag_kit.retriever import RetrievalResult

CHUNK = RetrievalResult(
    external_id="pg-tuning",
    text="shared_buffers sets the shared cache.",
    metadata={},
    fused_score=0.9,
    ranks={"dense": 1},
)
C = "[cite:pg-tuning]"
LIST = f"- shared_buffers is 25% of RAM {C}\n- work_mem is per sort {C}"
SEPARATORS = {"blank-line": "\n\n", "single-newline": "\n"}


def _generate(text: str) -> GeneratedAnswer | Refusal:
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn"
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: reply))
    return AnthropicGenerator(client=client).generate("q", [CHUNK], threshold=0.0)


# Each was refused on c5a014b over a fully cited list.
_LEADS = {
    "bold": "**Key settings:**",
    "italic-star": "*Key settings:*",
    "bold-underscore": "__Key settings:__",
    "fullwidth": "主要设置：",
    "bold-fullwidth": "**主要设置：**",
    "bold-trailing-space": "**Key settings:**  ",
    "quote-bold": "> **Key settings:**",
}


@pytest.mark.parametrize("sep", SEPARATORS.values(), ids=SEPARATORS.keys())
@pytest.mark.parametrize("lead", _LEADS.values(), ids=_LEADS.keys())
def test_closer_lead_in_over_cited_list_is_accepted(lead: str, sep: str) -> None:
    body = LIST.replace("- ", "> - ") if lead.startswith(">") else LIST
    if lead.startswith(">") and sep == "\n\n":
        sep = "\n>\n"  # a blank line inside a blockquote (#391)
    out = _generate(f"{lead}{sep}{body}")
    assert isinstance(out, GeneratedAnswer), getattr(out, "detail", out)


@pytest.mark.parametrize("lead", _LEADS.values(), ids=_LEADS.keys())
def test_closer_lead_in_rides_the_first_item(lead: str) -> None:
    # The lead-in is not its own sentence: it stays on the item after it, as a
    # plain `Settings:` does (#385).
    parts = split_sentences(
        f"{lead}\n\n{LIST.replace('- ', '> - ') if lead.startswith('>') else LIST}"
    )
    assert len(parts) == 2
    assert parts[0].startswith(lead.strip())
    assert C in parts[0]


# --- controls: green before and after #400 ---


@pytest.mark.parametrize("sep", SEPARATORS.values(), ids=SEPARATORS.keys())
def test_plain_colon_lead_in_still_accepted(sep: str) -> None:
    assert isinstance(_generate(f"Key settings:{sep}{LIST}"), GeneratedAnswer)


@pytest.mark.parametrize("sep", SEPARATORS.values(), ids=SEPARATORS.keys())
@pytest.mark.parametrize("line", ["Key: settings are tuned", "**Key:** settings are tuned"])
def test_mid_line_colon_is_not_a_lead_in(line: str, sep: str) -> None:
    out = _generate(f"{line}{sep}{LIST}")
    assert isinstance(out, Refusal)
    assert out.reason == "unparseable_output"


@pytest.mark.parametrize("lead", _LEADS.values(), ids=_LEADS.keys())
def test_uncited_item_under_closer_lead_in_is_still_refused(lead: str) -> None:
    quote = "> " if lead.startswith(">") else ""
    text = f"{lead}\n{quote}- shared_buffers is 25% of RAM {C}\n{quote}- work_mem is unlimited by default"
    assert isinstance(_generate(text), Refusal)
