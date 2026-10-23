"""The template answer for a markdown chunk passes its own reader (#399).

#385 made a heading, list item or blank line a sentence boundary, and exempted a
heading line and a numbered-list `1.` -- both read from the START of a line.
`TemplateGenerator` wrote `Per the retrieved context, ` in front of every
sentence and joined them with a space, so the `#` or the `1.` was no longer at a
line start, the re-split cut the prefixed heading off as its own sentence, and
the generator refused its own output for an ordinary markdown document. A fuzz
of 40,000 random markdown chunks refused 2,222 on c5a014b and none before #385.
"""

from __future__ import annotations

import random

import pytest

import rag_kit.generator as generator
from rag_kit.generator import GeneratedAnswer, Refusal, TemplateGenerator, split_sentences
from rag_kit.retriever import RetrievalResult


def _answer(*texts: str):
    retrieved = [
        RetrievalResult(
            external_id=f"doc{i}", text=t, metadata={}, fused_score=0.9, ranks={"dense": 1}
        )
        for i, t in enumerate(texts, start=1)
    ]
    return TemplateGenerator().generate("q", retrieved, threshold=0.0)


@pytest.mark.parametrize(
    "text",
    [
        "# Refund policy\n\nRefunds are issued within 30 days.",
        "Intro.\n\n## Details\n\nRefunds take 30 days.",
        "## Steps\n- Open the console\n- Click refund",
        "Do this.\n\n1. Open the console\n2. Click refund",
        "> # Policy\n>\n> Refunds take 30 days.",
    ],
)
def test_the_reported_chunks_are_answered(text: str) -> None:
    out = _answer(text)
    assert isinstance(out, GeneratedAnswer), out
    assert [c.external_id for c in out.citations] == ["doc1"]


def test_the_prefix_goes_after_the_heading_mark() -> None:
    out = _answer("# Refund policy\n\nRefunds are issued within 30 days.")
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == (
        "# Per the retrieved context, Refund policy\n\nRefunds are issued within 30 days [cite:doc1]."
    )


def test_a_numbered_item_starts_its_own_line() -> None:
    out = _answer("Do this.\n\n1. Open the console\n2. Click refund")
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == (
        "Per the retrieved context, Do this [cite:doc1].\n"
        "1. Per the retrieved context, Open the console [cite:doc1].\n"
        "2. Per the retrieved context, Click refund [cite:doc1]."
    )


def test_a_numbered_item_opening_the_next_chunk_starts_its_own_line() -> None:
    out = _answer("Refunds take 30 days.", "1. Open the console")
    assert isinstance(out, GeneratedAnswer), out
    assert "[cite:doc1].\n1. Per the retrieved context, Open the console [cite:doc2]." in out.text


@pytest.mark.parametrize("bare", ["#", "*", "2)"])
def test_a_bare_marker_line_does_not_become_a_break(bare: str) -> None:
    # ` [cite:` would make `#` a heading, `*` or `2)` a list item. One marker,
    # no space: the read-back fallback would answer too, with two.
    out = _answer(f"Refunds take 30 days\n{bare}")
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == f"Per the retrieved context, Refunds take 30 days\n{bare}[cite:doc1]."


def test_whitespace_the_terminator_strip_exposes_is_stripped_too() -> None:
    # `30.` is a line-start numbered marker in the chunk; stripping only the `?`
    # left `30.\n ` in front of the marker, a mid-line sentence end.
    out = _answer("30.\n ?")
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == "Per the retrieved context, 30 [cite:doc1]."


def test_a_heading_exemption_from_a_dropped_fragment_falls_back(monkeypatch) -> None:
    # `## !` is cut off and dropped (no alphanumeric), but its line is what made
    # the blank line below a non-boundary. The template line no longer starts
    # with `##`, so only the read-back fallback keeps every part cited.
    text = "## ! Refunds take 30 days\n\nThey are paid by card."
    assert split_sentences(text) == ["Refunds take 30 days\n\nThey are paid by card."]
    out = _answer(text)
    assert isinstance(out, GeneratedAnswer), out
    calls: list[str] = []
    real = generator._cite_every_cut
    monkeypatch.setattr(generator, "_cite_every_cut", lambda t, m: calls.append(t) or real(t, m))
    assert _answer(text) == out
    assert calls, "expected the read-back fallback to run"


_TOKENS = [
    "Refunds", "take", "30", "days", ". ", ".", "\n", "\n\n", "# ", "## ", "- ", "* ", "+ ",
    "1. ", "2) ", "10. ", "> ", ">", ":", "? ", "! ", '"', ")", "Dr. ", "e.g. ", "etc. ",
    "…", "。", "\r\n", "\r", "  ", "\t", "*", "**", "Then ",
]  # fmt: skip


def test_seeded_markdown_fuzz_never_refuses_a_grounded_chunk() -> None:
    rng = random.Random(399)
    refused: list[tuple[list[str], str]] = []
    for _ in range(3000):
        texts = [
            "".join(
                rng.choice(_TOKENS) + (" " if rng.random() < 0.5 else "")
                for _ in range(rng.randint(1, 10))
            )
            for _ in range(rng.randint(1, 3))
        ]
        if not any(ch.isalnum() for t in texts for ch in t):
            continue  # nothing grounded to cite: refused by design (control below)
        out = _answer(*texts)
        if isinstance(out, Refusal):
            refused.append((texts, out.detail))
    assert refused == [], refused[:5]


# Controls: the refusals that are by design, and plain prose byte for byte.


@pytest.mark.parametrize("text", ["", "   ", "...", "# \n\n- ", "> \n>\n"])
def test_a_chunk_with_nothing_to_cite_is_still_refused(text: str) -> None:
    out = _answer(text)
    assert isinstance(out, Refusal), out
    assert out.reason == "unparseable_output"


def test_plain_prose_is_byte_identical() -> None:
    out = _answer("Refunds take 30 days. Dr. Smith approves them!", 'He said "hi". Fine.')
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == (
        "Per the retrieved context, Refunds take 30 days [cite:doc1]. "
        "Per the retrieved context, Dr. Smith approves them [cite:doc1]. "
        'Per the retrieved context, He said "hi" [cite:doc2]. '
        "Per the retrieved context, Fine [cite:doc2]."
    )


def test_a_lead_in_list_already_answered_is_still_answered() -> None:
    out = _answer("Key Postgres settings:\n- shared_buffers\n- work_mem")
    assert isinstance(out, GeneratedAnswer), out
