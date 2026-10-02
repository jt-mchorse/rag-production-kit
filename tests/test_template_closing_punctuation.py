"""A sentence ending in a terminator plus a closing quote/bracket keeps its marker (#258).

`TemplateGenerator` stripped only bare terminators before appending
`[cite:<id>].`, so `... "restart the server."` became
`... "restart the server." [cite:doc1].` -- and `split_sentences`, which treats
a terminator plus a closer as a boundary (#161), cut the claim from its marker.
Measured on `main`: both rows below were refused as `unparseable_output`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_kit.generator import (
    GeneratedAnswer,
    TemplateGenerator,
    _template_sentence,
    split_sentences,
)
from rag_kit.retriever import RetrievalResult

ROOT = Path(__file__).resolve().parents[1]


def _answer(text: str):
    r = RetrievalResult(
        external_id="doc1", text=text, metadata={}, fused_score=1.0, ranks={"dense": 1}
    )
    return TemplateGenerator().generate("q", [r], threshold=0.0)


@pytest.mark.parametrize(
    ("text", "rendered"),
    [
        (
            'The manual says "restart the server."',
            'The manual says "restart the server [cite:doc1]."',
        ),
        ("Restart the server (see section 4.)", "Restart the server (see section 4 [cite:doc1].)"),
        ("He said “stop！”", "He said “stop [cite:doc1]！”"),
        ("It's 'done.'", "It's 'done [cite:doc1].'"),
    ],
)
def test_a_closing_tail_keeps_the_sentence_with_its_marker(text: str, rendered: str) -> None:
    out = _answer(text)
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == f"Per the retrieved context, {rendered}"
    assert [c.external_id for c in out.citations] == ["doc1"]


@pytest.mark.parametrize(
    ("text", "rendered"),
    [
        ("Restart the server.", "Restart the server [cite:doc1]."),
        ("Really?", "Really [cite:doc1]."),
    ],
)
def test_a_plain_ending_renders_exactly_as_before(text: str, rendered: str) -> None:
    assert _answer(text).text == f"Per the retrieved context, {rendered}"


def test_every_committed_corpus_sentence_renders_exactly_as_before() -> None:
    # The old rule, spelled out, against every sentence the committed eval corpus
    # holds: no published eval answer changes.
    def old(s: str, eid: str) -> str:
        return f"Per the retrieved context, {s.strip().rstrip('.!?…。！？؟')} [cite:{eid}]."

    checked = 0
    for line in (
        (ROOT / "evals" / "dataset" / "corpus_v1.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        doc = json.loads(line)
        for s in split_sentences(doc["text"]):
            assert _template_sentence(s, "x") == old(s, "x"), s
            checked += 1
    assert checked >= 10  # the committed corpus: one sentence per document
