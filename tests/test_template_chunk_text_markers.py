"""A `[cite:...]` already inside a chunk's text is not read as a citation (#283).

`TemplateGenerator` copies each chunk's sentences into its answer word for word
and then validates the answer with `enforce_citations` (and
`evals/run_eval.py` re-runs that check over `GeneratedAnswer.text`). Measured on
`main` before the fix: a chunk describing the marker syntax was refused as a
dangling citation, a chunk mentioning another chunk's marker credited that
other chunk with its sentence, and an unclosed `[cite:` swallowed the
template's own marker into one dangling id.
"""

from __future__ import annotations

import pytest

from rag_kit.generator import (
    GeneratedAnswer,
    Refusal,
    TemplateGenerator,
    enforce_citations,
)
from rag_kit.retriever import RetrievalResult


def _r(external_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        external_id=external_id, text=text, metadata={}, fused_score=1.0, ranks={"dense": 1}
    )


_ALONE = [
    ("Answers carry [cite:<external_id>] markers.", "Answers carry (cite:<external_id>) markers"),
    ("See [cite:doc2] for details.", "See (cite:doc2) for details"),
    ("An open [cite: marker here.", "An open (cite: marker here"),
    ("Two [cite:a] and [cite:b] markers.", "Two (cite:a) and (cite:b) markers"),
]


@pytest.mark.parametrize(("text", "copied"), _ALONE)
def test_a_marker_in_chunk_text_does_not_refuse_the_answer(text: str, copied: str) -> None:
    retrieved = [_r("doc1", text)]
    out = TemplateGenerator().generate("q", retrieved, threshold=0.0)
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == f"Per the retrieved context, {copied} [cite:doc1]."
    assert [c.external_id for c in out.citations] == ["doc1"]
    # The re-check `evals/run_eval.py` applies to the answer text agrees.
    assert [c.external_id for c in enforce_citations(out.text, retrieved)] == ["doc1"]


def test_a_marker_naming_another_retrieved_chunk_does_not_misattribute() -> None:
    retrieved = [_r("doc1", "See [cite:doc2] for the old limit."), _r("doc2", "Old limit was 5.")]
    out = TemplateGenerator().generate("q", retrieved, threshold=0.0)
    assert isinstance(out, GeneratedAnswer), out
    # On `main` this was ['doc2', 'doc1']: doc2 was credited with doc1's sentence.
    assert [c.external_id for c in out.citations] == ["doc1", "doc2"]
    assert out.text == (
        "Per the retrieved context, See (cite:doc2) for the old limit [cite:doc1]. "
        "Per the retrieved context, Old limit was 5 [cite:doc2]."
    )


def test_a_marker_whose_id_holds_a_sentence_boundary_is_split_consistently() -> None:
    # Neutralised BEFORE splitting: the splitter skips boundaries inside a real
    # marker, so neutralising afterwards would leave this sentence whole for the
    # template and cut in two for the validator.
    retrieved = [_r("doc1", "See [cite:faq.md#Q3. refunds] for details.")]
    out = TemplateGenerator().generate("q", retrieved, threshold=0.0)
    assert isinstance(out, GeneratedAnswer), out
    assert [c.external_id for c in out.citations] == ["doc1"]
    assert enforce_citations(out.text, retrieved)


def test_the_citation_error_branch_is_reachable_and_refuses() -> None:
    # The branch the old `pragma: no cover` hid: a punctuation-only chunk
    # yields no cited sentence.
    out = TemplateGenerator().generate("q", [_r("doc1", "...")], threshold=0.0)
    assert isinstance(out, Refusal)
    assert out.reason == "unparseable_output"
