"""A citation id with a terminator and a space is not split inside its marker (#256).

`Document` accepts `faq.md#Q3. refunds` and `_CITE_PATTERN` reads
`[cite:faq.md#Q3. refunds]` back to it (#197's readback), but `split_sentences`
split on terminal-punctuation-then-whitespace INSIDE the marker. Measured on
`main`: `TemplateGenerator().generate(...)` over such a chunk was refused with
"Refusal sentence has no [cite:...] marker: '... [cite:faq.md#Q3.'". The readback
runs the regex and not the splitter, which is why it could not see this.
"""

from __future__ import annotations

import pytest

from rag_kit import Document
from rag_kit.generator import (
    GeneratedAnswer,
    TemplateGenerator,
    enforce_citations,
    split_sentences,
)
from rag_kit.retriever import RetrievalResult

IDS = ["faq.md#Q3. refunds", "What is RAG? part 1", "notes.txt#v2! draft", "doc 1. intro"]


def _result(external_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        external_id=external_id,
        text=text,
        metadata={"source": "test"},
        fused_score=0.5,
        ranks={"lexical": 1, "dense": 1},
        rerank_score=None,
        rerank_rank=None,
    )


@pytest.mark.parametrize("eid", IDS)
def test_the_id_is_a_legal_document_id(eid: str) -> None:
    # The write seam accepts it, so the read side must handle it.
    Document(external_id=eid, text="Refunds take 14 days.", metadata={})


@pytest.mark.parametrize("eid", [*IDS, "docs/guide.md#3"])
def test_the_template_path_answers_with_one_citation(eid: str) -> None:
    result = TemplateGenerator().generate(
        "refunds?", [_result(eid, "Refunds take 14 days. Pro plans get 30.")], threshold=0.1
    )
    assert isinstance(result, GeneratedAnswer), result
    assert [c.external_id for c in result.citations] == [eid]


@pytest.mark.parametrize("eid", IDS)
def test_an_anthropic_shaped_answer_passes_enforcement(eid: str) -> None:
    answer = f"Refunds take 14 days [cite:{eid}]. Pro plans get 30 [cite:{eid}]."
    citations = enforce_citations(answer, [_result(eid, "Refunds take 14 days.")])
    assert [c.external_id for c in citations] == [eid]


def test_a_marker_is_never_split_and_splitting_outside_markers_is_unchanged() -> None:
    text = "Refunds take 14 days [cite:faq.md#Q3. refunds]. Dr. Smith agreed [cite:a]. Done!"
    assert split_sentences(text) == [
        "Refunds take 14 days [cite:faq.md#Q3. refunds].",
        "Dr. Smith agreed [cite:a].",
        "Done!",
    ]


def test_a_private_use_character_in_the_text_is_not_mistaken_for_a_placeholder() -> None:
    text = "A claim with 0 in it [cite:x]. Another [cite:y]."
    assert split_sentences(text) == ["A claim with 0 in it [cite:x].", "Another [cite:y]."]
