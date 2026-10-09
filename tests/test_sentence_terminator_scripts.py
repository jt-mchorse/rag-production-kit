"""Every script's full stop ends a sentence for citation enforcement (#301).

`_SENTENCE_SPLIT` knew only `.!?…。！？؟`, so an answer in Hindi, Urdu,
Amharic, Armenian, Burmese or Khmer was ONE sentence, and an uncited claim
passed `enforce_citations` on its neighbour's `[cite:...]` marker. Measured on
`main`: an uncited sentence plus a cited one, ending in `।`, `॥`, `۔`, `።`, `։`,
`။`, `។`, halfwidth `｡` or `‼`, all came back ACCEPTED with n=1 sentence, while
the English control was refused. The set is chunking-strategies-lab's csl#240.
"""

from __future__ import annotations

import pytest

from rag_kit import generator, rewriter, text
from rag_kit.generator import (
    CitationError,
    GeneratedAnswer,
    TemplateGenerator,
    enforce_citations,
    split_sentences,
)
from rag_kit.retriever import RetrievalResult


def _result(external_id: str, body: str) -> RetrievalResult:
    return RetrievalResult(
        external_id=external_id,
        text=body,
        metadata={"source": "test"},
        fused_score=0.9,
        ranks={"lexical": 1, "dense": 1},
    )


# (script, first sentence, terminator, second sentence) -- spaced scripts.
SPACED = [
    ("devanagari danda", "पेरिस राजधानी है", "।", "यह बड़ा है"),
    ("devanagari double danda", "पेरिस राजधानी है", "॥", "यह बड़ा है"),
    ("urdu full stop", "پیرس دارالحکومت ہے", "۔", "یہ بڑا ہے"),
    ("ethiopic full stop", "ፓሪስ ዋና ከተማ ናት", "።", "ትልቅ ናት"),
    ("ethiopic question mark", "ፓሪስ ዋና ከተማ ናት", "፧", "ትልቅ ናት"),
    ("armenian full stop", "Փարիզը մայրաքաղաքն է", "։", "Այն մեծ է"),
    ("armenian question mark", "Փարիզը մայրաքաղաքն է", "՞", "Այն մեծ է"),
    ("armenian exclamation", "Փարիզը մայրաքաղաքն է", "՜", "Այն մեծ է"),
    ("myanmar section", "ပါရီသည် မြို့တော်ဖြစ်သည်", "။", "ကြီးမားသည်"),
    ("khmer khan", "ប៉ារីសជារាជធានី", "។", "វាធំ"),
    ("khmer bariyoosan", "ប៉ារីសជារាជធានី", "៕", "វាធំ"),
    ("double exclamation", "Paris is the capital", "‼", "It is large"),
    ("double question", "Paris is the capital", "⁇", "It is large"),
    ("question exclamation", "Paris is the capital", "⁈", "It is large"),
    ("exclamation question", "Paris is the capital", "⁉", "It is large"),
]
IDS = [row[0] for row in SPACED]


@pytest.mark.parametrize(("script", "first", "stop", "second"), SPACED, ids=IDS)
def test_an_uncited_claim_is_refused(script: str, first: str, stop: str, second: str) -> None:
    answer = f"{first}{stop} {second} [cite:a]{stop}"
    assert len(split_sentences(answer)) == 2
    with pytest.raises(CitationError) as excinfo:
        enforce_citations(answer, [_result("a", "x")])
    assert excinfo.value.reason == "unparseable_output"


@pytest.mark.parametrize(("script", "first", "stop", "second"), SPACED, ids=IDS)
def test_a_fully_cited_answer_is_accepted(script: str, first: str, stop: str, second: str) -> None:
    answer = f"{first} [cite:a]{stop} {second} [cite:b]{stop}"
    citations = enforce_citations(answer, [_result("a", "x"), _result("b", "y")])
    assert [c.external_id for c in citations] == ["a", "b"]


def test_halfwidth_ideographic_stop_splits_without_a_space() -> None:
    # `｡` (U+FF61) is the halfwidth `。`; Japanese puts no space after either.
    answer = "巴黎是首都｡它很大[cite:a]｡"
    assert len(split_sentences(answer)) == 2
    with pytest.raises(CitationError):
        enforce_citations(answer, [_result("a", "x")])
    assert enforce_citations("巴黎是首都[cite:a]｡它很大[cite:a]｡", [_result("a", "x")])


@pytest.mark.parametrize(("script", "first", "stop", "second"), SPACED, ids=IDS)
def test_template_generator_output_passes_its_own_enforcement(
    script: str, first: str, stop: str, second: str
) -> None:
    # The writer strips and re-terminates with the reader's set: a template
    # sentence that kept `।` in front of its marker would now be cut there.
    chunk = _result("a", f"{first}{stop} {second}{stop}")
    out = TemplateGenerator().generate("q", [chunk], threshold=0.0)
    assert isinstance(out, GeneratedAnswer), out
    assert len(split_sentences(out.text)) == 2


def test_rewriter_then_split_sees_the_new_terminators() -> None:
    assert rewriter._split_then("पहला चरण पूरा करें। Then list the risks.") == [
        "पहला चरण पूरा करें।",
        "list the risks.",
    ]


def test_rewriter_does_not_stack_a_question_mark_on_a_danda() -> None:
    assert rewriter._split_question_and("What is the price। and where is the store।") == [
        "What is the price?",
        "where is the store?",
    ]


def test_one_terminator_set_everywhere() -> None:
    assert generator._TERMINATORS is text.SENTENCE_TERMINATORS
    assert rewriter._TERMINATORS is text.SENTENCE_TERMINATORS
    for _, _, stop, _ in SPACED:
        assert stop in text.SENTENCE_TERMINATORS
