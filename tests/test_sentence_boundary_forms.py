"""Three ways a model ends a sentence that `split_sentences` did not see (#262).

`_SENTENCE_SPLIT` knew one boundary shape: a terminator, at most one closer from
six, then whitespace (#144, #160). Each row in `_BYPASSES` was ACCEPTED on main
(acab504): an uncited first claim merged onto the cited second one and rode on
its marker. The shapes:

* Chinese/Japanese put no space after `。！？`;
* a closer run or a closer outside the six -- `.**` (markdown bold), `.")`,
  `.»`, `。」`;
* a marker straight after the terminator, `days.[cite:doc1] Next ...`, which
  `enforce_citations` accepts as a cited sentence, so the boundary is after it.
"""

from __future__ import annotations

import pytest

from rag_kit.generator import (
    CitationError,
    GeneratedAnswer,
    TemplateGenerator,
    enforce_citations,
    split_sentences,
)
from rag_kit.retriever import RetrievalResult


def _r(external_id: str, text: str = "x") -> RetrievalResult:
    return RetrievalResult(
        external_id=external_id, text=text, metadata={}, fused_score=1.0, ranks={"dense": 1}
    )


_RETRIEVED = [_r("doc1"), _r("doc2"), _r("faq.md#Q3. refunds")]

# (id, answer). Every row holds an uncited claim and must be refused.
_BYPASSES = [
    ("zh-no-space", "退款期为14天。专业版退款期为30天[cite:doc1]。"),
    ("ja-fullwidth-bang-no-space", "返金は14日です！プロは30日です[cite:doc1]。"),
    ("ja-fullwidth-question-no-space", "返金は14日ですか？プロは30日です[cite:doc1]。"),
    ("ja-quote-then-said", "彼は「返金は14日です。」と言った。プロは30日です[cite:doc1]。"),
    ("markdown-bold", "**Refunds take 14 days.** Pro plans get 30 days [cite:doc1]."),
    ("two-closers", 'The guide says ("refunds take 14 days.") Pro plans get 30 days [cite:doc1].'),
    ("guillemet", "Refunds take «14 days.» Pro plans get 30 days [cite:doc1]."),
    ("ja-corner-bracket-then-space", "「返金は14日です。」 プロは30日です[cite:doc1]。"),
    ("marker-after-terminator", "Refunds take 14 days.[cite:doc1] Pro plans get 30 days."),
    ("zh-marker-after-terminator", "退款期为14天。[cite:doc1]专业版退款期为30天。"),
]

# (id, answer). Fully cited: must stay accepted. The decimal, abbreviation and
# Japanese `。」と` rows are where a looser boundary would over-split.
_CITED = [
    ("marker-after-terminator-alone", "Pro plans get 30 days.[cite:doc1]"),
    (
        "marker-after-terminator-twice",
        "Refunds take 14 days.[cite:doc1] Pro plans get 30 days.[cite:doc2]",
    ),
    ("zh-marker-before-terminator", "退款期为14天[cite:doc1]。专业版退款期为30天[cite:doc2]。"),
    ("zh-marker-after-terminator", "退款期为14天。[cite:doc1]专业版退款期为30天。[cite:doc2]"),
    (
        "ja-quote-then-said",
        "彼は「返金は14日です。」と言った[cite:doc1]。プロは30日です[cite:doc2]。",
    ),
    ("decimals", "Latency fell 3.5 percent [cite:doc1]. It was 2.0x faster [cite:doc2]."),
    ("abbreviations", "Dr. Smith found it [cite:doc1]. The U.S. team agreed [cite:doc2]."),
    ("markdown-bold", "**Refunds take 14 days [cite:doc1].** Pro plans get 30 days [cite:doc2]."),
    (
        "two-closers",
        'The guide says ("refunds take 14 days [cite:doc1].") Pro plans get 30 days [cite:doc2].',
    ),
    (
        "terminator-and-space-inside-an-id",
        "Per the FAQ refunds take 14 days [cite:faq.md#Q3. refunds]. Pro gets 30 [cite:doc1].",
    ),
]


@pytest.mark.parametrize("text", [t for _, t in _BYPASSES], ids=[i for i, _ in _BYPASSES])
def test_an_uncited_claim_in_any_sentence_form_is_refused(text: str) -> None:
    with pytest.raises(CitationError) as excinfo:
        enforce_citations(text, _RETRIEVED)
    assert excinfo.value.reason == "unparseable_output"


@pytest.mark.parametrize("text", [t for _, t in _CITED], ids=[i for i, _ in _CITED])
def test_a_fully_cited_answer_in_any_sentence_form_is_accepted(text: str) -> None:
    assert enforce_citations(text, _RETRIEVED)


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        # Closers and a marker stay on the sentence they end.
        ("A is true.**  B is true.", ["A is true.**", "B is true."]),
        ('He said ("A.") B.', ['He said ("A.")', "B."]),
        ("A.[cite:doc1] B.", ["A.[cite:doc1]", "B."]),
        # A terminator run is one boundary, not two.
        ("真的吗？！是的。", ["真的吗？！", "是的。"]),
        ("Really?! Yes.", ["Really?!", "Yes."]),
        # No zero-width split after `.`, and none before a closer after `。`.
        ("Version 2.0 shipped.", ["Version 2.0 shipped."]),
        ("「です。」と言った。", ["「です。」と言った。"]),
    ],
)
def test_where_the_text_is_cut(text: str, sentences: list[str]) -> None:
    assert split_sentences(text) == sentences


@pytest.mark.parametrize(
    "source",
    [
        "Restart the server.**",
        'The guide says ("restart the server.")',
        "Refunds take «14 days.»",
        "彼は「返金は14日です。」",
        "退款期为14天。专业版退款期为30天。",
        "Run `make test.`",
        # A terminator RUN before the closer: the writer took one terminator
        # and left `?` / `..` in front of the marker, where the reader cut.
        'He said “Really?!”',
        "Wait...”",
        'She asked "why?!"',
        "Is it done?!)",
    ],
)
def test_the_template_writer_emits_what_the_reader_accepts(source: str) -> None:
    # `_TERMINATOR_THEN_CLOSERS` shares `_CLOSERS` with the splitter: a closer the
    # writer did not know would put the marker after the tail, where the reader
    # cuts it off -- #258's refusal through a new closer.
    out = TemplateGenerator().generate("q", [_r("doc1", source)], threshold=0.0)
    assert isinstance(out, GeneratedAnswer), out
    assert [c.external_id for c in out.citations] == ["doc1"]
