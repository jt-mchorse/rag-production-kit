"""Two renderings are distinct as numbers, not as strings (#356).

`render_comparison` returned once the strings differed, and `-0.0000` vs
`0.0000` do. Measured on main: `LexicalOverlapReranker(length_penalty=1e-5)`
scores a zero-overlap chunk -9.6875e-06, and `TemplateGenerator().generate(...,
threshold=0.0)` refused with `top_score=-0.0000 below threshold=0.0000`.
"""

from __future__ import annotations

import pytest

from rag_kit.comparison import render_comparison
from rag_kit.generator import Refusal, TemplateGenerator
from rag_kit.reranker import LexicalOverlapReranker
from rag_kit.retriever import RetrievalResult


@pytest.mark.parametrize(
    ("value", "other"),
    [(-9.6875e-06, 0.0), (9.6875e-06, -0.0), (-1e-12, 0.0), (0.0, -1e-12), (-4e-5, 4e-5)],
)
def test_the_renderings_read_back_distinct_and_in_order(value: float, other: float) -> None:
    a, b = render_comparison(value, other, places=4)
    assert float(a) != float(b)
    assert (float(a) < float(b)) == (value < other)


def test_equal_and_ordinary_pairs_are_unchanged() -> None:
    assert render_comparison(0.5, 0.5, places=4) == ("0.5000", "0.5000")
    assert render_comparison(0.12, 0.34, places=4) == ("0.1200", "0.3400")


def test_the_refusal_detail_is_true_as_written() -> None:
    results = [RetrievalResult("d1", "completely unrelated words here", {}, 0.03, {"dense": 1})]
    reranked = LexicalOverlapReranker(length_penalty=1e-5).rerank("refund policy", results)
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score < 0.0  # anti-vacuity: the #69 negative score
    out = TemplateGenerator().generate("refund policy", reranked, threshold=0.0)
    assert isinstance(out, Refusal)
    top, thr = out.detail.split("top_score=")[1].split(" below threshold=")
    assert float(top) < float(thr), out.detail
