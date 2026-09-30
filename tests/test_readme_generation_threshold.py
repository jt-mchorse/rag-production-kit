"""The README's generation snippet can actually answer (#237).

It fed `Retriever.search` output to `generate(..., threshold=0.05)`. Without a
reranker the threshold is compared with `fused_score`, and reciprocal-rank
fusion with `DEFAULT_K = 60` tops out at `2 / 61 = 0.0328` -- rank 1 in both
lists. So the documented call refused on every input, under a comment
promising a cited answer.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_kit import GeneratedAnswer, TemplateGenerator
from rag_kit.fusion import DEFAULT_K
from rag_kit.generator import _DEFAULT_THRESHOLD
from rag_kit.retriever import RetrievalResult

_README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
_FUSED_CEILING = 2 / (DEFAULT_K + 1)


def _best_case_fused() -> list[RetrievalResult]:
    """The highest fused scores `Retriever.search` can produce: ranks 1 and 2
    in both the lexical and the dense list."""
    return [
        RetrievalResult(
            external_id=f"doc-{rank}",
            text=text,
            metadata={},
            fused_score=2 / (DEFAULT_K + rank),
            ranks={"lex": rank, "dense": rank},
        )
        for rank, text in (
            (1, "Our refund policy gives Pro customers 14 days."),
            (2, "If a subscriber wants their money back, the window is two weeks."),
        )
    ]


def _snippet_call() -> str:
    block = next(
        b for b in re.findall(r"```python\n(.*?)```", _README, re.S) if "gen.generate(" in b
    )
    return re.search(r"gen\.generate\((.*)\)", block).group(1)  # type: ignore[union-attr]


def test_the_default_threshold_is_reachable_by_fused_only_retrieval() -> None:
    assert 0 < _DEFAULT_THRESHOLD < _FUSED_CEILING


def test_the_readme_generation_call_answers_on_best_case_retrieval() -> None:
    """Evaluate the snippet's own threshold (or the default, if it passes
    none) against the best retrieval the snippet can be handed."""
    args = _snippet_call()
    explicit = re.search(r"threshold\s*=\s*([0-9.]+)", args)
    threshold = float(explicit.group(1)) if explicit else _DEFAULT_THRESHOLD
    out = TemplateGenerator().generate(
        "when do refunds expire?", _best_case_fused(), threshold=threshold
    )
    assert isinstance(out, GeneratedAnswer), f"threshold={threshold} refuses: {out}"
    assert threshold < _FUSED_CEILING


def test_the_old_threshold_really_refused() -> None:
    """The premise of #237, kept so the fix is about a measured failure."""
    out = TemplateGenerator().generate(
        "when do refunds expire?", _best_case_fused(), threshold=0.05
    )
    assert not isinstance(out, GeneratedAnswer)
