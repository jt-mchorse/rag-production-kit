"""The correctness proxy does not credit tokens inside ``[cite:<id>]`` markers (#404).

A marker names a chunk; it is not answer content. Tokenized with the markers
in, a chunk id like ``pro-14-day-refund-window`` scored 5/7 against
"Pro customers get a 14-day refund window." for an answer whose prose said
nothing at all, and on the committed dataset ``rag_qa_007`` took ``export``
from ``[cite:doc-data-export]`` (its prose says "exported"), holding the
published mean at 0.90 instead of 0.89.

The only earlier arm used ``[cite:x]``, an id that cannot collide with any
expected token, so it could not see this.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evals import run_eval
from rag_kit import GeneratedAnswer, TemplateGenerator

_ROOT = Path(__file__).resolve().parent.parent

# Written out here rather than imported from the generator, so the arms below
# do not share the pattern the fix uses.
_MARKER = re.compile(r"\[cite:[^\]]*\]")


def _answer(text: str) -> GeneratedAnswer:
    return GeneratedAnswer(text=text, citations=(), used_threshold=0.0, top_score=1.0)


EXPECTED = "Pro customers get a 14-day refund window."


@pytest.mark.parametrize(
    "text",
    [
        "See the policy [cite:pro-14-day-refund-window].",
        "See the policy[cite:pro-14-day-refund-window].",
        "See the policy [cite:faq.md#pro customers refund].",
        "See the policy [cite:a] [cite:pro-14-day-refund-window].",
    ],
)
def test_tokens_only_inside_a_marker_earn_nothing(text: str) -> None:
    score, reasoning = run_eval._score_correctness(_answer(text), EXPECTED)
    assert score == 0.0, reasoning


def test_prose_tokens_still_count_beside_a_colliding_marker() -> None:
    # Control: stripping the marker must not strip the prose around it.
    text = "Pro customers get a 14 day refund window [cite:pro-14-day-refund-window]."
    score, _ = run_eval._score_correctness(_answer(text), EXPECTED)
    assert score == 1.0


def test_a_marker_between_two_words_keeps_them_apart() -> None:
    # Deleting the marker outright would glue `refund` and `window` into one
    # token `refundwindow`, which matches neither expected token.
    text = "Pro customers get a 14 day refund[cite:a]window."
    score, reasoning = run_eval._score_correctness(_answer(text), EXPECTED)
    assert score == 1.0, reasoning


def test_every_dataset_row_is_scored_on_its_prose_alone() -> None:
    """Through the real generator and retriever, each row's score is the
    overlap with the answer *minus its markers*, computed independently."""
    examples = run_eval._load_dataset(run_eval.DATASET_PATH)
    corpus = run_eval._load_corpus(run_eval.CORPUS_PATH)
    generator = TemplateGenerator()
    marker_rows = 0
    for ex in examples:
        retrieved = run_eval._retrieve_in_memory(ex.input, corpus, k=5)
        out = generator.generate(ex.input, retrieved, threshold=0.0)
        assert isinstance(out, GeneratedAnswer)
        marker_rows += bool(_MARKER.search(out.text))
        exp = set(run_eval._tokens(ex.expected)) - run_eval._STOP
        prose = set(run_eval._tokens(_MARKER.sub(" ", out.text))) - run_eval._STOP
        score, _ = run_eval._score_correctness(out, ex.expected)
        assert score == pytest.approx(len(exp & prose) / len(exp)), ex.example_id
    # The rule walked answers that carry markers, not an empty population.
    assert marker_rows == len(examples)


def test_rag_qa_007_no_longer_earns_export_from_its_chunk_id() -> None:
    (row,) = [
        r
        for r in next(s for s in run_eval.run_all_suites() if s.suite == "correctness").rows
        if r.example_id == "rag_qa_007"
    ]
    assert "'export'" not in row.reasoning
    assert row.score == pytest.approx(6 / 9)


@pytest.mark.parametrize("where", ["baselines", "current"])
def test_committed_correctness_artifacts_match_the_live_scorer(where: str) -> None:
    """The committed records were regenerated with the fixed scorer, so the
    README/benchmarks numbers locked against them are the stripped ones."""
    committed = json.loads((_ROOT / "evals" / where / "correctness.json").read_text())
    live = next(s for s in run_eval.run_all_suites() if s.suite == "correctness")
    live_rows = {r.example_id: round(r.score, 4) for r in live.rows}
    assert {r["example_id"]: r["score"] for r in committed["rows"]} == live_rows
