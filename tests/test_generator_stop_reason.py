"""An answer the model did not finish is refused, naming why (#305).

`AnthropicGenerator.generate` never read `message.stop_reason`. Measured on
main with an SDK-shaped fake (threshold 0.0, one chunk `a`):

    max_tokens, "Paris is the capital of France [cite:a]"            -> GeneratedAnswer
    max_tokens, "Paris is the capital [cite:a]. It is also the larg" -> Refusal unparseable_output
                                     "sentence has no [cite:...] marker: 'It is also the larg'"
    refusal,    []                                                    -> Refusal unparseable_output
                                     "answer text contained no sentences"
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_kit.generator import AnthropicGenerator, GeneratedAnswer, Refusal
from rag_kit.retriever import RetrievalResult

CHUNK = RetrievalResult(
    external_id="a",
    text="Paris is the capital of France.",
    metadata={},
    fused_score=0.9,
    ranks={"dense": 1},
)


def _generate(text: str | None, **message: Any) -> GeneratedAnswer | Refusal:
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    reply = SimpleNamespace(content=content, **message)
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: reply))
    return AnthropicGenerator(client=client, max_tokens=64).generate("q", [CHUNK], threshold=0.0)


@pytest.mark.parametrize(
    ("text", "stop_reason"),
    [
        ("Paris is the capital of France [cite:a]", "max_tokens"),
        ("Paris is the capital [cite:a]. It is also the larg", "max_tokens"),
        (None, "refusal"),
        ("Paris is the capital of France [cite:a].", "refusal"),
        ("Paris is the capital of France [cite:a].", "pause_turn"),
        ("Paris is the capital of France [cite:a].", "tool_use"),
    ],
)
def test_an_unfinished_answer_is_refused_naming_the_stop_reason(
    text: str | None, stop_reason: str
) -> None:
    out = _generate(text, stop_reason=stop_reason)
    assert isinstance(out, Refusal)
    assert out.reason == "unparseable_output"
    assert f"stop_reason={stop_reason!r}" in out.detail


def test_a_max_tokens_cut_names_the_cap() -> None:
    out = _generate("Paris is the capital of France [cite:a]", stop_reason="max_tokens")
    assert isinstance(out, Refusal)
    assert "max_tokens=64" in out.detail
    assert "incomplete" in out.detail


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
def test_a_finished_answer_is_judged_as_before(stop_reason: str) -> None:
    ok = _generate("Paris is the capital of France [cite:a].", stop_reason=stop_reason)
    assert isinstance(ok, GeneratedAnswer)
    assert [c.external_id for c in ok.citations] == ["a"]
    bad = _generate("Paris is the capital. It is old [cite:a].", stop_reason=stop_reason)
    assert isinstance(bad, Refusal)
    assert "no [cite:...] marker" in bad.detail


def test_the_models_own_refuse_line_still_wins() -> None:
    out = _generate("REFUSE: the context does not say", stop_reason="end_turn")
    assert isinstance(out, Refusal)
    assert out.reason == "insufficient_context"
    cut = _generate("REFUSE: the context does not s", stop_reason="max_tokens")
    assert isinstance(cut, Refusal)
    assert cut.reason == "insufficient_context"


def test_a_client_without_stop_reason_is_judged_as_before() -> None:
    assert isinstance(_generate("Paris is the capital of France [cite:a]."), GeneratedAnswer)
