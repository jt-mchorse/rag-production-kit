"""Tests for the query rewriter / decomposer (#3, D-014).

Two backends:

- `TemplateRewriter` — dep-free, deterministic, rule-based. Pattern
  coverage tests live here.
- `AnthropicRewriter` — production binding. Real-API tests live in a
  separate workflow; these tests inject a fake client so the SDK
  imports lazily and the parsing/validation code is exercised without
  an API key.

Integration with `Retriever.search(rewriter=...)` is covered in
``test_retriever_rewriter.py``.
"""

from __future__ import annotations

import json

import pytest

from rag_kit import AnthropicRewriter, RewriteResult, TemplateRewriter
from rag_kit.rewriter import Rewriter

# ----------------------------------------------------------------------
# TemplateRewriter: pattern coverage
# ----------------------------------------------------------------------


def test_template_rewriter_no_decomposition_for_simple_query():
    rw = TemplateRewriter()
    out = rw.rewrite("What is the refund window?")
    assert isinstance(out, RewriteResult)
    assert out.sub_queries == ("What is the refund window?",)
    assert out.reasoning == "no_decomposition"


def test_template_rewriter_compare_pattern_with():
    out = TemplateRewriter().rewrite("Compare Anthropic with OpenAI")
    assert out.reasoning == "compare_pattern"
    assert out.sub_queries == ("What is Anthropic?", "What is OpenAI?")


def test_template_rewriter_compare_pattern_to_versus():
    out = TemplateRewriter().rewrite("compare Postgres to MySQL")
    assert out.reasoning == "compare_pattern"
    assert out.sub_queries == ("What is Postgres?", "What is MySQL?")

    out = TemplateRewriter().rewrite("Compare BM25 versus dense retrieval?")
    assert out.reasoning == "compare_pattern"
    assert out.sub_queries == ("What is BM25?", "What is dense retrieval?")


def test_template_rewriter_then_pattern_strips_then_prefix():
    out = TemplateRewriter().rewrite(
        "Find the CEO of the company. Then describe their education."
    )
    assert out.reasoning == "sequential_then_pattern"
    assert out.sub_queries == (
        "Find the CEO of the company.",
        "describe their education.",
    )


def test_template_rewriter_multi_question_and_pattern():
    out = TemplateRewriter().rewrite(
        "Who founded Anthropic and where did they work before?"
    )
    assert out.reasoning == "multi_question_and_pattern"
    assert out.sub_queries == (
        "Who founded Anthropic?",
        "where did they work before?",
    )


def test_template_rewriter_and_inside_simple_phrase_is_not_split():
    """'wine and cheese pairings' must NOT split — only one half is a question."""
    out = TemplateRewriter().rewrite("Tell me about wine and cheese pairings.")
    assert out.reasoning == "no_decomposition"
    assert len(out.sub_queries) == 1


def test_template_rewriter_compare_pattern_takes_precedence_over_and():
    """'Compare X and Y' is the compare pattern, not the multi-question 'and' split."""
    out = TemplateRewriter().rewrite("Compare BM25 and dense retrieval")
    assert out.reasoning == "compare_pattern"
    assert out.sub_queries == ("What is BM25?", "What is dense retrieval?")


def test_template_rewriter_three_part_and_questions():
    out = TemplateRewriter().rewrite(
        "Who is the CTO and when did they join and what did they ship?"
    )
    assert out.reasoning == "multi_question_and_pattern"
    # Each conjunct must be a question; original "?" is restored to each part.
    assert out.sub_queries == (
        "Who is the CTO?",
        "when did they join?",
        "what did they ship?",
    )


def test_template_rewriter_rejects_empty_query():
    rw = TemplateRewriter()
    with pytest.raises(ValueError, match="non-empty"):
        rw.rewrite("")
    with pytest.raises(ValueError, match="non-empty"):
        rw.rewrite("   ")


def test_template_rewriter_normalizes_whitespace():
    out = TemplateRewriter().rewrite("What    is\tthe \n refund?")
    assert out.sub_queries == ("What is the refund?",)


def test_template_rewriter_implements_protocol():
    """Mypy-style: TemplateRewriter must be usable where the Rewriter protocol is."""
    rw: Rewriter = TemplateRewriter()
    result = rw.rewrite("hello world")
    assert isinstance(result, RewriteResult)


# ----------------------------------------------------------------------
# AnthropicRewriter: parsing / validation via injected fake client
# ----------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs) -> _FakeMessage:
        self.calls.append(kwargs)
        return _FakeMessage(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def _fake_client_for(payload: dict) -> _FakeClient:
    return _FakeClient(json.dumps(payload))


def test_anthropic_rewriter_parses_valid_json():
    payload = {
        "sub_queries": [
            "Who founded Anthropic?",
            "Where did the founders work before Anthropic?",
        ],
        "reasoning": "two_entity_decomposition",
    }
    rw = AnthropicRewriter(client=_fake_client_for(payload))
    result = rw.rewrite("Who founded Anthropic and where did they work before?")
    assert result.sub_queries == tuple(payload["sub_queries"])
    assert result.reasoning == "two_entity_decomposition"


def test_anthropic_rewriter_no_decomposition_passthrough():
    payload = {
        "sub_queries": ["What is the refund window?"],
        "reasoning": "no_decomposition",
    }
    rw = AnthropicRewriter(client=_fake_client_for(payload))
    result = rw.rewrite("What is the refund window?")
    assert result.sub_queries == ("What is the refund window?",)
    assert result.reasoning == "no_decomposition"


def test_anthropic_rewriter_truncates_to_max_sub_queries():
    payload = {
        "sub_queries": [f"sub {i}" for i in range(20)],
        "reasoning": "too_many",
    }
    rw = AnthropicRewriter(client=_fake_client_for(payload))
    result = rw.rewrite("decompose this")
    assert len(result.sub_queries) == 6  # _MAX_SUB_QUERIES


def test_anthropic_rewriter_strips_blank_entries():
    payload = {
        "sub_queries": ["real query", "  ", "another"],
        "reasoning": "mixed",
    }
    rw = AnthropicRewriter(client=_fake_client_for(payload))
    result = rw.rewrite("decompose")
    assert result.sub_queries == ("real query", "another")


def test_anthropic_rewriter_rejects_non_json():
    rw = AnthropicRewriter(client=_FakeClient("not json"))
    with pytest.raises(ValueError, match="non-JSON"):
        rw.rewrite("query")


def test_anthropic_rewriter_rejects_non_object_json():
    rw = AnthropicRewriter(client=_FakeClient("[1, 2, 3]"))
    with pytest.raises(ValueError, match="non-object"):
        rw.rewrite("query")


def test_anthropic_rewriter_rejects_missing_sub_queries():
    rw = AnthropicRewriter(client=_fake_client_for({"reasoning": "oops"}))
    with pytest.raises(ValueError, match="sub_queries"):
        rw.rewrite("query")


def test_anthropic_rewriter_rejects_empty_sub_queries_list():
    rw = AnthropicRewriter(
        client=_fake_client_for({"sub_queries": [], "reasoning": "blank"})
    )
    with pytest.raises(ValueError, match="sub_queries"):
        rw.rewrite("query")


def test_anthropic_rewriter_rejects_non_string_entry():
    rw = AnthropicRewriter(
        client=_fake_client_for({"sub_queries": ["fine", 42], "reasoning": "x"})
    )
    with pytest.raises(ValueError, match="not a string"):
        rw.rewrite("query")


def test_anthropic_rewriter_rejects_all_blank_entries():
    rw = AnthropicRewriter(
        client=_fake_client_for({"sub_queries": ["  ", "\t"], "reasoning": "blank"})
    )
    with pytest.raises(ValueError, match="all blank"):
        rw.rewrite("query")


def test_anthropic_rewriter_uses_injected_client_without_sdk():
    """Injected client path must not require the anthropic SDK to be installed."""
    rw = AnthropicRewriter(
        client=_fake_client_for({"sub_queries": ["q"], "reasoning": "r"}),
        model="claude-opus-4-7",
        max_tokens=123,
    )
    result = rw.rewrite("hello")
    assert result.sub_queries == ("q",)
    # The fake client recorded the call kwargs; check the model + max_tokens were forwarded.
    call = rw._client.messages.calls[0]
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 123


def test_anthropic_rewriter_rejects_empty_query():
    rw = AnthropicRewriter(client=_FakeClient("{}"))
    with pytest.raises(ValueError, match="non-empty"):
        rw.rewrite("   ")
