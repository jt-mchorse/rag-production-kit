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
from rag_kit.rewriter import Rewriter, _split_then

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
    out = TemplateRewriter().rewrite("Find the CEO of the company. Then describe their education.")
    assert out.reasoning == "sequential_then_pattern"
    assert out.sub_queries == (
        "Find the CEO of the company.",
        "describe their education.",
    )


def test_template_rewriter_then_connective_with_punctuation_is_stripped():
    # #92: the split fires on `then\b`, so a connective followed by punctuation
    # ("Then,", "Then;", "Then-") still split — but the old `startswith("then ")`
    # strip only handled a trailing space, leaking the connective into the
    # sub-query. Each of these must strip cleanly.
    for query, second in [
        ("Find the CEO. Then, describe their education.", "describe their education."),
        ("Find the CEO. Then; describe their education.", "describe their education."),
        ("Find the CEO. Then- describe their education.", "describe their education."),
    ]:
        out = TemplateRewriter().rewrite(query)
        assert out.reasoning == "sequential_then_pattern", query
        assert out.sub_queries == ("Find the CEO.", second), query


def test_split_then_strips_punctuated_connective_directly():
    # Unit-level guard on the helper, covering the exact leak cases from #92.
    assert _split_then("Do A. Then, do B.") == ["Do A.", "do B."]
    assert _split_then("Find X. Then; find Y.") == ["Find X.", "find Y."]
    assert _split_then("Do A. then-do B.") == ["Do A.", "do B."]
    # The plain space case (already covered end-to-end above) still works.
    assert _split_then("Do A. Then do B.") == ["Do A.", "do B."]


def test_split_then_does_not_false_split_on_thence_content_word():
    # `then\b` requires a word boundary, so a content word that merely *starts*
    # with "then" (e.g. "thence") is not the connective: no split fires and the
    # word is never stripped/mangled. Mirrors the split's own semantics.
    assert _split_then("Go home. Thence to the river.") is None


def test_template_rewriter_then_pattern_is_case_insensitive():
    # #96: _THEN_SPLIT enumerated `Then|then` literally without re.IGNORECASE,
    # so an emphatic all-caps / mixed-case connective silently failed to
    # decompose — unlike the sibling case-insensitive patterns and contrary to
    # the regex's own docstring. Every casing must decompose identically; the
    # already-case-insensitive _THEN_PREFIX strips the connective regardless.
    for query in [
        "Find the CEO. THEN describe their education.",
        "Find the CEO. ThEn describe their education.",
        "Find the CEO. then describe their education.",
    ]:
        out = TemplateRewriter().rewrite(query)
        assert out.reasoning == "sequential_then_pattern", query
        assert out.sub_queries == ("Find the CEO.", "describe their education."), query


def test_split_then_case_insensitive_split_and_strip():
    # Unit-level guard: the split fires and the connective is stripped for any
    # casing, while "thence" still never false-splits (word-boundary preserved).
    assert _split_then("Do A. THEN do B.") == ["Do A.", "do B."]
    assert _split_then("Do A. ThEn, do B.") == ["Do A.", "do B."]
    assert _split_then("Go home. THENCE to the river.") is None


def test_template_rewriter_multi_question_and_pattern():
    out = TemplateRewriter().rewrite("Who founded Anthropic and where did they work before?")
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


def test_template_rewriter_and_split_conjunct_with_dot_or_bang_is_well_formed():
    """#94: a question-like conjunct ending in `.` or `!` must not get a `?`
    stacked on top (the malformed "What is the price!?" double terminator)."""
    for q in [
        "What is the price. and is it worth it?",
        "What is the price! and is it worth it?",
    ]:
        out = TemplateRewriter().rewrite(q)
        assert out.reasoning == "multi_question_and_pattern", q
        assert out.sub_queries == ("What is the price?", "is it worth it?"), q
        for sub in out.sub_queries:
            assert sub.endswith("?"), sub
            assert not sub.endswith((".?", "!?", "??")), sub


def test_template_rewriter_and_split_does_not_mangle_internal_decimal():
    """The trailing-terminator strip must not touch internal punctuation: a
    decimal like `3.5` only ends in a terminator if the `.` is trailing."""
    out = TemplateRewriter().rewrite("What is 3.5 and is it worth it?")
    assert out.reasoning == "multi_question_and_pattern"
    assert out.sub_queries == ("What is 3.5?", "is it worth it?")


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


@pytest.mark.parametrize(
    "ender",
    [
        "？",  # U+FF1F full-width question mark (CJK IME)
        "！",  # U+FF01 full-width exclamation mark
        "。",  # U+3002 ideographic full stop
        "؟",  # U+061F Arabic question mark
    ],
)
def test_template_rewriter_and_split_non_ascii_terminator_is_well_formed(ender):
    """#111: a conjunct ending in a non-ASCII sentence terminator must not get a
    `?` stacked on top. The pre-fix ASCII-only strip left `？`/`！`/`。`/`؟` in
    place, yielding a doubled terminator like "where did they work？?"."""
    out = TemplateRewriter().rewrite(f"Who founded X and where did they work{ender}")
    assert out.reasoning == "multi_question_and_pattern"
    assert out.sub_queries == ("Who founded X?", "where did they work?")
    for sub in out.sub_queries:
        assert sub.endswith("?"), sub
        # No terminator (ASCII or non-ASCII) may precede the canonical "?".
        assert sub[-2:] not in {f"{t}?" for t in "?.!？！؟。"}, sub


def test_template_rewriter_and_split_non_ascii_does_not_mangle_internal_char():
    """The broadened terminator strip must only touch *trailing* terminators:
    a non-ASCII terminator that sits mid-conjunct is left untouched (#111)."""
    out = TemplateRewriter().rewrite("What is the 。 rule and is it worth it?")
    assert out.reasoning == "multi_question_and_pattern"
    assert out.sub_queries == ("What is the 。 rule?", "is it worth it?")


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
    rw = AnthropicRewriter(client=_fake_client_for({"sub_queries": [], "reasoning": "blank"}))
    with pytest.raises(ValueError, match="sub_queries"):
        rw.rewrite("query")


def test_anthropic_rewriter_rejects_non_string_entry():
    rw = AnthropicRewriter(client=_fake_client_for({"sub_queries": ["fine", 42], "reasoning": "x"}))
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
