"""Question contractions retain their original text when decomposed (#405)."""

from __future__ import annotations

import pytest

from rag_kit import TemplateRewriter


@pytest.mark.parametrize("apostrophe", ["'", "’"])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("What's the refund window", "what's the shipping time"),
        ("Who's the CEO", "where's the HQ"),
        ("Who're the founders", "what're their roles"),
        ("Doesn't the plan include support", "isn't the trial free"),
        ("Can't the plan be renewed", "won't the price change"),
        ("WHAT'S the refund window", "where's the policy"),
    ],
)
def test_contracted_questions_split_without_rewriting_words(first, second, apostrophe):
    first = first.replace("'", apostrophe)
    second = second.replace("'", apostrophe)
    result = TemplateRewriter().rewrite(f"{first} and {second}?")
    assert result.reasoning == "multi_question_and_pattern"
    assert result.sub_queries == (f"{first}?", f"{second}?")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Has the policy changed", "was the trial extended"),
        ("Have the founders left", "were their roles replaced"),
        ("Whose plan includes support", "whom should I contact"),
    ],
)
def test_auxiliary_and_wh_questions_split(first, second):
    result = TemplateRewriter().rewrite(f"{first} and {second}?")
    assert result.sub_queries == (f"{first}?", f"{second}?")
    assert result.reasoning == "multi_question_and_pattern"


@pytest.mark.parametrize(
    "query",
    [
        "McDonald's and Wendy's locations",
        "Will's biography and Will's filmography",
        "Will’s biography and Will’s filmography",
        "Can's albums and Can's members",
        "Can’s albums and Can’s members",
        "What's the wine and cheese pairing?",
        "What's the refund window and shipping time?",
        "WhatsApp plans and what's the price?",
        "Whereabouts and what's next?",
        "Who's the CEO! What's the HQ and where's the office?",
    ],
)
def test_contractions_do_not_turn_phrases_or_multiple_sentences_into_questions(query):
    result = TemplateRewriter().rewrite(query)
    assert result.sub_queries == (query,)
    assert result.reasoning == "no_decomposition"
