"""A blank query or rewriter sub-query is refused before any paid call (#424).

The emptiness checks were `if not query`, so a whitespace-only query passed
them. Measured on `d29bfde` with a fake connection and a call-counting embedder:

    Retriever.search(q, k=3) for " ", "\\t\\n", "\\u3000", "\\u00a0",
        "\\u200b", "\\ufeff"                     1 embed, 2 SQL, returned []
    search("q", rewriter=<("a", "   ")>)       2 embeds, 4 SQL, returned []
    search("q", rewriter=<("   ", "a")>)       2 embeds, 4 SQL, returned []
    RewriteResult(sub_queries=("",), ...)      constructed

"Blank" is defined once, `rag_kit.text.is_blank`: every character is
whitespace by `str.isspace` or a format character (category Cf). The catalog
sweep below pins why that is the right set: `str.isspace`, `str.strip()` and
`re`'s `\\s` agree codepoint for codepoint, and no blank codepoint yields a
lexical token, so the rule never refuses a query the lexical channel could
match. U+200B and U+FEFF are the arm `.strip()` alone would miss.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from typing import Any

import pytest

from rag_kit import AnthropicRewriter, RewriteResult, StreamingPipeline, TemplateRewriter
from rag_kit.reranker import Candidate, LexicalOverlapReranker
from rag_kit.text import is_blank, word_tokens
from tests.test_non_str_query_type_check import (
    _cohere,
    _CountingRetriever,
    _CountingRewriter,
    _FakeAnthropic,
    _FakeCohereClient,
    _retriever,
)

_BLANK: dict[str, str] = {
    "space": " ",
    "spaces": "   ",
    "tab-newline": "\t\n",
    "ideographic-space": "\u3000",
    "nbsp": "\xa0",
    "zwsp": "\u200b",
    "bom": "\ufeff",
    "word-joiner": "\u2060",
    "mixed": " \u200b\u3000\ufeff\t",
}
# The ones `.strip()` leaves non-empty: blank only by the Cf half of the rule.
_STRIP_MISSES = {"zwsp", "bom", "word-joiner", "mixed"}

# Not blank: each has a lexical token, so it is a real (if odd) query.
_NOT_BLANK: dict[str, str] = {
    "padded": "  refund  ",
    "zwsp-padded": "\u200brefund\u200b",
    # A lone combining mark is a token in `word_tokens` (category M).
    "combining-mark": "\u0301",
    # Hangul filler is Default_Ignorable but category Lo, so `isalnum` and a token.
    "hangul-filler": "\u3164",
    "digit": "0",
}

_ALL_CODEPOINTS = [chr(cp) for cp in range(sys.maxunicode + 1)]


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------


def test_the_three_stdlib_whitespace_notions_are_one_set() -> None:
    # The existing checks used `.strip()` and the rewriter's `_norm` uses
    # `re` `\s`: if these ever diverged, one check would let by what another
    # refuses.
    isspace = {c for c in _ALL_CODEPOINTS if c.isspace()}
    regex = {c for c in _ALL_CODEPOINTS if re.fullmatch(r"\s", c)}
    stripped = {c for c in _ALL_CODEPOINTS if c.strip() == ""}
    assert isspace == regex == stripped
    assert {"\xa0", "\u3000", "\u2028", "\x1c"} <= isspace
    assert "\u200b" not in isspace
    assert "\ufeff" not in isspace


def test_is_blank_is_exactly_whitespace_or_cf_over_the_catalog() -> None:
    blank = {c for c in _ALL_CODEPOINTS if is_blank(c)}
    expected = {c for c in _ALL_CODEPOINTS if c.isspace() or unicodedata.category(c) == "Cf"}
    assert blank == expected
    # No blank codepoint is a lexical token: blank is a subset of token-less.
    assert [hex(ord(c)) for c in blank if word_tokens(c)] == []


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_blank_variants_are_blank(name: str) -> None:
    assert is_blank(_BLANK[name])
    assert word_tokens(_BLANK[name]) == []
    assert (_BLANK[name].strip() == "") is (name not in _STRIP_MISSES)


@pytest.mark.parametrize("name", sorted(_NOT_BLANK))
def test_control_non_blank_boundary(name: str) -> None:
    assert not is_blank(_NOT_BLANK[name])
    assert word_tokens(_NOT_BLANK[name]) != []


def test_empty_is_blank() -> None:
    assert is_blank("")


# ---------------------------------------------------------------------------
# Retriever.search / rerank_pool / sub-queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_search_refuses_a_blank_query_before_embedding_or_sql(name: str) -> None:
    retriever, conn, embedder = _retriever()
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        retriever.search(_BLANK[name], k=3)
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_search_refuses_a_blank_query_before_the_rewriter(name: str) -> None:
    retriever, conn, embedder = _retriever()
    rewriter = _CountingRewriter()
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        retriever.search(_BLANK[name], k=3, rewriter=rewriter)
    assert rewriter.calls == 0
    assert embedder.calls == []


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_rerank_pool_refuses_a_blank_query_before_embedding(name: str) -> None:
    retriever, conn, embedder = _retriever()
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        retriever.rerank_pool(_BLANK[name], k=3)
    assert embedder.calls == []
    assert conn.executed == []


class _RawRewriter:
    """Returns a duck-typed result, so `RewriteResult.__post_init__` is skipped
    and only the retriever's own loop stands between a blank sub-query and the
    embedder."""

    def __init__(self, sub_queries: tuple[str, ...]) -> None:
        self.sub_queries = sub_queries

    def rewrite(self, query: str) -> Any:
        return self


@pytest.mark.parametrize("name", sorted(_BLANK))
@pytest.mark.parametrize("position", [0, 1])
def test_a_blank_sub_query_is_refused_before_any_is_embedded(name: str, position: int) -> None:
    subs = ["who wrote Hamlet", "who wrote Macbeth"]
    subs[position] = _BLANK[name]
    retriever, conn, embedder = _retriever()
    with pytest.raises(
        ValueError, match=rf"^rewriter sub_queries\[{position}\] must be non-empty; got "
    ):
        retriever.search("who wrote Hamlet and Macbeth", k=3, rewriter=_RawRewriter(tuple(subs)))
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_BLANK) + ["empty"])
def test_a_blank_single_sub_query_is_refused_before_embedding(name: str) -> None:
    retriever, conn, embedder = _retriever()
    sub = _BLANK.get(name, "")
    with pytest.raises(ValueError, match=r"^rewriter sub_queries\[0\] must be non-empty; got "):
        retriever.search("who wrote Hamlet", k=3, rewriter=_RawRewriter((sub,)))
    assert embedder.calls == []


def test_control_padded_query_and_sub_queries_still_run_both_channels() -> None:
    retriever, conn, embedder = _retriever()
    assert retriever.search("  refund  ", k=3) == []
    assert embedder.calls == ["  refund  "]
    assert len(conn.executed) == 2
    retriever, conn, embedder = _retriever()
    rewriter = _RawRewriter(("\u200bHamlet", " Macbeth "))
    assert retriever.search("q", k=3, rewriter=rewriter) == []
    assert embedder.calls == ["\u200bHamlet", " Macbeth "]


# ---------------------------------------------------------------------------
# RewriteResult and the two shipped rewriters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_BLANK) + ["empty"])
@pytest.mark.parametrize("position", [0, 1])
def test_rewrite_result_refuses_a_blank_sub_query(name: str, position: int) -> None:
    subs = ["a", "b"]
    subs[position] = _BLANK.get(name, "")
    with pytest.raises(ValueError, match=rf"^sub_queries\[{position}\] must be non-empty; got "):
        RewriteResult(sub_queries=tuple(subs), reasoning="")


def test_control_rewrite_result_leaves_types_and_emptiness_to_the_retriever() -> None:
    # A non-str element is the retriever's #422 check, `()` its "empty
    # sub_queries" check; neither moved here.
    assert RewriteResult(sub_queries=("a", b"b"), reasoning="").sub_queries == ("a", b"b")  # type: ignore[arg-type]
    assert RewriteResult(sub_queries=(), reasoning="").sub_queries == ()
    assert RewriteResult(sub_queries=[" a "], reasoning="").sub_queries == [" a "]  # type: ignore[arg-type]


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_template_rewriter_refuses_a_blank_query(name: str) -> None:
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        TemplateRewriter().rewrite(_BLANK[name])


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_anthropic_rewriter_refuses_a_blank_query_before_the_api(name: str) -> None:
    client = _FakeAnthropic()
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        AnthropicRewriter(client=client).rewrite(_BLANK[name])
    assert client.messages.calls == []


class _Message:
    def __init__(self, text: str) -> None:
        self.content = [{"type": "text", "text": text}]


class _ReplyingAnthropic:
    """A fake client whose reply is a fixed `{"sub_queries": [...]}` payload."""

    def __init__(self, sub_queries: list[str]) -> None:
        self._text = json.dumps({"sub_queries": sub_queries, "reasoning": "r"})
        self.messages = self

    def create(self, **kwargs: Any) -> _Message:
        return _Message(self._text)


def _reply(sub_queries: list[str]) -> _ReplyingAnthropic:
    return _ReplyingAnthropic(sub_queries)


def test_anthropic_rewriter_drops_a_zero_width_sub_query_it_used_to_keep() -> None:
    # `_norm` strips whitespace but keeps U+200B, and the filter was truthiness.
    out = AnthropicRewriter(client=_reply(["who wrote Hamlet", "\u200b", "  "])).rewrite("q")
    assert out.sub_queries == ("who wrote Hamlet",)


def test_anthropic_rewriter_all_blank_sub_queries_is_its_own_error() -> None:
    with pytest.raises(ValueError, match="all blank after normalization"):
        AnthropicRewriter(client=_reply(["\u200b", "\ufeff "])).rewrite("q")


# ---------------------------------------------------------------------------
# Rerankers and StreamingPipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_cohere_reranker_refuses_a_blank_query_before_the_api(name: str) -> None:
    client = _FakeCohereClient()
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        _cohere(client).rerank(_BLANK[name], [Candidate("d1", "text", {})])
    assert client.calls == []


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_lexical_reranker_refuses_a_blank_query_as_it_refuses_empty(name: str) -> None:
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        LexicalOverlapReranker().rerank(_BLANK[name], [Candidate("d1", "text", {})])


def test_control_lexical_reranker_keeps_input_order_for_a_tokenless_non_blank_query() -> None:
    # Punctuation has no token but is not blank: the old no-token arm still runs.
    cands = [Candidate("d1", "b", {}), Candidate("d2", "a", {})]
    out = LexicalOverlapReranker().rerank("?!", cands)
    assert [c.external_id for c in out] == ["d1", "d2"]


@pytest.mark.parametrize("name", sorted(_BLANK))
def test_streaming_pipeline_refuses_a_blank_query_before_the_first_event(name: str) -> None:
    retriever = _CountingRetriever()
    events = StreamingPipeline(retriever).run(_BLANK[name], k=3)
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        next(events)
    assert retriever.calls == 0


def test_control_streaming_pipeline_runs_a_padded_query() -> None:
    retriever = _CountingRetriever()
    kinds = [e.type for e in StreamingPipeline(retriever).run(" refund ", k=3)]
    assert kinds[0] == "retrieving"
    assert retriever.calls == 1
