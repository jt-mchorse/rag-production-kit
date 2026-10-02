"""`RewriteResult` refuses a bare-string `sub_queries` (#260).

`Retriever.search(..., rewriter=...)` iterates `sub_queries`. A custom rewriter
returning `RewriteResult(sub_queries="who wrote Macbeth", ...)` ran 17 hybrid
searches -- one per character -- and fused them. `RewriteResult` is the one
value every rewriter returns, so it is where the shape is refused.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_kit import HashEmbedder, Retriever
from rag_kit.rewriter import RewriteResult


@pytest.mark.parametrize("bare", ["who wrote Macbeth", b"who", ""])
def test_a_bare_string_is_refused_at_construction(bare: Any) -> None:
    with pytest.raises(ValueError, match="one character at a time"):
        RewriteResult(sub_queries=bare, reasoning="r")


def test_the_message_shows_the_working_spelling() -> None:
    with pytest.raises(ValueError, match=r"pass \('who wrote Macbeth',\)"):
        RewriteResult(sub_queries="who wrote Macbeth", reasoning="r")  # type: ignore[arg-type]


@pytest.mark.parametrize("subs", [("a b",), ("a b", "c d"), ["a b"]])
def test_sequences_are_unchanged(subs: Any) -> None:
    assert RewriteResult(sub_queries=subs, reasoning="r").sub_queries == subs


class _BareRewriter:
    def rewrite(self, query: str) -> RewriteResult:
        return RewriteResult(sub_queries=query, reasoning="custom")  # type: ignore[arg-type]


def test_retriever_search_runs_no_search_for_a_bare_string_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    retriever = Retriever(conn=None, embedder=HashEmbedder())
    monkeypatch.setattr(retriever, "_hybrid_search", lambda q, *a, **k: calls.append(q) or [])
    with pytest.raises(ValueError, match="one character at a time"):
        retriever.search("who wrote Macbeth", k=3, rewriter=_BareRewriter())
    assert calls == []
