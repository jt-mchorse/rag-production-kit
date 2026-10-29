"""A non-`str` query or `Document.text` is refused before any paid call (#422).

`_require_sendable` returned early for a non-`str`, and `Document.__post_init__`
type-checked `external_id` but not `text`. Measured on `094efe2` with a fake
connection and a call-counting embedder:

    Retriever.search(b"refund policy", k=3)   embed calls 1, then a bare error
    (same for 123, ["refund", "policy"], ("refund",); and for rerank_pool)
    search(<any of those>, rewriter=r)        r.rewrite() called, search SUCCEEDED
    sub_queries ("who wrote Hamlet", b"bad")  the first sub-query embedded first
    Document("d1", ["a", "list"])             ACCEPTED (and {"a": 1})
    AnthropicRewriter.rewrite(b"...")         messages.create sent "Query: b'...'"
    AnthropicGenerator.generate(b"...", ...)  messages.create sent "Query: b'...'"
    CohereReranker.rerank(b"...", ...)        client.rerank called

The type rule is `isinstance(x, str)`, which is how `Document.external_id` was
already checked: a `str` subclass is text, `True` is not.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_kit import (
    AnthropicGenerator,
    AnthropicRewriter,
    Document,
    HashEmbedder,
    Indexer,
    RetrievalResult,
    Retriever,
    RewriteResult,
    StreamingPipeline,
)
from rag_kit.reranker import Candidate, CohereReranker

_NON_STR: dict[str, Any] = {
    "bytes": b"refund policy",
    "bytearray": bytearray(b"refund policy"),
    "int": 123,
    "bool": True,
    "float": 1.5,
    "none": None,
    "list": ["refund", "policy"],
    "tuple": ("refund",),
    "dict": {"a": 1},
}
# Empty containers are named by their type, not as "non-empty" (#422).
_EMPTY_NON_STR: dict[str, Any] = {"empty-bytes": b"", "empty-list": [], "empty-tuple": ()}
_ALL_NON_STR = {**_NON_STR, **_EMPTY_NON_STR}


class _Text(str):
    """A `str` subclass: `isinstance(x, str)` holds, so it is text."""


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self._inner = HashEmbedder()

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._inner.embed(text)


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._conn.executed.append(sql)

    def executemany(self, sql: str, rows: list[Any]) -> None:
        self._conn.executed.append(sql)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return []


class _Conn:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        return None


class _CountingRewriter:
    def __init__(self, sub_queries: tuple[Any, ...] | None = None) -> None:
        self.calls = 0
        self._sub_queries = sub_queries

    def rewrite(self, query: str) -> RewriteResult:
        self.calls += 1
        return RewriteResult(sub_queries=self._sub_queries or ("ok",), reasoning="test")


def _retriever() -> tuple[Retriever, _Conn, _CountingEmbedder]:
    conn = _Conn()
    embedder = _CountingEmbedder()
    return Retriever(conn, embedder), conn, embedder


def _type_name(value: Any) -> str:
    return type(value).__name__


# ---------------------------------------------------------------------------
# Retriever.search / rerank_pool / rewriter sub-queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_search_refuses_a_non_str_query_before_embedding_or_sql(name: str) -> None:
    query = _ALL_NON_STR[name]
    retriever, conn, embedder = _retriever()
    with pytest.raises(ValueError, match=rf"^query must be a string; got {_type_name(query)}$"):
        retriever.search(query, k=3)
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_search_refuses_a_non_str_query_before_the_rewriter(name: str) -> None:
    # The worst arm on main: the rewriter replaced the query with strings, so
    # the search ran to completion on a list query.
    retriever, conn, embedder = _retriever()
    rewriter = _CountingRewriter()
    with pytest.raises(ValueError, match=r"^query must be a string; got "):
        retriever.search(_ALL_NON_STR[name], k=3, rewriter=rewriter)
    assert rewriter.calls == 0
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_rerank_pool_refuses_a_non_str_query_before_embedding(name: str) -> None:
    retriever, conn, embedder = _retriever()
    with pytest.raises(ValueError, match=r"^query must be a string; got "):
        retriever.rerank_pool(_ALL_NON_STR[name], k=3)
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_NON_STR))
def test_a_non_str_second_sub_query_is_refused_before_the_first_is_embedded(name: str) -> None:
    retriever, conn, embedder = _retriever()
    rewriter = _CountingRewriter(("who wrote Hamlet", _NON_STR[name]))
    with pytest.raises(ValueError, match=r"^rewriter sub_queries\[1\] must be a string; got "):
        retriever.search("who wrote Hamlet and Macbeth", k=3, rewriter=rewriter)
    assert embedder.calls == []
    assert conn.executed == []


@pytest.mark.parametrize("name", sorted(_NON_STR))
def test_a_non_str_single_sub_query_is_refused_before_embedding(name: str) -> None:
    retriever, conn, embedder = _retriever()
    rewriter = _CountingRewriter((_NON_STR[name],))
    with pytest.raises(ValueError, match=r"^rewriter sub_queries\[0\] must be a string; got "):
        retriever.search("who wrote Hamlet", k=3, rewriter=rewriter)
    assert embedder.calls == []


def test_control_empty_str_query_keeps_its_message() -> None:
    retriever, _, embedder = _retriever()
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        retriever.search("", k=3)
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        retriever.rerank_pool("", k=3)
    assert embedder.calls == []


def test_control_str_subclass_query_still_runs_both_channels() -> None:
    retriever, conn, embedder = _retriever()
    assert retriever.search(_Text("who wrote Macbeth"), k=3) == []
    assert embedder.calls == ["who wrote Macbeth"]
    assert len(conn.executed) == 2
    rewriter = _CountingRewriter(("who wrote Hamlet", _Text("who wrote Macbeth")))
    assert retriever.search("q", k=3, rewriter=rewriter) == []
    assert rewriter.calls == 1


# ---------------------------------------------------------------------------
# Document.text / Indexer.add_documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_document_refuses_non_str_text(name: str) -> None:
    text = _ALL_NON_STR[name]
    with pytest.raises(
        ValueError, match=rf"^Document\.text must be a string; got {_type_name(text)}$"
    ):
        Document("d1", text)


def test_a_non_str_text_document_never_reaches_add_documents() -> None:
    # The issue's batch: three good documents and a list-text one. On main the
    # fourth constructed, and add_documents embedded all four before writing.
    good = [Document(f"d{i}", f"text {i}") for i in range(3)]
    with pytest.raises(ValueError, match=r"^Document\.text must be a string; got list$"):
        good.append(Document("d3", ["a", "list"]))  # type: ignore[arg-type]
    conn, embedder = _Conn(), _CountingEmbedder()
    assert Indexer(conn, embedder).add_documents(good) == 3
    assert embedder.calls == ["text 0", "text 1", "text 2"]


def test_control_str_subclass_text_and_empty_text_message() -> None:
    assert Document("d1", _Text("chunk")).text == "chunk"
    with pytest.raises(ValueError, match=r"^Document\.text must be non-empty$"):
        Document("d1", "")


# ---------------------------------------------------------------------------
# Paid seams that take a query directly
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise RuntimeError("the fake client was called")


class _FakeAnthropic:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


_ONE_CHUNK = [RetrievalResult("d1", "Refunds take 5 days.", {}, 1.0, {})]


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_anthropic_rewriter_refuses_a_non_str_query_before_the_api(name: str) -> None:
    client = _FakeAnthropic()
    with pytest.raises(ValueError, match=r"^query must be a string; got "):
        AnthropicRewriter(client=client).rewrite(_ALL_NON_STR[name])
    assert client.messages.calls == []


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_anthropic_generator_refuses_a_non_str_query_before_the_api(name: str) -> None:
    client = _FakeAnthropic()
    with pytest.raises(ValueError, match=r"^query must be a string; got "):
        AnthropicGenerator(client=client).generate(_ALL_NON_STR[name], _ONE_CHUNK, threshold=0.0)
    assert client.messages.calls == []


def test_control_anthropic_seams_still_call_the_api_for_a_str_query() -> None:
    for call in (
        lambda c: AnthropicRewriter(client=c).rewrite(_Text("refund policy")),
        lambda c: AnthropicGenerator(client=c).generate(
            _Text("refund policy"), _ONE_CHUNK, threshold=0.0
        ),
    ):
        client = _FakeAnthropic()
        with pytest.raises(RuntimeError, match="the fake client was called"):
            call(client)
        assert len(client.messages.calls) == 1


class _FakeCohereClient:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def rerank(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise RuntimeError("the fake client was called")


def _cohere(client: _FakeCohereClient) -> CohereReranker:
    rr = CohereReranker.__new__(CohereReranker)
    rr.client = client
    rr.model = "rerank-test"
    rr.batch_size = 100
    rr.timeout_s = 5.0
    return rr


@pytest.mark.parametrize("name", sorted(_NON_STR))
def test_cohere_reranker_refuses_a_non_str_query_before_the_api(name: str) -> None:
    client = _FakeCohereClient()
    with pytest.raises(ValueError, match=r"^query must be a string; got "):
        _cohere(client).rerank(_NON_STR[name], [Candidate("d1", "text", {})])
    assert client.calls == []


def test_control_cohere_reranker_still_calls_the_api_for_a_str_query() -> None:
    client = _FakeCohereClient()
    with pytest.raises(RuntimeError, match="the fake client was called"):
        _cohere(client).rerank(_Text("refund"), [Candidate("d1", "text", {})])
    assert len(client.calls) == 1


class _CountingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, k: int = 5, *, reranker: Any = None) -> list[RetrievalResult]:
        self.calls += 1
        return []


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_streaming_pipeline_refuses_a_non_str_query_before_the_first_event(name: str) -> None:
    retriever = _CountingRetriever()
    events = StreamingPipeline(retriever).run(_ALL_NON_STR[name], k=3)
    with pytest.raises(ValueError, match=r"^query must be a string; got "):
        next(events)
    assert retriever.calls == 0


def test_control_streaming_pipeline_runs_a_str_subclass_query() -> None:
    retriever = _CountingRetriever()
    kinds = [e.type for e in StreamingPipeline(retriever).run(_Text("refund"), k=3)]
    assert kinds[0] == "retrieving"
    assert retriever.calls == 1
