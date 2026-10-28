"""The SSE path reranks the pool the library reranks (#417).

`StreamingPipeline` asked `retriever.search` for `k * RERANK_OVERFETCH` rows,
and `Retriever._hybrid_search` over-fetched every channel by
`_CANDIDATE_MULTIPLIER` again, so with a reranker each channel was read 16k
deep instead of the library's 4k. The fused pool differed, and the same query
cited a different chunk through `/api/stream` than through
`Retriever.search(..., reranker=...)`. Every streaming test used a fake
retriever with no internal over-fetch, so none could see it. These drive the
real `Retriever` over a fake connection that honours each SQL `LIMIT`.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from rag_kit.reranker import LexicalOverlapReranker
from rag_kit.retriever import Retriever
from rag_kit.streaming import StreamingPipeline


class _Conn:
    """Returns per-channel rankings, cut at the `LIMIT` the SQL asks for."""

    def __init__(self, lexical: list[str], dense: list[str], text: dict[str, str]) -> None:
        self.lexical, self.dense, self.text = lexical, dense, text
        self.limits: list[int] = []

    def transaction(self) -> Any:
        return _Null()

    def cursor(self) -> _Cur:
        return _Cur(self)


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


class _Cur:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn
        self.rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> _Cur:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        n = params[-1]
        self.conn.limits.append(n)
        if "ts_rank_cd" in sql:
            self.rows = [(i, self.conn.text[i], {}) for i in self.conn.lexical[:n]]
        else:
            self.rows = [
                (i, self.conn.text[i], {}, float(r)) for r, i in enumerate(self.conn.dense[:n])
            ]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _Embedder:
    dim = 64

    def embed(self, text: str) -> list[float]:
        return [0.1] * 64


def _issue_conn() -> _Conn:
    lexical = ["L1", "L2", "X", "L4", "L5", "L6", "L7", "L8"]
    dense = ["D1", "D2", "D3", "D4", "X", "D6", "D7", "D8"]
    text = {i: "unrelated filler" for i in {*lexical, *dense}}
    text["X"] = "refund window policy"
    return _Conn(lexical, dense, text)


def _streamed(retriever: Any, query: str, k: int) -> tuple[list[dict], list[dict]]:
    events = list(StreamingPipeline(retriever, reranker=LexicalOverlapReranker()).run(query, k=k))
    by_type = {e.type: e.payload for e in events}
    assert "error" not in by_type, by_type.get("error")
    return by_type["retrieved"]["chunks"], by_type["reranked"]["chunks"]


def _view(r: Any) -> tuple[Any, ...]:
    return (r.external_id, r.fused_score, r.ranks, r.rerank_score, r.rerank_rank)


def _event_view(c: dict) -> tuple[Any, ...]:
    return (c["external_id"], c["fused_score"], c["ranks"], c["rerank_score"], c["rerank_rank"])


def test_the_issue_case_cites_the_same_chunk_on_both_paths() -> None:
    conn = _issue_conn()
    lib = Retriever(conn, _Embedder()).search(
        "refund window", k=1, reranker=LexicalOverlapReranker()
    )
    lib_limits, conn.limits = conn.limits, []
    _, reranked = _streamed(Retriever(conn, _Embedder()), "refund window", 1)
    assert conn.limits == lib_limits == [4, 4]
    assert [_event_view(c) for c in reranked] == [_view(r) for r in lib]


@pytest.mark.parametrize("seed", range(60))
def test_streaming_reranks_exactly_the_library_pool(seed: int) -> None:
    rng = random.Random(seed)
    words = ["refund", "window", "policy", "shipping", "time", "filler", "other", "text"]
    ids = [f"d{i:02d}" for i in range(rng.randint(3, 40))]
    text = {i: " ".join(rng.sample(words, rng.randint(1, 4))) for i in ids}
    lexical = rng.sample(ids, rng.randint(0, len(ids)))
    dense = rng.sample(ids, len(ids))
    k = rng.randint(1, 6)
    query = " ".join(rng.sample(words, 2))

    conn = _Conn(lexical, dense, text)
    retriever = Retriever(conn, _Embedder())
    lib = retriever.search(query, k=k, reranker=LexicalOverlapReranker())
    pool = retriever.rerank_pool(query, k)
    retrieved, reranked = _streamed(retriever, query, k)

    assert [_event_view(c) for c in reranked] == [_view(r) for r in lib]
    assert [c["external_id"] for c in retrieved] == [r.external_id for r in pool]


def test_rerank_pool_is_what_search_reranks() -> None:
    """One definition: the pool's order and size are search's pre-rerank cut."""
    conn = _issue_conn()
    retriever = Retriever(conn, _Embedder())
    pool = retriever.rerank_pool("refund window", 2)
    assert conn.limits == [8, 8]  # k * _CANDIDATE_MULTIPLIER, not squared
    assert len(pool) == 8
    # Same channel depth as `search(k=2)`, whose no-reranker answer is the head.
    assert [_view(r) for r in pool[:2]] == [
        _view(r) for r in retriever.search("refund window", k=2)
    ]
    assert all(r.rerank_score is None for r in pool)


@pytest.mark.parametrize(("query", "k"), [("", 3), ("q", 0), ("q", True), ("q", -1)])
def test_rerank_pool_validates_like_search(query: str, k: Any) -> None:
    with pytest.raises(ValueError, match="k must be|query must be"):
        Retriever(_issue_conn(), _Embedder()).rerank_pool(query, k)


class _NoOverfetchRetriever:
    """A duck-typed retriever with no `rerank_pool`: the old call is kept."""

    def __init__(self) -> None:
        self.asked: list[int] = []

    def search(self, query: str, k: int = 5, *, reranker: Any = None) -> list[Any]:
        self.asked.append(k)
        return []


def test_a_retriever_without_rerank_pool_is_still_asked_for_k_times_the_overfetch() -> None:
    fake = _NoOverfetchRetriever()
    list(StreamingPipeline(fake, reranker=LexicalOverlapReranker()).run("q", k=3))
    assert fake.asked == [3 * StreamingPipeline.RERANK_OVERFETCH]


def test_without_a_reranker_streaming_reads_k_as_before() -> None:
    conn = _issue_conn()
    list(StreamingPipeline(Retriever(conn, _Embedder())).run("refund window", k=2))
    assert conn.limits == [8, 8]
