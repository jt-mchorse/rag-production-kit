"""The terminal `error` event survives an exception whose `__str__` raises (#251).

`run()`'s `except` built `{"message": str(e)}`. When `str(e)` raised, the
generator raised out after `retrieving` -- no `error`, no `done` -- breaking
the class's promise that an SSE client always sees a clean terminal frame.
Measured on `2ff3f6f`: events `['retrieving']`, then `RuntimeError: str broke`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag_kit.streaming import StreamingPipeline, to_sse


class _Unstringable(RuntimeError):
    def __str__(self) -> str:
        raise RuntimeError("str broke")


class _FailingRetriever:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def search(self, query: str, k: int = 5, **kwargs: object) -> list:
        raise self.exc


class _OkRetriever:
    def search(self, query: str, k: int = 5, **kwargs: object) -> list:
        return []


def _failing_stream(query: str, results: object) -> Iterator[str]:
    yield "partial"
    raise _Unstringable()


@pytest.mark.parametrize("where", ["retriever", "token-stream"])
def test_an_unstringable_exception_still_ends_in_one_error_event(where: str) -> None:
    if where == "retriever":
        pipeline = StreamingPipeline(_FailingRetriever(_Unstringable()))
    else:
        pipeline = StreamingPipeline(_OkRetriever(), token_stream=_failing_stream)
    events = list(pipeline.run("q"))
    assert events[-1].type == "error"
    assert [e.type for e in events].count("error") == 1
    assert "done" not in [e.type for e in events]
    assert events[-1].payload["exception"] == "_Unstringable"
    assert events[-1].payload["message"] == "<unstringable object>"
    # And the frame reaches the wire.
    to_sse(events[-1]).encode("utf-8")


def test_an_ordinary_exception_keeps_its_message() -> None:
    events = list(StreamingPipeline(_FailingRetriever(ValueError("bad k"))).run("q"))
    assert events[-1].type == "error"
    assert events[-1].payload == {"message": "bad k", "exception": "ValueError"}
