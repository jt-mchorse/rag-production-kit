"""`Document.metadata` must be storable as Postgres JSON, checked before embedding (#318).

`metadata` is written as `Jsonb(...)`, which is `json.dumps`: NaN and Infinity
become tokens Postgres's json parser refuses (`SELECT '{"a": NaN}'::jsonb` is
`invalid input syntax for type json`), and a date, a set or a tuple key raise at
dump time. Each passed `Document()`, so the whole batch was embedded before the
write refused it, the cost #308 measured for a NUL.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import pytest

from rag_kit.embedder import HashEmbedder
from rag_kit.indexer import Document, Indexer


def _cyclic() -> dict[str, Any]:
    d: dict[str, Any] = {}
    d["self"] = d
    return d


@pytest.mark.parametrize(
    "metadata",
    [
        {"score": math.nan},
        {"score": math.inf},
        {"nested": {"deep": [1, -math.inf]}},
        {"when": datetime.date(2026, 1, 1)},
        {"tags": {1, 2}},
        {(1, 2): "tuple key"},
        _cyclic(),
    ],
    ids=["nan", "inf", "nested-neg-inf", "date", "set", "tuple-key", "cyclic"],
)
def test_unstorable_metadata_is_refused_at_construction(metadata: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Document.metadata cannot be stored as Postgres JSON"):
        Document("doc1", "text", metadata)


def test_ordinary_metadata_still_constructs() -> None:
    meta = {
        "n": 1,
        "f": 0.5,
        "flag": True,
        "none": None,
        "list": [1, "a"],
        "d": {"k": 2},
        3: "int key",
    }
    assert Document("doc1", "text", meta).metadata == meta


class _Counting(HashEmbedder):
    calls = 0

    def embed(self, text: str) -> list[float]:
        _Counting.calls += 1
        return super().embed(text)


class _NoConn:
    def cursor(self) -> Any:  # pragma: no cover - never reached
        raise AssertionError("nothing may be written")


def test_the_batch_is_never_embedded() -> None:
    _Counting.calls = 0

    def batch() -> Any:
        for i in range(1000):
            yield Document(f"doc{i}", f"chunk {i} text")
        yield Document("doc-nan", "a chunk", {"score": math.nan})

    with pytest.raises(ValueError, match="Postgres JSON"):
        Indexer(_NoConn(), _Counting()).add_documents(batch())
    assert _Counting.calls == 0
