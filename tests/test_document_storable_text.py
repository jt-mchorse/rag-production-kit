"""A Document Postgres cannot store is refused before anything is embedded (#308).

`Document` accepted a NUL or a lone surrogate in its id, text or metadata, and
`Indexer.add_documents` embeds the whole batch before writing. Measured on
Postgres 17.6 (a text domain standing in for `vector`) with a counting embedder:

    1000 good + text with \\x00     DataError (cannot contain NUL)        embed calls 1001
    500 good + metadata "a\\x00b"   UntranslatableCharacter               embed calls 501
    200 good + text "a\\ud800b"     UnicodeEncodeError                    embed calls 201
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_kit.embedder import HashEmbedder
from rag_kit.indexer import Document, Indexer


@pytest.mark.parametrize(
    ("args", "where"),
    [
        (("doc1", "binary-ish \x00 chunk"), "Document.text"),
        (("doc1", "a\ud800b"), "Document.text"),
        (("doc1", "\udfff trailing low half"), "Document.text"),
        (("id\x00x", "ok"), "Document.external_id"),
        (("id\udc80", "ok"), "Document.external_id"),
        (("doc1", "ok", {"src": "a\x00b"}), "Document.metadata['src']"),
        (("doc1", "ok", {"k": "a\udc80"}), "Document.metadata['k']"),
        (("doc1", "ok", {"a\x00": 1}), "Document.metadata key"),
        (("doc1", "ok", {"tags": ["fine", "bad\x00"]}), "Document.metadata['tags'][1]"),
        (
            ("doc1", "ok", {"nested": {"deep": ("x", "y\ud800")}}),
            "Document.metadata['nested']['deep'][1]",
        ),
    ],
)
def test_unstorable_text_is_refused_at_construction(args: tuple[Any, ...], where: str) -> None:
    with pytest.raises(ValueError, match="which Postgres cannot store") as e:
        Document(*args)
    assert str(e.value).startswith(where)


@pytest.mark.parametrize(
    "text",
    [
        "naïve café",
        "日本語のテキスト",
        "emoji \U0001f600 ok",
        "line one\nline two\ttab",
        "  separator",
    ],
)
def test_ordinary_text_still_constructs(text: str) -> None:
    assert Document("doc1", text, {"note": text, "n": 1, "flag": True, "none": None}).text == text


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
        yield Document("doc-nul", "binary-ish \x00 chunk")

    with pytest.raises(ValueError, match="NUL"):
        Indexer(_NoConn(), _Counting()).add_documents(batch())
    assert _Counting.calls == 0
