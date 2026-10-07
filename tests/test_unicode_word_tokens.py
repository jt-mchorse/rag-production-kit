"""The dep-free lexical paths tokenize non-Latin text (#285).

`LexicalOverlapReranker` used `[A-Za-z0-9]+` and the hermetic eval retriever
`[a-z0-9]+`. Measured on `main` by a hunt agent and re-run here:

    query 'Python ошибка импорта'
      Russian chunk containing all three words  -> 0.99902, rank 2
      'Python release notes'                    -> 0.99905, rank 1
    'café crème' -> ['caf', 'cr', 'me'], so a chunk reading 'caf cr me'
                    outranked 'Le café crème'.
"""

from __future__ import annotations

import unicodedata

import pytest

from rag_kit.reranker import Candidate, LexicalOverlapReranker
from rag_kit.text import word_tokens


def _c(eid: str, text: str) -> Candidate:
    return Candidate(external_id=eid, text=text, metadata={})


def test_a_cyrillic_query_ranks_the_chunk_that_contains_it_first() -> None:
    ranked = LexicalOverlapReranker().rerank(
        "Python ошибка импорта",
        [
            _c("en", "Python release notes"),
            _c("ru", "В Python ошибка импорта возникает, когда модуль не найден."),
        ],
    )
    assert [r.external_id for r in ranked] == ["ru", "en"]


def test_accented_words_are_whole_tokens() -> None:
    ranked = LexicalOverlapReranker().rerank(
        "café crème", [_c("junk", "caf cr me"), _c("real", "Le café crème du matin")]
    )
    assert ranked[0].external_id == "real"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python ошибка импорта", ["python", "ошибка", "импорта"]),
        ("café crème", ["café", "crème"]),
        ("नमस्ते दुनिया", ["नमस्ते", "दुनिया"]),
        ("東京 タワー", ["東京", "タワー"]),
    ],
)
def test_word_tokens(text: str, expected: list[str]) -> None:
    assert word_tokens(text) == expected


def test_nfc_and_nfd_are_the_same_tokens() -> None:
    assert word_tokens(unicodedata.normalize("NFD", "café")) == word_tokens("café") == ["café"]


@pytest.mark.parametrize(
    "text",
    ["Hybrid retrieval: BM25 + dense_vectors (v2)!", "", "   ", "a_b-c.d", "Python 3.12 release"],
)
def test_ascii_is_exactly_the_old_patterns(text: str) -> None:
    import re

    assert (
        word_tokens(text)
        == re.findall(r"[A-Za-z0-9]+", text.lower())
        == re.findall(r"[a-z0-9]+", text.lower())
    )


def test_the_hermetic_eval_retriever_tokenizes_the_same_way() -> None:
    from evals.run_eval import _tokens

    assert _tokens("Python ошибка импорта") == ["python", "ошибка", "импорта"]
