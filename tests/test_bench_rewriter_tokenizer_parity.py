"""`bench_rewriter`'s retriever ranks as its twin in `evals/run_eval.py` does (#312).

The README says the rewriter bench runs "against the same in-memory
token-overlap retriever the eval suite uses". #285 made the eval copy
Unicode-aware and left this one on `[a-z0-9]+`, so a Cyrillic query
tokenized to nothing and the bench ranked by tie-break order. The two copies
are kept separate (different corpus types), so the parity is checked by
ranking the same corpora through both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.run_eval import _Chunk  # noqa: E402
from evals.run_eval import _retrieve_in_memory as eval_retrieve  # noqa: E402
from scripts.bench_rewriter import _CORPUS  # noqa: E402
from scripts.bench_rewriter import _retrieve_in_memory as bench_retrieve  # noqa: E402

_MIXED = (
    ("a-en", "Postgres tuning: shared_buffers and work_mem."),
    ("b-ru", "Ошибка импорта модуля возникает, если пакет не установлен."),
    ("c-fr", "Le café crème du matin, servi à huit heures."),
    ("d-junk", "caf cr me"),
    ("e-ja", "東京 は 日本 の 首都 です"),
    ("f-el", "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας."),
)


def _eval_order(query: str, corpus: tuple[tuple[str, str], ...], k: int) -> list[str]:
    chunks = [_Chunk(external_id=e, text=t) for e, t in corpus]
    return [r.external_id for r in eval_retrieve(query, chunks, k)]


def test_a_cyrillic_query_finds_its_chunk() -> None:
    assert bench_retrieve("ошибка импорта модуля", _MIXED, 1) == ["b-ru"]


@pytest.mark.parametrize(
    "query",
    [
        "ошибка импорта модуля",
        "café crème",
        "Αθήνα πρωτεύουσα",
        "東京 首都",
        "postgres work_mem tuning",
    ],
)
def test_the_bench_and_eval_retrievers_rank_a_mixed_corpus_identically(query: str) -> None:
    k = len(_MIXED)
    assert bench_retrieve(query, _MIXED, k) == _eval_order(query, _MIXED, k)


def test_the_shipped_fixture_ranks_identically_for_every_corpus_text() -> None:
    # Every chunk's own text as a query, over the bench's own ASCII corpus: the
    # case the published table rests on.
    k = len(_CORPUS)
    for _ext_id, text in _CORPUS:
        assert bench_retrieve(text, _CORPUS, k) == _eval_order(text, _CORPUS, k)
