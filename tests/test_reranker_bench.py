"""`scripts/bench_reranker.py` and the table it publishes (#234).

The script measures `LexicalOverlapReranker`'s recall@k lift over the
retriever's own order on two synthetic fixtures. It is deterministic (no
timings in the table), so the committed section of `docs/benchmarks.md` is
locked to a live re-render, and the Status row's quoted value to the row it
summarises.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.bench_reranker as bench  # noqa: E402
import scripts.bench_rewriter as rewriter_bench  # noqa: E402
from rag_kit.reranker import Candidate, ScoredCandidate  # noqa: E402

_DOC = (_REPO_ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")


def _committed_region() -> str:
    m = re.search(
        r"<!-- bench-reranker:table:begin -->\n(.*?)\n<!-- bench-reranker:table:end -->",
        _DOC,
        re.S,
    )
    assert m, "docs/benchmarks.md lost the bench-reranker markers"
    return m.group(1)


def test_the_committed_table_is_the_live_render() -> None:
    fx, rows = bench.run()
    rendered = bench.render_markdown(fx, rows, candidates=bench.DEFAULT_CANDIDATES)
    assert _committed_region() == rendered.rstrip(), (
        "regenerate with: python -m scripts.bench_reranker --output md"
    )


def test_the_status_row_quotes_the_row_it_summarises() -> None:
    _, rows = bench.run()
    (row,) = [r for r in rows if r.fixture == "multi-hop" and r.k == 3]
    status = next(line for line in _DOC.splitlines() if "Reranker quality lift" in line)
    assert f"{row.recall_reranked - row.recall_fused:+.3f} recall@3" in status


def test_the_fixtures_are_imported_not_copied() -> None:
    """One corpus, one question set, one retriever: the README's rewriter
    table and this one measure the same thing."""
    assert bench._MULTI_HOP_CORPUS is rewriter_bench._CORPUS
    assert bench._MULTI_HOP_QUERIES is rewriter_bench._QUERIES
    assert bench._multi_hop_retrieve is rewriter_bench._retrieve_in_memory


class _Identity:
    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[ScoredCandidate]:
        return [
            ScoredCandidate(
                external_id=c.external_id,
                text=c.text,
                metadata=c.metadata,
                rerank_score=float(-i),
                rerank_rank=i + 1,
            )
            for i, c in enumerate(candidates)
        ]


class _Reverse(_Identity):
    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[ScoredCandidate]:
        return super().rerank(query, list(reversed(candidates)))


def test_a_reranker_that_changes_nothing_measures_no_lift() -> None:
    """Vacuity control: the measurement must be able to report zero."""
    for fixture in bench.fixtures():
        for row in bench.measure(fixture, _Identity(), ks=(1, 3, 5), candidates=10):
            assert row.recall_reranked == row.recall_fused
            assert (row.improved, row.regressed) == (0, 0)


def test_a_reranker_that_reverses_the_order_measures_a_regression() -> None:
    """And it must be able to report harm, on the fixture that has headroom."""
    (multi_hop, eval_set) = bench.fixtures()
    rows = bench.measure(multi_hop, _Reverse(), ks=(1, 3), candidates=10)
    assert all(r.recall_reranked < r.recall_fused for r in rows)
    assert all(r.regressed > 0 for r in rows)
    # The saturated fixture regresses too: saturation bounds lift, not harm.
    rows = bench.measure(eval_set, _Reverse(), ks=(1,), candidates=10)
    assert rows[0].recall_reranked < 1.0


def test_the_saturation_note_is_derived_from_the_numbers() -> None:
    fx, rows = bench.run()
    rendered = bench.render_markdown(fx, rows, candidates=10)
    eval_line = next(line for line in rendered.splitlines() if line.startswith("- **eval"))
    multi_line = next(line for line in rendered.splitlines() if line.startswith("- **multi"))
    assert "cannot show an improvement" in eval_line
    assert "cannot show an improvement" not in multi_line


def test_the_saturation_note_does_not_deny_what_the_bench_can_measure() -> None:
    """#243: the note said "cannot show lift in either direction" while
    `test_a_reranker_that_reverses_the_order_measures_a_regression` measures a
    regression on this exact fixture. Ask the bench, then read the note."""
    (_, eval_set) = bench.fixtures()
    (row,) = bench.measure(eval_set, _Reverse(), ks=(1,), candidates=10)
    assert row.regressed > 0, "the premise: harm is measurable on the saturated fixture"
    fx, rows = bench.run()
    rendered = bench.render_markdown(fx, rows, candidates=10)
    eval_line = next(line for line in rendered.splitlines() if line.startswith("- **eval"))
    assert "either direction" not in eval_line
    assert "a regression would still show" in eval_line
    assert "either direction" not in bench.__doc__


def test_a_repeated_k_is_refused_at_parse_time(capsys: pytest.CaptureFixture[str]) -> None:
    """#243: `--k 1,1` reported `n_queries: 16` for an 8-question fixture."""
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--k", "1,1", "--output", "json"])
    assert excinfo.value.code == 2
    assert "each k may appear once" in capsys.readouterr().err


@pytest.mark.parametrize("ks", [(1, 1), (3, 1, 3), (5, 5, 5)])
def test_a_repeated_k_is_refused_by_measure(ks: tuple[int, ...]) -> None:
    """The library seam too: a caller of `measure` would double-count as well."""
    (multi_hop, _) = bench.fixtures()
    with pytest.raises(ValueError, match="each k may appear once"):
        bench.measure(multi_hop, _Identity(), ks=ks, candidates=10)


def test_n_queries_is_each_fixtures_question_count() -> None:
    fx, rows = bench.run()
    for f in fx:
        assert {r.n_queries for r in rows if r.fixture == f.name} == {len(f.queries)}


def test_too_few_candidates_is_an_operator_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert bench.main(["--k", "1,5", "--candidates", "3"]) == 2
    assert "--candidates (3) must be >= the largest k (5)" in capsys.readouterr().err


def test_the_readme_sentence_quotes_the_rows() -> None:
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    sentence = readme.split("**Reranker lift over fused-only**", 1)[1].split("\n\n", 1)[0]
    sentence = " ".join(sentence.split())
    _, rows = bench.run()
    by_k = {r.k: r for r in rows if r.fixture == "multi-hop"}
    assert (
        f"recall@3 on the same multi-hop fixture from {by_k[3].recall_fused:.3f} to {by_k[3].recall_reranked:.3f}"
        in sentence
    )
    assert f"recall@1 from {by_k[1].recall_fused:.3f} to {by_k[1].recall_reranked:.3f}" in sentence
