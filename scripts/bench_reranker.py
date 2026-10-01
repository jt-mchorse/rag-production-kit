"""Measure recall@k lift from reranking over fused-only retrieval (#234).

`docs/benchmarks.md` listed "Reranker quality lift over fused-only" as pending
against #2, which shipped the reranking layer and closed without a number. This
is that number, hermetically: no API key, no Postgres.

For each query, the in-memory token-overlap retriever returns ``--candidates``
ids (the over-fetch a reranker needs something to reorder). Two paths then take
the top ``k``:

1. **fused-only** -- the retriever's own order.
2. **reranked** -- the same candidates reordered by ``LexicalOverlapReranker``.

Recall@k is the fraction of a query's gold ids in its top k, averaged over the
fixture. Both fixtures and both retrievers are *imported* from where they live
(`scripts/bench_rewriter.py`, `evals/run_eval.py`), not copied, so this measures
the same corpora the README's other tables do.

**What the number is, and is not.** `LexicalOverlapReranker` is the dep-free
stand-in the pipeline uses in CI; it scores query-token *coverage* where the
retriever scores overlap *density*, and that difference is all the lift can come
from. This is the lift of that stand-in on two synthetic fixtures, not a claim
about a cross-encoder. The eval golden set is saturated -- fused-only recall is
already 1.000 at every k -- so it cannot show lift in either direction, and the
rendered table says so from the numbers rather than dropping the row.

Usage:
    python -m scripts.bench_reranker
    python -m scripts.bench_reranker --output md
    python -m scripts.bench_reranker --k 1,3,5 --candidates 10 --output json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.run_eval import CORPUS_PATH, DATASET_PATH, _load_corpus, _load_dataset  # noqa: E402
from evals.run_eval import _retrieve_in_memory as _eval_retrieve  # noqa: E402
from rag_kit.reranker import Candidate, LexicalOverlapReranker, Reranker  # noqa: E402
from scripts.bench_rewriter import _CORPUS as _MULTI_HOP_CORPUS  # noqa: E402
from scripts.bench_rewriter import _QUERIES as _MULTI_HOP_QUERIES  # noqa: E402
from scripts.bench_rewriter import _recall  # noqa: E402
from scripts.bench_rewriter import _retrieve_in_memory as _multi_hop_retrieve  # noqa: E402

DEFAULT_KS = (1, 3, 5)
DEFAULT_CANDIDATES = 10


@dataclass(frozen=True)
class Fixture:
    name: str
    description: str
    queries: tuple[tuple[str, tuple[str, ...]], ...]  # (query, gold ids)
    retrieve: Callable[[str, int], list[tuple[str, str]]]  # -> [(id, text)] in rank order


def _multi_hop() -> Fixture:
    text = dict(_MULTI_HOP_CORPUS)

    def retrieve(query: str, n: int) -> list[tuple[str, str]]:
        return [(i, text[i]) for i in _multi_hop_retrieve(query, _MULTI_HOP_CORPUS, n)]

    return Fixture(
        name="multi-hop",
        description=f"{len(_MULTI_HOP_CORPUS)} chunks, {len(_MULTI_HOP_QUERIES)} questions "
        "(scripts/bench_rewriter.py)",
        queries=tuple((q.query, q.gold) for q in _MULTI_HOP_QUERIES),
        retrieve=retrieve,
    )


def _eval_golden_set() -> Fixture:
    corpus = _load_corpus(CORPUS_PATH)
    examples = [e for e in _load_dataset(DATASET_PATH) if e.gold_chunk_ids]

    def retrieve(query: str, n: int) -> list[tuple[str, str]]:
        return [(r.external_id, r.text) for r in _eval_retrieve(query, corpus, n)]

    return Fixture(
        name="eval golden set",
        description=f"{len(corpus)} chunks, {len(examples)} questions (evals/dataset, rag-qa-v0.1)",
        queries=tuple((e.input, e.gold_chunk_ids) for e in examples),
        retrieve=retrieve,
    )


def fixtures() -> list[Fixture]:
    return [_multi_hop(), _eval_golden_set()]


@dataclass(frozen=True)
class Row:
    fixture: str
    k: int
    n_queries: int
    recall_fused: float
    recall_reranked: float
    improved: int
    regressed: int


def measure(
    fixture: Fixture, reranker: Reranker, *, ks: Sequence[int], candidates: int
) -> list[Row]:
    if candidates < max(ks):
        raise ValueError(f"--candidates ({candidates}) must be >= the largest k ({max(ks)})")
    per_k: dict[int, list[tuple[float, float]]] = {k: [] for k in ks}
    for query, gold in fixture.queries:
        pool = fixture.retrieve(query, candidates)
        fused = [i for i, _ in pool]
        reranked = [
            s.external_id
            for s in reranker.rerank(
                query, [Candidate(external_id=i, text=t, metadata={}) for i, t in pool]
            )
        ]
        for k in ks:
            per_k[k].append((_recall(fused[:k], gold), _recall(reranked[:k], gold)))
    return [
        Row(
            fixture=fixture.name,
            k=k,
            n_queries=len(pairs),
            recall_fused=statistics.fmean(f for f, _ in pairs),
            recall_reranked=statistics.fmean(r for _, r in pairs),
            improved=sum(1 for f, r in pairs if r > f),
            regressed=sum(1 for f, r in pairs if r < f),
        )
        for k, pairs in per_k.items()
    ]


def run(
    *, ks: Sequence[int] = DEFAULT_KS, candidates: int = DEFAULT_CANDIDATES
) -> tuple[list[Fixture], list[Row]]:
    fx = fixtures()
    reranker = LexicalOverlapReranker()
    return fx, [row for f in fx for row in measure(f, reranker, ks=ks, candidates=candidates)]


def _saturated(rows: Sequence[Row], fixture: str) -> bool:
    return all(r.recall_fused == 1.0 for r in rows if r.fixture == fixture)


def render_markdown(fx: Sequence[Fixture], rows: Sequence[Row], *, candidates: int) -> str:
    lines = [
        f"Over-fetch {candidates} candidates from the in-memory token-overlap retriever, "
        "then take the top k in the retriever's order (**fused-only**) or after "
        "`LexicalOverlapReranker` (**reranked**).",
        "",
        "| fixture | k | fused-only recall@k | reranked recall@k | Δ | improved / regressed |",
        "| ------- | -: | ------------------: | ----------------: | -: | -------------------- |",
    ]
    for r in rows:
        delta = r.recall_reranked - r.recall_fused
        lines.append(
            f"| {r.fixture} | {r.k} | {r.recall_fused:.3f} | {r.recall_reranked:.3f} | "
            f"{delta:+.3f} | {r.improved} / {r.regressed} |"
        )
    lines.append("")
    for f in fx:
        note = f"- **{f.name}**: {f.description}."
        if _saturated(rows, f.name):
            note += (
                " Fused-only recall is already 1.000 at every k, so this fixture "
                "cannot show lift in either direction."
            )
        lines.append(note)
    return "\n".join(lines) + "\n"


def _parse_ks(raw: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(part) for part in raw.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--k must be comma-separated integers; got {raw!r}"
        ) from None
    if not ks or any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError(f"every k must be >= 1; got {raw!r}")
    return ks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=_parse_ks, default=DEFAULT_KS, help="default: 1,3,5")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", choices=("text", "md", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        fx, rows = run(ks=args.k, candidates=args.candidates)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.output == "json":
        print(json.dumps([asdict(r) for r in rows], indent=2))
    elif args.output == "md":
        print(render_markdown(fx, rows, candidates=args.candidates), end="")
    else:
        for r in rows:
            print(
                f"{r.fixture:16} k={r.k}  fused={r.recall_fused:.3f}  "
                f"reranked={r.recall_reranked:.3f}  +{r.improved}/-{r.regressed}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
