"""Measure recall@k uplift from `TemplateRewriter` on a synthetic multi-hop corpus.

Runs the in-memory token-overlap retriever (same one used by
``evals/run_eval.py`` for hermetic CI) over a tiny synthetic corpus
that's been deliberately shaped so single-shot retrieval struggles on
multi-hop questions. Compares two paths:

1. Baseline: hybrid (single-shot) retrieval on the original query.
2. Rewriter: ``TemplateRewriter.rewrite(query)`` → per-sub-query
   retrieval → RRF fusion across sub-queries.

The script reports per-query gold-hit deltas and mean recall@k. Numbers
are real wall-clock measurements over a deterministic fixture; nothing
is baked in. Re-running on a different machine produces the same
recall numbers but different latencies — that's expected.

Usage:
    python -m scripts.bench_rewriter
    python -m scripts.bench_rewriter --k 5 --output md
    python -m scripts.bench_rewriter --output json

Real-LLM (`AnthropicRewriter`) numbers are intentionally out of scope
for this script — they need an operator-supplied ``ANTHROPIC_API_KEY``
budget and live in a separate workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.fusion import reciprocal_rank_fusion  # noqa: E402
from rag_kit.rewriter import Rewriter, TemplateRewriter  # noqa: E402

# ----------------------------------------------------------------------
# Synthetic multi-hop fixture
# ----------------------------------------------------------------------

# 18 chunks, each one fact. The multi-hop questions below are
# designed so each requires evidence from ≥2 distinct chunks. Token
# vocabulary is chosen so that single-shot retrieval on the combined
# question scatters across distractors that share weak overlap with
# both halves but support neither fully — the canonical failure mode
# decomposition is supposed to fix.
_CORPUS: tuple[tuple[str, str], ...] = (
    ("hq-anthropic", "Anthropic is headquartered in San Francisco."),
    ("hq-openai", "OpenAI is headquartered in San Francisco."),
    ("hq-cohere", "Cohere is headquartered in Toronto."),
    ("hq-mistral", "Mistral is headquartered in Paris."),
    ("founder-anthropic", "Dario Amodei and Daniela Amodei founded Anthropic in 2021."),
    ("founder-openai", "Sam Altman led the founding of OpenAI in 2015."),
    ("founder-cohere", "Aidan Gomez co-founded Cohere in 2019."),
    ("prior-amodei", "Before founding Anthropic, Dario Amodei was VP of Research at OpenAI."),
    ("prior-altman", "Before OpenAI, Sam Altman was president of Y Combinator."),
    ("prior-gomez", "Before founding Cohere, Aidan Gomez was a researcher at Google Brain."),
    ("model-anthropic", "Anthropic builds the Claude family of large language models."),
    ("model-openai", "OpenAI develops the GPT family of large language models."),
    ("model-cohere", "Cohere ships the Command family of large language models."),
    ("model-mistral", "Mistral releases open-weight large language models."),
    ("api-anthropic", "Anthropic exposes its Claude models via a public Messages API."),
    ("api-openai", "OpenAI exposes its GPT models via a public Chat Completions API."),
    ("api-cohere", "Cohere exposes its Command models via a public Chat API."),
    ("api-mistral", "Mistral exposes its open-weight models via la Plateforme."),
)


@dataclass(frozen=True)
class _Query:
    query: str
    gold: tuple[str, ...]


# Each query is multi-hop: its gold set is two chunks from distinct
# topic groups (e.g., founder + prior, HQ + model, etc.). Single-shot
# retrieval has to surface both with a single token-overlap pass — the
# rewriter splits into halves that each see one of the two facts.
_QUERIES: tuple[_Query, ...] = (
    _Query(
        "Who founded Anthropic and where did they work before?",
        ("founder-anthropic", "prior-amodei"),
    ),
    _Query(
        "Who founded Cohere and where did they work before?",
        ("founder-cohere", "prior-gomez"),
    ),
    _Query(
        "Where is OpenAI headquartered and what models do they build?",
        ("hq-openai", "model-openai"),
    ),
    _Query(
        "Where is Mistral headquartered and how do they ship their models?",
        ("hq-mistral", "api-mistral"),
    ),
    _Query(
        "Who founded OpenAI and where did they work before?",
        ("founder-openai", "prior-altman"),
    ),
    _Query(
        "What family of models does Anthropic build and how do they expose them?",
        ("model-anthropic", "api-anthropic"),
    ),
    _Query(
        "Compare Cohere with Mistral",
        ("hq-cohere", "hq-mistral"),
    ),
    _Query(
        "Where is Anthropic headquartered and who founded it?",
        ("hq-anthropic", "founder-anthropic"),
    ),
)


# ----------------------------------------------------------------------
# Token-overlap retriever (same shape as evals/run_eval.py)
# ----------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _retrieve_in_memory(query: str, corpus: Sequence[tuple[str, str]], k: int) -> list[str]:
    """Return top-k external_ids by token-overlap score, ties broken by id."""
    q_tokens = set(_tokens(query))
    scored: list[tuple[float, str]] = []
    for ext_id, text in corpus:
        c_tokens = _tokens(text)
        overlap = sum(1 for tok in c_tokens if tok in q_tokens)
        score = overlap / (len(c_tokens) + 1)
        scored.append((score, ext_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [ext_id for _, ext_id in scored[:k]]


# ----------------------------------------------------------------------
# Bench paths
# ----------------------------------------------------------------------


def _baseline_top_k(query: str, k: int) -> list[str]:
    return _retrieve_in_memory(query, _CORPUS, k)


def _rewriter_top_k(query: str, k: int, rewriter: Rewriter, over_fetch: int = 4) -> list[str]:
    """Single-shot if the rewriter chooses not to decompose; else per-sub-query + RRF fuse."""
    rewrite = rewriter.rewrite(query)
    if len(rewrite.sub_queries) <= 1:
        return _retrieve_in_memory(rewrite.sub_queries[0], _CORPUS, k)
    per_sub_k = max(k * over_fetch, k * 2)
    rankings: dict[str, list[str]] = {}
    for i, sq in enumerate(rewrite.sub_queries):
        rankings[f"subquery_{i}"] = _retrieve_in_memory(sq, _CORPUS, per_sub_k)
    fused = reciprocal_rank_fusion(rankings, k=60)
    return [ext_id for ext_id, _, _ in fused[:k]]


def _recall(top_ids: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for g in gold if g in top_ids)
    return hit / len(gold)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _QueryResult:
    query: str
    gold: tuple[str, ...]
    baseline_top: list[str]
    rewriter_top: list[str]
    baseline_recall: float
    rewriter_recall: float
    sub_queries: tuple[str, ...]


def _run(k: int) -> list[_QueryResult]:
    rewriter = TemplateRewriter()
    out: list[_QueryResult] = []
    for q in _QUERIES:
        base = _baseline_top_k(q.query, k)
        rew_split = rewriter.rewrite(q.query)
        rew = _rewriter_top_k(q.query, k, rewriter)
        out.append(
            _QueryResult(
                query=q.query,
                gold=q.gold,
                baseline_top=base,
                rewriter_top=rew,
                baseline_recall=_recall(base, q.gold),
                rewriter_recall=_recall(rew, q.gold),
                sub_queries=rew_split.sub_queries,
            )
        )
    return out


def _summary(results: Sequence[_QueryResult]) -> dict:
    base_recalls = [r.baseline_recall for r in results]
    rew_recalls = [r.rewriter_recall for r in results]
    return {
        "n_queries": len(results),
        "mean_recall_baseline": statistics.fmean(base_recalls) if base_recalls else 0.0,
        "mean_recall_rewriter": statistics.fmean(rew_recalls) if rew_recalls else 0.0,
        "queries_with_improvement": sum(
            1 for r in results if r.rewriter_recall > r.baseline_recall
        ),
        "queries_with_regression": sum(1 for r in results if r.rewriter_recall < r.baseline_recall),
    }


def _print_text(results: Sequence[_QueryResult], k: int, elapsed_ms: float) -> None:
    s = _summary(results)
    print(f"# bench_rewriter (k={k}, n={s['n_queries']}, elapsed={elapsed_ms:.1f}ms)")
    print()
    print("query | baseline_recall | rewriter_recall | delta")
    print("------|-----------------|-----------------|------")
    for r in results:
        delta = r.rewriter_recall - r.baseline_recall
        print(
            f"{r.query[:60]!r} | {r.baseline_recall:.2f} | {r.rewriter_recall:.2f} | {delta:+.2f}"
        )
    print()
    print(f"mean recall@{k} baseline: {s['mean_recall_baseline']:.3f}")
    print(f"mean recall@{k} rewriter: {s['mean_recall_rewriter']:.3f}")
    print(
        f"improvements/regressions: {s['queries_with_improvement']}/{s['queries_with_regression']}"
    )


def _print_md(results: Sequence[_QueryResult], k: int, elapsed_ms: float) -> None:
    s = _summary(results)
    print(f"## bench_rewriter (k={k}, n={s['n_queries']})\n")
    print(
        f"Mean recall@{k}: baseline **{s['mean_recall_baseline']:.3f}** → "
        f"rewriter **{s['mean_recall_rewriter']:.3f}** "
        f"(improvements: {s['queries_with_improvement']}, "
        f"regressions: {s['queries_with_regression']}, "
        f"elapsed: {elapsed_ms:.1f}ms)\n"
    )
    print("| Query | Baseline recall | Rewriter recall | Δ |")
    print("|---|---|---|---|")
    for r in results:
        delta = r.rewriter_recall - r.baseline_recall
        sign = "+" if delta >= 0 else ""
        print(
            f"| `{r.query}` | {r.baseline_recall:.2f} | {r.rewriter_recall:.2f} | "
            f"{sign}{delta:.2f} |"
        )


def _print_json(results: Sequence[_QueryResult], k: int, elapsed_ms: float) -> None:
    out = {
        "k": k,
        "elapsed_ms": round(elapsed_ms, 2),
        "summary": _summary(results),
        "queries": [
            {
                "query": r.query,
                "gold": list(r.gold),
                "sub_queries": list(r.sub_queries),
                "baseline_top": r.baseline_top,
                "rewriter_top": r.rewriter_top,
                "baseline_recall": round(r.baseline_recall, 4),
                "rewriter_recall": round(r.rewriter_recall, 4),
            }
            for r in results
        ],
    }
    print(json.dumps(out, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bench TemplateRewriter recall@k on multi-hop fixture."
    )
    parser.add_argument("--k", type=int, default=5, help="top-k cut for recall measurement")
    parser.add_argument(
        "--output", choices=("text", "md", "json"), default="text", help="output format"
    )
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k must be positive")

    start = time.perf_counter()
    results = _run(args.k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if args.output == "md":
        _print_md(results, args.k, elapsed_ms)
    elif args.output == "json":
        _print_json(results, args.k, elapsed_ms)
    else:
        _print_text(results, args.k, elapsed_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
