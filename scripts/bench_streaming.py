"""Drive N synthetic queries through `StreamingPipeline`; print p50/p95 per phase.

The numbers this prints are real wall-clock measurements on whatever
machine runs it — the script doesn't bake in a target or print a number
unless it actually ran. The corpus and reranker are in-memory and
dep-free so the result depends only on host CPU and Python version,
not on network or DB tuning. That's the point: it isolates pipeline
overhead from retrieval-backend overhead.

Usage:
    python -m scripts.bench_streaming --n 200
    python -m scripts.bench_streaming --n 50 --k 5

The output table is a stable shape — `docs/benchmarks.md` reproduces
it under "Streaming pipeline" with the date and host the run was done
on. Don't paste numbers in without rerunning.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.reranker import LexicalOverlapReranker  # noqa: E402
from rag_kit.retriever import RetrievalResult  # noqa: E402
from rag_kit.streaming import PhaseTimings, StreamingPipeline  # noqa: E402

# Realistic-shape corpus: ~40 short technical chunks. Long enough that
# the reranker isn't trivially measuring zero work, short enough that
# the script finishes in seconds even at --n 1000.
_CORPUS_TEXT = """\
Postgres tuning starts with shared_buffers, work_mem, effective_cache_size.
Reciprocal rank fusion combines lexical and dense retrieval channels.
HNSW index parameters: M controls graph degree, ef_search trades latency for recall.
Cross-encoder reranking improves NDCG at the cost of additional latency per query.
Citation enforcement: every claim in a generated answer should reference a chunk id.
Server-sent events let a browser render intermediate pipeline phases as they happen.
Faithfulness, recall@k, and answer correctness are the three RAG eval axes.
Token cost telemetry: prompt + completion tokens times the per-million-token price.
Voyage embeddings outperform OpenAI text-embedding-3 on legal domain corpora.
BGE small is a strong open-weight embedding model under one billion params.
pgvector supports hnsw and ivfflat index types with tunable build parameters.
Hybrid retrieval beats pure dense on out-of-distribution query distributions.
Chunk overlap of 10 to 20 percent helps when context windows are narrow.
Recursive chunking respects natural document structure better than fixed-size.
Semantic chunking uses embedding boundaries to keep ideas in one chunk.
Late chunking embeds the full document then splits, preserving global context.
LLM-as-judge can drift across model upgrades, so calibrate against human labels.
Drift detection on production traffic samples catches regressions early.
Pytest plugins make eval suites runnable as part of CI checks per pull request.
Snapshot testing for prompts catches semantic drift between model versions.
SSE keepalive comments prevent intermediaries from closing idle streams.
Backpressure via bounded asyncio queues prevents tool dispatch from overwhelming.
Async batching via httpx with bounded concurrency unlocks 5 to 20 times wins.
MCP servers expose tools to agents over a standard stdio or websocket protocol.
Tool registries with at least five tools test agent planning under variety.
Postgres-aware MCP servers introspect schemas and answer read-only questions.
Filesystem sandbox servers enforce allow-lists before any tool call returns.
Streaming text generation in Next.js uses React Server Components plus suspense.
Partial JSON parsing supports progressive rendering of tool-call payloads.
Mocking the Anthropic API deterministically replays recorded responses.
Playwright tests for streaming UIs benefit from semantic assertions.
Prompt caching reduces input token costs by reusing tokens across requests.
Semantic cache invalidation uses embedding similarity plus a TTL.
Model routing escalates to a stronger model only on logprob-entropy spikes.
Batch APIs apply when latency is not the bottleneck, e.g., nightly evals.
Vector search at scale benchmarks pgvector, Qdrant, and one more across millions.
HNSW M of 16 to 32 is a typical starting point for balanced index builds.
Query latency under load tests concurrency at 1, 10, and 100 simultaneous.
Cost per query at scale dominates the cost story for many production RAG apps.
Terraform for benchmark infra makes the numbers reproducible across runs.
"""


def _build_corpus() -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for i, line in enumerate(_CORPUS_TEXT.strip().splitlines()):
        text = line.strip()
        if text:
            chunks.append((f"chunk-{i:03d}", text))
    return chunks


_QUERIES = [
    "postgres tuning shared buffers",
    "reciprocal rank fusion hybrid retrieval",
    "cross encoder reranking latency",
    "citation enforcement chunk ids",
    "server sent events streaming pipeline",
    "faithfulness recall correctness evals",
    "token cost telemetry prompt completion",
    "voyage embeddings legal domain corpora",
    "pgvector hnsw index parameters",
    "chunk overlap recursive semantic",
    "drift detection production traffic",
    "prompt caching semantic cache TTL",
    "mcp server tool registry agents",
    "playwright streaming ui semantic assertions",
    "vector search scale millions concurrent",
]


class _BenchRetriever:
    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        self._corpus = corpus

    def search(self, query: str, k: int = 5, *, reranker=None) -> list[RetrievalResult]:
        query_terms = {t.lower() for t in query.split() if t}
        scored: list[tuple[str, str, float]] = []
        for ext_id, text in self._corpus:
            overlap = sum(
                1 for tok in text.lower().split() if tok.strip(".,") in query_terms
            )
            scored.append((ext_id, text, float(overlap) + 0.01))
        scored.sort(key=lambda r: r[2], reverse=True)
        out: list[RetrievalResult] = []
        for i, (ext_id, text, sc) in enumerate(scored[:k]):
            out.append(
                RetrievalResult(
                    external_id=ext_id,
                    text=text,
                    metadata={},
                    fused_score=sc,
                    ranks={"lexical": i + 1, "dense": i + 1},
                )
            )
        return out


def _token_stream(query: str, retrieved: Sequence[RetrievalResult]) -> Iterable[str]:
    if not retrieved:
        text = "I don't have context to answer that."
    else:
        text = f"Based on {retrieved[0].external_id}: {retrieved[0].text}"
    for tok in text.split(" "):
        yield tok + " "


def run(n: int, k: int) -> PhaseTimings:
    corpus = _build_corpus()
    timings = PhaseTimings()
    pipe = StreamingPipeline(
        _BenchRetriever(corpus),
        reranker=LexicalOverlapReranker(length_penalty=0.0),
        token_stream=_token_stream,
        timings=timings,
    )
    for i in range(n):
        q = _QUERIES[i % len(_QUERIES)]
        # `list(...)` drains the generator so every phase event is yielded
        # and every timings.record() fires — same code path as a real client.
        for _ in pipe.run(q, k=k):
            pass
    return timings


def _fmt(ms: float | None) -> str:
    return "—" if ms is None else f"{ms:7.2f}"


def print_table(t: PhaseTimings) -> None:
    summary = t.summary()
    header = f"{'phase':<12} {'n':>5} {'p50_ms':>10} {'p95_ms':>10}"
    print(header)
    print("-" * len(header))
    for phase in ("retrieving", "reranking", "generating", "total"):
        s = summary[phase]
        print(f"{phase:<12} {s['n']:>5} {_fmt(s['p50_ms']):>10} {_fmt(s['p95_ms']):>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark StreamingPipeline phases")
    ap.add_argument("--n", type=int, default=200, help="number of queries to run")
    ap.add_argument("--k", type=int, default=3, help="final top-k after rerank")
    args = ap.parse_args()

    t0 = time.perf_counter()
    timings = run(args.n, args.k)
    wall = time.perf_counter() - t0

    n_queries = args.n
    extra = ""
    if timings.retrieving:
        rps = n_queries / wall if wall > 0 else float("inf")
        mean_total = statistics.mean(timings.total)
        extra = f"  ({n_queries} queries in {wall*1000:.0f} ms; {rps:.1f} q/s; mean total {mean_total:.2f} ms)"
    print(f"Streaming pipeline benchmark · n={args.n} · k={args.k}{extra}")
    print()
    print_table(timings)


if __name__ == "__main__":
    main()
