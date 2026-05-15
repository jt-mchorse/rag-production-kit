"""Cross-encoder reranking layer.

A `Reranker` takes a query + a list of `Candidate`s (the output of hybrid
retrieval) and returns the same candidates re-sorted by a higher-quality
relevance signal. The contract is intentionally narrow — one method,
deterministic input, deterministic output shape — so backends can be
swapped without changing call sites.

Two backends ship in this PR (D-006):

- `LexicalOverlapReranker` — local fallback, dep-free. Token-overlap
  heuristic that lets the reranking flow be exercised end-to-end in CI
  without an API key. Not "good"; just deterministic and hermetic.
- `CohereReranker` — production binding, lazy-imports the `cohere` SDK
  so the module loads without it. Configurable model id, batch size,
  timeout. Recorded as D-005.

Wire into the retriever via `Retriever.search(query, k, reranker=...)`.
The reranker is always opt-in (D-007) so existing callers keep their
hybrid-only behavior.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Candidate:
    """Input to the reranker: one chunk surfaced by retrieval."""

    external_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScoredCandidate:
    """Output of the reranker: a candidate with the reranker's score + new rank."""

    external_id: str
    text: str
    metadata: dict[str, Any]
    rerank_score: float  # backend-specific scale; higher = more relevant
    rerank_rank: int  # 1-indexed position in the reranked list


class Reranker(Protocol):
    """Single-method seam for swapping reranker backends."""

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[ScoredCandidate]:
        """Return candidates re-sorted by relevance to query, with score + new rank."""


# ----------------------------------------------------------------------
# Local fallback: lexical overlap (dep-free, deterministic)
# ----------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


class LexicalOverlapReranker:
    """Reranker that scores by lowercase-token overlap with the query.

    Score formula: ``|query_tokens ∩ candidate_tokens| / |query_tokens|`` with
    a small length-penalty term to break ties toward shorter, more focused
    chunks. Deterministic given the same inputs.

    Not a substitute for a real cross-encoder. Use it for CI so the rerank
    flow is exercised hermetically; use `CohereReranker` (or your own
    backend) for production retrieval quality.
    """

    def __init__(self, *, length_penalty: float = 0.001) -> None:
        # Tiny coefficient so the penalty only shows up as a tie-breaker.
        if length_penalty < 0:
            raise ValueError("length_penalty must be non-negative")
        self.length_penalty = length_penalty

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[ScoredCandidate]:
        if not query:
            raise ValueError("query must be non-empty")
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            # Query has no scoreable tokens — preserve input order.
            return [
                ScoredCandidate(
                    external_id=c.external_id,
                    text=c.text,
                    metadata=c.metadata,
                    rerank_score=0.0,
                    rerank_rank=i + 1,
                )
                for i, c in enumerate(candidates)
            ]

        scored: list[tuple[float, Candidate]] = []
        for c in candidates:
            c_tokens = set(_tokenize(c.text))
            overlap = len(q_tokens & c_tokens) / len(q_tokens)
            penalty = self.length_penalty * len(c.text)
            scored.append((overlap - penalty, c))

        # Stable sort so equal scores preserve input order — tests rely on this.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ScoredCandidate(
                external_id=c.external_id,
                text=c.text,
                metadata=c.metadata,
                rerank_score=score,
                rerank_rank=i + 1,
            )
            for i, (score, c) in enumerate(scored)
        ]


# ----------------------------------------------------------------------
# Cohere production backend (lazy-imported)
# ----------------------------------------------------------------------


class CohereReranker:
    """Production reranker backed by Cohere's /rerank API.

    Requires the `rerank-cohere` optional dependency:
        pip install 'rag-production-kit[rerank-cohere]'

    The API key is read from `COHERE_API_KEY`. Model id, batch size, and
    timeout are configurable.
    """

    DEFAULT_MODEL = "rerank-english-v3.0"
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_TIMEOUT_S = 30.0

    def __init__(
        self,
        *,
        model: str | None = None,
        batch_size: int | None = None,
        timeout_s: float | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "CohereReranker requires the optional 'rerank-cohere' extra. "
                "Install with: pip install 'rag-production-kit[rerank-cohere]'"
            ) from e

        self._cohere_module = cohere
        self.client = cohere.ClientV2(api_key=api_key or os.environ.get("COHERE_API_KEY"))
        self.model = model or self.DEFAULT_MODEL
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.timeout_s = timeout_s if timeout_s is not None else self.DEFAULT_TIMEOUT_S

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[ScoredCandidate]:
        if not query:
            raise ValueError("query must be non-empty")
        candidates_list = list(candidates)
        if not candidates_list:
            return []

        # Cohere's /rerank takes the documents inline and returns indices into
        # that list, sorted by relevance. We chunk into `batch_size` requests
        # so very large candidate lists don't trip request-size limits, then
        # merge by score (the API returns scores on a comparable scale within
        # one model + version).
        merged: list[tuple[float, Candidate]] = []
        for start in range(0, len(candidates_list), self.batch_size):
            batch = candidates_list[start : start + self.batch_size]
            documents = [c.text for c in batch]
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                request_options={"timeout_in_seconds": self.timeout_s},
            )
            for r in response.results:
                merged.append((float(r.relevance_score), batch[r.index]))

        merged.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ScoredCandidate(
                external_id=c.external_id,
                text=c.text,
                metadata=c.metadata,
                rerank_score=score,
                rerank_rank=i + 1,
            )
            for i, (score, c) in enumerate(merged)
        ]


# ----------------------------------------------------------------------
# Telemetry: how much did the reranker actually move things?
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RerankDelta:
    """Telemetry on how much the reranker reordered the input."""

    n_input: int
    top_k_overlap: int  # how many of the top-k were already in input top-k
    top_k_size: int  # k used for the comparison
    ndcg_displacement: float  # 1.0 = no change, 0.0 = total flip


def rerank_delta_ndcg(
    before: Sequence[str],
    after: Sequence[str],
    *,
    k: int = 5,
) -> RerankDelta:
    """Compute how much the reranker moved the top-k.

    `before` is the list of `external_id`s as ranked by retrieval; `after`
    is the same after the reranker. Returns nDCG-style displacement using
    the input position as relevance — so if the reranker put the input's
    top item at the bottom, displacement is low; if it kept the order,
    displacement is 1.0.

    `top_k_overlap` is the cardinality of the intersection of the top-k
    before vs. top-k after — useful when nDCG hides large reordering inside
    the top set.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    before_list = list(before)
    after_list = list(after)
    n = max(len(before_list), len(after_list))
    if n == 0:
        return RerankDelta(n_input=0, top_k_overlap=0, top_k_size=0, ndcg_displacement=1.0)

    eff_k = min(k, len(before_list), len(after_list))
    overlap = len(set(before_list[:eff_k]) & set(after_list[:eff_k]))

    # Use `before` ranks as relevance: the input top is the most relevant.
    # rel(id) = (n - input_position(id)) for ids in `before`, else 0.
    rel: dict[str, float] = {}
    for i, ext_id in enumerate(before_list):
        rel[ext_id] = float(n - i)

    def dcg(seq: Sequence[str]) -> float:
        return sum(rel.get(ext_id, 0.0) / math.log2(i + 2) for i, ext_id in enumerate(seq))

    ideal = dcg(before_list)
    actual = dcg(after_list)
    displacement = actual / ideal if ideal > 0 else 1.0

    return RerankDelta(
        n_input=len(before_list),
        top_k_overlap=overlap,
        top_k_size=eff_k,
        ndcg_displacement=displacement,
    )
