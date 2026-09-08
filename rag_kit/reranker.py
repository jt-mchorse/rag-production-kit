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
        """Return candidates re-sorted by relevance to query, with score + new rank.

        **Among equal scores, the input order is preserved** (#207). Scores tie
        by construction rather than by coincidence — two candidates carrying the
        same `text` are the same document to any scorer that only sees the text
        — so "sorted by score" does not by itself define an output. Leaving the
        rest to insertion order makes the ranking a function of whatever built
        that order, which for a batching backend is the API's arbitrary tie
        ordering and the operator's `batch_size`.

        Input order is the rule rather than `fusion.py`'s doc-id tie-break
        because the input here is already a ranking — `Retriever.search`'s fused
        list — and it carries signal a lexicographic rule would discard. RRF has
        no such incoming order to inherit, which is why the two seams answer
        differently.

        `tests/test_reranker_tie_order_contract.py` runs this against every
        backend in this module, so a third one inherits the contract instead of
        re-deriving it.
        """


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
        # Finiteness, not just sign: `NaN < 0` and `inf < 0` are both False, so
        # a non-finite penalty slips the sign check and poisons every score
        # (`overlap - length_penalty * len(text)` → NaN/-Inf). All-NaN scores
        # sort as a no-op (NaN comparisons are false), so the relevant chunk is
        # silently not surfaced first. Mirrors the finiteness sweep already
        # applied to fusion's `k`, `telemetry.ModelPrice`, and latency.
        if not math.isfinite(length_penalty) or length_penalty < 0:
            raise ValueError(f"length_penalty must be a finite number >= 0.0; got {length_penalty}")
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
            # Bound the length factor to [0, 1) so the penalty stays in
            # [0, length_penalty) — strictly below the smallest overlap quantum
            # (1 / len(q_tokens)) for any tiny coefficient. A *raw* char-count
            # penalty (length_penalty * len(c.text)) is unbounded and routinely
            # exceeds the gap between distinct overlap levels at realistic chunk
            # sizes, so a less-relevant short chunk would outrank a more-relevant
            # long one — defeating the "penalty is only a tie-breaker" contract
            # (#90). The factor is still strictly monotonic in length, so equal-
            # overlap chunks still break ties toward shorter.
            penalty = self.length_penalty * (len(c.text) / (len(c.text) + 1))
            scored.append((overlap - penalty, c))

        # Stable sort so equal scores preserve input order — the `Reranker`
        # Protocol's tie rule, which this backend gets for free because
        # `scored` is built by iterating `candidates` in order. `CohereReranker`
        # does not: it appends per batch in *API-response* order, so it has to
        # carry the input position explicitly (#207).
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
        # Validate the configurable construction args BEFORE importing the
        # optional `cohere` extra. Two reasons: (1) a bad value then fails loud
        # with a clear ValueError even when the extra isn't installed, instead of
        # the validation being unreachable behind the ImportError; (2) it never
        # reaches the batch loop. `batch_size <= 0` is silently wrong, not just
        # odd: `rerank` chunks via `range(0, n, self.batch_size)`, so a negative
        # batch_size makes that range empty and `rerank` returns [] — every
        # candidate silently dropped, the API never called, no error. The prior
        # `batch_size or DEFAULT` idiom additionally swallowed an explicit 0 into
        # the default, masking operator error; an explicit `is None` check keeps
        # `None` meaning "use the default" while rejecting 0. Guard like the `k`
        # (fusion/rerank_delta_ndcg), `length_penalty`, and non-finite
        # relevance_score guards elsewhere in this module — fail loud at the seam.
        if batch_size is not None and (
            not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0
        ):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(f"timeout_s must be a finite number > 0.0, got {timeout_s!r}")

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
        self.batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
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
        # `(score, input_position, candidate)`. The position is what makes the
        # Protocol's tie rule hold here: `merged` is filled per batch in the
        # order the API returned each batch's rows, which is by relevance and
        # not by input position, so a stable sort on score alone left tied docs
        # ordered by the API's own arbitrary tie choice — and moved them again
        # whenever `batch_size` put them in different requests (#207).
        merged: list[tuple[float, int, Candidate]] = []
        for start in range(0, len(candidates_list), self.batch_size):
            batch = candidates_list[start : start + self.batch_size]
            documents = [c.text for c in batch]
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                request_options={"timeout_in_seconds": self.timeout_s},
            )
            # `relevance_score` is guarded below because the API is an external,
            # uncontrolled source. `index` is the OTHER field read off the same
            # row, and it decides *which document a score is attached to* — in a
            # kit whose premise is that a citation points at the chunk the claim
            # came from. It was fed straight into `batch[r.index]` (#186).
            #
            # The Protocol above says `rerank` returns "candidates re-sorted",
            # i.e. a permutation of the input. Measured against 3 candidates
            # (D0, D1, D2), eight response shapes broke that and five broke it
            # SILENTLY:
            #
            #   index = -1      -> ['D2', 'D1', 'D2']   D2 twice, D0 gone
            #   index = -3      -> ['D0', 'D1', 'D2']   looks perfect; the 0.9
            #                                           score belongs to a doc
            #                                           three positions away
            #   duplicate index -> ['D0', 'D0', 'D2']   D0 twice, D1 gone
            #   fewer results   -> ['D0']               2 candidates dropped
            #   empty results   -> []                   whole retrieval gone
            #   more results    -> 6 rows out of 3 in
            #   index = 7       -> raw IndexError
            #   index = 1.0/'1'/None -> raw TypeError
            #
            # The `-3` row is why a "less than length" bounds check is not
            # enough: Python's negative indexing makes an out-of-range index
            # look like a perfectly ordinary result. And the empty/short rows
            # reproduce a harm this module has already named as unacceptable on
            # the operator-supplied road — see the `batch_size` guard's "every
            # candidate silently dropped ... no error".
            #
            # Per BATCH, not per call: `rerank` chunks by `batch_size`, so each
            # request is independently a permutation of its own slice.
            results = list(response.results)
            if len(results) != len(batch):
                raise ValueError(
                    f"Cohere rerank returned {len(results)} result(s) for a batch of "
                    f"{len(batch)} document(s); the reranker contract is a re-sort of "
                    "its input, so a short response silently drops candidates and a "
                    "long one duplicates them"
                )
            seen_indices: set[int] = set()
            for r in results:
                # `bool` is an `int` subclass, so `index=True` indexed as 1 and
                # returned the wrong document without tripping any check.
                if isinstance(r.index, bool) or not isinstance(r.index, int):
                    raise ValueError(
                        f"Cohere rerank returned a non-integer index {r.index!r} "
                        f"({type(r.index).__name__}); it is used to look a document up "
                        "by position and would otherwise raise a raw TypeError deep in "
                        "the merge"
                    )
                if not 0 <= r.index < len(batch):
                    raise ValueError(
                        f"Cohere rerank returned index {r.index} for a batch of "
                        f"{len(batch)} document(s); a negative index silently resolves "
                        "to a different document rather than failing, so the score "
                        "would be attributed to a chunk that did not earn it"
                    )
                if r.index in seen_indices:
                    raise ValueError(
                        f"Cohere rerank returned index {r.index} more than once in one "
                        "batch; the reranker contract is a permutation, and a repeated "
                        "index returns one document twice while dropping another "
                        "entirely"
                    )
                seen_indices.add(r.index)

            for r in results:
                # The Cohere API is an external, uncontrolled source: a malformed
                # or erroring response can hand back a non-finite relevance_score.
                # Unguarded, a NaN flows into ScoredCandidate.rerank_score, then
                # into generator._top_score's max(), then the refusal gate
                # `top < threshold` — which is False for NaN, so the generator
                # answers from chunks it should have refused. _validate_threshold
                # (#78) already closed the operator-supplied-threshold half of this
                # exact gate; this is the API-supplied-score half. Fail loud at the
                # seam, like LexicalOverlapReranker's length_penalty guard and the
                # #80/#82 external-value finiteness guards. (LexicalOverlap scores
                # are finite by construction; only the Cohere path is exposed.)
                score = float(r.relevance_score)
                if not math.isfinite(score):
                    raise ValueError(
                        f"Cohere rerank returned a non-finite relevance_score ({score!r}); "
                        "a NaN/Inf score would poison the generator's refusal gate "
                        "(top < threshold is False for NaN, answering when it should refuse)"
                    )
                # `start` is the batch's offset into `candidates_list` and
                # `r.index` is validated in `[0, len(batch))` above, so the sum
                # is the candidate's position in the caller's own list.
                merged.append((score, start + r.index, batch[r.index]))

        # `(-score, position)` rather than `reverse=True`: reversing would
        # reverse the tie-break too, ranking the *last* tied candidate first.
        merged.sort(key=lambda row: (-row[0], row[1]))
        return [
            ScoredCandidate(
                external_id=c.external_id,
                text=c.text,
                metadata=c.metadata,
                rerank_score=score,
                rerank_rank=i + 1,
            )
            for i, (score, _position, c) in enumerate(merged)
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
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    before_list = list(before)
    after_list = list(after)

    # A valid ranking is a set of distinct ids: a reranker returns a permutation
    # of its distinct inputs. A duplicated id is degenerate — and silently wrong,
    # not just odd: `rel` is keyed by external_id and the ideal is dcg(before_list),
    # so a repeated id in `after` re-adds that id's full relevance into the actual
    # DCG while the ideal is unchanged, pushing ndcg_displacement past its
    # documented 1.0 ceiling (a dashboard would read "improved beyond the input
    # ideal", which is impossible); a duplicate in `before` double-counts the ideal
    # and distorts the ratio downward. Fail loud at the seam, matching the `k`,
    # `length_penalty`, and Cohere non-finite-score guards in this module (#98).
    for _name, _seq in (("before", before_list), ("after", after_list)):
        if len(set(_seq)) != len(_seq):
            raise ValueError(
                f"{_name} contains duplicate external_ids; rerank_delta_ndcg expects "
                "each ranking to be distinct ids (duplicates push ndcg_displacement "
                "past its documented [0.0, 1.0] range)"
            )

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
