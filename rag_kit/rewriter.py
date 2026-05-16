"""Pre-retrieval query rewriting / decomposition.

A `Rewriter` takes the raw user query and returns 1..K sub-queries plus a
short reasoning string explaining the decomposition. Multi-hop questions
benefit from being split into independently retrievable parts: "Who is
the CEO of X and where did they work before?" retrieves better when
issued as the two sub-queries "Who is the CEO of X" and "Where did <CEO>
work before" — at minimum, two retrieval passes each see a query whose
embedding is concentrated on one entity instead of being split between
two.

The protocol is intentionally narrow — one method, structured input,
structured output — so backends can be swapped without changing call
sites. Same single-method-protocol seam as `Reranker` (D-005) and
`Generator` (D-008).

Two rewriters ship in this module:

- `TemplateRewriter` — local default, dep-free, deterministic.
  Rule-based decomposition over a handful of common multi-hop patterns
  ("X and Y", "A. Then B.", "Compare X with Y"). The hermetic-CI
  rationale of D-006 / D-013 applies here too: the full rewrite-and-
  retrieve flow is exercised in CI without an API key, and the rewrite
  output is reproducible across runs.
- `AnthropicRewriter` — production binding, lazy-imports the
  `anthropic` SDK so the module loads without it. Gated by the existing
  `rag-anthropic` extra (same extra as `AnthropicGenerator`).

Wire into `Retriever.search(query, k, *, rewriter=...)`. The rewriter is
always opt-in (mirrors D-007) so existing callers keep their behavior.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RewriteResult:
    """Output of a rewriter: 1..K sub-queries plus a reasoning string.

    A single-element `sub_queries` means the rewriter chose not to
    decompose (the query already retrieves well as-is). Callers that
    treat the rewriter as opt-in pre-retrieval can use that length to
    decide whether the retrieval path branches into per-sub-query
    hybrid search + fusion, or stays single-shot.
    """

    sub_queries: tuple[str, ...]
    reasoning: str


class Rewriter(Protocol):
    """Single-method seam for swapping rewriter backends."""

    def rewrite(self, query: str) -> RewriteResult:
        """Return 1..K sub-queries for `query`, with a short reasoning string."""


# ----------------------------------------------------------------------
# Local default: rule-based decomposition (dep-free, deterministic)
# ----------------------------------------------------------------------


_WHITESPACE = re.compile(r"\s+")

# "compare X with Y", "compare X and Y", "compare X to Y"
_COMPARE_RE = re.compile(
    r"^\s*compare\s+(?P<a>.+?)\s+(?:with|to|and|vs\.?|versus)\s+(?P<b>.+?)\s*\??\s*$",
    re.IGNORECASE,
)

# "X. Then Y." — sequential-step decomposition. Matches a sentence-ending
# punctuation followed by "Then " (case-insensitive).
_THEN_SPLIT = re.compile(r"(?<=[.!?])\s+(?=Then\b|then\b)")

# Conjunction split for "X and Y" / "X, and Y" — bounded so we don't
# over-fragment a query like "wine and cheese pairings". Only fire when
# at least one side parses to a question shape ("who/what/where/when/
# why/how/which/is/are/does/did/can/will/should").
_AND_SPLIT_RE = re.compile(r"\s*,?\s+and\s+", re.IGNORECASE)

_QUESTION_PREFIXES = (
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "is",
    "are",
    "does",
    "did",
    "do",
    "can",
    "could",
    "will",
    "would",
    "should",
)


def _norm(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _looks_like_question(s: str) -> bool:
    head = _norm(s).lower()
    if not head:
        return False
    first_word = head.split(" ", 1)[0].rstrip(",?:")
    return first_word in _QUESTION_PREFIXES


def _split_then(query: str) -> list[str] | None:
    parts = _THEN_SPLIT.split(query)
    if len(parts) < 2:
        return None
    cleaned = []
    for raw in parts:
        s = _norm(raw)
        # Strip a leading "Then " so the sub-query reads as an independent question.
        if s.lower().startswith("then "):
            s = s[len("then ") :].lstrip()
        if s:
            cleaned.append(s)
    if len(cleaned) >= 2:
        return cleaned
    return None


def _split_compare(query: str) -> list[str] | None:
    match = _COMPARE_RE.match(query)
    if not match:
        return None
    a = _norm(match.group("a"))
    b = _norm(match.group("b"))
    if not a or not b:
        return None
    # Synthesize independent sub-queries about each entity. We don't try
    # to invent an attribute axis — that's the rewriter's caller's job.
    return [f"What is {a}?", f"What is {b}?"]


def _split_question_and(query: str) -> list[str] | None:
    """Split a multi-question conjunction like "Who is X and where did Y work?".

    Only fires when *both* halves of the split look like questions —
    otherwise "wine and cheese pairings" would split into two useless
    halves. Returns None if the heuristic doesn't apply.
    """
    stripped = query.strip().rstrip("?").rstrip()
    parts = _AND_SPLIT_RE.split(stripped)
    if len(parts) < 2:
        return None
    out: list[str] = []
    for part in parts:
        s = _norm(part)
        if not s:
            return None
        if not _looks_like_question(s):
            return None
        # Re-append the question mark we stripped above so each sub-query
        # is a well-formed question.
        if not s.endswith("?"):
            s = s + "?"
        out.append(s)
    return out if len(out) >= 2 else None


class TemplateRewriter:
    """Deterministic, dep-free decomposer for hermetic CI and offline tests.

    Tries a sequence of rule-based patterns and returns the first match's
    decomposition. If no pattern matches, returns the original query as a
    single-element `sub_queries` tuple — i.e., a no-op rewrite — with
    reasoning ``"no_decomposition"``.

    Patterns (in priority order):

    1. `compare X with Y` / `compare X and Y` → `[What is X?, What is Y?]`.
    2. `A. Then B.` (sequential-step) → `[A, B]` with the leading "Then "
       stripped from B so each part is independently readable.
    3. `Who is X and where did Y work?` (multi-question conjunction) →
       split on `" and "` only if *both* halves look like questions
       (start with `who/what/where/...`).
    4. Fallthrough: return `(query,)` with reasoning ``"no_decomposition"``.

    Not a substitute for an LLM-based rewriter. Use it for CI and offline
    benchmarks; use `AnthropicRewriter` (or your own implementation) in
    production. Same hermetic-CI rationale as `LexicalOverlapReranker`
    (D-006) and `TemplateGenerator` (D-008/D-013).
    """

    _MAX_SUB_QUERIES = 8

    def rewrite(self, query: str) -> RewriteResult:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        normalized = _norm(query)

        compare_split = _split_compare(normalized)
        if compare_split:
            return RewriteResult(
                sub_queries=tuple(compare_split[: self._MAX_SUB_QUERIES]),
                reasoning="compare_pattern",
            )

        then_split = _split_then(normalized)
        if then_split:
            return RewriteResult(
                sub_queries=tuple(then_split[: self._MAX_SUB_QUERIES]),
                reasoning="sequential_then_pattern",
            )

        and_split = _split_question_and(normalized)
        if and_split:
            return RewriteResult(
                sub_queries=tuple(and_split[: self._MAX_SUB_QUERIES]),
                reasoning="multi_question_and_pattern",
            )

        return RewriteResult(sub_queries=(normalized,), reasoning="no_decomposition")


# ----------------------------------------------------------------------
# Production: Anthropic-backed rewriter (lazy-imported via `rag-anthropic`)
# ----------------------------------------------------------------------


class AnthropicRewriter:
    """Production rewriter backed by the Anthropic API.

    Asks the model to decompose the user query into 1..K sub-queries and
    return strict JSON of the form
    ``{"sub_queries": [...], "reasoning": "..."}``. Output is parsed and
    validated; malformed responses raise ``ValueError`` rather than
    silently returning the original query so the caller can decide
    whether to fall back.

    The `anthropic` SDK is imported lazily so the module loads without
    the `rag-anthropic` extra installed. Real-API tests live in a
    separate workflow gated on `ANTHROPIC_API_KEY`; unit tests for this
    class use a fake client.
    """

    _SYSTEM_PROMPT = (
        "You decompose user questions into 1..K independently retrievable sub-queries. "
        "Rules:\n"
        "- If the question is simple (one entity, one attribute), return ONE sub-query "
        "  that equals the original question, with reasoning 'no_decomposition'.\n"
        "- If the question asks about multiple entities or multiple attributes that "
        "  benefit from separate retrieval, return 2..6 sub-queries, each a complete, "
        "  standalone question.\n"
        "- Never invent facts. Never resolve references across sub-queries (a pronoun "
        "  like 'they' should be left as a literal phrase to be resolved at the "
        "  next retrieval step, not replaced with a guessed entity).\n"
        "Respond with ONLY a JSON object of the form "
        '{"sub_queries": ["...", "..."], "reasoning": "..."}. '
        "Do not wrap the JSON in markdown or any prose."
    )

    _MAX_SUB_QUERIES = 6

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_tokens: int = 400,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = client  # injectable for tests; otherwise built on first call

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised when extra not installed
            raise ImportError(
                "install the `rag-anthropic` extra to use AnthropicRewriter"
            ) from e
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set and no api_key provided")
        self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def rewrite(self, query: str) -> RewriteResult:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        client = self._ensure_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
        raw_text = ""
        for block in getattr(message, "content", []) or []:
            chunk_text = getattr(block, "text", None)
            if chunk_text is None and isinstance(block, dict):
                chunk_text = block.get("text")
            if chunk_text:
                raw_text += chunk_text

        stripped = raw_text.strip()
        if not stripped:
            raise ValueError("AnthropicRewriter returned empty response")

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"AnthropicRewriter returned non-JSON output: {stripped!r}") from e
        if not isinstance(payload, dict):
            raise ValueError(f"AnthropicRewriter returned non-object JSON: {payload!r}")

        sub_queries_raw = payload.get("sub_queries")
        if not isinstance(sub_queries_raw, list) or not sub_queries_raw:
            raise ValueError(
                f"AnthropicRewriter response missing non-empty sub_queries: {payload!r}"
            )
        cleaned: list[str] = []
        for entry in sub_queries_raw:
            if not isinstance(entry, str):
                raise ValueError(f"sub_queries entry is not a string: {entry!r}")
            text = _norm(entry)
            if text:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("AnthropicRewriter sub_queries were all blank after normalization")
        cleaned = cleaned[: self._MAX_SUB_QUERIES]
        reasoning = payload.get("reasoning") or ""
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        return RewriteResult(sub_queries=tuple(cleaned), reasoning=reasoning)


__all__ = [
    "AnthropicRewriter",
    "RewriteResult",
    "Rewriter",
    "TemplateRewriter",
]
