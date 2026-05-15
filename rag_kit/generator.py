"""Answer generation with citation enforcement and weak-context refusal.

A `Generator` takes a user query plus the retrieved chunks (output of
`Retriever.search`) and returns either:

- a `GeneratedAnswer` whose text carries inline `[cite:<external_id>]`
  markers, with one validated `Citation` per claim sentence, or
- a `Refusal` that explains why an answer wasn't produced (currently:
  `insufficient_context` when retrieval scores fall below the caller's
  `threshold`; `unparseable_output` when the LLM produces text with no
  citations or dangling references that the validator rejects).

Two generators ship in this module:

- `TemplateGenerator` — local fallback, dep-free, deterministic. The
  same hermetic-CI rationale as D-006's `LexicalOverlapReranker`: CI
  exercises the full citation + refusal flow without an API key, and
  the answer text is reproducible across runs.
- `AnthropicGenerator` — production binding, lazy-imports the
  `anthropic` SDK so the module loads without it. Gated by the
  `rag-anthropic` extra.

Two refusal paths are deliberate. The threshold check happens *before*
the LLM is called: if retrieval is weak, we never pay for a generation
that would hallucinate. The citation-enforcement check happens *after*:
if the LLM's output cannot be reconciled with the retrieved chunks, we
refuse rather than emit a citation-less answer.

Citation enforcement is exposed as the free function `enforce_citations`
so it can be reused by alternative generators (or by a downstream eval
harness; see issue #7).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .retriever import RetrievalResult

_DEFAULT_THRESHOLD = 0.02  # tuned against the in-repo retrieval tests; >0 by construction
_CITE_PATTERN = re.compile(r"\[cite:([^\]]+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Citation:
    """One validated citation: a chunk that supports a claim."""

    external_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAnswer:
    """Successful generation: text with inline citation markers + validated citations."""

    text: str
    citations: tuple[Citation, ...]
    used_threshold: float
    top_score: float


@dataclass(frozen=True)
class Refusal:
    """Structured refusal: no answer produced, with a machine-readable reason."""

    reason: str  # one of: insufficient_context | unparseable_output
    detail: str
    used_threshold: float
    top_score: float


class CitationError(Exception):
    """Raised by enforce_citations; carries the same `reason` codes as Refusal."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class Generator(Protocol):
    """Single-method seam for swapping generator backends."""

    def generate(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
        *,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> GeneratedAnswer | Refusal:
        """Produce an answer for `query` given `retrieved` chunks, or refuse."""


def _top_score(retrieved: Sequence[RetrievalResult]) -> float:
    """Return the maximum confidence score across retrieved chunks.

    Uses `rerank_score` when present (it's a higher-quality signal than the
    fused score), falls back to `fused_score` otherwise. Returns 0.0 for an
    empty list so the refusal path is triggered.
    """
    if not retrieved:
        return 0.0
    best = 0.0
    for r in retrieved:
        score = r.rerank_score if r.rerank_score is not None else r.fused_score
        if score > best:
            best = score
    return best


def split_sentences(text: str) -> list[str]:
    """Split text into claim sentences for citation validation.

    Whitespace-only fragments are dropped. The split is intentionally simple
    — generators are instructed to produce one claim per sentence ending in
    `.`/`!`/`?`, and we don't need a full NLP tokenizer for that contract.
    """
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


def enforce_citations(
    text: str,
    retrieved: Sequence[RetrievalResult],
) -> tuple[Citation, ...]:
    """Validate that every claim sentence cites at least one retrieved chunk.

    Raises `CitationError("unparseable_output", ...)` if any sentence is
    missing a `[cite:...]` marker, or if a marker references an id that
    isn't in `retrieved` (dangling reference).

    Returns the deduplicated list of `Citation`s actually used by the text,
    in the order they first appear, so the caller can render them.
    """
    allowed: dict[str, RetrievalResult] = {r.external_id: r for r in retrieved}
    sentences = split_sentences(text)
    if not sentences:
        raise CitationError("unparseable_output", "answer text contained no sentences")

    seen: dict[str, Citation] = {}
    for sentence in sentences:
        ids_in_sentence = _CITE_PATTERN.findall(sentence)
        if not ids_in_sentence:
            raise CitationError(
                "unparseable_output",
                f"sentence has no [cite:...] marker: {sentence!r}",
            )
        for cite_id in ids_in_sentence:
            if cite_id not in allowed:
                raise CitationError(
                    "unparseable_output",
                    f"dangling citation {cite_id!r} not in retrieved chunks",
                )
            if cite_id not in seen:
                src = allowed[cite_id]
                seen[cite_id] = Citation(
                    external_id=src.external_id,
                    text=src.text,
                    metadata=src.metadata,
                )
    return tuple(seen.values())


def _refusal(reason: str, detail: str, threshold: float, top: float) -> Refusal:
    return Refusal(reason=reason, detail=detail, used_threshold=threshold, top_score=top)


class TemplateGenerator:
    """Deterministic, dep-free generator for hermetic CI and refusal-path tests.

    Produces one citation per retrieved chunk, in the order they appear.
    The output text is a fixed-shape template — *not* an answer to the
    query — but it's good enough to exercise the citation/refusal flow
    end-to-end without an API key, exactly like `LexicalOverlapReranker`
    does for the reranking layer.
    """

    def __init__(self, max_chunks: int = 3) -> None:
        if max_chunks <= 0:
            raise ValueError(f"max_chunks must be positive, got {max_chunks}")
        self.max_chunks = max_chunks

    def generate(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
        *,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> GeneratedAnswer | Refusal:
        top = _top_score(retrieved)
        if not retrieved:
            return _refusal("insufficient_context", "no chunks retrieved", threshold, top)
        if top < threshold:
            return _refusal(
                "insufficient_context",
                f"top_score={top:.4f} below threshold={threshold:.4f}",
                threshold,
                top,
            )

        chosen = list(retrieved)[: self.max_chunks]
        sentences = [
            f"Per the retrieved context, {c.text.strip().rstrip('.')} [cite:{c.external_id}]."
            for c in chosen
        ]
        text = " ".join(sentences)
        try:
            citations = enforce_citations(text, retrieved)
        except CitationError as e:  # pragma: no cover - template can't violate its own contract
            return _refusal(e.reason, e.detail, threshold, top)
        return GeneratedAnswer(
            text=text,
            citations=citations,
            used_threshold=threshold,
            top_score=top,
        )


class AnthropicGenerator:
    """Production generator backed by the Anthropic API.

    The `anthropic` SDK is imported lazily so the module loads without the
    `rag-anthropic` extra installed. Real-API tests live in a separate
    workflow gated on `ANTHROPIC_API_KEY`; unit tests for this class use
    a fake client.
    """

    _SYSTEM_PROMPT = (
        "You are a careful RAG assistant. Answer the user's query using ONLY the provided "
        "context chunks. Every sentence in your answer must end with at least one "
        "[cite:<external_id>] marker referencing the chunk that supports it. If the "
        "context is not sufficient to answer, respond with exactly: REFUSE: <one-line reason>."
    )

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_tokens: int = 512,
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
                "install the `rag-anthropic` extra to use AnthropicGenerator"
            ) from e
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set and no api_key provided")
        self._client = anthropic.Anthropic(api_key=key)
        return self._client

    @staticmethod
    def _format_context(retrieved: Sequence[RetrievalResult]) -> str:
        return "\n\n".join(
            f"<chunk id={r.external_id!r}>\n{r.text}\n</chunk>" for r in retrieved
        )

    def generate(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
        *,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> GeneratedAnswer | Refusal:
        top = _top_score(retrieved)
        if not retrieved:
            return _refusal("insufficient_context", "no chunks retrieved", threshold, top)
        if top < threshold:
            return _refusal(
                "insufficient_context",
                f"top_score={top:.4f} below threshold={threshold:.4f}",
                threshold,
                top,
            )

        client = self._ensure_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nContext:\n{self._format_context(retrieved)}"
                    ),
                }
            ],
        )
        # The Anthropic SDK exposes content as a list of blocks; we want the text blocks
        # joined. Handle both real SDK objects (block.text attr) and dict-shaped fakes.
        raw_text = ""
        for block in getattr(message, "content", []) or []:
            chunk_text = getattr(block, "text", None)
            if chunk_text is None and isinstance(block, dict):
                chunk_text = block.get("text")
            if chunk_text:
                raw_text += chunk_text

        stripped = raw_text.strip()
        if stripped.startswith("REFUSE:"):
            return _refusal("insufficient_context", stripped[len("REFUSE:"):].strip(), threshold, top)

        try:
            citations = enforce_citations(stripped, retrieved)
        except CitationError as e:
            return _refusal(e.reason, e.detail, threshold, top)
        return GeneratedAnswer(
            text=stripped,
            citations=citations,
            used_threshold=threshold,
            top_score=top,
        )


__all__ = [
    "AnthropicGenerator",
    "Citation",
    "CitationError",
    "GeneratedAnswer",
    "Generator",
    "Refusal",
    "TemplateGenerator",
    "enforce_citations",
    "split_sentences",
]
