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

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .retriever import RetrievalResult

_DEFAULT_THRESHOLD = 0.02  # tuned against the in-repo retrieval tests; >0 by construction
_CITE_PATTERN = re.compile(r"\[cite:([^\]]+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Tokens that end in a period but are NOT sentence boundaries. The `_SENTENCE_SPLIT`
# regex treats every period-then-whitespace as a boundary, so an ordinary
# abbreviation ("Dr.", "U.S.", "e.g.") strands a leading claim-less fragment
# ("Dr.") that survives the alphanumeric filter and then trips
# `enforce_citations` into a false `unparseable_output` refusal of an otherwise
# fully-cited answer (#110). Matched case-insensitively against the last
# whitespace token with its trailing dots removed. Curated on purpose and kept
# deliberately lenient: a genuine sentence that ends in one of these is
# vanishingly rare, and the cost of that rare over-merge (one claim riding on a
# neighbour's citation) is far smaller than refusing every answer that says
# "Dr." or "U.S.".
_ABBREVIATIONS = frozenset(
    {
        # Titles
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "rev",
        "gen",
        "sen",
        "rep",
        # Latin / editorial
        "e.g",
        "i.e",
        "etc",
        "cf",
        "vs",
        "al",
        "viz",
        # Org / measure
        "inc",
        "ltd",
        "llc",
        "corp",
        "co",
        "no",
        "vol",
        "fig",
        "eq",
        "pp",
        # Dotted initialisms (geo / time)
        "u.s",
        "u.k",
        "u.n",
        "e.u",
        "a.m",
        "p.m",
    }
)


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
    # Take the actual maximum — seeding `best` at 0.0 clamped a genuinely
    # negative maximum up to 0.0. `LexicalOverlapReranker` can score a long,
    # low-overlap chunk negative (overlap - length_penalty·len(text)), so an
    # all-negative set used to report top_score=0.0 in the refusal and, at a
    # non-positive `threshold`, pass the `top < threshold` gate and answer
    # from chunks that should have been refused as insufficient context.
    return max(r.rerank_score if r.rerank_score is not None else r.fused_score for r in retrieved)


def _ends_with_abbreviation(fragment: str) -> bool:
    """True if `fragment`'s last token is a known abbreviation or a single-letter
    initial — i.e. its trailing period is *not* a sentence boundary.

    See `_ABBREVIATIONS`: the trailing dots are stripped and the token matched
    case-insensitively (so `"U.S."` -> `"u.s"`, `"e.g."` -> `"e.g"`, `"Dr."` ->
    `"dr"`). A lone capital initial (the `"J."` in `"Dr. J. Smith"`) also isn't a
    boundary. Only `.`-terminated tokens can qualify — `!`/`?` are unambiguous
    sentence ends and never abbreviations.
    """
    tokens = fragment.split()
    if not tokens:
        return False
    last = tokens[-1].rstrip(".")
    if not last:
        return False
    if last.lower() in _ABBREVIATIONS:
        return True
    # Single capital-letter initial, e.g. the "J." in "Dr. J. Smith". Restricted
    # to uppercase letters so numeric fragments ("Section 5.") aren't merged.
    return len(last) == 1 and last.isupper()


def split_sentences(text: str) -> list[str]:
    """Split text into claim sentences for citation validation.

    Fragments with no claim content are dropped: whitespace-only, and also
    punctuation-only (no alphanumeric character). A stray terminator can split
    off a claim-less fragment like ``"."`` — keeping it made `enforce_citations`
    demand a `[cite:...]` marker on something that asserts nothing, falsely
    refusing an otherwise fully-cited answer. A real claim requires a
    word/number, and digits are alphanumeric, so a bare-number fragment stays
    under enforcement — dropping word-less fragments can't mask an uncited claim.

    The regex splits on every terminal-punctuation-then-whitespace, which also
    fires *inside* an abbreviation ("Dr. Smith" -> "Dr.", "Smith"). A merge pass
    re-joins a fragment with the next when it ends in a known abbreviation or a
    single-letter initial (`_ends_with_abbreviation`), so an abbreviation-bearing
    claim stays one sentence and its lone `[cite:...]` marker satisfies
    enforcement instead of stranding a claim-less "Dr." fragment that falsely
    refuses the whole answer (#110). The merge is deliberately lenient — see
    `_ABBREVIATIONS`.

    The split is intentionally simple — generators are instructed to produce one
    claim per sentence ending in `.`/`!`/`?`, and we don't need a full NLP
    tokenizer for that contract.
    """
    parts = _SENTENCE_SPLIT.split(text.strip())
    merged: list[str] = []
    for part in parts:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [p for p in merged if any(ch.isalnum() for ch in p)]


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
        for raw_cite_id in ids_in_sentence:
            # Strip incidental whitespace inside the marker. The LLM path
            # (`AnthropicGenerator`) routinely emits `[cite: doc1]` / `[cite:doc1 ]`;
            # without this, the padded id ` doc1` mismatches the real external_id
            # `doc1`, the marker reads as *dangling*, and a fully-grounded answer is
            # falsely refused as `unparseable_output` (#88). Stripping is strictly
            # more lenient and cannot create a false-accept: a genuinely-unknown id
            # still misses `allowed`, and corpus external_ids never carry
            # leading/trailing whitespace, so two distinct valid ids can't collide.
            cite_id = raw_cite_id.strip()
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


def _validate_threshold(threshold: float) -> None:
    """Reject a non-finite refusal threshold at the generator boundary.

    The gate is `top < threshold`. A `NaN` threshold makes that comparison
    `False` for every `top`, silently bypassing the refusal path so the
    generator answers from chunks it should have refused; `-inf` does the
    same, and `+inf` forces an unconditional refusal. None of those are a
    meaningful confidence floor, so fail loud — matching the finiteness/sign
    guards on `fusion.k` (#38), `LexicalOverlapReranker.length_penalty`
    (#75), and `max_chunks` (#41). A *finite* negative threshold stays
    valid: `_top_score` can be genuinely negative (#69), so a negative floor
    is a legitimate "accept low-overlap chunks" configuration.
    """
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be a finite number; got {threshold!r}")


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
        # Extend #41 sign-only contract to this construction site. Sign-only
        # accepted bool (`True` silently bound `self.max_chunks = True`, then
        # `list(retrieved)[: self.max_chunks]` sliced to first item only;
        # `False` raised the old "must be positive" message but masked the
        # type bug) and accepted float (`2.5` silently bound, then surfaced
        # as `TypeError: slice indices must be integers` deep in `generate`).
        if not isinstance(max_chunks, int) or isinstance(max_chunks, bool) or max_chunks <= 0:
            raise ValueError(f"max_chunks must be a positive integer; got {max_chunks!r}")
        self.max_chunks = max_chunks

    def generate(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
        *,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> GeneratedAnswer | Refusal:
        _validate_threshold(threshold)
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
        # Emit one cited template sentence per sentence *inside* each chunk,
        # using the same splitter `enforce_citations` validates with. A chunk's
        # `text` is whole-document prose (the indexer does no sentence-level
        # chunking), so wrapping it in a single template sentence would let
        # `split_sentences` fragment it and leave every sentence but the last
        # uncited — a false "unparseable_output" refusal. Each chunk still
        # yields exactly one deduped Citation.
        sentences = [
            f"Per the retrieved context, {s.strip().rstrip('.!?')} [cite:{c.external_id}]."
            for c in chosen
            for s in split_sentences(c.text)
        ]
        text = " ".join(sentences)
        try:
            citations = enforce_citations(text, retrieved)
        # pragma: no cover - defensive: only reachable if every chosen chunk is
        # empty / punctuation-only, which yields no cited sentences.
        except CitationError as e:  # pragma: no cover
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
            raise ImportError("install the `rag-anthropic` extra to use AnthropicGenerator") from e
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set and no api_key provided")
        self._client = anthropic.Anthropic(api_key=key)
        return self._client

    @staticmethod
    def _format_context(retrieved: Sequence[RetrievalResult]) -> str:
        return "\n\n".join(f"<chunk id={r.external_id!r}>\n{r.text}\n</chunk>" for r in retrieved)

    def generate(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
        *,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> GeneratedAnswer | Refusal:
        _validate_threshold(threshold)
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
                    "content": (f"Query: {query}\n\nContext:\n{self._format_context(retrieved)}"),
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
            return _refusal(
                "insufficient_context", stripped[len("REFUSE:") :].strip(), threshold, top
            )

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
