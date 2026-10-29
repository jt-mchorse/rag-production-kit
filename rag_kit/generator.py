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
import numbers
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .comparison import render_comparison
from .io_utils import copy_json_value
from .retriever import RetrievalResult
from .text import SENTENCE_CLOSERS, SENTENCE_TERMINATORS, is_blank

_DEFAULT_THRESHOLD = 0.02  # tuned against the in-repo retrieval tests; >0 by construction
# Decimal places the refusal detail has always rendered at, and still renders
# at whenever four is enough to tell `top` and `threshold` apart. Passed
# explicitly rather than defaulted inside `render_comparison`, so this
# module's width stays this module's business (#225, D-021).
_DETAIL_PLACES = 4
_CITE_PATTERN = re.compile(r"\[cite:([^\]]+)\]")
# One terminator set for the splitter, the template writer and the rewriter
# (#301); the per-script table lives with it in `text.py`.
_TERMINATORS = SENTENCE_TERMINATORS
# Sentence terminators. Besides ASCII `.!?`, this includes the unicode
# terminators `…` (U+2026 ellipsis), `。` (ideographic full stop), `！`/`？`
# (fullwidth), and `؟` (U+061F Arabic question mark), so a claim ending in one
# is split off and individually citation-enforced instead of merging onto the
# next sentence's [cite:...] marker and bypassing enforcement (#144/#148,
# sibling of #143/#139/#133). `؟` reaches parity with the rewriter's
# `_TERMINATORS` (#147), which already recognized it here. Like `!`/`?`, these
# are unambiguous ends and never abbreviations, so the `_ends_with_abbreviation`
# merge pass (dot-only) leaves them untouched — no regression on ASCII behavior.
# A closing quote or bracket may sit between the terminator and the whitespace:
# `."`, `.'`, `.)`, `.]`, and the smart-quote variants `.”` / `.’`. A bare
# `(?<=[.!?…])\s+` requires whitespace *immediately* after the terminator, so
# that intervening quote/bracket hides the boundary — the two sentences merge
# and an uncited first claim rides on the next sentence's [cite:...] marker,
# bypassing enforcement (the same false-accept as the abbreviation vein #144/
# #148, reached through punctuation rather than an abbreviation). Match an
# optional closing quote/bracket via a SECOND fixed-width lookbehind alternated
# with the bare one (Python `re` forbids a variable-width lookbehind, and this
# keeps the quote *attached* to the preceding sentence rather than consuming it,
# unlike a `["'…]?` inside the split delimiter). Quoting source text and ending
# the sentence at the closing quote is standard English punctuation, so this is
# reachable in ordinary generated answers, not a synthetic input.
#
# That rule was one closer from six (#160), and model output ends sentences in
# three more ways it did not see, each merging an uncited claim onto the next
# sentence's marker (#262): a RUN of closers or a closer outside the six
# (`.**` -- markdown bold, which Claude writes by default -- `.")`, `.»`, `。」`);
# a `[cite:...]` marker straight after the terminator (`days.[cite:doc1] Next`,
# which `enforce_citations` accepts as a cited sentence, so the boundary is
# AFTER the marker); and Chinese/Japanese, which put no space after `。！？`.
#
# So a boundary is a terminator run, then any run of closers and markers, then
# the GAP the text is cut at: whitespace, or -- after `。！？` only -- nothing at
# all, before the next character that is not whitespace, a closer or `[`. `.`
# never splits without whitespace (decimals). A closer straight after a CJK
# terminator still needs whitespace: Japanese `「…です。」と言った` continues
# the sentence, and an over-merge is the cheaper error here (`_ABBREVIATIONS`).
# The gap is a capture group because the terminators and closers it follows
# stay attached to the sentence they end; Python `re` has no variable-width
# lookbehind to express that.
_CLOSERS = SENTENCE_CLOSERS  # shared with the rewriter (#320)
_CJK_TERMINATORS = "。！？｡"
_MARKER = r"\[cite:[^\]]+\]"
_SENTENCE_SPLIT = re.compile(
    rf"[{re.escape(_TERMINATORS)}]+(?:[{re.escape(_CLOSERS)}]|{_MARKER})*(?P<gap>\s+)"
    rf"|[{_CJK_TERMINATORS}]+(?:{_MARKER})*(?P<cjk>)(?=[^\s{re.escape(_CLOSERS)}\[])"
)

# A line or paragraph break was never a boundary: only a terminator split, so a
# bullet or paragraph with no closing punctuation merged into the next sentence
# and an uncited claim rode its [cite:...] marker (#385, the same false-accept
# as #190/#262). Markdown is the model's default shape (#262) and the system
# prompt never asks for terminated sentences. So the newline in front of an
# UNAMBIGUOUS structural break is a boundary too: a blank line, a list item
# (`-`, `*`, `+`, `1.`, `1)`), or an ATX heading (`#`..`######` then a space).
# A bare single newline is not one: a hard-wrapped sentence would be cut in
# two and its first half refused, so it stays the cheaper over-merge.
#
# The same breaks inside a blockquote (#391). A line holding only `>`s is
# markdown's blank line within a quote, and `> - item` / `> # h` are its list
# items and headings, so the lookahead first skips any `>` prefix. Without it
# `> claim\n>\n> cited [cite:x]` was one sentence. `_QUOTE_PREFIX` is that
# prefix, and `_QUOTE_BLANK` a line that is blank once it is removed.
_QUOTE_PREFIX = re.compile(r"(?:[ \t]*>)*")
_QUOTE_BLANK = re.compile(r"[ \t>]*")
_STRUCTURAL_BREAK = re.compile(
    r"\n(?=(?:[ \t]*>)*[ \t]*(?:\n|(?:[-*+]|\d{1,9}[.)])[ \t]|#{1,6}[ \t]))"
)
# The text in front of a break that it does NOT cut after (#385): a heading line,
# or a lead-in ending in `:` ("Key Postgres settings:"). Neither got its own
# sentence before #385 (each rode the text after it), and they carry no marker
# of their own in a normal answer, so cutting there would refuse every cited
# list that has a heading or a lead-in. They stay attached to what follows,
# which is exactly how they behaved before. A claim written INTO one still
# rides; telling such a claim from a label needs a classifier, out of scope here.
# The `:` need not be the last character (#400). Markdown emphasis closes AFTER
# it (`**Key settings:**`, `*...:*`, `__...:__`), and a CJK answer writes the
# fullwidth `：`, so a lead-in is `:` or `：` then any run of `_CLOSERS` -- the
# same terminator-then-closer shape the splitter reads (#161/#262/#314). Without
# it each of those lines was cut off as its own uncited sentence and a fully
# cited list under it refused. A colon mid-line (`Key: settings are`) is still
# not one, and neither is a bold label with no colon (`**Key settings**`, #389).
_HEADING_LINE = re.compile(r"[ \t]{0,3}#{1,6}[ \t].*")
_LEAD_IN = re.compile(rf".*[:：][{re.escape(_CLOSERS)}]*[ \t]*")
# An ordered-list marker at the start of a line (`1. item`). `_SENTENCE_SPLIT`
# reads its `.` plus space as a sentence end, so `'1.'` was cut off as a
# fragment and, holding a digit, refused a fully cited numbered list (#385).
# Behind a blockquote prefix too (`> 1. item`, #391).
_ORDERED_MARKER = re.compile(r"(?:[ \t]*>)*[ \t]*\d{1,9}")

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

# The subset of `_ABBREVIATIONS` whose legitimate sense is a *numeric reference*
# — they abbreviate a label that is followed by a number ("No. 5", "Vol. 3",
# "Fig. 2", "Eq. 4", "pp. 12"). At least one ("no") is ALSO a common English
# word that naturally ends a claim ("The answer is no."). Treating these as
# unconditional non-boundaries let an uncited claim ending in the word merge
# into the next sentence and ride on ITS `[cite:...]` marker, bypassing
# enforcement — the same false-accept #126 fixed for the lone capital initial
# ("vitamin C."). So they count as a non-boundary only when a digit follows
# (the numeric sense); the word sense stays a real boundary. False-refusing the
# rare non-numeric use ("See No. IV") is the safe direction (#126); false-
# accepting an uncited claim is the bug.
#
# CORRECTION (#190). This comment used to say that the name/proper-noun-context
# abbreviations ("co"→"Acme Co.", "st"→"Main St.", "al"→"et al.") were
# "deliberately NOT gated: they are followed by proper nouns, not digits, and
# are not common standalone claim-endings". Every one of those three has since
# been shown to be a common standalone claim-ending. "al" was gated first, by a
# *different* discriminator (lowercase follow-on) — being followed by a proper
# noun is what made it unsafe, not what made it safe. "co" is now gated the same
# way, with the rest of the company suffixes (`_ORG_SUFFIX_ABBREVIATIONS`).
# "st" still is not, and #191 records why: unlike the others its mid-sentence
# sense takes a *capitalized* continuation ("St. Peter"), so the follow-on
# signal carries no information for it. The sentence stays here, corrected
# rather than deleted, because being wrong in a comment is what kept anyone
# from re-checking for four issues.
#
# The rule this leaves: "not a digit-gated case" is not the same statement as
# "safe to merge unconditionally". Each abbreviation needs the discriminator
# that separates *its* two senses, or an explicit record of why none exists.
_NUMERIC_REFERENCE_ABBREVIATIONS = frozenset({"no", "vol", "fig", "eq", "pp"})

# The subset of `_ABBREVIATIONS` that also spells a common *unit* which naturally
# ends a measurement claim. "ms" is the courtesy title "Ms." (followed by a
# proper name) but ALSO the milliseconds unit ("5 ms") — and a latency/telemetry
# RAG kit ends claims in "... N ms." constantly. Treating "ms" as an
# unconditional non-boundary let an uncited measurement claim ("The p50 was 5
# ms.") merge into the next (cited) sentence and ride on ITS `[cite:...]` marker,
# bypassing enforcement — the same false-accept #126/#130 fixed for "vitamin C."
# and the word "no". Unlike the numeric-reference set (gated on a digit that
# *follows*), the two senses are told apart by the token's own CASE: the courtesy
# title is always capitalized "Ms." while the SI-style unit is lowercase "ms".
# So a lowercase "ms" is always a real boundary — "5 ms." AND "reported in ms."
# (no preceding number) alike — and only a capitalized "Ms." with no preceding
# number is the title (a non-boundary). A number-precedes check alone (the
# original #138 shape) missed the unit forms where no number immediately precedes
# ("in ms.", "measured in ms."), mis-reading them as the title. False-refusing a
# rare capitalized "5 Ms." is the safe direction; false-accepting an uncited
# claim is the bug.
_UNIT_COLLISION_ABBREVIATIONS = frozenset({"ms"})

# The subset of `_ABBREVIATIONS` that spells a *time-of-day* marker which very
# commonly ENDS a claim ("The outage started at 5 p.m."). Unlike "ms" (which
# also has a title sense "Ms." with no preceding number), "a.m"/"p.m" are
# essentially always the time sense, so the `_UNIT_COLLISION` preceding-number
# gate can't distinguish a boundary from a mid-sentence use — a number precedes
# in BOTH "...at 5 p.m. The root cause..." (boundary) and "...at 9 a.m. sharp"
# (mid-sentence). Treating them as unconditional non-boundaries let an uncited
# claim ending in a time ("The outage started at 5 p.m.") merge into the next
# (cited) sentence and ride on ITS `[cite:...]` marker, bypassing enforcement —
# the same false-accept #126/#130/#139 fixed for "vitamin C.", "no", and "5 ms.".
# A latency/telemetry RAG kit ends claims in "... at N a.m./p.m." as often as it
# does in units (#139's rationale). The distinguishing signal is what FOLLOWS: a
# lowercase continuation ("9 a.m. sharp") is mid-sentence (non-boundary), while a
# capitalized/digit/empty follow-on ("5 p.m. The root...") is a real boundary.
# False-refusing the rare "9 a.m. Monday ..." is the safe direction (#126);
# false-accepting an uncited claim is the bug.
_TIME_ABBREVIATIONS = frozenset({"a.m", "p.m"})

# The subset of `_ABBREVIATIONS` that spells an *enumeration* marker which very
# commonly ENDS a claim ("We support JSON, CSV, etc."). Unlike "Dr." — a
# genuinely rare claim-ender, since a title is always followed by a name —
# "etc." is one of the MOST common sentence-ending abbreviations in English, so
# treating it as an unconditional non-boundary let an uncited claim ending in it
# ("We support JSON, CSV, etc.") merge into the next (cited) sentence and ride on
# ITS `[cite:...]` marker, bypassing enforcement — the same false-accept
# #126/#130/#139/#152 fixed for "vitamin C.", "no", "5 ms.", and "in ms.". Like
# the time markers (and unlike the unit's own-case signal), the two senses are
# told apart by what FOLLOWS: a lowercase continuation ("…, etc. are fruits") is
# a genuine mid-clause list (non-boundary), while a capitalized/digit/empty
# follow-on ("…, etc. The system…") is a real boundary. False-refusing the rare
# mid-sentence "etc. Foo" is the safe direction (#126); false-accepting an
# uncited claim is the bug.
_ENUMERATION_ABBREVIATIONS = frozenset({"etc"})

# The subset of `_ABBREVIATIONS` that spells an *attribution* marker which very
# commonly ENDS a claim ("The method was introduced by Vaswani et al."). In a
# paper/research RAG "X et al." is a routine LLM attribution phrasing and the
# NEXT sentence is very often the cited one, so treating "al" as an unconditional
# non-boundary let an uncited attribution ("… by Vaswani et al.") merge into the
# next (cited) sentence and ride on ITS `[cite:...]` marker, bypassing
# enforcement — the same false-accept #154 fixed for "etc.". Told apart, like the
# enumeration/time markers, by what FOLLOWS: a lowercase continuation ("Smith et
# al. found that …") is a genuine mid-clause use (non-boundary), while a
# capitalized/digit/bracket/empty follow-on ("… et al. It uses …") is a real
# boundary. False-refusing a rare mid-sentence "et al. Foo" is the safe direction;
# false-accepting an uncited claim is the bug. (The other ungated Latin markers —
# "e.g."/"i.e."/"cf."/"viz." — stay unconditional non-boundaries: they INTRODUCE
# a clause and are commonly followed by a *capitalized* example, so this
# lowercase-follow-on gate would wrongly split them.)
_ATTRIBUTION_ABBREVIATIONS = frozenset({"al"})

# The subset of `_ABBREVIATIONS` that spells a *company suffix* which very
# commonly ENDS a claim ("The vendor of record is Acme Inc."). A filings /
# contracts / company-news corpus is squarely this kit's target, and an LLM
# naming an entity it did not retrieve is exactly the hallucination the citation
# layer exists to catch — so treating these as unconditional non-boundaries let
# an uncited entity claim merge into the next (cited) sentence and ride on ITS
# `[cite:...]` marker, bypassing enforcement. Measured, all five bypassed (#190).
#
# Told apart, like the enumeration/time/attribution markers, by what FOLLOWS: a
# lowercase continuation ("Acme Inc. reported a loss", "Acme Ltd. and Beta Corp.
# merged") is a genuine mid-clause use, while a capitalized/digit/bracket/empty
# follow-on ("… is Acme Inc. The filing …") is a real boundary. The one form this
# costs is a title-case continuation, "Acme Inc. Reports Record Revenue" — a
# headline, not a claim sentence, and false-refusing is the safe direction (#126).
#
# This is the case the `_NUMERIC_REFERENCE_ABBREVIATIONS` comment above ruled out
# by name, on a rationale that had already been disproved for one of the three
# members it listed. See the correction there.
#
# Deliberately NOT extended to the geo initialisms ("u.s"/"u.k"/"u.n"/"e.u") or
# "st", which have the identical false-accept: the discriminator that works for
# every other subset carries no information for them, because their attributive
# sense takes a CAPITALIZED continuation — "U.S. Federal Reserve", "U.N. Security
# Council", "St. Peter". Gating them would trade one false-accept for a frequent
# false-refusal of correctly-grounded answers. That is a posture question with no
# clean discriminator, filed as #191 rather than decided here;
# `tests/test_citation_org_suffix_boundary.py` pins their current behaviour so
# the gap is recorded rather than forgotten.
_ORG_SUFFIX_ABBREVIATIONS = frozenset({"inc", "ltd", "llc", "corp", "co"})

_HAS_DIGIT_RE = re.compile(r"\d")


def _looks_numeric(token: str) -> bool:
    """True if `token` carries a digit — a measurement like "5", "1.5", "500"
    (the unit sense of "ms"), as opposed to a name or ordinary word (the title
    sense). A digit anywhere is enough: decimals and unit-suffixed forms all
    qualify while proper names never do."""
    return bool(_HAS_DIGIT_RE.search(token))


@dataclass(frozen=True)
class Citation:
    """One validated citation: a chunk that supports a claim."""

    external_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy *metadata* so the frozen record owns it (#227, D-022).

        `frozen=True` stops a caller rebinding the attribute and says nothing
        about the object it points at, so without this the caller's dict — and
        anything nested inside it — stays editable through the reference they
        still hold. Deep over `dict`/`list`, because `Any` proves nothing about
        the values and a shallow copy *is* the defect.

        `_extract_citations` builds a `Citation` from the `RetrievalResult`
        it cites, so before this the two records shared one dict and editing a
        citation's metadata edited the retrieval result's.
        """
        object.__setattr__(self, "metadata", copy_json_value(self.metadata))


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
    scores = [r.rerank_score if r.rerank_score is not None else r.fused_score for r in retrieved]
    # Guard the consumer, not one producer (#372). #86 rejected a non-finite
    # Cohere score at its seam, but `Reranker` is a public protocol and
    # `Retriever.search(reranker=...)` passes any custom `rerank_score` (or a
    # NaN fused score) straight here. `max()` keeps a NaN only when it comes
    # first, and `NaN < threshold` is False, so the verdict flipped with input
    # order and a NaN batch answered when it should refuse. `+inf` would pass
    # every threshold too. Fail loud, like `_validate_threshold` (#78).
    checked: list[float] = []
    for r, raw in zip(retrieved, scores, strict=True):
        field = "rerank_score" if r.rerank_score is not None else "fused_score"
        # The type half (#380): `True` passed the finiteness gate as 1.0 and a
        # str escaped `math.isfinite` as a bare TypeError naming no chunk. Any
        # other `numbers.Real` is a score: sentence-transformers'
        # `CrossEncoder.predict` -- the usual custom reranker -- returns
        # `numpy.float32`, which is not a `float` subclass. Coerce once, so the
        # finiteness check, `max()` and the refusal detail all see a float.
        score = _as_finite_float(raw)
        if score is None:
            raise ValueError(
                f"chunk {r.external_id!r} has a non-numeric {field} ({raw!r}); "
                "the refusal gate compares it against a numeric threshold"
            )
        if not math.isfinite(score):
            raise ValueError(
                f"chunk {r.external_id!r} has a non-finite {field} ({score!r}); "
                "a NaN/Inf score would poison the refusal gate "
                "(top < threshold is False for NaN, answering when it should refuse)"
            )
        checked.append(score)
    return max(checked)


def _as_finite_float(value: object) -> float | None:
    """Coerce a real, non-bool number to `float`; `None` if it is not one (#380).

    `bool` is a `numbers.Real` but is refused: `True` is a flag, not a 1.0
    score or threshold. The result may still be NaN/Inf -- the caller owns the
    finiteness message. A real too large for a double (`10**400`) overflows
    `float()`; it comes back as `inf` so it fails the caller's finiteness check
    with a ValueError instead of escaping as a bare OverflowError.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _ends_with_abbreviation(fragment: str, following: str = "") -> bool:
    """True if `fragment`'s last token is a known abbreviation or a single-letter
    initial — i.e. its trailing period is *not* a sentence boundary.

    See `_ABBREVIATIONS`: the trailing dots are stripped and the token matched
    case-insensitively (so `"U.S."` -> `"u.s"`, `"e.g."` -> `"e.g"`, `"Dr."` ->
    `"dr"`). A lone capital initial (the `"J."` in `"Dr. J. Smith"`) also isn't a
    boundary, but ONLY in a name context (see below). Only `.`-terminated tokens
    can qualify — `!`/`?` are unambiguous sentence ends and never abbreviations.

    `following` is the next candidate fragment (the one that would be merged in).
    It gates the numeric-reference abbreviations in
    `_NUMERIC_REFERENCE_ABBREVIATIONS` ("No.", "Vol.", ...): those are a
    non-boundary only when a number follows, so an uncited claim ending in the
    word "no" ("The answer is no.") stays a real boundary and can't ride on the
    next sentence's citation (#130, sibling of #126).
    """
    tokens = fragment.split()
    if not tokens:
        return False
    last = tokens[-1].rstrip(".")
    if not last:
        return False
    if last.lower() in _ABBREVIATIONS:
        if last.lower() in _NUMERIC_REFERENCE_ABBREVIATIONS:
            # Non-boundary only in the numeric sense: a digit must follow.
            stripped = following.lstrip()
            return bool(stripped) and stripped[0].isdigit()
        if last.lower() in _UNIT_COLLISION_ABBREVIATIONS:
            # Non-boundary only in the *capitalized title* sense ("Ms." + a
            # name). The reliable discriminator is the token's own case: the
            # courtesy title is always written "Ms." while the SI-style
            # millisecond unit is lowercase "ms". A number immediately before is
            # ALSO the unit sense ("5 ms."), but a number doesn't always precede
            # the unit ("latency reported in ms.", "measured in ms.") — so the
            # numeric-prev check alone (the #138 shape) mis-read those as the
            # title and merged the uncited measurement claim into the next
            # sentence, letting it ride on that sentence's citation. Requiring
            # capitalization keeps a lowercase "ms" a real boundary regardless of
            # what precedes (see `_UNIT_COLLISION_ABBREVIATIONS`).
            prev = tokens[-2] if len(tokens) >= 2 else ""
            return last[:1].isupper() and not _looks_numeric(prev)
        if last.lower() in _TIME_ABBREVIATIONS:
            # Non-boundary only when the continuation is unambiguously
            # mid-clause: a lowercase-letter follow-on ("9 a.m. sharp"). A
            # capitalized/digit/bracket/empty follow-on ("5 p.m. The root...")
            # is a real boundary so an uncited time-of-day claim can't ride on
            # the next sentence's citation (see `_TIME_ABBREVIATIONS`).
            stripped = following.lstrip()
            return bool(stripped) and stripped[0].islower()
        if last.lower() in _ENUMERATION_ABBREVIATIONS:
            # Non-boundary only when the continuation is unambiguously
            # mid-clause: a lowercase-letter follow-on ("..., etc. are fruits").
            # A capitalized/digit/bracket/empty follow-on ("..., etc. The
            # system...") is a real boundary so an uncited enumeration claim
            # ("We support JSON, CSV, etc.") can't ride on the next sentence's
            # citation (see `_ENUMERATION_ABBREVIATIONS`).
            stripped = following.lstrip()
            return bool(stripped) and stripped[0].islower()
        if last.lower() in _ATTRIBUTION_ABBREVIATIONS:
            # Non-boundary only when the continuation is unambiguously mid-clause:
            # a lowercase-letter follow-on ("Smith et al. found that ..."). A
            # capitalized/digit/bracket/empty follow-on ("... et al. It uses ...")
            # is a real boundary so an uncited attribution claim ("The method was
            # introduced by Vaswani et al.") can't ride on the next sentence's
            # citation (see `_ATTRIBUTION_ABBREVIATIONS`).
            stripped = following.lstrip()
            return bool(stripped) and stripped[0].islower()
        if last.lower() in _ORG_SUFFIX_ABBREVIATIONS:
            # Non-boundary only when the continuation is unambiguously mid-clause:
            # a lowercase-letter follow-on ("Acme Inc. reported a loss"). A
            # capitalized/digit/bracket/empty follow-on ("... is Acme Inc. The
            # filing ...") is a real boundary so an uncited entity claim ("The
            # vendor of record is Acme Inc.") can't ride on the next sentence's
            # citation (see `_ORG_SUFFIX_ABBREVIATIONS`, #190).
            stripped = following.lstrip()
            return bool(stripped) and stripped[0].islower()
        return True
    # A lone capital-letter token, e.g. the "J." in "Dr. J. Smith". A bare
    # `len==1 and isupper()` test also fires on ordinary claim endings —
    # "vitamin C.", "hepatitis B.", "grade A.", "type O." — and merging those
    # into the next sentence lets an *uncited* claim ride on the following
    # sentence's [cite:...] marker, bypassing enforcement (#126). Only treat it
    # as a non-boundary in a NAME context: preceded by a title abbreviation
    # ("Dr. J.") or another single-letter initial ("J. K."). False-refusing an
    # ambiguous name is the safe direction; false-accepting an uncited claim
    # (the bug) is not.
    if len(last) == 1 and last.isupper():
        prev = tokens[-2].rstrip(".") if len(tokens) >= 2 else ""
        return prev.lower() in _ABBREVIATIONS or (len(prev) == 1 and prev.isupper())
    return False


def _last_content_line(text: str) -> str:
    """The last line of `text` that is not blank, with any `>` prefix removed.

    What the #385 heading / `:` lead-in check reads in front of a break. A line
    of only `>`s is blank inside a blockquote (#391), so `> Settings:\n>\n> - a`
    finds `Settings:` and the cited list under it stays attached, as it does
    outside a quote.
    """
    for line in reversed(text.split("\n")):
        if not _QUOTE_BLANK.fullmatch(line):
            return line[_QUOTE_PREFIX.match(line).end() :].rstrip()
    return ""


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

    A markdown structural break is a boundary too, whether or not a terminator
    precedes it: a blank line, a list item, or a heading (`_STRUCTURAL_BREAK`,
    #385). A heading line and a lead-in ending in `:` stay attached to the text
    after them, and the abbreviation merge never re-joins across a break. The
    same holds inside a blockquote, where a `>`-only line is the blank line, and
    for CRLF or CR line endings (#391).

    The split is intentionally simple. It is not a full NLP tokenizer, and a
    claim the model leaves unterminated can only be separated at one of these
    markdown breaks. A bare single newline does not separate it.
    """
    # Never split INSIDE a citation marker (#256). `Document` accepts an id like
    # `faq.md#Q3. refunds` and `_CITE_PATTERN` reads it back, but the split
    # fires on terminal-punctuation-then-whitespace wherever it occurs, so
    # `[cite:faq.md#Q3. refunds]` became `[cite:faq.md#Q3.` + `refunds].` and a
    # fully grounded answer was refused as having an uncited sentence. Split
    # points inside a marker's span are skipped -- by position, not by masking
    # the markers with placeholders, which would corrupt any text that already
    # contained the placeholder characters.
    stripped = text.strip()
    # Boundaries are found on `lines`, a copy with CRLF and lone-CR line endings
    # read as `\n` (#391): `\r\n\r\n` is the same blank line as `\n\n`, but no
    # `\n`-anchored break matched it, so an uncited paragraph rode the next one's
    # marker. The copy keeps every offset (`\r\n` -> ` \n`, `\r` -> `\n`), so the
    # parts are still cut from `stripped` and come back byte-for-byte.
    lines = stripped.replace("\r\n", " \n").replace("\r", "\n")
    marker_spans = [m.span() for m in _CITE_PATTERN.finditer(lines)]
    breaks = [
        m.start()
        for m in _STRUCTURAL_BREAK.finditer(lines)
        if not any(lo < m.start() < hi for lo, hi in marker_spans)  # #256, as below
    ]
    cuts: list[tuple[int, int]] = []
    for boundary in _SENTENCE_SPLIT.finditer(lines):
        gap_lo, gap_hi = boundary.span("gap" if boundary.group("gap") is not None else "cjk")
        if any(lo < gap_lo < hi for lo, hi in marker_spans):
            continue
        line_start = lines.rfind("\n", 0, boundary.start()) + 1
        if boundary.group().startswith(".") and _ORDERED_MARKER.fullmatch(
            lines, line_start, boundary.start()
        ):
            continue  # the `1.` of a numbered-list item, not a sentence end (#385)
        cuts.append((gap_lo, gap_hi))
    # Structural breaks (#385). One that lies inside a terminator's gap
    # (`done.\n\n- next`) is already cut there; the cut is only marked as
    # structural, so the merge pass below cannot re-join across it.
    for pos in breaks:
        if any(lo <= pos < hi for lo, hi in cuts):
            continue
        last_line = _last_content_line(lines[:pos])  # `Settings:\n\n- a`
        if _HEADING_LINE.fullmatch(last_line) or _LEAD_IN.fullmatch(last_line):
            continue
        cuts.append((pos, pos + 1))
    cuts.sort()
    parts: list[str] = []
    start = 0
    for lo, hi in cuts:
        parts.append(stripped[start:lo])
        start = hi
    parts.append(stripped[start:])
    merged: list[str] = []
    for i, part in enumerate(parts):
        structural = i > 0 and any(cuts[i - 1][0] <= pos < cuts[i - 1][1] for pos in breaks)
        if merged and not structural and _ends_with_abbreviation(merged[-1], following=part):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [p for p in merged if any(ch.isalnum() for ch in p)]


# A terminator followed by closing punctuation -- a boundary `_SENTENCE_SPLIT`
# splits after (#161). The same `_CLOSERS` as the splitter, so the writer cannot
# emit a tail the reader cuts at a different place (#262).
# A terminator RUN, as the splitter reads it: `?!”` or `...”` with one
# terminator here left `?` / `..` in front of the marker, where the splitter
# cut it off -- a fully grounded chunk refused (#262).
# And whatever terminators or closers follow that (#314): `up?".` and
# `FAQ.).` end in a terminator AFTER the closer, so a tail anchored on closers
# matched nothing, the strip took only the last `.`, and the marker landed after
# `?"` -- where the splitter cut the claim from it again.
_TERMINATOR_THEN_CLOSERS = re.compile(
    rf"[{re.escape(_TERMINATORS)}]+[{re.escape(_CLOSERS)}]+"
    rf"[{re.escape(_TERMINATORS + _CLOSERS)}]*$"
)

# What `_STRUCTURAL_BREAK` reads at the start of a line (#399): `>` quote
# prefixes, list markers and ATX heading marks, each before a space, in any
# nesting. `_template_sentence` keeps a sentence's run of them at its start and
# writes the template prefix after it, and `TemplateGenerator.generate` starts
# such a sentence on its own line, so the #385/#391 line-start reads see the
# same heading, numbered item or quote in the answer as in the chunk.
_BLOCK_PREFIX = re.compile(r"(?:[ \t]*(?:>|(?:[-*+]|\d{1,9}[.)]|#{1,6})(?=[ \t])))*[ \t]*")
# A line holding only such a marker, which one more space turns into a break.
_BARE_BLOCK_MARKER = re.compile(r"(?:[ \t]*>)*[ \t]*(?:[-*+]|\d{1,9}[.)]|#{1,6})")


def _template_sentence(sentence: str, external_id: str) -> str:
    """One cited template sentence for one source sentence (#258).

    Normally the trailing terminators are stripped and `[cite:...]` plus a full
    stop is appended -- unchanged. But a sentence ending in a terminator and a
    closing quote or bracket (`... "restart the server."`) cannot be handled
    that way: the strip leaves `."` in place, the appended marker lands after
    it, and `split_sentences` -- which treats terminator+closer as a boundary --
    cut the claim from its own marker, so `enforce_citations` refused a fully
    grounded chunk. The marker goes *before* that tail instead, which keeps the
    quote balanced and the sentence and its marker in one piece.

    A markdown sentence keeps its block prefix in front of the template prefix
    (`# Per the retrieved context, Policy`), so the #385 heading and numbered-
    item reads still see it (#399). And the result is read back with
    `split_sentences`: when some part of it is still uncited -- the reader cut
    it where it had not cut the chunk, because a heading mark or a `30.` it
    keyed on now sits elsewhere -- every point the reader can cut at gets a
    marker of its own (`_cite_every_cut`). Derived from the reader rather than
    restated, like `_marker_readback` (#197).
    """
    marker = f"[cite:{external_id}]"
    s = sentence.strip()
    # The prefix goes after any block prefix, not in front of it (#399): the
    # #385 reads are anchored at a line start, so `Per the retrieved context,
    # # Policy` was no longer a heading and `..., 1. Open` no longer a list item.
    lead = s[: _BLOCK_PREFIX.match(s).end()]
    s = s[len(lead) :]
    tail = _TERMINATOR_THEN_CLOSERS.search(s)
    if tail is not None:
        head, end = s[: tail.start()], s[tail.start() :]
    else:
        head, end = s.rstrip(_TERMINATORS), "."
    # And any whitespace the strip exposed, with the terminators behind it
    # (#399): `30.\n ?` left `30.\n `, so the marker sat after a blank line
    # (a break) or `30.` became a mid-line sentence end, uncited either way.
    head = head.rstrip(_TERMINATORS + " \t\r\n")
    body = f"{lead}Per the retrieved context, {head}"
    text = f"{body}{_marker_gap(body)}{marker}{end}"
    if all(_CITE_PATTERN.search(part) for part in split_sentences(text)):
        return text
    body = _cite_every_cut(body, marker)
    return f"{body}{_marker_gap(body)}{marker}{end}"


def _marker_gap(text: str) -> str:
    """The space in front of a marker appended to `text`, or none (#399).

    ` [cite:` after a bare `#` / `*` / `2)` line makes it a heading or list
    item: a new structural break, which cut the claim from its own marker.
    """
    last_line = re.split(r"\r\n|\r|\n", text)[-1]
    return "" if _BARE_BLOCK_MARKER.fullmatch(last_line) else " "


def _cite_every_cut(text: str, marker: str) -> str:
    """`text` with `marker` in front of every place `split_sentences` may cut it.

    Those are a terminator run the splitter reads as a sentence end, and the end
    of a line that holds something (#399). Each part the reader can cut off then
    carries a marker, whichever cuts its heading and numbered-item rules make.
    The fallback of `_template_sentence`: it reads worse than one marker, so it
    is only used when the template's own output did not read back cited.
    """
    lines = text.replace("\r\n", " \n").replace("\r", "\n")  # as `split_sentences`
    spots = {m.start() for m in _SENTENCE_SPLIT.finditer(lines)}
    for m in re.finditer(r"[ \t]*\n", lines):
        line = lines[lines.rfind("\n", 0, m.start()) + 1 : m.start()]
        if not _QUOTE_BLANK.fullmatch(line):
            spots.add(m.start())
    out: list[str] = []
    start = 0
    for pos in sorted(spots):
        out.append(text[start:pos])
        out.append(_marker_gap(text[:pos]) + marker)
        start = pos
    out.append(text[start:])
    return "".join(out)


_MARKER_OPEN = "[cite:"


def _neutralise_markers(text: str) -> str:
    """Chunk text with every `[cite:` in it made unreadable as a marker (#283).

    `TemplateGenerator` copies a chunk's sentences into its answer word for
    word and then validates the answer with `enforce_citations` -- and
    `evals/run_eval.py` re-runs that check over `GeneratedAnswer.text`. So a
    `[cite:X]` already present in a chunk's *text* (a documentation corpus
    that describes the marker syntax, this repo's own README among them) was
    read as a real citation: refused as dangling when `X` was not retrieved,
    and, when it was, credited `X` with a sentence that came from another
    chunk. An *unclosed* `[cite:` in the text is caught too: the template's
    own marker closes it, so `[cite: marker here [cite:doc1]` read as one
    dangling id.

    A complete marker becomes `(cite:X)` and any leftover opener `(cite:`,
    which `_CITE_PATTERN` cannot match. Applied to the chunk text BEFORE
    `split_sentences`: the splitter refuses to cut inside a real marker, so
    neutralising afterwards would leave a `[cite:faq.md#Q3. refunds]`
    sentence whole for the template but cut in two for the validator.
    """
    text = _CITE_PATTERN.sub(lambda m: f"(cite:{m.group(1)})", text)
    return text.replace(_MARKER_OPEN, "(cite:")


def _marker_readback(external_id: str) -> str | None:
    """What a `[cite:<external_id>]` marker actually resolves to, or `None`.

    Synthesises the marker an answer would carry for this id, then runs the
    *real* reader over it: `_CITE_PATTERN` for the capture, and the same
    `.strip()` `enforce_citations` applies to every captured id. `None` means
    the grammar does not match at all, i.e. no marker can name this id.

    Derived rather than described, on purpose (#197). #182 identified two ways
    an id fails to be citable back -- surrounding whitespace, defeated by the
    strip, and a `]`, defeated by `_CITE_PATTERN` stopping at the first one --
    and stated them as two separate rules. The read-side collision check then
    re-stated one of them, and the half it left out silently cited the wrong
    chunk. A helper that *calls* the grammar cannot fall behind the grammar, so
    a future marker-syntax change is covered without anyone remembering this
    file exists.

    `tests/test_citation_id_readback.py` pins the write seam against this
    helper, so `Document.__post_init__`'s two operator-facing messages and this
    function cannot disagree about which ids are legal.
    """
    match = _CITE_PATTERN.search(f"[cite:{external_id}]")
    if match is None:
        return None
    return match.group(1).strip()


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

    # Two retrieved ids that a `[cite:...]` marker cannot tell apart make every
    # citation to either of them a coin flip, and the loser is rendered as the
    # source of a claim it did not come from. Measured on a corpus holding both
    # 'doc1' and ' doc1', a citation to ' doc1' resolved to 'doc1' and rendered
    # that chunk's text (#182). Refuse instead -- a refusal is a recoverable
    # answer, a citation pointing at the wrong chunk is a confidently wrong one,
    # and it is invisible downstream because the resulting `Citation` is
    # perfectly well-formed.
    #
    # #182 closed this at the write seam, where `Document.__post_init__` rejects
    # both shapes it identified: a padded id (defeated by the reader's `.strip()`
    # below) and one containing `]` (defeated by `_CITE_PATTERN`, which stops at
    # the first `]`). This function is handed rows from the *database*, not
    # `Document` objects, so a corpus indexed before that guard still needs
    # checking here.
    #
    # That backstop used to be written as its own rule -- ids equal after
    # `.strip()` -- and so it covered exactly one of the two causes #182 names
    # in the same breath. A corpus holding 'doc' and 'doc]1' passed it, and
    # `[cite:doc]1]` was ACCEPTED, citing 'doc' and rendering 'doc's text as the
    # source (#197). Two hand-listed rules on the write side and one on the read
    # side is what produced that gap, and a third grammar feature would produce
    # it again.
    #
    # So the rule is derived, not restated: ask what a citation to each id would
    # actually resolve to by running the *real* grammar over a synthesised
    # marker, and refuse when two distinct ids answer the same. Whitespace, `]`,
    # and whatever the marker grammar grows next are covered by construction,
    # because `_marker_readback` calls `_CITE_PATTERN` and the same `.strip()`
    # the loop below applies rather than describing them.
    readback_owners: dict[str, str] = {}
    for r in retrieved:
        key = _marker_readback(r.external_id)
        if key is None:
            # No `[cite:<id>]` marker can name this id at all (the grammar does
            # not match). Such an id is unreachable rather than ambiguous: a
            # citation aimed at it reads as dangling and the answer is refused,
            # which is the safe direction, so it is not this check's business.
            continue
        first = readback_owners.get(key)
        if first is not None and first != r.external_id:
            raise CitationError(
                "unparseable_output",
                f"retrieved chunks contain ids a citation cannot tell apart: "
                f"{first!r} and {r.external_id!r} both read back as {key!r} from a "
                "[cite:...] marker; a citation could not be attributed to one of "
                "them unambiguously",
            )
        readback_owners.setdefault(key, r.external_id)

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
            # falsely refused as `unparseable_output` (#88).
            #
            # Stripping is strictly more lenient, and it cannot create a
            # false-accept *given* that no two retrieved ids are equal after
            # stripping. That used to be asserted here as "corpus external_ids
            # never carry leading/trailing whitespace" -- and nothing enforced
            # the "never", so `Document(' doc1')` was accepted and a corpus
            # holding both 'doc1' and ' doc1' resolved a citation to the wrong
            # chunk (#182). Both halves are now closed: `Document.__post_init__`
            # rejects a padded id at the write seam, and the collision check at
            # the top of this function covers rows indexed before that guard,
            # which the write seam structurally cannot reach.
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


def _validate_threshold(threshold: float) -> float:
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

    And the type half (#380). Two of the three guards named above refuse a
    `bool`; this one checked finiteness only, so `threshold=True` was a 1.0
    floor (refusing a 0.5 top score) and `False` a 0.0 one, while a str or
    `None` escaped `math.isfinite` as a bare TypeError -- the same split
    #303/#362 closed for telemetry and `PhaseTimings`. Any other
    `numbers.Real` (a `numpy.float32` read from config, a `Fraction`) is
    accepted and returned as a `float`, so the gate, `Refusal.used_threshold`
    and the refusal detail never see the original type.
    """
    value = _as_finite_float(threshold)
    if value is None or not math.isfinite(value):
        shown = threshold if value is None else value
        raise ValueError(f"threshold must be a finite number; got {shown!r}")
    return value


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
        # Free, but held to `AnthropicGenerator`'s query rules (#422, #426) so
        # the `Generator` protocol means one thing whichever backend is wired.
        if not isinstance(query, str):
            raise ValueError(f"query must be a string; got {type(query).__name__}")
        if is_blank(query):
            raise ValueError("query must be non-empty")
        threshold = _validate_threshold(threshold)
        top = _top_score(retrieved)
        if not retrieved:
            return _refusal("insufficient_context", "no chunks retrieved", threshold, top)
        if top < threshold:
            top_s, thr_s = render_comparison(top, threshold, places=_DETAIL_PLACES)
            return _refusal(
                "insufficient_context",
                f"top_score={top_s} below threshold={thr_s}",
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
        # The chunk's own `[cite:...]` text is neutralised first, so the only
        # markers in the answer are the ones `_template_sentence` appends (#283).
        sentences = [
            _template_sentence(s, c.external_id)
            for c in chosen
            for s in split_sentences(_neutralise_markers(c.text))
        ]
        # A sentence that opens with a heading, list marker or `>` starts a new
        # line (#399): joined by a space, `1. Open` sat mid-line, `_ORDERED_MARKER`
        # missed it, and `1.` was cut off as an uncited fragment.
        text = "".join(
            ("\n" if _BLOCK_PREFIX.match(s).end() else " ") + s if i else s
            for i, s in enumerate(sentences)
        )
        try:
            citations = enforce_citations(text, retrieved)
        # Reached when every chosen chunk is empty / punctuation-only (no cited
        # sentences), and when `retrieved` holds two ids a marker cannot tell
        # apart (the collision check at the top of `enforce_citations`). Until
        # #256 it was also reached by any id containing a terminator and a
        # space, which `split_sentences` split inside its own marker; until
        # #258 by a sentence ending in a terminator plus a closing
        # quote/bracket, which `_template_sentence` now keeps with its marker;
        # until #283 by a chunk whose own text contained a `[cite:` marker; until
        # #399 by a markdown heading, list or numbered item, whose mark the
        # template prefix pushed off the line start the #385 reads key on.
        except CitationError as e:
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
        "[cite:<external_id>] marker referencing the chunk that supports it, where "
        '<external_id> is copied character for character from that chunk\'s id="..." '
        "attribute. If the "
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
        # The id goes in verbatim. It used to be written with `!r`, which shows
        # `docs\guide.md` as `'docs\\guide.md'` and a tab as `\t`, so a model that
        # copied the id it was shown cited a string `enforce_citations` does not
        # know, and a grounded answer was refused as dangling (#282). Whatever
        # sits between the quotes is exactly what a `[cite:...]` must contain.
        return "\n\n".join(f'<chunk id="{r.external_id}">\n{r.text}\n</chunk>' for r in retrieved)

    def generate(
        self,
        query: str,
        retrieved: Sequence[RetrievalResult],
        *,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> GeneratedAnswer | Refusal:
        # Before the paid call (#422): the prompt is an f-string, so any object
        # was sent -- `b"..."` as the text `Query: b'...'`.
        if not isinstance(query, str):
            raise ValueError(f"query must be a string; got {type(query).__name__}")
        # `Retriever.search`'s rule (#424, #426): a blank query was sent as a
        # paid call with an empty `Query:` line.
        if is_blank(query):
            raise ValueError("query must be non-empty")
        threshold = _validate_threshold(threshold)
        top = _top_score(retrieved)
        if not retrieved:
            return _refusal("insufficient_context", "no chunks retrieved", threshold, top)
        if top < threshold:
            top_s, thr_s = render_comparison(top, threshold, places=_DETAIL_PLACES)
            return _refusal(
                "insufficient_context",
                f"top_score={top_s} below threshold={thr_s}",
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

        # An answer the model did not finish is not judged as if it were (#305).
        # Cut at `max_tokens` straight after a marker, it passed enforcement and
        # was published as a complete grounded answer; cut mid-sentence, the
        # refusal blamed its citations; a model refusal read as "no sentences".
        # Same reason as any unusable output -- the documented pair is unchanged
        # -- with a detail that names the real cause. A client that reports no
        # `stop_reason` (a duck-typed fake) is judged as before.
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason is not None and stop_reason not in _FINISHED_STOP_REASONS:
            detail = f"the model stopped with stop_reason={stop_reason!r}"
            if stop_reason == "max_tokens":
                detail += f" (max_tokens={self.max_tokens}); the answer is incomplete"
            return _refusal("unparseable_output", detail, threshold, top)

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


#: The `stop_reason`s that mean the model finished its answer (#305): it ended
#: its turn, or hit a stop sequence. `AnthropicGenerator` sends no tools and no
#: stop sequences of its own, so anything else -- `max_tokens`, `refusal`,
#: `pause_turn`, `tool_use` -- leaves an answer that is not the model's whole one.
_FINISHED_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})


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
