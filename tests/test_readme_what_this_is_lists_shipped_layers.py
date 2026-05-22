"""README 'What this is' section ↔ shipped layers snapshot (#27).

Before #27, the closing paragraph of the What-this-is section claimed
``Everything beyond #1 + #2 + #3 + #4 + #5 + #6 is staged in follow-up
issues``, with #7 (eval-harness integration) described as pending and
#8 (Next.js demo) unmentioned — even though both shipped on 2026-05-16
and 2026-05-18 respectively. The Architecture section already said
``Eight runtime layers ship today`` and listed both. Top-to-bottom
readers got contradictory state-of-the-repo claims from the same file.

This test locks the canonical set of bold sub-sections inside the
What-this-is section to the eight shipped layers, hard-pins absence
of the stale ``staged in follow-up issues`` wording, and cross-checks
that the Architecture section's eight-layer claim still parses cleanly
and agrees on the count.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

# The canonical eight-layer set: each entry is the leading text of a
# bold-section opener inside the What-this-is section, with the parens
# trailer (e.g. " (#7)") stripped. Matched as a *prefix* of the bold
# text, so a future copy-edit that adds an "and …" clause keeps passing.
CANONICAL_LAYERS: frozenset[str] = frozenset(
    {
        "Hybrid retrieval",
        "Cross-encoder reranking",
        "Citation enforcement and weak-context refusal",
        "Streaming intermediate events",
        "Query rewriting and decomposition",
        "Cost telemetry",
        "Eval-harness integration",
        "Next.js demo frontend",
    }
)

EIGHT_LAYERS_PHRASE = "Eight runtime layers ship today"
# Stale wording matched whitespace-insensitively because the pre-#27 README
# wrapped the phrase mid-line as ``staged in follow-up\nissues``.
STALE_PHRASE_RE = re.compile(r"staged\s+in\s+follow-up\s+issues")

_BOLD_OPENER_RE = re.compile(r"^\*\*([^*]+)\*\*", re.MULTILINE)
_SECTION_RE = re.compile(r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)


@pytest.fixture(scope="module")
def sections() -> dict[str, str]:
    text = README.read_text(encoding="utf-8")
    return {m.group(1).strip(): m.group(2) for m in _SECTION_RE.finditer(text)}


@pytest.fixture(scope="module")
def what_this_is(sections: dict[str, str]) -> str:
    assert "What this is" in sections, "README missing '## What this is' section"
    return sections["What this is"]


@pytest.fixture(scope="module")
def architecture(sections: dict[str, str]) -> str:
    assert "Architecture" in sections, "README missing '## Architecture' section"
    return sections["Architecture"]


def _bold_openers(section_text: str) -> list[str]:
    """The leading text of every ``**...**`` opener at the start of a line.

    Only matches bold runs at column 0 (true section openers), not
    mid-paragraph bold emphasis.
    """
    return [m.group(1).strip() for m in _BOLD_OPENER_RE.finditer(section_text)]


def test_what_this_is_lists_all_eight_shipped_layers(what_this_is: str) -> None:
    """Each canonical layer must appear as a bold sub-section opener."""
    openers = _bold_openers(what_this_is)
    missing = [
        layer
        for layer in CANONICAL_LAYERS
        if not any(opener.startswith(layer) for opener in openers)
    ]
    assert not missing, (
        "README 'What this is' section is missing bold sub-sections for "
        f"these shipped layers: {sorted(missing)}. The repo ships eight "
        "layers today (#1 through #8); each needs its own bold sub-section "
        "of the same shape as the existing ones. Found openers: "
        f"{openers!r}"
    )


def test_what_this_is_has_no_extra_bold_openers(what_this_is: str) -> None:
    """Bold openers in the What-this-is section must be exactly the canonical set.

    Catches the inverse drift: a future edit that adds a bold sub-section
    for a layer that hasn't actually shipped.
    """
    openers = _bold_openers(what_this_is)
    extras = [
        opener
        for opener in openers
        if not any(opener.startswith(layer) for layer in CANONICAL_LAYERS)
    ]
    assert not extras, (
        "README 'What this is' section has bold sub-section openers that "
        f"do not match any canonical shipped layer: {extras!r}. If a new "
        "layer shipped, add it to CANONICAL_LAYERS in this test. If a bold "
        "run is mid-paragraph emphasis (not a sub-section opener), refactor "
        "so it doesn't sit at column 0."
    )


def test_what_this_is_does_not_claim_layers_are_staged(what_this_is: str) -> None:
    """Hard-pin the original drift: closing paragraph must not say a layer is pending.

    The exact wording the pre-#27 README used wrapped mid-line as
    ``staged in follow-up\\nissues``. Match the phrase whitespace-insensitively
    so a future copy-paste from an old version of the section, in either
    wrapped or unwrapped form, can't silently reintroduce the bug.
    """
    match = STALE_PHRASE_RE.search(what_this_is)
    assert match is None, (
        f"README 'What this is' section contains the stale phrase "
        f"`{match.group(0)!r}` (matched against /staged\\s+in\\s+follow-up\\s+issues/), "
        "which was the original drift in #27. If a layer is genuinely "
        "deferred to a follow-up issue, rephrase so the claim is anchored "
        "to the specific layer (with a `[#N]` link) and update "
        "CANONICAL_LAYERS in this test."
    )


def test_architecture_section_agrees_eight_layers_ship(architecture: str) -> None:
    """Architecture must still claim eight runtime layers ship today.

    Cross-check between sections. If the canonical set ever shrinks or
    grows, the Architecture section's intro line is the other place that
    has to be updated; pinning it here makes that requirement load-bearing.
    """
    assert EIGHT_LAYERS_PHRASE in architecture, (
        f"README 'Architecture' section no longer contains the phrase "
        f"`{EIGHT_LAYERS_PHRASE!r}`. The What-this-is section's canonical "
        "layer set is sized to eight; if the architecture changed, update "
        "both the Architecture intro line and CANONICAL_LAYERS in this test."
    )
    assert len(CANONICAL_LAYERS) == 8, (
        "CANONICAL_LAYERS must contain exactly eight entries — matches the "
        "Architecture section's 'Eight runtime layers ship today' claim."
    )
