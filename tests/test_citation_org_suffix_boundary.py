"""An uncited claim ending in a company suffix rode on the next citation (#190).

`split_sentences` merges a fragment into the next when it ends in a known
abbreviation, so an abbreviation-bearing claim stays one sentence and its lone
`[cite:...]` marker satisfies enforcement. Five refined subsets have since been
carved out of `_ABBREVIATIONS` — each because the abbreviation *also* spells
something that naturally **ends** a claim, and merging unconditionally let an
uncited claim ride on the following sentence's marker (#130, #139, #152, #154).

The company suffixes were the sixth. `_NUMERIC_REFERENCE_ABBREVIATIONS`' comment
had ruled them out by name:

    Name/proper-noun-context abbreviations ("co"->"Acme Co.", "st"->"Main St.",
    "al"->"et al.") are deliberately NOT gated: they are followed by proper
    nouns, not digits, and are not common standalone claim-endings ...

`"al"` was gated anyway, by a *different* discriminator, because being followed
by a proper noun is what made it unsafe rather than what made it safe. Two of
the three members of that one sentence still carried the disproved rationale.

Everything below is one two-sided table driven through the real
`enforce_citations`. `REFUSE` means an **uncited** claim ends there and must not
ride on the next sentence's marker; `ACCEPT` means it is genuinely mid-sentence
and the whole thing is one cited claim. A table rather than separate one-off
tests because the failure mode being guarded is *asymmetry between abbreviations
that should behave the same way* — which a list of individual tests hides and a
grid does not.
"""

from __future__ import annotations

import pytest

from rag_kit.generator import (
    _ABBREVIATIONS,
    _ATTRIBUTION_ABBREVIATIONS,
    _ENUMERATION_ABBREVIATIONS,
    _NUMERIC_REFERENCE_ABBREVIATIONS,
    _ORG_SUFFIX_ABBREVIATIONS,
    _TIME_ABBREVIATIONS,
    _UNIT_COLLISION_ABBREVIATIONS,
    CitationError,
    enforce_citations,
)
from rag_kit.retriever import RetrievalResult

REFUSE = "refuse"
ACCEPT = "accept"

#: The sentence an uncited claim would ride on. Starts with a capital, so under
#: the follow-on discriminator it is an unambiguous new sentence.
CITED_NEXT = "The filing was made in 2020 [cite:doc-1]."


@pytest.fixture
def retrieved() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            external_id="doc-1", text="t", metadata={}, fused_score=0.9, ranks={"dense": 1}
        )
    ]


def _verdict(text: str, retrieved: list[RetrievalResult]) -> str:
    try:
        enforce_citations(text, retrieved)
    except CitationError:
        return REFUSE
    return ACCEPT


# --- the fix ----------------------------------------------------------------

ORG_SUFFIX_CASES = [
    # An uncited entity claim ends in the suffix. All five were ACCEPTED before.
    (REFUSE, f"The vendor of record is Acme Inc. {CITED_NEXT}"),
    (REFUSE, f"The counterparty is Acme Ltd. {CITED_NEXT}"),
    (REFUSE, f"The filing entity is Acme LLC. {CITED_NEXT}"),
    (REFUSE, f"The parent is Acme Corp. {CITED_NEXT}"),
    (REFUSE, f"The supplier is Acme Co. {CITED_NEXT}"),
    # Genuinely mid-sentence: a lowercase continuation. These must not regress
    # into false refusals -- closing a false-accept by refusing everything would
    # be a worse layer, not a better one.
    (ACCEPT, "Acme Inc. reported a loss in 2020 [cite:doc-1]."),
    (ACCEPT, "Acme Ltd. and Beta Corp. merged in 2019 [cite:doc-1]."),
    (ACCEPT, "Acme LLC. was dissolved in 2021 [cite:doc-1]."),
    (ACCEPT, "Acme Corp. acquired the unit [cite:doc-1]."),
    (ACCEPT, "Acme Co. supplies the parts [cite:doc-1]."),
]


@pytest.mark.parametrize(
    ("want", "text"), ORG_SUFFIX_CASES, ids=[t[:44] for _w, t in ORG_SUFFIX_CASES]
)
def test_org_suffix_boundary(want: str, text: str, retrieved: list[RetrievalResult]) -> None:
    assert _verdict(text, retrieved) == want


def test_the_cost_of_the_rule_is_a_title_case_continuation(
    retrieved: list[RetrievalResult],
) -> None:
    """The one form this discriminator gets wrong, pinned rather than hidden.

    A title-case continuation reads as a boundary, so a headline splits. That is
    a headline, not a claim sentence, and false-refusing is the direction this
    module has chosen five times (#126) -- but it is a real behaviour change and
    belongs in the record, not in a footnote.
    """
    assert _verdict("Acme Inc. Reports Record Revenue [cite:doc-1].", retrieved) == REFUSE


# --- the known gap, pinned at current behaviour (#191) -----------------------

KNOWN_GAP_CASES = [
    # These have the IDENTICAL false-accept, and are deliberately not fixed:
    # their attributive sense takes a CAPITALIZED continuation, so the follow-on
    # signal that separates the two senses everywhere else carries no
    # information here. Gating them would trade one false-accept for a frequent
    # false-refusal of correctly-grounded answers -- a posture question filed as
    # #191 rather than decided in #190.
    (ACCEPT, f"The company is headquartered in the U.S. {CITED_NEXT}"),
    (ACCEPT, f"The subsidiary operates in the U.K. {CITED_NEXT}"),
    (ACCEPT, f"The programme is administered by the U.N. {CITED_NEXT}"),
    (ACCEPT, f"The rule applies across the E.U. {CITED_NEXT}"),
    (ACCEPT, f"The office is on Main St. {CITED_NEXT}"),
]


@pytest.mark.parametrize(
    ("want", "text"), KNOWN_GAP_CASES, ids=[t[:44] for _w, t in KNOWN_GAP_CASES]
)
def test_geo_and_street_gap_is_recorded_not_forgotten(
    want: str, text: str, retrieved: list[RetrievalResult]
) -> None:
    """Pins the **current** (leaky) behaviour on purpose.

    These rows are a false-accept, not a desired outcome. They are asserted so
    the gap is a recorded fact with a pointer to #191 rather than an absence,
    and so whoever changes it does so deliberately and updates this file --
    exactly the situation where a silent behaviour change would otherwise look
    like an unrelated regression.
    """
    assert _verdict(text, retrieved) == want


ATTRIBUTIVE_USES_THE_GAP_PROTECTS = [
    "The U.S. government filed the brief [cite:doc-1].",
    "The U.S. Federal Reserve raised rates [cite:doc-1].",
    "The U.N. Security Council met in March [cite:doc-1].",
    "St. Peter's Basilica was completed in 1626 [cite:doc-1].",
]


@pytest.mark.parametrize("text", ATTRIBUTIVE_USES_THE_GAP_PROTECTS, ids=lambda t: t[:44])
def test_the_attributive_uses_that_argue_against_closing_the_gap(
    text: str, retrieved: list[RetrievalResult]
) -> None:
    """The other half of #191's tradeoff, as running code.

    Every one of these takes a *capitalized* continuation, so a lowercase-follow-on
    gate on the geo initialisms or `st` would flip each of them to a refusal --
    a correctly-grounded answer rejected. This is the evidence the decision-revisit
    rests on; if someone closes #191 by adding the gate, these four are what they
    are trading away, and the test says so before the argument has to be
    reconstructed.
    """
    assert _verdict(text, retrieved) == ACCEPT


# --- the four already-refined subsets must be untouched ---------------------

REGRESSION_CASES = [
    (REFUSE, f"We support JSON, CSV, etc. {CITED_NEXT}"),
    (ACCEPT, "We support JSON, CSV, etc. in the parser [cite:doc-1]."),
    (REFUSE, f"The outage started at 5 p.m. {CITED_NEXT}"),
    (ACCEPT, "The alert fired at 9 a.m. sharp on Monday [cite:doc-1]."),
    (REFUSE, f"The p50 was 5 ms. {CITED_NEXT}"),
    (ACCEPT, "The report names Ms. Chen as the author [cite:doc-1]."),
    (REFUSE, f"The answer is no. {CITED_NEXT}"),
    (ACCEPT, "See No. 5 for the breakdown [cite:doc-1]."),
    (REFUSE, f"The method was introduced by Vaswani et al. {CITED_NEXT}"),
    (ACCEPT, "Smith et al. found the same effect [cite:doc-1]."),
    (ACCEPT, "The study was led by Dr. Chen in 2020 [cite:doc-1]."),
]


@pytest.mark.parametrize(
    ("want", "text"), REGRESSION_CASES, ids=[t[:44] for _w, t in REGRESSION_CASES]
)
def test_already_refined_subsets_are_unaffected(
    want: str, text: str, retrieved: list[RetrievalResult]
) -> None:
    assert _verdict(text, retrieved) == want


# --- structure ---------------------------------------------------------------


def test_every_refined_subset_is_inside_the_parent_set() -> None:
    """A member of a refined subset that is not in `_ABBREVIATIONS` is dead code:
    `_ends_with_abbreviation` only consults the subsets inside the parent's
    `if`, so the gate would never run for it."""
    for name, subset in (
        ("numeric", _NUMERIC_REFERENCE_ABBREVIATIONS),
        ("unit", _UNIT_COLLISION_ABBREVIATIONS),
        ("time", _TIME_ABBREVIATIONS),
        ("enumeration", _ENUMERATION_ABBREVIATIONS),
        ("attribution", _ATTRIBUTION_ABBREVIATIONS),
        ("org suffix", _ORG_SUFFIX_ABBREVIATIONS),
    ):
        assert subset <= _ABBREVIATIONS, f"{name} subset has a member outside _ABBREVIATIONS"


def test_the_refined_subsets_are_disjoint() -> None:
    """`_ends_with_abbreviation` checks the subsets in a fixed order and returns
    from the first match, so an abbreviation in two subsets would silently get
    whichever discriminator happens to be checked first."""
    subsets = [
        _NUMERIC_REFERENCE_ABBREVIATIONS,
        _UNIT_COLLISION_ABBREVIATIONS,
        _TIME_ABBREVIATIONS,
        _ENUMERATION_ABBREVIATIONS,
        _ATTRIBUTION_ABBREVIATIONS,
        _ORG_SUFFIX_ABBREVIATIONS,
    ]
    seen: set[str] = set()
    for subset in subsets:
        overlap = seen & subset
        assert not overlap, f"abbreviation in two refined subsets: {sorted(overlap)}"
        seen |= subset
