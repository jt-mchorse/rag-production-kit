"""The refusal detail asserts an ordering — it has to stay readable as one.

`generate` decides at full float precision (`top < threshold`) and explains
the decision in `Refusal.detail`. Every verdict in every colliding case is
*correct*, which is why nothing in the suite could go red over #225: the
defect lived only in the sentence, never in the gate. So these arms assert on
the rendered string, and the gate arms elsewhere in `test_generator.py` stay
as they are.

Three things each arm here has to survive, beyond the unfixed tree:

* **The "widen one side" neighbour.** Widening only `top` passes any arm that
  merely checks the two strings differ. It is caught by the *structural*
  same-precision arm, not by an ordering arm — see
  `test_both_sides_render_at_the_same_precision`.
* **The "wider fixed width" neighbour.** Moving `.4f` to `.8f` passes the
  0.84999999 case and fails 0.8499999999. Both margins are in the table.
* **A silent re-render of ordinary refusals.** Pinned byte-for-byte by
  `test_ordinary_refusal_detail_is_unchanged`, which is *green* against the
  unfixed tree on purpose: it is the arm that rejects a fix that widened
  everything unconditionally.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest

from rag_kit.comparison import COMPARISON_MAX_PLACES, render_comparison
from rag_kit.generator import AnthropicGenerator, Refusal, TemplateGenerator
from rag_kit.retriever import RetrievalResult

_DETAIL_RE = re.compile(r"top_score=(\S+) below threshold=(\S+)")

# Both shipped backends. Hand-listed here *and* cross-checked against an
# AST-discovered population in `test_every_generate_backend_is_covered`, so a
# third backend fails this module instead of quietly not being tested by it.
# `AnthropicGenerator` reaches the refusal before `_ensure_client`, so the
# threshold path needs no API key and no SDK.
BACKENDS = (TemplateGenerator, AnthropicGenerator)


def _chunk(score: float) -> RetrievalResult:
    return RetrievalResult(
        external_id="d1", text="A sentence.", metadata={}, fused_score=score, ranks={}
    )


def _refuse(backend: type, top: float, threshold: float) -> Refusal:
    result = backend().generate("q", [_chunk(top)], threshold=threshold)
    assert isinstance(result, Refusal), f"{backend.__name__} answered instead of refusing"
    assert result.reason == "insufficient_context"
    return result


# `top` is strictly below `threshold` in every row, so every row is a refusal
# and the sentence claims an ordering. The two orientations matter separately:
# see the same-precision arm.
COLLAPSING_MARGINS = [
    pytest.param(0.84999999, 0.85, id="score-carries-the-expansion"),
    pytest.param(0.8499999999, 0.85, id="score-carries-more-than-.8f-would-show"),
    pytest.param(0.85, 0.8500001, id="threshold-carries-the-expansion"),
    pytest.param(0.02 - 1e-12, 0.02, id="at-the-default-threshold"),
    pytest.param(-1234.567890123, -1234.5678901, id="negative-top-score-#69-region"),
]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
@pytest.mark.parametrize(("top", "threshold"), COLLAPSING_MARGINS)
def test_detail_never_renders_the_two_sides_identically(
    backend: type, top: float, threshold: float
) -> None:
    """The sentence says one number is below another; they must not read equal."""
    detail = _refuse(backend, top, threshold).detail
    match = _DETAIL_RE.fullmatch(detail)
    assert match is not None, detail
    rendered_top, rendered_threshold = match.groups()
    assert rendered_top != rendered_threshold, detail


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
@pytest.mark.parametrize(("top", "threshold"), COLLAPSING_MARGINS)
def test_both_sides_render_at_the_same_precision(
    backend: type, top: float, threshold: float
) -> None:
    """Structural, not an ordering check — and that is the whole point.

    Measured, by building the "widen only `top`" neighbour and running this
    module against it: the ordering arm below goes red on **2 of the 5** rows
    and this one goes red on all 5. Three rows stay green there because `top`
    is the side carrying the long expansion, so widening it alone still comes
    out ordered correctly — and `top` is the side that carries it whenever the
    threshold is a round configured number, which `_DEFAULT_THRESHOLD = 0.02`
    makes the common case. Of the two rows that do catch it, only
    `threshold-carries-the-expansion` catches it for the reason you would
    predict; `negative-top-score-#69-region` catches it because rounding the
    *narrow* side away from zero flips the comparison, which is a second way
    the mismatch lies and one a hand-built table of positive scores would
    never have surfaced.

    Counting decimal places instead catches the neighbour in every row. Same
    lesson `prompt-regression-suite#175` and `ai-app-integration-tests#125`
    measured: a structural arm beats an outcome arm when the property is a
    relationship between two renderings rather than a fact about one.

    The `repr` branch is the one case where the two strings can differ in
    length, because `repr` round-trips each side exactly. No row here reaches
    it, and this arm asserts that rather than skipping over it; the branch has
    its own arm below.
    """
    detail = _refuse(backend, top, threshold).detail
    match = _DETAIL_RE.fullmatch(detail)
    assert match is not None, detail
    rendered_top, rendered_threshold = match.groups()
    # Every row above is separable in fixed point, so asserting that rather
    # than skipping past the `repr` branch: a skip here would be a silent hole
    # the day a row stops reaching the code this arm is about.
    assert "e" not in rendered_top, detail
    assert "e" not in rendered_threshold, detail
    assert "." in rendered_top, detail
    assert "." in rendered_threshold, detail
    assert len(rendered_top.split(".")[1]) == len(rendered_threshold.split(".")[1]), detail


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
@pytest.mark.parametrize(("top", "threshold"), COLLAPSING_MARGINS)
def test_the_rendered_ordering_matches_the_verdict(
    backend: type, top: float, threshold: float
) -> None:
    """Read the sentence back as a reader would and check it is true."""
    detail = _refuse(backend, top, threshold).detail
    match = _DETAIL_RE.fullmatch(detail)
    assert match is not None, detail
    rendered_top, rendered_threshold = match.groups()
    assert float(rendered_top) < float(rendered_threshold), detail


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
def test_ordinary_refusal_detail_is_unchanged(backend: type) -> None:
    """Byte-identical to what this repo published before #225.

    GREEN against the unfixed tree, deliberately. This is the arm that rejects
    the neighbour which "fixes" the class by widening every refusal to some
    larger fixed width: a four-place refusal is the overwhelmingly common one,
    and republishing it as `0.50000000` would be a regression the collapse
    arms above cannot see. The equivalent lock in `llm-eval-harness` is what
    caught exactly that neighbour there (#252).
    """
    assert _refuse(backend, 0.5, 0.85).detail == "top_score=0.5000 below threshold=0.8500"


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.__name__)
def test_full_precision_values_survive_on_the_dataclass(backend: type) -> None:
    """The data was never the defect; pin that the fix did not become one.

    `Refusal.top_score` / `.used_threshold` carry the real floats. A fix that
    rounded the *stored* values to match the sentence would make the gate less
    precise to make the message consistent — the standing anti-pattern this
    portfolio has now rejected in five repos.
    """
    refusal = _refuse(backend, 0.84999999, 0.85)
    assert refusal.top_score == 0.84999999
    assert refusal.used_threshold == 0.85


def test_repr_fallback_is_reachable_at_1e_minus_5() -> None:
    """The divergence from the sibling repos, pinned as a fact about doubles.

    `prompt-regression-suite` D-012 and `llm-eval-harness` D-026 bound the
    widening loop at 17 places and argue it always separates two distinct
    values, because their operands live near magnitude 1. This repo's do not:
    `_top_score` is documented negative-capable (#69) and `_validate_threshold`
    accepts any finite float. At 1e-5 — an unremarkable fused score — adjacent
    doubles still collide at 17 places and need 25, so the `repr` fallback is
    load-bearing here rather than a subnormal-scale formality.

    Asserted against `math.nextafter` rather than a typed-in literal, so the
    arm states the property instead of a transcription of it.
    """
    small = 1e-5
    neighbour = math.nextafter(small, math.inf)
    assert small != neighbour
    assert f"{small:.{COMPARISON_MAX_PLACES}f}" == f"{neighbour:.{COMPARISON_MAX_PLACES}f}"

    left, right = render_comparison(small, neighbour, places=4)
    assert left != right
    assert (left, right) == (repr(small), repr(neighbour))
    assert float(left) < float(right)


def test_caller_width_past_the_ceiling_is_still_honoured() -> None:
    """`places` above `COMPARISON_MAX_PLACES` must not fall straight to `repr`.

    A bare `range(places, COMPARISON_MAX_PLACES + 1)` is empty there, which
    would silently discard the width the caller asked for. No site in this repo
    passes such a width today; the arm exists because the failure is invisible
    — `repr` output is *correct*, just not what was requested.
    """
    assert render_comparison(0.1, 0.2, places=20) == (
        f"{0.1:.20f}",
        f"{0.2:.20f}",
    )


def test_equal_inputs_are_not_widened() -> None:
    """Unreachable through the strict `<` gate; the helper is total regardless."""
    assert render_comparison(0.85, 0.85, places=4) == ("0.8500", "0.8500")


def test_negative_places_is_rejected() -> None:
    with pytest.raises(ValueError, match="places must be non-negative"):
        render_comparison(0.1, 0.2, places=-1)


# ----------------------------------------------------------------------
# Population arms — discovered, not transcribed
# ----------------------------------------------------------------------

_GENERATOR_SRC = Path(__file__).resolve().parent.parent / "rag_kit" / "generator.py"


def _fixed_precision_spec_count(node: ast.JoinedStr) -> int:
    """Number of interpolations in one f-string carrying a `.Nf` format spec."""
    count = 0
    for value in node.values:
        if not isinstance(value, ast.FormattedValue) or value.format_spec is None:
            continue
        spec = "".join(
            part.value for part in value.format_spec.values if isinstance(part, ast.Constant)
        )
        if re.search(r"\.\d+f", spec):
            count += 1
    return count


def test_no_f_string_in_generator_renders_a_comparison_pair_inline() -> None:
    """The rule, over the whole module rather than the two known call sites.

    #225's two sites were byte-identical, and a third generator backend is the
    obvious next change to this file. A test naming `TemplateGenerator` and
    `AnthropicGenerator` would pass on the day a third one ships with the same
    inline f-string, so the population is discovered instead.

    Scoped to `generator.py` and to *two or more* fixed-precision
    interpolations in one f-string: a single formatted value in a string is not
    this class — the repo's benchmark scripts are full of correct ones — and a
    pair in one sentence is. This is what the collapsed detail looked like::

        f"top_score={top:.4f} below threshold={threshold:.4f}"
    """
    tree = ast.parse(_GENERATOR_SRC.read_text(encoding="utf-8"))
    offenders = [
        (node.lineno, _fixed_precision_spec_count(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr) and _fixed_precision_spec_count(node) >= 2
    ]
    assert offenders == [], (
        f"{_GENERATOR_SRC.name} renders a comparison pair inline at line(s) "
        f"{[lineno for lineno, _ in offenders]}; route it through "
        "rag_kit.comparison.render_comparison (#225, D-021)"
    )


def test_every_generate_backend_is_covered() -> None:
    """Anti-vacuity for the arm above *and* for `BACKENDS`.

    Two separate ways this module could go quietly hollow: the AST rule could
    be walking a corpus with no `generate` methods in it at all, and `BACKENDS`
    could drift behind the module. One assertion closes both — the set of
    classes in `generator.py` defining a `generate` method must be exactly the
    set this module parametrizes over.

    `Generator` itself is excluded: it is the `Protocol`, whose `generate` is a
    docstring-only declaration with no refusal path to render.
    """
    tree = ast.parse(_GENERATOR_SRC.read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(child, ast.FunctionDef) and child.name == "generate" for child in node.body
        )
    } - {"Generator"}
    assert defined, "no generate backends discovered — the AST arms walk an empty corpus"
    assert defined == {b.__name__ for b in BACKENDS}, defined
