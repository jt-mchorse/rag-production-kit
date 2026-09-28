"""Rendering a comparison the surrounding prose asserts an ordering about.

`generator.py` decides at full float precision — the refusal gate is
`top < threshold` — and then explains that decision in a string handed back
to the caller as `Refusal.detail`. Rendering both sides of the explanation at
a fixed width made the explanation contradict itself at a near miss (#225)::

    top=0.84999999  threshold=0.85  -> refuse
    -> "top_score=0.8500 below threshold=0.8500"

A near-threshold refusal is the ordinary shape of a marginal retrieval, and
it is exactly when someone debugging "why did this query refuse?" reads the
sentence most carefully. So the width is chosen by asking whether the two
values actually render differently, and widening while they do not.

Duplicated from `prompt-regression-suite`'s D-012 rather than shared, the
same call `llm-eval-harness` D-026 made. The three are separate
distributions with no dependency between them, and manufacturing one so a
short formatter could be imported would be a worse trade than the
duplication. Saying so here so it does not read as accidental.

**What does not transfer from those two.** Both of them bound the widening
loop at 17 decimal places and justify it by their operating region: a cosine
lives in ``[-1, 1]`` and a threshold in ``(0, 1]``, where the gap between
adjacent doubles is ~2.2e-16, so 17 places always separates two distinct
values and the `repr` fallback is a subnormal-scale formality. Neither bound
holds here. `_top_score` is documented as genuinely negative for a long,
low-overlap chunk (`overlap - length_penalty·len(text)`, #69), and
`_validate_threshold` accepts *any finite float* on purpose. At a magnitude
of 1e-5 — an unremarkable fused score, and well inside what a caller may set
`threshold` to — two adjacent doubles still render identically at 17 places
and need 25. So in this module the `repr` fallback is not a formality; it is
the only thing making the function total, and it is reachable at ordinary
magnitudes. It is pinned by a test at 1e-5 rather than at subnormal scale.
"""

from __future__ import annotations

#: Ceiling on widening the fixed-point rendering. Unlike the sibling repos
#: this is a *budget*, not a proof of separation — see the module docstring.
#: Past this width a fixed-point rendering of a small-magnitude value is long
#: enough to be worse to read than the `repr` the fallback gives instead.
COMPARISON_MAX_PLACES = 17


def render_comparison(value: float, other: float, *, places: int) -> tuple[str, str]:
    """Render two numbers so an ordering stated between them stays visible.

    Widens from `places` only while the two render identically, and always
    returns both sides at the same precision.

    `places` is a **required** keyword argument, not a default. Centralising
    inline formatters onto one helper with a hardcoded width silently
    re-renders every call site that disagreed with that width — the
    regression `llm-eval-harness#252` shipped, where five sites at three
    places and three at four were collapsed onto three and a published
    column narrowed. This module's one caller renders at four places and the
    sibling repos render at three; nothing here should have an opinion about
    which is right.

    **A wider fixed width is not the same fix.** `.4f` here is already wider
    than `prompt-regression-suite`'s `.3f` and collided just the same; `.8f`
    would move the margin again without closing the class. Deciding on the
    rendered strings cannot drift from what the reader sees, because it *is*
    what the reader sees.

    **Both sides, at the same precision, is the half that is easy to miss.**
    Widening only the side that needs it looks correct for as long as that
    side carries the long decimal expansion — which is what happens whenever
    the threshold is a round configured number, the orientation
    `_DEFAULT_THRESHOLD = 0.02` always gives. In the other orientation, a
    threshold derived by arithmetic, that neighbour renders
    ``top_score=0.8500000000 below threshold=0.8500``, which read as written
    states the reverse of the verdict it is explaining.

    Returns two renderings that are either both fixed-point at one width, or
    both `repr`. The `repr` branch is the one case where the two strings can
    have different lengths, and it is deliberate: `repr` round-trips a float
    by definition, so it is the strongest available form of "both sides shown
    exactly", not a weaker one.

    Equal inputs return the narrow rendering unwidened: there is nothing to
    distinguish, and widening would imply a difference that is not there. The
    caller never relies on that — the refusal detail is reached only when
    ``top < threshold`` strictly — but the function is total, so it says what
    it does.
    """
    if places < 0:
        raise ValueError(f"places must be non-negative; got {places!r}")
    if value == other:
        return (f"{value:.{places}f}", f"{other:.{places}f}")
    # `max(...)` so a caller asking for a width past the ceiling still gets its
    # own width tried. A bare `range(places, COMPARISON_MAX_PLACES + 1)` is
    # empty for `places > COMPARISON_MAX_PLACES`, which would send every such
    # call straight to `repr` — silently ignoring the width it asked for.
    for width in range(places, max(places, COMPARISON_MAX_PLACES) + 1):
        rendered = (f"{value:.{width}f}", f"{other:.{width}f}")
        if rendered[0] != rendered[1]:
            return rendered
    # Two distinct floats no fixed-point width in the budget separates. Not
    # only a subnormal-scale corner here: see the module docstring.
    return (repr(value), repr(other))
