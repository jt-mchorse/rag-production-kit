"""Public-surface tests for ``rag_kit/__init__.py``.

``rag_kit`` re-exports 42 names from nine submodules (``embedder``,
``fusion``, ``generator``, ``indexer``, ``reranker``, ``retriever``,
``rewriter``, ``streaming``, ``telemetry``) and declares them in
``__all__`` + ``__version__``. Every other test in this suite imports
submodules directly (``from rag_kit.reranker import LexicalOverlapReranker``),
so silent renames or accidental ``__all__`` drops in ``__init__.py``
don't fail any test — but they break the README's quoted quickstart
(``from rag_kit import Document, HashEmbedder, Indexer, Retriever``) and
any downstream importer that uses the top-level surface.

These tests lock that surface:

1. ``__version__`` is set to a semver-ish string.
2. Every name in ``__all__`` is bound on the package and non-None.
3. ``__all__`` agrees with the actual top-level relative ``from .X import …``
   names — guards against a future export being added to the imports
   block but not ``__all__`` (or vice versa).
4. The README's quickstart imports succeed — **discovered** from
   ``README.md`` rather than transcribed beside the test (#223), in both
   spellings the README uses (top-level and dotted-submodule), with an
   anti-vacuity arm pinning the *corpus* it discovered.
5. Anchor names from each re-exported submodule are reachable via
   ``rag_kit`` — guards against a submodule being split or renamed
   without updating ``__init__.py``.

Same hygiene posture as the public-surface snapshots landed in
``llm-eval-harness`` (#25), ``llm-cost-optimizer`` (#23), and
``prompt-regression-suite`` (#20) this week. Adapted for relative
imports — ``rag_kit`` uses ``from .embedder import …`` rather than
``from rag_kit.embedder import …``, so the AST parser filters on
``level >= 1`` rather than ``module.startswith("rag_kit.")`` (same
adaptation prompt-regression-suite#20 used).

No ``importlib.reload`` workaround needed: ``rag_kit`` doesn't ship a
pytest plugin via entry-points, so ``__init__.py`` is instrumented by
``pytest-cov`` from the start (already at 100%).
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

import rag_kit

_INIT_PATH = Path(rag_kit.__file__)
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+].+)?$")

# The document a reader copy-pastes a quickstart out of. A *document*
# rather than a list of *names*: the names are discovered from it below.
#
# What this replaces was four names transcribed once, under a comment
# pinning them to "line 115 in README.md". By the time #223 read it the
# snippet was at line 150, the README carried **six** `from rag_kit import
# …` snippets naming 16 distinct top-level names, and a seventh import line
# used the dotted spelling. The tuple walked 4 of 16 names across 1 of 6
# snippets. A check that can only fail when someone edits the check is not a
# lock on the thing it names.
_REPO_ROOT = _INIT_PATH.parent.parent
QUICKSTART_DOC = "README.md"

# `from rag_kit …`, in both spellings the README uses:
#
#   README:150  from rag_kit import Document, HashEmbedder, Indexer, Retriever
#   README:151  from rag_kit.db import connect
#
# Two details of this pattern are load-bearing, and both were found by
# running it rather than by reading it:
#
# * **Leading whitespace is allowed.** `README:279` is indented — it sits
#   inside a continued example block. The sibling fix in
#   chunking-strategies-lab#194 anchors its regex at `^from` under
#   `re.MULTILINE`, which is correct for *that* README and silently walks 5
#   of 6 snippets here, dropping `aggregate_telemetry` — one of the two
#   names #223 notes is in neither the old tuple nor `SUBMODULE_ANCHORS`.
# * **The dotted group is optional and captured.** `rag_kit.db` is in none
#   of the nine `SUBMODULE_ANCHORS`, and `connect` is not a top-level
#   re-export, so the second line of the quickstart is pinned by nothing at
#   all today. A reader copy-pasting the quickstart runs both lines.
#
# An inline backticked mention in prose cannot match: it starts with a
# backtick, not with `from`.
_IMPORT_STMT = re.compile(
    r"^[ \t]*from[ \t]+rag_kit(?P<sub>(?:\.[A-Za-z_][A-Za-z0-9_]*)+)?[ \t]+"
    r"import[ \t]+(?P<body>\([^)]*\)|[^\n(]+?)[ \t]*$",
    re.MULTILINE,
)

# The names the README quoted when #223 was written, as a *floor*. The
# document is the source of truth; this set is the control on the discovery:
# if the regex or the `ast` parse regresses to finding fewer names, this says
# so instead of every downstream assertion passing over a shrunken set.
# The four the old tuple transcribed are the first line of it.
KNOWN_TOP_LEVEL_NAMES = frozenset(
    {
        "Document",
        "HashEmbedder",
        "Indexer",
        "Retriever",
        "GeneratedAnswer",
        "Refusal",
        "TemplateGenerator",
        "TemplateRewriter",
        "LexicalOverlapReranker",
        "StreamingPipeline",
        "to_sse",
        "CostRecord",
        "ModelPrice",
        "PriceTable",
        "TelemetryStore",
        "aggregate_telemetry",
    }
)

# Same floor for the dotted spelling. One statement, one name, and nothing
# else in this file looks at `rag_kit.db`.
KNOWN_DOTTED_IMPORTS = frozenset({("rag_kit.db", "connect")})

# Lower bounds on the corpus, read off the README as it stood for #223: six
# top-level statements and one dotted one. Bounds rather than equalities so
# that *adding* a snippet to the README is not a test failure, while losing
# one — or a discovery that has stopped discovering — is.
MIN_TOP_LEVEL_STATEMENTS = 6
MIN_DOTTED_STATEMENTS = 1

# Anchor names that prove each re-exported submodule survived. One name
# per submodule; if ``__init__.py`` ever drops a submodule's whole
# block, the corresponding anchor goes missing.
SUBMODULE_ANCHORS = {
    "embedder": "HashEmbedder",
    "fusion": "reciprocal_rank_fusion",
    "generator": "TemplateGenerator",
    "indexer": "Indexer",
    "reranker": "LexicalOverlapReranker",
    "retriever": "Retriever",
    "rewriter": "TemplateRewriter",
    "streaming": "StreamingPipeline",
    "telemetry": "TelemetryStore",
}


def _parse_init_relative_imports() -> set[str]:
    """Return the set of names imported into ``__init__.py`` via
    top-level relative ``from .X import (...)`` blocks."""
    tree = ast.parse(_INIT_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        # Relative import — ``level`` is the number of leading dots.
        if isinstance(node, ast.ImportFrom) and node.level >= 1:
            for alias in node.names:
                # An aliased import (``from .telemetry import aggregate
                # as aggregate_telemetry``) adds the alias to the public
                # surface, not the original name.
                names.add(alias.asname or alias.name)
    return names


def _discover_quickstart_imports() -> list[tuple[str, tuple[str, ...]]]:
    """Return one entry per ``from rag_kit …`` statement in the quickstart doc.

    Each entry is ``(module, names)`` — ``("rag_kit", (...))`` for the
    top-level spelling and ``("rag_kit.db", ("connect",))`` for the dotted
    one. A *list*, not a set union, so the caller can assert on the corpus
    (how many statements, in which spelling) and not only on the union of
    names. That distinction is what lets the vacuity arm below tell "found
    one snippet" apart from "found all six" — the failure mode this change
    is most likely to introduce.

    The statement body is handed to ``ast`` rather than split on commas, so
    a trailing comma, a parenthesised multi-line block or an ``as`` alias
    parse the way Python parses them. This is the technique
    ``test_all_matches_actual_top_level_imports`` already uses on
    ``__init__.py``, pointed at the document instead.
    """
    text = (_REPO_ROOT / QUICKSTART_DOC).read_text(encoding="utf-8")
    found: list[tuple[str, tuple[str, ...]]] = []
    for match in _IMPORT_STMT.finditer(text):
        module = "rag_kit" + (match.group("sub") or "")
        stmt = f"from {module} import {match.group('body')}"
        names: list[str] = []
        for node in ast.parse(stmt).body:
            assert isinstance(node, ast.ImportFrom)
            for alias in node.names:
                # The *imported* name is what has to exist on the module; an
                # ``as`` alias renames it only in the reader's own script.
                names.append(alias.name)
        found.append((module, tuple(names)))
    return found


def test_version_is_set_to_semver_ish_string() -> None:
    """``__version__`` is published; downstream importers and PyPI
    builds rely on it."""
    assert hasattr(rag_kit, "__version__"), (
        "rag_kit.__version__ is missing — packaging tools and downstream "
        "`rag_kit.__version__` lookups will break."
    )
    version = rag_kit.__version__
    assert isinstance(version, str), (
        f"rag_kit.__version__ should be a string, got {type(version).__name__}: {version!r}."
    )
    assert version, "rag_kit.__version__ is an empty string."
    assert _SEMVER_PATTERN.match(version), (
        f"rag_kit.__version__ = {version!r} doesn't look like semver "
        f"(expected MAJOR.MINOR.PATCH[-prerelease][+build])."
    )


def test_all_names_are_bound_and_non_none() -> None:
    """Every name in ``__all__`` must be importable and non-None.

    Catches the silent-failure where someone removes a re-import line
    but leaves the name in ``__all__``.
    """
    missing: list[str] = []
    none_valued: list[str] = []
    for name in rag_kit.__all__:
        if not hasattr(rag_kit, name):
            missing.append(name)
            continue
        if getattr(rag_kit, name) is None:
            none_valued.append(name)
    assert not missing, (
        f"rag_kit.__all__ advertises names that are not bound on the "
        f"package: {missing}. The most likely cause is a re-import line "
        f"was deleted from __init__.py but __all__ wasn't updated."
    )
    assert not none_valued, (
        f"rag_kit.__all__ entries bound to None: {none_valued}. A "
        f"re-import probably resolved to a missing submodule attribute."
    )


def test_all_matches_actual_top_level_imports() -> None:
    """``__all__`` should equal the set of top-level re-exports.

    Catches the inverse drift: someone adds a new ``from .X import Y``
    but forgets to add ``Y`` to ``__all__``, so ``import *`` silently
    misses the export.
    """
    advertised = set(rag_kit.__all__)
    imported = _parse_init_relative_imports()
    only_imported = imported - advertised
    only_advertised = advertised - imported
    assert not only_imported, (
        f"Names imported into rag_kit/__init__.py but missing from "
        f"__all__: {sorted(only_imported)}. Add them to __all__ or stop "
        f"importing them at the top level."
    )
    assert not only_advertised, (
        f"Names in rag_kit.__all__ but not imported at the top of "
        f"__init__.py: {sorted(only_advertised)}. Add the import or "
        f"remove the __all__ entry."
    )


def test_the_discovery_finds_every_quickstart_statement() -> None:
    """Anti-vacuity: pin the corpus before anything is asserted about it.

    Every other assertion in this file about the quickstart is a statement
    over ``_discover_quickstart_imports()``. If the regex matches **zero**
    blocks, all of them pass for free — and if it matches *most* blocks they
    still pass, which is the subtler and likelier regression. So this arm
    pins the corpus rather than the result: how many statements, in which
    spelling, and a floor on the names.

    The counts are lower bounds. Adding a snippet to the README is not a
    test failure; losing one, or a parser that has stopped parsing, is.

    Green against the pre-#223 tree in the sense that matters — the property
    it pins (the README carries six top-level snippets and one dotted one)
    was already true there. That is precisely the property the four-name
    tuple did not walk.
    """
    found = _discover_quickstart_imports()
    top_level = [entry for entry in found if entry[0] == "rag_kit"]
    dotted = [entry for entry in found if entry[0] != "rag_kit"]

    assert len(top_level) >= MIN_TOP_LEVEL_STATEMENTS, (
        f"the quickstart-import discovery found {len(top_level)} top-level "
        f"`from rag_kit import …` statements in {QUICKSTART_DOC}, expected at "
        f"least {MIN_TOP_LEVEL_STATEMENTS}. Either a snippet was removed from "
        f"the README, or `_IMPORT_STMT` stopped matching a spelling it uses "
        f"— an indented statement inside a continued example block is the one "
        f"that has actually been missed before (#223)."
    )
    assert len(dotted) >= MIN_DOTTED_STATEMENTS, (
        f"the discovery found {len(dotted)} dotted `from rag_kit.X import …` "
        f"statements, expected at least {MIN_DOTTED_STATEMENTS}. The "
        f"quickstart's second line is `from rag_kit.db import connect`; if it "
        f"is gone from the README that is fine, but if the pattern stopped "
        f"capturing the dotted spelling then `rag_kit.db` is unpinned again."
    )

    union = {name for _, names in top_level for name in names}
    assert union >= KNOWN_TOP_LEVEL_NAMES, (
        f"the discovery is missing top-level names it found when #223 was "
        f"written: {sorted(KNOWN_TOP_LEVEL_NAMES - union)}. The README is the "
        f"source of truth, but a *shrinking* result means the parser "
        f"regressed, not that the document did."
    )
    dotted_pairs = {(module, name) for module, names in dotted for name in names}
    assert dotted_pairs >= KNOWN_DOTTED_IMPORTS, (
        f"the discovery is missing dotted imports it found when #223 was "
        f"written: {sorted(KNOWN_DOTTED_IMPORTS - dotted_pairs)}."
    )


def test_quickstart_top_level_imports_resolve() -> None:
    """Every top-level name the quickstarts import must exist on ``rag_kit``.

    The names come from the README, not from a list beside this test. The
    tuple this replaces named four of them and pointed at "line 115"; the
    snippet was at 150 and five further snippets named twelve more names,
    including `to_sse` and `aggregate_telemetry`, which are in neither the
    tuple nor ``SUBMODULE_ANCHORS``. Three failures it could not see: a name
    added to one of the five unwalked snippets, a re-export dropped from
    ``__init__.py`` for any of those twelve, and its own drift.

    If any of these names disappears from the top-level surface, every
    reader who copy-pastes that snippet hits an ``ImportError``.
    """
    missing = sorted(
        {
            name
            for module, names in _discover_quickstart_imports()
            if module == "rag_kit"
            for name in names
            if not hasattr(rag_kit, name)
        }
    )
    assert not missing, (
        f"{QUICKSTART_DOC} imports names that are not on the top-level "
        f"surface: {missing}. A reader copy-pasting the snippet gets an "
        f"ImportError — either restore the exports or fix the snippet."
    )


def test_quickstart_dotted_imports_resolve() -> None:
    """The dotted spelling resolves too — on its own submodule.

    ``from rag_kit.db import connect`` is the second line of the quickstart,
    and ``db`` is in none of the nine ``SUBMODULE_ANCHORS`` (it is not a
    re-exported submodule), so nothing in this suite looked at it before
    #223. A reader copy-pasting the quickstart runs both lines, so both
    lines are part of the same claim.

    ``importlib`` rather than ``hasattr(rag_kit, "db")``: the submodule is
    not imported by ``__init__.py``, so the attribute does not exist on the
    package until something imports it.
    """
    unresolved: list[str] = []
    for module, names in _discover_quickstart_imports():
        if module == "rag_kit":
            continue
        try:
            imported = importlib.import_module(module)
        except ImportError:  # pragma: no cover - fails the assert below
            unresolved.extend(f"{module} (module missing)" for _ in names or [None])
            continue
        unresolved.extend(f"{module}.{name}" for name in names if not hasattr(imported, name))
    assert not unresolved, (
        f"{QUICKSTART_DOC} imports dotted names that do not resolve: "
        f"{sorted(unresolved)}. A reader copy-pasting the quickstart gets an "
        f"ImportError on the line after the top-level import."
    )


@pytest.mark.parametrize(
    ("submodule", "anchor"),
    sorted(SUBMODULE_ANCHORS.items()),
    ids=sorted(SUBMODULE_ANCHORS.keys()),
)
def test_submodule_anchor_re_exported(submodule: str, anchor: str) -> None:
    """One anchor per re-exported submodule survives at the top level.

    If a submodule is split or renamed (``rewriter.py`` →
    ``rewriter/__init__.py``, ``telemetry.py`` → ``observability.py``,
    etc.) and ``__init__.py`` isn't updated, the anchor name vanishes
    from ``rag_kit``.
    """
    assert hasattr(rag_kit, anchor), (
        f"`{anchor}` from `rag_kit.{submodule}` is no longer re-exported "
        f"at the top level. Did `{submodule}` move or get renamed? "
        f"Update `rag_kit/__init__.py` to re-export from the new path."
    )
