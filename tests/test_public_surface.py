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

These five tests lock that surface:

1. ``__version__`` is set to a semver-ish string.
2. Every name in ``__all__`` is bound on the package and non-None.
3. ``__all__`` agrees with the actual top-level relative ``from .X import …``
   names — guards against a future export being added to the imports
   block but not ``__all__`` (or vice versa).
4. The README's quickstart imports succeed.
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
import re
from pathlib import Path

import pytest

import rag_kit

_INIT_PATH = Path(rag_kit.__file__)
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+].+)?$")

# README's quickstart (line 115 in README.md) quotes these four names
# as importable directly from the top-level package.
README_QUICKSTART_NAMES = ("Document", "HashEmbedder", "Indexer", "Retriever")

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


def test_readme_quickstart_imports_resolve() -> None:
    """README quickstart must keep working as written.

    The README literally quotes (line 115)::

        from rag_kit import Document, HashEmbedder, Indexer, Retriever

    If any of those four names disappears from the top-level surface,
    every reader who copy-pastes the quickstart hits an ImportError.
    """
    for name in README_QUICKSTART_NAMES:
        assert hasattr(rag_kit, name), (
            f"rag_kit.{name} is missing from the top-level surface. The "
            f"README's quickstart imports it directly — either restore "
            f"the export or update the README quickstart."
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
