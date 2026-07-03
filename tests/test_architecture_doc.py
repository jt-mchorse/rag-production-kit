"""Architecture-doc lock: catch drift between `docs/architecture.md` and
the actual shipped surface of the repo.

Sister to the architecture-doc locks shipped this same week in
``embedding-model-shootout`` (PR #20), ``vector-search-at-scale``
(PR #22), ``llm-eval-harness`` (PR #30), ``prompt-regression-suite``
(PR #25), and ``llm-cost-optimizer`` (PR #28), plus the JS variants
in ``mcp-server-cookbook``, ``nextjs-streaming-ai-patterns``, and
``ai-app-integration-tests``.

This repo's doc uses BOTH ``(#NN)`` issue references AND ``D-NNN``
core-decision references, so the lock has *two* coverage axes (in
contrast to ``llm-cost-optimizer`` PR #28 which is D-NNN-only and
``vector-search-at-scale`` PR #22 which is #NN-only).

Four invariants pinned:

1. **Path-token reachability.** Every backtick-quoted token that starts
   with one of the ``RESOLVABLE_PREFIXES`` resolves on disk. Operator-
   supplied future artifacts are allow-listed in
   ``OPERATOR_SUPPLIED_PATHS``. Placeholder shapes (``<...>``, ``{...}``)
   skipped.

2. **Closed-feature-issue coverage.** Every issue in
   ``KNOWN_SHIPPED_ISSUES`` is referenced at least once. So a future
   ninth core deliverable can't ship without the doc updating.

3. **Active-decision coverage.** Every non-superseded ``D-NNN`` in
   ``MEMORY/core_decisions_ai.md`` (excluding ``D-001`` scope baseline)
   is referenced at least once.

4. **Banned-phrase absence.** Phrases that characterized pre-fix drift
   elsewhere in the portfolio.

Hard-pin tests lock each of the constants so a future loose edit can't
silently weaken the guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "architecture.md"
DECISIONS = REPO_ROOT / "MEMORY" / "core_decisions_ai.md"


BANNED_PHRASES = (
    "this pr",
    "pending downstream",
    "(unfiled)",
    "to-be-filed",
)


RESOLVABLE_PREFIXES = (
    "rag_kit/",
    "evals/",
    "scripts/",
    "frontend/",
    "docs/",
    "tests/",
    ".github/",
)


# Operator-supplied artifacts: paths the doc names as the file an operator
# commits *after* running a real workload. These deliberately don't exist
# in-repo (no-fabricated-benchmarks posture). Hard-pinned so future
# additions are intentional, not accidents. Empty by default — added on
# demand when authoring uncovers a legitimate operator-only path.
OPERATOR_SUPPLIED_PATHS: tuple[str, ...] = ()


# Core deliverables (handoff §2). Each shipped surface is annotated in
# the doc with its origin issue number; this set is the inventory.
KNOWN_SHIPPED_ISSUES = (1, 2, 3, 4, 5, 6, 7, 8, 50)


MIN_ACTIVE_DECISION_ID = 2


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def active_decisions() -> tuple[int, ...]:
    """Parse ``MEMORY/core_decisions_ai.md`` for non-superseded ``D-NNN``
    entries whose numeric id is ``>= MIN_ACTIVE_DECISION_ID``.
    """
    text = DECISIONS.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=- id:)", text)
    active: list[int] = []
    for block in blocks:
        id_match = re.search(r"- id:\s*D-(\d+)", block)
        if not id_match:
            continue
        sup_match = re.search(r"superseded_by:\s*(\S+)", block)
        is_active = (sup_match is None) or (sup_match.group(1).strip().lower() == "null")
        if is_active:
            n = int(id_match.group(1))
            if n >= MIN_ACTIVE_DECISION_ID:
                active.append(n)
    return tuple(sorted(active))


def _extract_backtick_paths(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"`([^`\n]+)`", text):
        token = match.group(1).strip()
        for prefix in RESOLVABLE_PREFIXES:
            if token.startswith(prefix):
                while token and token[-1] in ".,;:":
                    token = token[:-1]
                token = re.sub(r"\(\)$", "", token)
                if "<" in token or "{" in token:
                    break
                if token:
                    found.add(token)
                break
    return found


def _resolves_on_disk(token: str) -> bool:
    return (REPO_ROOT / token).exists()


# CamelCase symbols the doc legitimately names that are NOT owned by this
# package — cross-repo references. `RunResult` is `eval-harness`'s type (the
# evals-pipeline section: "each writes one `RunResult` JSON via `eval-harness`");
# rag_kit's own result types are `RetrievalResult` / `RewriteResult`. Excluded
# from the public-surface check so a correct external reference isn't a false
# positive. Hard-pinned by `test_external_symbols_hard_pin_set` so this seam
# can't silently grow into a hole that hides real drift.
EXTERNAL_SYMBOLS = ("RunResult",)


def _extract_camel_symbols(text: str) -> set[str]:
    """Backtick-quoted multi-word CamelCase identifiers (an internal
    lowercase->uppercase boundary, e.g. `HashEmbedder`, `UnknownModelError`).

    Single-word capitalized tokens (`Backend`-in-prose) and all-caps tokens
    (`SCORE`) are excluded so only genuine class-name claims are checked. Bare
    snake_case is not locked — it collides with field/column names the doc
    quotes, which are not importable symbols.
    """
    found: set[str] = set()
    for match in re.finditer(r"`([^`\n]+)`", text):
        token = match.group(1).strip()
        token = re.sub(r"\(\)$", "", token)
        while token and token[-1] in ".,;:":
            token = token[:-1]
        if re.fullmatch(r"[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*", token) and re.search(
            r"[a-z][A-Z]", token
        ):
            found.add(token)
    return found


def test_doc_exists() -> None:
    assert DOC.exists(), f"missing {DOC}"


def test_decisions_file_exists() -> None:
    assert DECISIONS.exists(), f"missing {DECISIONS}"


def test_backtick_paths_resolve_on_disk(doc_text: str) -> None:
    tokens = _extract_backtick_paths(doc_text)
    operator_set = set(OPERATOR_SUPPLIED_PATHS)
    unresolved = sorted(t for t in tokens if not _resolves_on_disk(t) and t not in operator_set)
    assert not unresolved, (
        "docs/architecture.md quotes paths that don't exist on disk:\n"
        + "\n".join(f"  - `{t}`" for t in unresolved)
        + "\n(regenerate the doc to match the current layout, fix the typo, "
        "or — if this is an operator-supplied future artifact — add it to "
        "OPERATOR_SUPPLIED_PATHS in tests/test_architecture_doc.py)"
    )


def test_operator_supplied_paths_actually_absent() -> None:
    landed = [p for p in OPERATOR_SUPPLIED_PATHS if (REPO_ROOT / p).exists()]
    assert not landed, (
        "These paths are listed as operator-supplied in "
        "tests/test_architecture_doc.py but exist on disk:\n"
        + "\n".join(f"  - `{p}`" for p in landed)
        + "\n(drop them from OPERATOR_SUPPLIED_PATHS so the resolvability "
        "check covers them as literal paths)"
    )


def test_doc_symbol_refs_resolve(doc_text: str) -> None:
    """Every symbol the doc names resolves to a real package attribute.

    ``test_backtick_paths_resolve_on_disk`` validates slash-path tokens only;
    a *symbol* reference — a fully-qualified ``rag_kit.<module>.<symbol>`` or a
    CamelCase public type — was unguarded. That is the drift class portfolio-ops
    #55 catalogued (a doc naming a nonexistent class stays green in CI).
    Propagates the embedding-model-shootout #71 / llm-eval-harness #140 lock,
    adapted to the two citation styles this doc uses, with an
    ``EXTERNAL_SYMBOLS`` allowlist for cross-repo references (#118).
    """
    import importlib

    pkg = importlib.import_module("rag_kit")
    dotted = set(re.findall(r"rag_kit\.([a-z_]+)\.([A-Za-z_][A-Za-z0-9_]*)", doc_text))
    camel = _extract_camel_symbols(doc_text) - set(EXTERNAL_SYMBOLS)
    assert dotted or camel, (
        "expected at least one symbol reference (`rag_kit.<module>.<symbol>` or "
        "a CamelCase public type) in docs/architecture.md — the resolver would "
        "otherwise be vacuously green"
    )

    unresolved: list[str] = []
    for module_name, symbol in sorted(dotted):
        try:
            module = importlib.import_module(f"rag_kit.{module_name}")
        except ModuleNotFoundError:
            unresolved.append(f"rag_kit.{module_name}.{symbol} (module not importable)")
            continue
        if not hasattr(module, symbol):
            unresolved.append(f"rag_kit.{module_name}.{symbol}")
    for symbol in sorted(camel):
        if not hasattr(pkg, symbol):
            unresolved.append(f"{symbol} (not in the rag_kit public surface)")

    assert not unresolved, (
        "docs/architecture.md names symbols that don't exist in the package:\n"
        + "\n".join(f"  - {u}" for u in unresolved)
        + "\n(fix the doc to match the shipped symbol, update the rename that "
        "orphaned it, or — if it's a legitimate cross-repo reference — add it to "
        "EXTERNAL_SYMBOLS in tests/test_architecture_doc.py)"
    )


def test_external_symbols_hard_pin_set() -> None:
    assert EXTERNAL_SYMBOLS == ("RunResult",)


def test_external_symbols_absent_from_public_surface() -> None:
    # An allow-listed external symbol that IS in the rag_kit surface would be a
    # stale exemption hiding real coverage — drop it from EXTERNAL_SYMBOLS then.
    import importlib

    pkg = importlib.import_module("rag_kit")
    shadowed = [s for s in EXTERNAL_SYMBOLS if hasattr(pkg, s)]
    assert not shadowed, (
        "these EXTERNAL_SYMBOLS now exist in the rag_kit public surface; drop "
        "them from the allowlist so the symbol check covers them:\n"
        + "\n".join(f"  - {s}" for s in shadowed)
    )


def test_every_shipped_issue_referenced(doc_text: str) -> None:
    referenced = {int(m.group(1)) for m in re.finditer(r"#(\d+)\b", doc_text)}
    missing = sorted(set(KNOWN_SHIPPED_ISSUES) - referenced)
    assert not missing, (
        "docs/architecture.md doesn't reference these closed-feature-issues "
        "even once:\n"
        + "\n".join(f"  - #{n}" for n in missing)
        + "\n(every shipped layer should have its origin issue annotated "
        "in the doc; add a `(#NN)` to the relevant component bullet or "
        "diagram node)"
    )


def test_every_active_decision_referenced(doc_text: str, active_decisions: tuple[int, ...]) -> None:
    referenced = {int(m.group(1)) for m in re.finditer(r"\bD-0*(\d+)\b", doc_text)}
    missing = sorted(set(active_decisions) - referenced)
    assert not missing, (
        "docs/architecture.md doesn't reference these active "
        "(non-superseded) core decisions even once:\n"
        + "\n".join(f"  - D-{n:03d}" for n in missing)
        + "\n(every shipped layer / posture in MEMORY/core_decisions_ai.md "
        "should be annotated in the doc where the relevant code lives; "
        "add a `D-NNN` reference to the relevant bullet)"
    )


def test_no_banned_phrases(doc_text: str) -> None:
    lowered = doc_text.lower()
    hits = [p for p in BANNED_PHRASES if p in lowered]
    assert not hits, (
        "docs/architecture.md contains drift phrases:\n"
        + "\n".join(f"  - {p!r}" for p in hits)
        + "\n(these phrases describe a pre-shipping state; the doc is a "
        "steady-state reference, not a PR description)"
    )


def test_banned_phrases_hard_pin_set() -> None:
    assert BANNED_PHRASES == (
        "this pr",
        "pending downstream",
        "(unfiled)",
        "to-be-filed",
    )


def test_resolvable_prefixes_hard_pin_set() -> None:
    assert RESOLVABLE_PREFIXES == (
        "rag_kit/",
        "evals/",
        "scripts/",
        "frontend/",
        "docs/",
        "tests/",
        ".github/",
    )


def test_known_shipped_issues_hard_pin_set() -> None:
    assert KNOWN_SHIPPED_ISSUES == (1, 2, 3, 4, 5, 6, 7, 8, 50)


def test_min_active_decision_id_hard_pin() -> None:
    assert MIN_ACTIVE_DECISION_ID == 2


def test_operator_supplied_paths_hard_pin_set() -> None:
    assert OPERATOR_SUPPLIED_PATHS == ()
