"""README decision-range upper-bound lock.

Sister to ``chunking-strategies-lab`` ``test_readme_snapshot.py``
``test_decision_range_cites_latest_active`` (pattern leader), plus the
propagations in ``llm-eval-harness``, ``llm-cost-optimizer``, and
``prompt-regression-suite``.

The README's architecture-section summary cites a range like
``D-002…D-NNN``; the upper bound must equal the highest active
(non-superseded) ``D-NNN`` in ``MEMORY/core_decisions_ai.md``. A new
decision landing without the README being updated fails this test
loud — the same drift class that ``test_architecture_doc.py`` catches
inside ``docs/architecture.md``, but for the README's range citation.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
DECISIONS = REPO_ROOT / "MEMORY" / "core_decisions_ai.md"


def _max_active_decision_id() -> int:
    """Highest non-superseded ``D-NNN`` in ``MEMORY/core_decisions_ai.md``."""
    text = DECISIONS.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=- id:)", text)
    best = 0
    for block in blocks:
        id_match = re.search(r"- id:\s*D-(\d+)", block)
        if not id_match:
            continue
        sup_match = re.search(r"superseded_by:\s*(\S+)", block)
        is_active = (sup_match is None) or (sup_match.group(1).strip().lower() == "null")
        if is_active:
            n = int(id_match.group(1))
            if n > best:
                best = n
    return best


def test_decision_range_cites_latest_active() -> None:
    body = README.read_text(encoding="utf-8")
    pattern = re.compile(r"D-0*2\s*(?:…|\.\.\.)\s*D-0*(\d+)")
    matches = pattern.findall(body)
    assert matches, (
        "README.md must cite the active-decision range as "
        "`D-002…D-NNN` somewhere (the architecture-section summary "
        "paragraph by convention). Not found."
    )
    cited = max(int(m) for m in matches)
    latest = _max_active_decision_id()
    assert cited == latest, (
        f"README.md cites decision range up to D-{cited:03d}, but the "
        f"highest active D-NNN in MEMORY/core_decisions_ai.md is "
        f"D-{latest:03d}. Update the README's architecture-section "
        f"summary to D-002…D-{latest:03d}."
    )
