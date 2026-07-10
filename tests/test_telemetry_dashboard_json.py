"""The dashboard ``/json`` endpoint must emit strict-valid JSON (#134).

``json.dumps`` defaults to ``allow_nan=True``, emitting the bare tokens
``NaN`` / ``Infinity`` — which are **invalid JSON**, so a browser's
``fetch().then(r => r.json())`` rejects the whole ``/json`` response and the
dashboard blanks. ``CostRecord`` has no ``__post_init__``, so a record
constructed directly (a documented public API) bypasses ``build``'s #108
finiteness guard and can carry a non-finite ``per_phase_ms`` phase value that
reaches the JSON egress. This pins the ``/json`` body to strict-valid JSON —
the egress sibling of the #81/#82/#87/#106/#108 non-finite-at-the-seam sweep.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.telemetry import CostRecord  # noqa: E402
from scripts.telemetry_dashboard import _json_response_body  # noqa: E402


def _reject_non_finite(token: str) -> float:
    # Emulate a strict JSON parser (the browser's JSON.parse): raise on any bare
    # NaN / Infinity / -Infinity token instead of accepting it.
    raise ValueError(f"invalid JSON token {token!r}")


def _record(per_phase_ms: dict[str, float]) -> CostRecord:
    return CostRecord(
        ts=1e12,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=1,
        completion_tokens=1,
        prompt_usd=0.0,
        completion_usd=0.0,
        total_usd=0.0,
        total_latency_ms=1.0,
        per_phase_ms=per_phase_ms,
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_json_body_is_strict_valid_with_non_finite_per_phase(bad: float) -> None:
    body = _json_response_body([_record({"retrieving": bad, "generating": 12.0})]).decode("utf-8")
    # A strict parser must accept it (no bare NaN/Infinity tokens survive).
    parsed = json.loads(body, parse_constant=_reject_non_finite)
    phases = parsed["records"][0]["per_phase_ms"]
    assert phases["retrieving"] is None  # non-finite -> null, like JSON.stringify(NaN)
    assert phases["generating"] == 12.0  # finite sibling preserved


def test_json_body_preserves_all_finite_values() -> None:
    # Over-sanitization guard: a fully-finite record round-trips unchanged.
    body = _json_response_body([_record({"retrieving": 3.5, "generating": 8.25})]).decode("utf-8")
    parsed = json.loads(body, parse_constant=_reject_non_finite)
    rec = parsed["records"][0]
    assert rec["per_phase_ms"] == {"retrieving": 3.5, "generating": 8.25}
    assert rec["total_latency_ms"] == 1.0
    assert rec["query"] == "q"


def test_json_body_empty_records_is_valid() -> None:
    body = _json_response_body([]).decode("utf-8")
    assert json.loads(body, parse_constant=_reject_non_finite) == {"records": []}
