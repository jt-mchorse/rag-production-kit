"""PhaseTimings.percentile and telemetry.percentile agree exactly (#287).

`telemetry.percentile`'s docstring says it "matches
`rag_kit.streaming.PhaseTimings.percentile` so a 24-hour aggregate window and a
streaming-pipeline snapshot agree on the number". They used two interpolation
formulas -- `lo * (1 - frac) + hi * frac` vs `lo + (hi - lo) * frac` -- that
differ in the last bits. Measured on `main` (a hunt agent, re-run here): p95 of
`[204.41] * 3` was 204.40999999999997 from PhaseTimings, below every sample, and
8,893 of 60k random (sample, percentile) pairs disagreed with telemetry.
"""

from __future__ import annotations

import random

import pytest

from rag_kit.streaming import PhaseTimings
from rag_kit.telemetry import percentile


def _timings(values: list[float]) -> PhaseTimings:
    t = PhaseTimings()
    for v in values:
        t.record("retrieving", v)
    return t


def test_a_constant_sample_returns_the_constant() -> None:
    assert _timings([204.41] * 3).percentile("retrieving", 95) == 204.41


@pytest.mark.parametrize("p", [1, 50, 95, 99])
def test_parity_over_random_samples(p: int) -> None:
    rng = random.Random(287)
    for _ in range(2_000):
        values = [round(rng.uniform(0, 500), rng.randint(0, 3)) for _ in range(rng.randint(1, 9))]
        assert _timings(values).percentile("retrieving", p) == percentile(values, p / 100.0)


def test_never_outside_the_sample_range() -> None:
    rng = random.Random(7)
    for _ in range(5_000):
        values = [rng.choice([0.2, 7.7, 204.41, 1.1]) for _ in range(rng.randint(1, 5))]
        got = _timings(values).percentile("retrieving", 95)
        assert min(values) <= got <= max(values)


def test_edges_and_empty_are_unchanged() -> None:
    t = _timings([3.0, 1.0, 2.0])
    assert t.percentile("retrieving", 0) == 1.0
    assert t.percentile("retrieving", 100) == 3.0
    assert PhaseTimings().percentile("retrieving", 50) is None
