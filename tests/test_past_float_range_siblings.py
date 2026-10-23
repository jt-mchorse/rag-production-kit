"""The #395 siblings outside telemetry refuse an int past float range in words (#396).

`math.isfinite` (and `float()`) convert an int to float first, so an int past
float range raised a raw `OverflowError` before the guard could name the field.
#395 fixed `rag_kit/telemetry.py` only. Measured on main (c3dd45e), Python 3.11:

    PhaseTimings().record("total", 10**400)
    PhaseTimings(total=[10**400]).percentile("total", 50)
    CohereReranker(api_key="x", timeout_s=10**400)
    to_pgvector([..., 10**400, ...])
        -> OverflowError: int too large to convert to float

The bound is float range, so the largest int `float()` accepts is still valid.
"""

from __future__ import annotations

import pytest

from rag_kit.db import EMBEDDING_DIM, to_pgvector
from rag_kit.reranker import CohereReranker
from rag_kit.streaming import PhaseTimings

HUGE = 10**400  # past float range: float(HUGE) raises OverflowError
# The largest int float() accepts: 2**1024 - 2**970 rounds up to 2**1024.
FLOAT_MAX_INT = 2**1024 - 2**970 - 1
BIG = 10**5000  # past 4300 digits, where str() itself raises (so: explicit ids)

PAST = [HUGE, -HUGE, FLOAT_MAX_INT + 1, BIG]
PAST_IDS = ["1e400", "-1e400", "max+1", "5000-digit"]
HELD = [2**1023, FLOAT_MAX_INT]
HELD_IDS = ["2^1023", "max"]


def _vec(component: object) -> list:
    vec: list = [0.0] * EMBEDDING_DIM
    vec[3] = component
    return vec


# --- red against main: a ValueError naming the field ------------------------


@pytest.mark.parametrize("value", PAST, ids=PAST_IDS)
def test_record_refuses_an_int_past_float_range(value: int) -> None:
    pt = PhaseTimings()
    with pytest.raises(
        ValueError, match=r"^ms must be a finite non-negative number; got an int of"
    ):
        pt.record("total", value)
    assert pt.total == []


@pytest.mark.parametrize("value", PAST, ids=PAST_IDS)
def test_percentile_refuses_an_int_past_float_range_on_a_direct_list(value: int) -> None:
    # The public init field bypasses `record`, as the #168 comment says.
    pt = PhaseTimings(total=[1.0, value])
    with pytest.raises(
        ValueError,
        match=r"^values must all be finite non-negative numbers; got \[1\.0, an int of \d+ bits\]$",
    ):
        pt.percentile("total", 50)


@pytest.mark.parametrize("value", PAST, ids=PAST_IDS)
def test_cohere_timeout_refuses_an_int_past_float_range(value: int) -> None:
    # Before the optional `cohere` import, like the other construction guards.
    with pytest.raises(
        ValueError, match=r"^timeout_s must be a finite number > 0\.0, got an int of"
    ):
        CohereReranker(api_key="x", timeout_s=value)


@pytest.mark.parametrize("value", PAST, ids=PAST_IDS)
def test_to_pgvector_refuses_a_component_past_float_range(value: int) -> None:
    with pytest.raises(
        ValueError, match=r"^embedding component at index 3 must be finite; got an int of \d+ bits"
    ):
        to_pgvector(_vec(value))


# --- controls: ints that float() accepts behave as before -------------------


@pytest.mark.parametrize("value", HELD, ids=HELD_IDS)
def test_record_and_percentile_accept_a_large_int_float_can_hold(value: int) -> None:
    pt = PhaseTimings()
    pt.record("total", value)
    assert pt.percentile("total", 50) == value


def test_percentile_still_names_a_bad_value_by_its_repr() -> None:
    pt = PhaseTimings(total=[-5.0, float("nan")])
    with pytest.raises(ValueError, match=r"; got \[-5\.0, nan\]$"):
        pt.percentile("total", 50)


@pytest.mark.parametrize("value", HELD, ids=HELD_IDS)
def test_cohere_timeout_accepts_a_large_int_float_can_hold(value: int) -> None:
    # Clears the guard and reaches the optional-extra import.
    with pytest.raises(ImportError, match="rerank-cohere"):
        CohereReranker(api_key="x", timeout_s=value)


@pytest.mark.parametrize("value", HELD, ids=HELD_IDS)
def test_to_pgvector_accepts_a_large_int_float_can_hold(value: int) -> None:
    assert repr(float(value)) in to_pgvector(_vec(value))


def test_to_pgvector_still_names_infinity_by_its_repr() -> None:
    with pytest.raises(ValueError, match=r"index 3 must be finite; got inf "):
        to_pgvector(_vec(float("inf")))
