"""The guards that never bind to SQLite refuse an int past float range in words (#395).

#392 put `_check_sqlite_int_range` before `math.isfinite` at every site that
binds an int into SQLite. The sites that never bind were left over:
`isfinite` converts an int to float first, so an int past float range raised
there before the guard could name the field. Measured on main (58dd718),
Python 3.11:

    ModelPrice(10**400, 0.0), ModelPrice(0.0, 10**400)
    percentile([10**400], 0.5)
    aggregate([CostRecord(..., total_usd=10**400)])
    aggregate([CostRecord(..., total_latency_ms=10**400)])
        -> OverflowError: int too large to convert to float

The bound here is float range, not int64: `2**63` is a valid price, and so is
the largest int `float()` accepts.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_kit.telemetry import CostRecord, ModelPrice, aggregate, percentile

HUGE = 10**400  # past float range: float(HUGE) raises OverflowError
# The largest int float() accepts: 2**1024 - 2**970 rounds up to 2**1024.
FLOAT_MAX_INT = 2**1024 - 2**970 - 1
BIG = 10**5000  # past 4300 digits, where str() itself raises (so: explicit ids)


def _record(**overrides: Any) -> CostRecord:
    # Constructed directly, as `aggregate`'s own comments say a caller may.
    kwargs: dict[str, Any] = {
        "ts": 1_700_000_000.0,
        "query": "q",
        "model": "m",
        "retrieved_count": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "prompt_usd": 0.0,
        "completion_usd": 0.0,
        "total_usd": 0.0,
        "total_latency_ms": 12.0,
    }
    return CostRecord(**{**kwargs, **overrides})


# --- red against main: a ValueError naming the field ------------------------


@pytest.mark.parametrize(
    "value", [HUGE, -HUGE, FLOAT_MAX_INT + 1, BIG], ids=["1e400", "-1e400", "max+1", "5000-digit"]
)
@pytest.mark.parametrize("name", ["prompt_per_million", "completion_per_million"])
def test_model_price_refuses_an_int_past_float_range(name: str, value: int) -> None:
    rates = {"prompt_per_million": 0.0, "completion_per_million": 0.0, name: value}
    with pytest.raises(ValueError, match=rf"^{name} must be a finite number >= 0.0; got an int"):
        ModelPrice(**rates)


@pytest.mark.parametrize("value", [HUGE, -HUGE, BIG], ids=["1e400", "-1e400", "5000-digit"])
def test_percentile_refuses_an_int_past_float_range(value: int) -> None:
    with pytest.raises(ValueError, match=r"^values must all be finite numbers; got an int of"):
        percentile([1.0, value], 0.5)


def test_aggregate_refuses_a_total_usd_past_float_range() -> None:
    with pytest.raises(ValueError, match=r"^total_usd values must all be finite numbers"):
        aggregate([_record(), _record(total_usd=HUGE)])


def test_aggregate_refuses_a_latency_past_float_range() -> None:
    # Delegated to `percentile`, with the message #80's tests pin.
    with pytest.raises(ValueError, match=r"^values must all be finite numbers"):
        aggregate([_record(), _record(total_latency_ms=HUGE)])


# --- controls: ints that float() accepts behave as before -------------------


@pytest.mark.parametrize("value", [2**63, 2**1023, FLOAT_MAX_INT], ids=["2^63", "2^1023", "max"])
def test_model_price_accepts_a_large_int_float_can_hold(value: int) -> None:
    price = ModelPrice(value, value)
    assert price.prompt_per_million == value


def test_model_price_still_refuses_a_negative_rate_by_value() -> None:
    with pytest.raises(ValueError, match=r"^prompt_per_million .*; got -1.0$"):
        ModelPrice(-1.0, 0.0)


@pytest.mark.parametrize(
    "value", [2**1023, FLOAT_MAX_INT, -FLOAT_MAX_INT], ids=["2^1023", "max", "-max"]
)
def test_percentile_accepts_a_large_int_float_can_hold(value: int) -> None:
    assert percentile([value], 0.5) == value


def test_percentile_still_refuses_infinity_with_the_80_message() -> None:
    with pytest.raises(ValueError, match=r"^values must all be finite numbers; got \[inf\]$"):
        percentile([float("inf")], 0.5)


def test_aggregate_accepts_a_large_int_total_usd_float_can_hold() -> None:
    agg = aggregate([_record(total_usd=2**1023)])
    assert agg.n == 1
