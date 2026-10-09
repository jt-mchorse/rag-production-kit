"""`ModelPrice` and `total_latency_ms` carry the type check `per_phase_ms` has (#303).

`CostRecord.build` refused a bool or non-number `per_phase_ms` value by name,
but its sibling `total_latency_ms` and both `ModelPrice` rates were checked for
finiteness only. Measured on main:

    ModelPrice(True, False)   -> accepted, cost(1M, 1M) = (1.0, 0.0)
    ModelPrice('1.0', 2.0)    -> TypeError: must be real number, not str
    total_latency_ms=True     -> accepted; read back from the store as 1.0
    total_latency_ms='12.5'   -> TypeError: must be real number, not str
"""

from __future__ import annotations

import pytest

from rag_kit.telemetry import CostRecord, ModelPrice, PriceTable

NOT_NUMBERS = [True, False, "1.0", None, [1.0]]


@pytest.mark.parametrize("bad", NOT_NUMBERS, ids=repr)
@pytest.mark.parametrize("field", ["prompt_per_million", "completion_per_million"])
def test_model_price_refuses_a_non_number_by_name(field: str, bad: object) -> None:
    kwargs: dict[str, object] = {
        "prompt_per_million": 3.0,
        "completion_per_million": 15.0,
        field: bad,
    }
    with pytest.raises(ValueError, match=rf"^{field} must be a finite number >= 0\.0; got "):
        ModelPrice(**kwargs)  # type: ignore[arg-type]


def test_price_table_add_goes_through_the_same_guard() -> None:
    with pytest.raises(ValueError, match="prompt_per_million"):
        PriceTable().add("m", True, 15.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("good", [0, 3, 0.0, 3.5])
def test_model_price_accepts_ints_and_floats(good: float) -> None:
    assert ModelPrice(good, good).cost(1_000_000, 1_000_000) == (float(good), float(good))


def _build(total_latency_ms: object) -> CostRecord:
    return CostRecord.build(
        ts=1.0,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=1,
        completion_tokens=1,
        total_latency_ms=total_latency_ms,  # type: ignore[arg-type]
        per_phase_ms=None,
        price_table=PriceTable({"m": ModelPrice(3.0, 15.0)}),
    )


@pytest.mark.parametrize("bad", NOT_NUMBERS, ids=repr)
def test_total_latency_refuses_a_non_number_by_name(bad: object) -> None:
    with pytest.raises(
        ValueError, match=r"^total_latency_ms must be a finite non-negative number; got "
    ):
        _build(bad)


@pytest.mark.parametrize("good", [0, 12, 0.0, 12.5])
def test_total_latency_accepts_ints_and_floats(good: float) -> None:
    assert _build(good).total_latency_ms == good


def test_the_three_fields_and_per_phase_refuse_the_same_inputs() -> None:
    # One contract: whatever per_phase_ms refuses, its siblings refuse.
    for bad in NOT_NUMBERS:
        with pytest.raises(ValueError, match="per_phase_ms"):
            CostRecord.build(
                ts=1.0,
                query="q",
                model="m",
                retrieved_count=1,
                prompt_tokens=1,
                completion_tokens=1,
                total_latency_ms=1.0,
                per_phase_ms={"retrieve": bad},  # type: ignore[dict-item]
                price_table=PriceTable({"m": ModelPrice(3.0, 15.0)}),
            )
        with pytest.raises(ValueError, match="total_latency_ms"):
            _build(bad)
        with pytest.raises(ValueError, match="prompt_per_million"):
            ModelPrice(bad, 1.0)  # type: ignore[arg-type]
