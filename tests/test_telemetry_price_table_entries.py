"""`PriceTable({...})` checks its entries when it is built, as `add()` does (#342).

`add()` builds a `ModelPrice`, so a bad rate is refused at configuration time.
The constructor copied its mapping as given. Measured on main:

    PriceTable({"m": (5.0, 25.0)})          -> built; cost("m", ...) AttributeError
    PriceTable({"m": {"prompt_per_million": 5.0, ...}}) -> same
    PriceTable({"m": 5.0})                  -> same

A table loaded from JSON config fails on its first request instead of at load,
with an error naming neither the model nor the field.
"""

from __future__ import annotations

import pytest

from rag_kit.telemetry import ModelPrice, PriceTable

NOT_MODEL_PRICES = {
    "tuple-of-rates": (5.0, 25.0),
    "list-of-rates-from-json": [5.0, 25.0],
    "dict-of-rates-from-json": {"prompt_per_million": 5.0, "completion_per_million": 25.0},
    "bare-float": 5.0,
    "none": None,
}


@pytest.mark.parametrize("bad", list(NOT_MODEL_PRICES.values()), ids=list(NOT_MODEL_PRICES))
def test_a_value_that_is_not_a_model_price_is_refused_at_construction(bad: object) -> None:
    with pytest.raises(ValueError, match=r"^price for model 'claude-x' must be a ModelPrice; got "):
        PriceTable({"ok": ModelPrice(3.0, 15.0), "claude-x": bad})  # type: ignore[dict-item]


def test_a_key_that_is_not_a_str_is_refused_at_construction() -> None:
    # A model id is a str, so an int key could never be looked up.
    with pytest.raises(ValueError, match=r"^price table model ids must be str; got 7$"):
        PriceTable({7: ModelPrice(3.0, 15.0)})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "prices",
    [
        None,
        {},
        {"claude-opus-4-7": ModelPrice(5.0, 25.0), "claude-sonnet-4-6": ModelPrice(3.0, 15.0)},
        [("m", ModelPrice(1.0, 4.0))],
    ],
    ids=["none", "empty", "two-models", "pairs"],
)
def test_valid_tables_still_build(prices: object) -> None:
    pt = PriceTable(prices)  # type: ignore[arg-type]
    for model in pt.known_models():
        assert isinstance(pt.cost(model, 1_000_000, 1_000_000), tuple)


def test_add_is_unchanged() -> None:
    pt = PriceTable()
    pt.add("m", 5.0, 25.0)
    assert pt.cost("m", 1_000_000, 1_000_000) == (5.0, 25.0)
