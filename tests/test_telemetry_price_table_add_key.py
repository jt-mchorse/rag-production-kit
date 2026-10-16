"""`PriceTable.add()` refuses a non-str model id, as the constructor does (#350).

#342 made `PriceTable({...})` refuse a key that is not a `str`, and its comment
said the constructor checked entries "as `add()` checks its own". `add()` checked
only the price. Measured on main:

    PriceTable({None: ModelPrice(3.0, 15.0)})  -> ValueError
    t.add(None, 3.0, 15.0); t.add(123, 1.0, 2.0) -> known_models() == (None, 123)
    CostRecord.build(model=None, price_table=t) -> priced
    TelemetryStore(...).record(rec)            -> sqlite3.IntegrityError (model NOT NULL)
"""

from __future__ import annotations

import pytest

from rag_kit.telemetry import CostRecord, ModelPrice, PriceTable, UnknownModelError

NON_STR_KEYS = {"none": None, "int": 123, "bytes": b"claude-x", "float": 1.5}


def _outcome(fn) -> str:  # type: ignore[no-untyped-def]
    try:
        fn()
    except ValueError as e:
        return f"ValueError: {e}"
    return "accepted"


@pytest.mark.parametrize("key", list(NON_STR_KEYS.values()), ids=list(NON_STR_KEYS))
def test_add_refuses_a_non_str_key_with_the_constructors_message(key: object) -> None:
    table = PriceTable({"ok": ModelPrice(3.0, 15.0)})
    with pytest.raises(ValueError, match=r"^price table model ids must be str; got "):
        table.add(key, 3.0, 15.0)  # type: ignore[arg-type]
    assert table.known_models() == ("ok",)


@pytest.mark.parametrize(
    "key",
    [*NON_STR_KEYS.values(), "claude-x", ""],
    ids=[*NON_STR_KEYS, "str", "empty-str"],
)
def test_add_and_the_constructor_agree_on_every_key(key: object) -> None:
    via_ctor = _outcome(lambda: PriceTable({key: ModelPrice(3.0, 15.0)}))  # type: ignore[dict-item]
    via_add = _outcome(lambda: PriceTable().add(key, 3.0, 15.0))  # type: ignore[arg-type]
    assert via_add == via_ctor


def test_add_still_refuses_a_bad_rate_for_a_good_key() -> None:
    with pytest.raises(ValueError, match="prompt_per_million"):
        PriceTable().add("claude-x", -1.0, 15.0)


def test_a_none_model_cannot_be_priced_through_a_table_built_by_add() -> None:
    # The end of the chain the issue measured: with `add(None, ...)` refused,
    # `build(model=None)` is an unknown model, not a row headed for a NOT NULL
    # column.
    table = PriceTable()
    with pytest.raises(ValueError, match="model ids must be str"):
        table.add(None, 3.0, 15.0)  # type: ignore[arg-type]
    with pytest.raises(UnknownModelError):
        CostRecord.build(
            ts=None,
            query="q",
            model=None,  # type: ignore[arg-type]
            retrieved_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_latency_ms=1.0,
            per_phase_ms=None,
            price_table=table,
        )
