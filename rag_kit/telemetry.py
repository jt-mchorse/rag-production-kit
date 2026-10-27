"""Cost telemetry: per-request token / dollar / latency capture + a small SQLite store.

Three pieces in this module:

- ``CostRecord`` — the per-request blob (timestamp, query, model, token
  counts, USD breakdown, total latency, per-phase latency dict).
- ``PriceTable`` — operator-supplied mapping of model id →
  ``(prompt_per_million, completion_per_million)`` USD prices. **No
  defaults are shipped** (D-015); querying an unconfigured model
  raises ``UnknownModelError``. Same rationale as D-013 — the
  no-fabricated-benchmarks rule extends to no-fabricated-prices.
- ``TelemetryStore`` — thin stdlib ``sqlite3`` wrapper, schema below.
  Single file, no daemon, dep-free; consistent with D-002 (only
  required runtime dep is ``psycopg``; everything else opt-in).

Aggregate latency percentiles use the same linear-interp NIST type-7
math as ``PhaseTimings`` in ``rag_kit.streaming`` so the two layers
report comparable numbers when the operator looks at a 24-hour window.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_kit.indexer import _lone_surrogate

# Moved to `io_utils` so the streaming, reranker and db guards can share them
# without an import cycle (#396).
from rag_kit.io_utils import _describe_number, _past_float_range, atomic_write_text

# ----------------------------------------------------------------------
# Price table
# ----------------------------------------------------------------------


class UnknownModelError(KeyError):
    """Raised by ``PriceTable.cost`` when the model id is not configured.

    Inherits ``KeyError`` for backwards compatibility with callers that
    only catch ``KeyError``; the explicit subclass lets newer callers
    pattern-match on the more specific type.
    """


@dataclass(frozen=True)
class ModelPrice:
    """USD-per-million-tokens for one model."""

    prompt_per_million: float
    completion_per_million: float

    def __post_init__(self) -> None:
        # D-015 prevents silent-zero via UnknownModelError; this guard extends
        # the same posture to silent-negative and silent-NaN/Infinity (#38).
        # A NaN rate propagates through cost() into CostRecord.total_usd = NaN,
        # which sums NaN through aggregate() and renders as "NaN" on the cost
        # dashboard — same shape of silent corruption as the silent-zero case
        # D-015 addressed, just one layer of arithmetic later. Tightened from
        # sign-only to finiteness mirroring the portfolio's contract-tightening
        # sweep (llm-cost-optimizer#37, llm-eval-harness#43, etc).
        for name, value in (
            ("prompt_per_million", self.prompt_per_million),
            ("completion_per_million", self.completion_per_million),
        ):
            # The type half of the contract (#303), as `CostRecord.build`
            # states it for `per_phase_ms`: bool is an int subclass, so
            # `ModelPrice(True, False)` priced at $1/$0, and a str escaped
            # `math.isfinite` as a bare TypeError naming no field.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number >= 0.0; got {value!r}")
            # An int past float range is refused before `isfinite` converts it (#395).
            if _past_float_range(value) or not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be a finite number >= 0.0; got {_describe_number(value)}"
                )

    def cost(self, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
        """Return ``(prompt_usd, completion_usd)`` rounded to 6 decimal places."""
        # Non-negative *integer* counts, not sign-only: a sign-only check lets a
        # NaN through (`NaN < 0` is False), which then propagates
        # `NaN * rate / 1_000_000` → prompt_usd → CostRecord.total_usd →
        # aggregate().total_usd, and `dump_aggregate_json` serializes it as the
        # bare token `NaN` — invalid JSON that a strict log-tailer rejects whole.
        # Same silent-corruption shape the rate guard above (#38) and percentile
        # (#80) already close; this is the token-count seam that sweep missed.
        # Reject bool too (it is an int subclass). Mirrors the dim/max_chunks/k
        # non-negative-int contracts elsewhere in the kit.
        for name, value in (("prompt", prompt_tokens), ("completion", completion_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"token counts must be non-negative integers; got {name}={value!r}"
                )
        prompt_usd = round(prompt_tokens * self.prompt_per_million / 1_000_000, 6)
        completion_usd = round(completion_tokens * self.completion_per_million / 1_000_000, 6)
        return prompt_usd, completion_usd


def _checked_model_id(model: object) -> str:
    """A price table key, refused unless it is a ``str`` (#342, #350).

    A key that is not a ``str`` can never match a model id. It was refused by
    ``PriceTable(...)`` only, so ``add(None, ...)`` stored one, and a
    ``CostRecord.build(model=None)`` priced against it reached
    ``TelemetryStore.record`` as a raw ``IntegrityError`` on the NOT NULL
    ``model`` column. Both entry points call this, so they cannot disagree.
    """
    if not isinstance(model, str):
        raise ValueError(f"price table model ids must be str; got {model!r}")
    return model


class PriceTable:
    """Operator-supplied mapping of model id → :class:`ModelPrice`.

    No default entries (D-015). Public-list prices change frequently
    and the repo can't be the source of truth for a downstream
    deployment's actual contract. Configure explicitly:

        prices = PriceTable({
            "claude-opus-4-7":  ModelPrice(15.0, 75.0),  # operator values, not shipped defaults
            "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
        })
    """

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None) -> None:
        self._prices: dict[str, ModelPrice] = dict(prices or {})
        # Each entry is checked here (#342). The mapping used to be copied as
        # given, so a table loaded from config (`{"m": [5.0, 25.0]}`, or a dict
        # of the two rates) built fine and failed on the first request, inside
        # `CostRecord.build`, as an AttributeError naming neither the model nor
        # the field. `add()` checks its price by building the `ModelPrice`, and
        # its key through the same `_checked_model_id` as here (#350).
        for model, entry in self._prices.items():
            _checked_model_id(model)
            if not isinstance(entry, ModelPrice):
                raise ValueError(
                    f"price for model {model!r} must be a ModelPrice; "
                    f"got {type(entry).__name__} {entry!r}"
                )

    def add(self, model: str, prompt_per_million: float, completion_per_million: float) -> None:
        self._prices[_checked_model_id(model)] = ModelPrice(
            prompt_per_million, completion_per_million
        )

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
        """Return ``(prompt_usd, completion_usd)``. Raises ``UnknownModelError`` for unconfigured models."""
        try:
            entry = self._prices[model]
        except KeyError as e:
            raise UnknownModelError(
                f"no price configured for model {model!r}; call PriceTable.add() "
                "or pass an entry to PriceTable()"
            ) from e
        return entry.cost(prompt_tokens, completion_tokens)

    def known_models(self) -> tuple[str, ...]:
        return tuple(self._prices.keys())


# ----------------------------------------------------------------------
# Per-request record
# ----------------------------------------------------------------------


#: SQLite's INTEGER range (#387). `sqlite3` binds every Python `int` as a
#: signed 64-bit INTEGER -- even into a REAL column -- so an int outside it
#: escaped `record` as a raw `OverflowError`, after `build` had accepted it.
_SQLITE_INT_MIN = -(2**63)
_SQLITE_INT_MAX = 2**63 - 1


def _check_sqlite_int_range(value: Any, name: str) -> None:
    """Refuse an ``int`` SQLite cannot bind (#387); a ``float`` is never one.

    Call it BEFORE ``math.isfinite`` / ``math.isnan`` (#392): both convert an
    ``int`` to ``float`` first, so one past float range (``10**400``) raised a
    raw ``OverflowError`` from the finiteness check and never reached this one.
    Such an int is named by its size, not its digits: past 4300 digits ``str()``
    itself raises, and the message would lose the field name.
    """
    if isinstance(value, int) and not _SQLITE_INT_MIN <= value <= _SQLITE_INT_MAX:
        got = value if value.bit_length() <= 128 else f"an int of {value.bit_length()} bits"
        raise ValueError(
            f"{name} must fit SQLite's signed 64-bit INTEGER range "
            f"[{_SQLITE_INT_MIN}, {_SQLITE_INT_MAX}]; got {got}"
        )


def _checked_ts(ts_value: Any, name: str = "ts") -> float:
    """``ts`` as ``build`` and ``TelemetryStore.record`` both require it (#184, #245).

    Also ``last_24h``'s ``now`` (#354), which is the same kind of value.
    """
    if isinstance(ts_value, bool) or not isinstance(ts_value, (int, float)):
        raise ValueError(
            f"{name} must be a finite number of seconds since the Unix epoch (UTC); "
            f"got {ts_value!r}"
        )
    _check_sqlite_int_range(ts_value, name)  # before isfinite (#392)
    if not math.isfinite(ts_value):
        raise ValueError(
            f"{name} must be a finite number of seconds since the Unix epoch (UTC); got {ts_value}"
        )
    return ts_value


def _checked_floor(value: Any, name: str) -> float:
    """A window floor for ``since`` / ``dump_aggregate_json`` (#354).

    #184's reason for ``ts`` applies to the read-side operand too: an ISO-8601
    string, a ``datetime`` or NaN was bound into ``WHERE ts >= ?``, where SQLite
    sorts every REAL below every TEXT and binds NaN as NULL, so each came back as
    a well-formed EMPTY window -- a zero-spend aggregate, indistinguishable from
    a quiet day. ``True`` read as 1970. Unlike a stored ``ts``, an infinite
    floor means what it says (``-inf`` everything, ``inf`` nothing), so it is
    allowed; NaN is not a floor at all.

    An ``int`` floor must fit SQLite's INTEGER range, as a stored ``ts`` must
    (#392): ``WHERE ts >= ?`` binds it, so ``2**63`` escaped as a raw
    ``OverflowError`` -- and ``10**400`` already did so at ``math.isnan``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a number of seconds since the Unix epoch (UTC); got {value!r}"
        )
    _check_sqlite_int_range(value, name)  # before isnan (#392)
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN; it matches no row, so the window is empty")
    return value


#: The `REAL NOT NULL` columns other than `ts`, which `record` checks for NaN
#: (#245). SQLite stores a NaN as NULL, so one of these reached the NOT NULL
#: constraint and escaped `record` as a raw `sqlite3.IntegrityError` naming a
#: column. Since #368 `record` checks their type too: affinity is a
#: preference, so a str `total_usd` was stored as TEXT and the HTML dashboard
#: then raised a TypeError on every request, and `None`/`Decimal` escaped as
#: raw `IntegrityError`/`ProgrammingError`. `tests/test_telemetry_record_seam.py`
#: derives the REAL columns from `_SCHEMA_SQL` and fails if this list misses one.
_NAN_CHECKED_REAL_FIELDS = ("prompt_usd", "completion_usd", "total_usd", "total_latency_ms")

#: The `TEXT NOT NULL` and `INTEGER NOT NULL` columns (#352). #245 checked the
#: REAL ones only, and SQLite type affinity is a preference, not a constraint:
#: a `bytes` query was stored as a BLOB and came back out of `since()` as
#: `bytes`, which took down both dashboard endpoints (`html.escape` and
#: `json.dumps` both refuse it), and a `None` escaped `record` as a raw
#: `sqlite3.IntegrityError`. `per_phase_json` is TEXT too, but `record` writes
#: it itself with `json.dumps`, so it is never a caller's value.
#: `tests/test_telemetry_text_integer_columns.py` derives both lists from
#: `_SCHEMA_SQL`.
_TEXT_CHECKED_FIELDS = ("query", "model")
_INTEGER_CHECKED_FIELDS = ("retrieved_count", "prompt_tokens", "completion_tokens")


def _checked_columns(values: Mapping[str, Any]) -> None:
    """Refuse a TEXT column that is not a ``str`` and an INTEGER column that is
    not a non-negative ``int`` (#352). Shared by ``build`` and ``record``, so the
    checked path and the bare constructor are refused in the same words.

    A ``str`` holding a lone surrogate is refused too (#368): SQLite binds text
    as UTF-8, which cannot encode one, so it escaped ``record`` as a raw
    ``UnicodeEncodeError``. A NUL is not refused -- SQLite stores it intact.
    Nor is an INTEGER above ``2**63 - 1`` (#387): it escaped as ``OverflowError``."""
    for name in _TEXT_CHECKED_FIELDS:
        value = values[name]
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a str; got {type(value).__name__} {value!r}")
        reason = _lone_surrogate(value)
        if reason is not None:
            raise ValueError(f"{name} must be a str SQLite can store; it contains {reason}")
    for name in _INTEGER_CHECKED_FIELDS:
        value = values[name]
        # bool is an int subclass; a stray True must not pose as a count.
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative int; got {value!r}")
        _check_sqlite_int_range(value, name)


def _checked_phases(per_phase_ms: Mapping[Any, Any]) -> None:
    """Refuse a ``per_phase_ms`` map ``build`` would refuse. Shared by ``build``
    and ``record`` (#406), so the bare constructor is refused in the same words.

    Per-phase value guard (#108): the same finiteness/sign contract as
    total_latency_ms, applied to each per_phase_ms value. Without it a
    non-finite phase value reaches TelemetryStore.record, which persists the
    map via json.dumps(... allow_nan=True) -- writing the bare token
    NaN/Infinity (invalid JSON). bool is an int subclass; reject it so a stray
    True/False can't pose as a millisecond value.

    Keys must be `str`, as the annotation says (#387). `record` persists the map
    with `json.dumps`, which raised a raw TypeError for a tuple key and coerced
    the rest to strings: `{None: 5.0}` came back as `{"null": 5.0}`, and
    `{1: 5.0, "1": 6.0}` as `{"1": 6.0}`, the 5.0 silently gone.

    At ``record`` this is the whole rule, not only what SQLite cannot store as
    for ``total_latency_ms`` (#245): ``aggregate`` guards that column's
    infinities and negatives on read (#80, #213), but no read path checks
    ``per_phase_ms`` (the dashboard's ``/json`` only maps a non-finite value to
    ``null``), so the write seam is the only place it can be checked.
    """
    for phase, value in per_phase_ms.items():
        if not isinstance(phase, str):
            raise ValueError(
                f"per_phase_ms keys must be str phase names; got {type(phase).__name__} {phase!r}"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"per_phase_ms[{phase!r}] must be a finite non-negative number; got {value!r}"
            )
        # The same int range as `total_latency_ms` (#392). An int past float
        # range made `math.isfinite` itself raise a raw OverflowError.
        _check_sqlite_int_range(value, f"per_phase_ms[{phase!r}]")
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"per_phase_ms[{phase!r}] must be a finite non-negative number; got {value}"
            )


@dataclass(frozen=True)
class CostRecord:
    """One per-request telemetry blob.

    Latency fields are floats in milliseconds. ``per_phase_ms`` is the
    phase-name → ms mapping used by ``rag_kit.streaming.PhaseTimings``
    so a downstream consumer can chart per-phase latency over a window
    without an extra schema.
    """

    # Seconds since the Unix epoch, UTC. `build` rejects a non-finite, non-real
    # or `bool` value (#184) -- this is the key every read path filters and
    # orders on, so a corrupt one is not merely a wrong column, it decides
    # whether the row is in a window at all.
    ts: float
    query: str
    model: str
    retrieved_count: int
    prompt_tokens: int
    completion_tokens: int
    prompt_usd: float
    completion_usd: float
    total_usd: float
    total_latency_ms: float
    per_phase_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy *per_phase_ms* so the frozen record owns it (#227, D-022).

        `dict(...)` and deliberately **not** `copy_json_value`, which is the one
        asymmetry in this change. The declared element type is `float`, which is
        immutable, so one level is the whole depth — the same "check the copy's
        depth against the declared element type" question the other four records
        answer the other way, because `dict[str, Any]` proves nothing about its
        values. `Mapping` is also not necessarily a `dict`, and
        `copy_json_value` would hand any other mapping straight back.

        `CostRecord.build` has done exactly this since it was written; the bare
        constructor is the unguarded half, and it is public via `__all__`.
        `TelemetryStore.append` serializes this field, so an alias reaches the
        persisted row.

        `tests/test_frozen_record_metadata_aliasing.py` locks the annotation:
        widening it past `Mapping[str, float]` makes the shallow copy wrong, and
        nothing else would say so.
        """
        object.__setattr__(self, "per_phase_ms", dict(self.per_phase_ms))

    @staticmethod
    def build(
        *,
        ts: float | None,
        query: str,
        model: str,
        retrieved_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_latency_ms: float,
        per_phase_ms: Mapping[str, float] | None,
        price_table: PriceTable,
    ) -> CostRecord:
        """Build a record from raw inputs, computing USD from the price table.

        Raises ``UnknownModelError`` if the model is not in the price table.
        ``ts`` defaults to ``time.time()`` when ``None`` (test injection point).
        """
        # `ts` is the field every read path in this module is keyed on --
        # `since()` filters `WHERE ts >= ?`, orders `ORDER BY ts ASC`, and
        # `last_24h()` is defined in terms of it -- and it was the one float on
        # this record that no seam validated (#184). The rule below is not new:
        # it is the guard already stated for `total_latency_ms` immediately
        # after this, and for every `per_phase_ms` value below that, applied to
        # the operand the sweep skipped.
        #
        # Measured before this guard, with `build` accepting each one:
        #
        #   nan       -> record() raised sqlite3.IntegrityError:
        #                NOT NULL constraint failed: cost_records.ts
        #                (SQLite stores NaN as NULL; an exception type outside
        #                this module's contract, naming a column rather than
        #                the caller's mistake)
        #   inf       -> stored, and in EVERY last_24h() window forever:
        #                last_24h(now=9_000_000_000) -- the year 2255 -- still
        #                returns it. The dashboard's headline 24h spend is
        #                permanently inflated by a row that is not from the
        #                last 24 hours and never will be.
        #   -inf      -> stored, and in NO window, ever.
        #   '2026-08-24' -> stored. `ts REAL NOT NULL` does not stop it: SQLite
        #                type affinity is a preference, not a constraint, and
        #                `typeof(ts)` comes back 'text'. TEXT sorts above every
        #                REAL, so it too is in every window forever, and it
        #                round-trips out of `since()` as a `str` in a field
        #                annotated `float`.
        #   True      -> stored as 1.0, i.e. 1970-01-01T00:00:01Z. Rejected for
        #                exactly the reason the `per_phase_ms` guard below
        #                gives: bool is an int subclass, and a stray True must
        #                not pose as a value in a numeric field.
        #
        # An ISO-8601 string is the obvious wrong-type guess for a field named
        # `ts`, and `time.time_ns()` instead of `time.time()` is the classic
        # unit slip; both were silent.
        #
        # Deliberately NOT bounded from above. `time.time_ns()` gives a finite
        # 1.7e18 that `time.gmtime` cannot format, but the representable range
        # is a property of the platform's `time_t`, not of this module, so an
        # input-domain ceiling here would pin a host property as a contract.
        # That half is guarded at the outcome instead, in
        # `scripts/telemetry_dashboard.py`, so one unformattable row does not
        # take the whole page down with it.
        #
        # Negative values are allowed: a pre-1970 timestamp is unusual but
        # well-defined, and this module has no business deciding an operator's
        # backfill window is wrong.
        #
        # One definition, shared with `TelemetryStore.record` (#245): the bare
        # constructor skips this method, and `ts`'s harm happens inside SQL, so
        # the store's write seam is the only other place it can be closed.
        ts_value = _checked_ts(ts if ts is not None else time.time())
        _checked_columns(
            {
                "query": query,
                "model": model,
                "retrieved_count": retrieved_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
        # Finiteness guard (#38): NaN latency propagates through percentile()
        # which sorts a list with NaN — Python's sort is stable but NaN
        # comparisons are all false, so the returned percentile is implementation-
        # defined and silently wrong. Mirrors the ModelPrice guard above.
        # And its type half (#303), as `per_phase_ms` below has it: `True` was
        # stored as 1.0 ms and a str escaped as a bare TypeError.
        if isinstance(total_latency_ms, bool) or not isinstance(total_latency_ms, (int, float)):
            raise ValueError(
                f"total_latency_ms must be a finite non-negative number; got {total_latency_ms!r}"
            )
        _check_sqlite_int_range(total_latency_ms, "total_latency_ms")  # before isfinite (#392)
        if not math.isfinite(total_latency_ms) or total_latency_ms < 0:
            raise ValueError(
                f"total_latency_ms must be a finite non-negative number; got {total_latency_ms}"
            )
        _checked_phases(per_phase_ms or {})
        prompt_usd, completion_usd = price_table.cost(model, prompt_tokens, completion_tokens)
        return CostRecord(
            ts=ts_value,
            query=query,
            model=model,
            retrieved_count=retrieved_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_usd=prompt_usd,
            completion_usd=completion_usd,
            total_usd=round(prompt_usd + completion_usd, 6),
            total_latency_ms=total_latency_ms,
            per_phase_ms=dict(per_phase_ms or {}),
        )


# ----------------------------------------------------------------------
# SQLite store
# ----------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cost_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    model TEXT NOT NULL,
    retrieved_count INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    prompt_usd REAL NOT NULL,
    completion_usd REAL NOT NULL,
    total_usd REAL NOT NULL,
    total_latency_ms REAL NOT NULL,
    per_phase_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cost_records_ts_idx ON cost_records (ts);
"""


class TelemetryStore:
    """Thin SQLite wrapper for ``CostRecord`` persistence.

    Schema is owned by this module — there's no migration story, just a
    single table that grows. For demo / dashboard purposes that's fine;
    a production deployment with retention requirements can build on
    top by writing to a managed store and using this as a reference.

    The store is not safe for concurrent writers (SQLite locks the file
    on write); that's an acceptable simplification for a single-process
    demo. A multi-process production setup should use a dedicated
    metrics backend.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TelemetryStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def record(self, rec: CostRecord) -> int:
        """Insert one record; return its assigned id.

        Raises ``ValueError`` for a ``ts`` outside ``CostRecord.build``'s rule,
        any other ``REAL`` column that is not a non-``bool`` ``int``/``float``
        or is NaN (#245, #368), or a ``TEXT``/``INTEGER`` column of the wrong
        type or a ``TEXT`` one holding a lone surrogate (#352, #368), or an
        ``int`` in any column outside SQLite's signed 64-bit range (#387), or a
        ``per_phase_ms`` that ``build`` refuses (#406): a non-``str`` key, or a
        value that is not a finite non-negative number. ``build``
        already refuses them; the bare constructor is public (`rag_kit.__all__`) and validates
        neither, and ``ts`` cannot be guarded at a read boundary the way
        ``aggregate`` guards latency: ``since()`` filters and orders on it in
        SQL, so an ``inf`` was in every ``last_24h()`` window and a ``-inf`` in
        none before any Python saw the row. Not in ``__post_init__``, because
        ``since()`` rebuilds stored rows through the constructor and must keep
        reading rows already written.
        """
        _checked_ts(rec.ts)
        _checked_columns(
            {name: getattr(rec, name) for name in _TEXT_CHECKED_FIELDS + _INTEGER_CHECKED_FIELDS}
        )
        for name in _NAN_CHECKED_REAL_FIELDS:
            value = getattr(rec, name)
            # bool is an int subclass; a stray True must not pose as 1.0 USD.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a real number; got {type(value).__name__} {value!r}"
                )
            if isinstance(value, float) and math.isnan(value):
                raise ValueError(
                    f"{name} must not be NaN; SQLite stores NaN as NULL, which the "
                    f"NOT NULL column {name} refuses"
                )
            _check_sqlite_int_range(value, name)
        _checked_phases(rec.per_phase_ms)
        cur = self._conn.execute(
            """
            INSERT INTO cost_records (
                ts, query, model, retrieved_count, prompt_tokens, completion_tokens,
                prompt_usd, completion_usd, total_usd, total_latency_ms, per_phase_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.ts,
                rec.query,
                rec.model,
                rec.retrieved_count,
                rec.prompt_tokens,
                rec.completion_tokens,
                rec.prompt_usd,
                rec.completion_usd,
                rec.total_usd,
                rec.total_latency_ms,
                json.dumps(dict(rec.per_phase_ms)),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def since(self, ts_floor: float) -> list[CostRecord]:
        """Return all records with ``ts >= ts_floor`` in chronological order.

        Raises ``ValueError`` for a floor that is not a real number or is NaN
        (#354); ``-inf`` and ``inf`` are accepted.
        """
        _checked_floor(ts_floor, "ts_floor")
        cur = self._conn.execute(
            """
            SELECT ts, query, model, retrieved_count, prompt_tokens, completion_tokens,
                   prompt_usd, completion_usd, total_usd, total_latency_ms, per_phase_json
            FROM cost_records
            WHERE ts >= ?
            ORDER BY ts ASC
            """,
            (ts_floor,),
        )
        rows = cur.fetchall()
        out: list[CostRecord] = []
        for row in rows:
            ts, query, model, n_retr, p_tok, c_tok, p_usd, c_usd, tot_usd, lat_ms, phase_json = row
            try:
                phases = json.loads(phase_json) if phase_json else {}
            except json.JSONDecodeError:
                phases = {}
            out.append(
                CostRecord(
                    ts=ts,
                    query=query,
                    model=model,
                    retrieved_count=n_retr,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    prompt_usd=p_usd,
                    completion_usd=c_usd,
                    total_usd=tot_usd,
                    total_latency_ms=lat_ms,
                    per_phase_ms=phases,
                )
            )
        return out

    def last_24h(self, now: float | None = None) -> list[CostRecord]:
        # `float(...)` (#392): an int `now` that `_checked_ts` accepts can still
        # fall below SQLite's INTEGER range once the 24 hours are subtracted --
        # `now=-2**63` is a `ts` that `record` stores, but the int cutoff escaped
        # `since` as a raw OverflowError. `ts` is a REAL column, so a float
        # cutoff is the comparison SQLite makes anyway.
        cutoff = float(_checked_ts(now, "now") if now is not None else time.time()) - 24 * 3600
        return self.since(cutoff)

    def dump_aggregate_json(self, path: str | Path, *, since_ts: float | None = None) -> None:
        """Write the current window aggregate to ``path`` as JSON (#50).

        Atomic on POSIX via ``rag_kit.io_utils.atomic_write_text`` —
        the same helper the eval-action's composite sticky comment
        relies on (#44) — so a SIGINT / disk-full / OOM between
        truncate and flush can't leave the log-tailer (or the
        ``scripts/telemetry_dashboard.py`` build artifact) reading a
        half-written file.

        ``since_ts`` defaults to last-24h (the same window
        ``last_24h()`` returns and the dashboard renders), so an
        operator wiring a cron job that simply writes
        ``telemetry.json`` every minute gets the rolling 24-hour
        aggregate by default. Pass an explicit ``since_ts`` to widen
        or narrow the window.

        On-disk shape is ``Aggregate.to_dict()`` with sorted keys,
        indent=2, trailing newline — byte-shape parity with the
        cost-optimizer's ``dump_aggregate_json`` and
        ``dump_stats_json`` so one log-parsing config consumes all
        portfolio observability artifacts.
        """
        floor = (
            _checked_floor(since_ts, "since_ts")
            if since_ts is not None
            else (time.time() - 24 * 3600)
        )
        records = self.since(floor)
        agg = aggregate(records)
        payload = json.dumps(agg.to_dict(), sort_keys=True, indent=2) + "\n"
        atomic_write_text(path, payload)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """Window aggregate: counts, totals, and latency percentiles."""

    n: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_usd: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-stable dict for observability/logging sinks (#50).

        Mirrors the pattern set by ``CacheTelemetry.to_dict`` in
        ``llm-cost-optimizer`` (#50 there) and ``CacheStats.to_dict``
        (#52 there): one stable-keys dict per long-lived state object,
        consumed by metric backends or written to disk via
        ``TelemetryStore.dump_aggregate_json``. The seven fields match
        the dataclass exactly so a future addition without serializer
        update is caught loud by the field-set lock test.
        """
        return {
            "n": self.n,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_usd": self.total_usd,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }


def percentile(values: Sequence[float], q: float) -> float:
    """NIST type-7 (linear interp) percentile, no numpy dependency.

    Matches `rag_kit.streaming.PhaseTimings.percentile` so a 24-hour
    aggregate window and a streaming-pipeline snapshot agree on the
    number when given the same sample. Edges (q=0, q=1) clamp.

    A non-finite value (NaN / +/-Infinity) is rejected (#80). `CostRecord.build`
    already guards `total_latency_ms` finiteness at ingestion (#38), but a
    record constructed directly (the dataclass has no `__post_init__`) bypasses
    it, and this function is exported on its own. Unguarded, `sorted()` leaves
    the NaN in an implementation-defined slot (all NaN comparisons are False),
    so the returned percentile is silently wrong and position-dependent, and
    `dump_aggregate_json` then serializes the `nan` as the bare token `NaN` —
    invalid JSON a strict log-tailer rejects whole. Fail loud at the metric
    boundary, the same posture as the q-range guard below and `PhaseTimings.record`.
    """
    # `q` first, before the empty-window return (#330). Checked after it, a bad
    # `q` -- `95` for the 95th percentile -- returned 0.0 for as long as the
    # window was empty and raised only once the first record landed, so the
    # same call behaved two ways depending on how much data was stored.
    # `PhaseTimings.percentile` checks `p` first and refuses bool by name (#42);
    # a bare range check let `True`/`False` through as q=1.0/0.0 and a `str`
    # crash as a raw TypeError. NaN fails the range check below.
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise ValueError(f"q must be a number in [0.0, 1.0]; got {q!r}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0.0, 1.0]; got {q}")
    if not values:
        return 0.0
    # `_past_float_range` first (#395): `isfinite` raised a raw OverflowError on
    # `10**400`. Such a value is named by its size; its repr can itself raise.
    huge = [_describe_number(v) for v in values if _past_float_range(v)]
    if huge:
        raise ValueError(f"values must all be finite numbers; got {', '.join(huge)}")
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"values must all be finite numbers; got {list(values)!r}")
    s = sorted(values)
    if q == 0.0:
        return s[0]
    if q == 1.0:
        return s[-1]
    idx = q * (len(s) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return s[int(idx)]
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def aggregate(records: Iterable[CostRecord]) -> Aggregate:
    rs = list(records)
    if not rs:
        return Aggregate(
            n=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_usd=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
        )
    latencies = [r.total_latency_ms for r in rs]
    costs = [r.total_usd for r in rs]
    # `total_usd` fails loud at the metric boundary, the cost sibling of the
    # `total_latency_ms` percentile guard below (#80). `CostRecord.build` guards
    # cost finiteness at ingestion (#38/#58), but a record constructed directly
    # (the dataclass has no `__post_init__`) bypasses it, and a ±Infinity survives
    # the store round-trip — SQLite's `total_usd REAL NOT NULL` only rejects NaN
    # (stored as NULL), it passes ±Infinity through `record()`/`since()` to here.
    # Unguarded, `dump_aggregate_json` then serializes the non-finite sum as the
    # bare token `Infinity` / `NaN` (json.dumps defaults to allow_nan=True) —
    # invalid JSON that breaks the "one log-parsing config consumes all portfolio
    # observability artifacts" contract. Reject it here rather than emit an
    # unparseable artifact, the same posture as the latency guard (#80); #135's
    # dashboard `/json` egress sibling sanitizes at the presentation boundary.
    #
    # Not unreachable after #387/#392, as #395 guessed: `aggregate` takes any
    # records, and a directly-constructed `total_usd=10**400` made `isfinite`
    # raise a raw OverflowError. Refused first, named by its size (#395).
    huge = [_describe_number(c) for c in costs if _past_float_range(c)]
    if huge:
        raise ValueError(f"total_usd values must all be finite numbers; got {', '.join(huge)}")
    if any(not math.isfinite(c) for c in costs):
        raise ValueError(f"total_usd values must all be finite numbers; got {costs!r}")
    # The cost domain, the clause #213 carried for latency and token counts and
    # not for cost (#410). `ModelPrice` refuses a negative rate ("a finite number
    # >= 0.0", #36) and `PriceTable.cost` a negative count, so `build` can never
    # produce a negative `total_usd`; `record` leaves the sign to this read side
    # by design (#245). A directly-constructed `total_usd=-5.0` therefore reached
    # the sum: measured on 49660aa, with one real record of 0.002 it published
    # `total_usd = -4.998` through `dump_aggregate_json` and the dashboard
    # headline. Checked per record, not on the sum: a -5.0 beside a +10.0 sums
    # positive and is just as wrong. Negativity only, like the latency clause
    # below, so the two finiteness messages above stay as their tests quote them.
    #
    # `total_usd` is the only money field anything here sums or publishes.
    # `prompt_usd` / `completion_usd` are stored and read back, but no `Aggregate`
    # field, `/json` row or dashboard cell reads them, so they have no read
    # boundary to agree with.
    if any(c < 0 for c in costs):
        raise ValueError(f"total_usd values must all be non-negative; got {costs!r}")
    # The latency domain, at the same boundary and for the same reason (#213).
    # `CostRecord.build` rejects "a finite non-negative number"; every sweep
    # since (#38, #58, #63, #80, #81, #82, #87, #106, #108, #135, #168) carried
    # the non-finite half of that sentence to the read boundaries and none
    # carried the other half. Latency finiteness is delegated to `percentile`
    # below; non-negativity cannot be, because `percentile` is a general helper
    # over `Sequence[float]` whose signed answers are correct -- a guard there
    # would reject valid input, which is worse than this gap. So the clause goes
    # where the *domain* is known, which is here, beside the `total_usd` guard
    # whose own comment already makes this argument about directly-constructed
    # records.
    #
    # Measured on b74753c: a `CostRecord(total_latency_ms=-5000.0)` (the
    # dataclass has no `__post_init__`, by design, so `build` is bypassed)
    # survived `record()` -> SQLite -> `since()` and reached here. With two
    # samples the negative *became* the reported `latency_p50_ms`; with twenty
    # it was invisible and still moved p50/p95/p99 down by 0.50/0.05/0.01 ms,
    # with nothing in the output naming the sample that did it.
    #
    # Negativity ONLY, not finiteness: `percentile` below already rejects a
    # non-finite latency with the message `test_aggregate_with_non_finite_latency`
    # pins (#80), and re-raising that case here with different wording would
    # change a contract two tests quote while adding nothing. The two failure
    # modes also want different diagnostics -- a NaN latency is a broken
    # measurement, a negative one is a broken clock or a hand-built record -- so
    # they read better as separate messages than as one merged sentence.
    if any(lat < 0 for lat in latencies):
        raise ValueError(f"total_latency_ms values must all be non-negative; got {latencies!r}")
    # Token counts, the same clause at the same boundary (#213). `PriceTable.cost`
    # rejects "token counts must be non-negative integers" on the way in, and the
    # sums below had no guard, so a directly-constructed record published a
    # NEGATIVE TOTAL: measured, two real records of 100 prompt tokens plus one
    # `prompt_tokens=-5000` gave `total_prompt_tokens = -4800`, and
    # `dump_aggregate_json` wrote `{"total_prompt_tokens": -4900}`.
    #
    # This site was not in #213's hand-written list of three. It was found by the
    # inventory lock in `tests/test_latency_domain_boundaries.py`, which collects
    # every "non-negative" rejection in the package from the AST -- and the member
    # the hand list missed has the *less* arguable consequence of the four, since
    # a negative duration can at least be read as a clock artifact and a negative
    # token count cannot be read at all.
    bad_tokens = [
        (r.prompt_tokens, r.completion_tokens)
        for r in rs
        if r.prompt_tokens < 0 or r.completion_tokens < 0
    ]
    if bad_tokens:
        raise ValueError(
            f"token counts must all be non-negative; got (prompt, completion) pairs {bad_tokens!r}"
        )
    return Aggregate(
        n=len(rs),
        total_prompt_tokens=sum(r.prompt_tokens for r in rs),
        total_completion_tokens=sum(r.completion_tokens for r in rs),
        total_usd=round(sum(costs), 6),
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
        latency_p99_ms=percentile(latencies, 0.99),
    )


__all__ = [
    "Aggregate",
    "CostRecord",
    "ModelPrice",
    "PriceTable",
    "TelemetryStore",
    "UnknownModelError",
    "aggregate",
    "percentile",
]
