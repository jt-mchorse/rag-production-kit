"""`to_sse` is total: every frame parses, and every frame encodes (#188, D-017).

`_json_safe` is the documented single chokepoint for SSE frame validity -- its
own docstring says the wire serializer is "the correct single chokepoint to
enforce frame validity (#106) … guaranteeing every frame parses". Two structural
properties defeated that. It walked **values only** --
`{k: _json_safe(v) for k, v in obj.items()}` passed every key straight through
-- and it was **recursive** over caller-supplied data.

Measured on `main` before this change, `metadata` on a `retrieved` event::

    input at metadata                to_sse()              frame.encode()           wire
    CONTROL str key + str value      ok                    ok                       correct
    CONTROL non-finite VALUE (#106)  ok                    ok                       {"score": null}
    non-finite float KEY             ok                    ok                       {"Infinity": "x"}
    int key colliding with a str     ok                    ok                       {"1": "a", "1": "b"}  <- DUPLICATE NAME
    tuple key                        RAISES TypeError      -                        no frame
    frozenset key                    RAISES TypeError      -                        no frame
    lone surrogate in a value        ok                    RAISES UnicodeEncodeError  no frame
    lone surrogate in a key          ok                    RAISES UnicodeEncodeError  no frame
    deeply nested (3000 levels)      RAISES RecursionError -                        no frame
    circular reference               RAISES RecursionError -                        no frame

The sanitizer was less robust than the call it wraps::

    json.dumps alone, circular dict    -> ValueError: Circular reference detected
    recursive _json_safe, circular     -> RecursionError

Depth is bounded by this module's own `_MAX_DEPTH` rather than by out-recursing
`json.dumps`, because how deep `json.dumps` can go is a property of the
interpreter version -- see
`test_depth_beyond_the_limit_is_truncated_rather_than_raising`.

`default=str` cannot rescue any of the key rows: `json.dumps` raises
`TypeError: keys must be str, int, float, bool or None` *before* consulting
`default=`, which is only ever called for values.

And the failures are worse than a normal exception, because
`demo/streaming/server.py:197` reads::

    frame = to_sse(event).encode("utf-8")
    try:
        self.wfile.write(frame)
    except BrokenPipeError:
        return

`to_sse` and `.encode` are *outside* the `try`, and the 200 plus headers have
already gone out -- so the client got a truncated `text/event-stream` with no
`error` frame and no `done` frame, byte-indistinguishable from a network drop.
`StreamingPipeline.run`'s own `except Exception -> yield StreamEvent("error")`
arm cannot help either: it wraps the generator body, and `to_sse` runs after
each event is yielded. The one seam promising frame validity was the one seam
outside the error handling.

This module asserts the property, not the implementation: for every row, the
frame must satisfy `json.loads` **and** `.encode("utf-8")`, and carry no
duplicate JSON name.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import pytest

from rag_kit.streaming import _MAX_DEPTH as MAX_DEPTH
from rag_kit.streaming import _TOO_DEEP as TOO_DEEP_MARKER
from rag_kit.streaming import StreamEvent, to_sse

# Built from a codepoint, never written literally: a source file containing a
# literal lone surrogate has no UTF-8 encoding and cannot be saved.
SURROGATE = chr(0xD800)
REPLACEMENT = "�"


def _deep(levels: int) -> dict[str, Any]:
    node: Any = {"leaf": 1}
    for _ in range(levels):
        node = {"n": node}
    return node


def _circular() -> dict[str, Any]:
    node: dict[str, Any] = {}
    node["self"] = node
    return node


def _shared_but_acyclic() -> dict[str, Any]:
    """A DAG. Must NOT be mistaken for a cycle and truncated."""
    inner = {"x": 1}
    return {"a": inner, "b": inner}


# (label, metadata) -- every row must produce a parseable, encodable frame.
TABLE: list[tuple[str, Any]] = [
    ("CONTROL str key and str value", {"src": "doc1"}),
    ("CONTROL non-finite value (#106)", {"score": float("inf")}),
    ("CONTROL nan value (#106)", {"score": float("nan")}),
    ("CONTROL nested list and dict", {"xs": [1, {"y": [None, True]}]}),
    ("CONTROL non-ASCII value", {"src": "café 日本語 \U0001f389"}),
    ("CONTROL empty containers", {"a": {}, "b": []}),
    ("non-finite float key", {float("inf"): "x"}),
    ("nan float key", {float("nan"): "x"}),
    ("int key", {1: "one"}),
    ("bool key", {True: "t"}),
    ("None key", {None: "n"}),
    ("int key colliding with a str key", {1: "from-int", "1": "from-str"}),
    ("tuple key", {(1, 2): "x"}),
    ("frozenset key", {frozenset({1}): "x"}),
    ("lone surrogate in a value", {"src": "doc" + SURROGATE}),
    ("lone surrogate in a key", {"k" + SURROGATE: "v"}),
    ("lone surrogate nested in a list", {"xs": ["ok", "bad" + SURROGATE]}),
    ("tuple value", {"xs": ("a", "b")}),
    ("deeply nested (3000 levels)", _deep(3000)),
    ("circular reference", _circular()),
    ("shared but acyclic (a DAG)", _shared_but_acyclic()),
]


def _frame(metadata: Any) -> str:
    return to_sse(StreamEvent(type="retrieved", payload={"metadata": metadata}, elapsed_ms=1.0))


def _data_line(frame: str) -> str:
    assert frame.startswith("event: ")
    assert frame.endswith("\n\n")
    return frame.split("data: ", 1)[1].rstrip("\n")


@pytest.mark.parametrize(("label", "metadata"), TABLE, ids=[r[0] for r in TABLE])
def test_every_frame_parses_and_encodes(label: str, metadata: Any) -> None:
    """The property the docstring claims, run over the table that broke it."""
    frame = _frame(metadata)
    frame.encode("utf-8")  # the operation at demo/streaming/server.py:197
    json.loads(_data_line(frame))  # the operation a browser's EventSource runs


@pytest.mark.parametrize(("label", "metadata"), TABLE, ids=[r[0] for r in TABLE])
def test_no_frame_carries_a_duplicate_json_name(label: str, metadata: Any) -> None:
    """RFC 8259 leaves duplicate names undefined and `JSON.parse` keeps the last.

    `{1: "a", "1": "b"}` is two entries in Python and one JSON name, and
    `json.dumps` emitted both -- so an entry vanished from `metadata` with no
    diagnostic anywhere.
    """
    seen_duplicate: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        names = [k for k, _ in pairs]
        if len(set(names)) != len(names):
            seen_duplicate.append(label)
        return dict(pairs)

    json.loads(_data_line(_frame(metadata)), object_pairs_hook=hook)
    assert seen_duplicate == [], f"{label} put a duplicate JSON name on the wire"


# ----------------------------------------------------------------------
# The frame's *content*, not just its validity
# ----------------------------------------------------------------------


def _metadata_of(metadata: Any) -> Any:
    return json.loads(_data_line(_frame(metadata)))["payload"]["metadata"]


def test_an_all_clean_payload_is_byte_identical() -> None:
    """The change is a sanitizer, not a re-serializer. Nothing about an
    already-valid frame moves -- key order, separators, non-ASCII literals."""
    clean = {"src": "doc1", "score": 0.42, "tags": ["a", "b"], "note": "café 日本語"}
    frame = _frame(clean)
    expected_data = json.dumps(
        {"payload": {"metadata": clean}, "elapsed_ms": 1.0}, ensure_ascii=False
    )
    assert _data_line(frame) == expected_data
    assert frame == f"event: retrieved\ndata: {expected_data}\n\n"


def test_non_finite_values_still_become_null() -> None:
    """#106's rule is preserved exactly, not merely 'still valid JSON'."""
    assert _metadata_of({"a": float("inf"), "b": float("-inf"), "c": float("nan")}) == {
        "a": None,
        "b": None,
        "c": None,
    }


def test_finite_floats_are_untouched() -> None:
    assert _metadata_of({"a": 0.0, "b": -1.5, "c": 1e308}) == {"a": 0.0, "b": -1.5, "c": 1e308}


@pytest.mark.parametrize(
    ("key", "expected_name"),
    [
        (1, "1"),
        (-7, "-7"),
        (True, "true"),
        (False, "false"),
        (None, "null"),
        (2.5, "2.5"),
    ],
)
def test_json_native_key_types_keep_json_dumps_own_spelling(key: Any, expected_name: str) -> None:
    """These are the key types `json.dumps` coerces itself. Doing the coercion
    in `_safe_key` must not change what lands on the wire -- in particular
    `True` stays `"true"` rather than becoming `"1"` via its `int` base."""
    assert list(_metadata_of({key: "v"})) == [expected_name]


def test_a_non_finite_key_becomes_null_not_the_string_infinity() -> None:
    """The guard covered one operand. `_json_safe` mapped non-finite floats to
    `null` as *values* and passed them through as *keys*, so `{inf: "x"}`
    reached the wire as the string name `"Infinity"` -- a key no caller could
    have written."""
    assert _metadata_of({float("inf"): "x"}) == {"null": "x"}
    assert _metadata_of({float("nan"): "x"}) == {"null": "x"}


def test_a_key_type_json_dumps_rejects_becomes_a_string_instead_of_raising() -> None:
    assert _metadata_of({(1, 2): "x"}) == {"(1, 2)": "x"}


def test_a_coerced_key_collision_keeps_the_last_entry_as_json_parse_did() -> None:
    """Wire *semantics* are unchanged -- `JSON.parse` already kept the last of
    the duplicate names. What changes is that the frame is now well-defined
    JSON instead of relying on undefined behaviour."""
    assert _metadata_of({1: "from-int", "1": "from-str"}) == {"1": "from-str"}


def test_an_unencodable_character_is_replaced_not_dropped_and_not_a_question_mark() -> None:
    """U+FFFD, deliberately -- `errors="replace"` substitutes `"?"` on the
    encode side, and `"?"` is a character a caller can legitimately have
    written, so it would make a substitution indistinguishable from real data."""
    assert _metadata_of({"src": "doc" + SURROGATE + "end"}) == {"src": "doc" + REPLACEMENT + "end"}
    assert _metadata_of({"k" + SURROGATE: "v"}) == {"k" + REPLACEMENT: "v"}
    assert _metadata_of({"xs": ["ok", "bad" + SURROGATE]}) == {"xs": ["ok", "bad" + REPLACEMENT]}


def test_surrounding_text_is_untouched_by_a_single_bad_codepoint() -> None:
    payload = "a long piece of café 日本語 text " + SURROGATE + " and more text after it"
    assert _metadata_of({"src": payload}) == {"src": payload.replace(SURROGATE, REPLACEMENT)}


def test_a_cycle_is_named_rather_than_raising() -> None:
    """`json.dumps` alone gives `ValueError: Circular reference detected`; the
    recursive walk turned that into an opaque `RecursionError` before
    `json.dumps` was ever reached. Naming it in the frame tells the operator
    more than either, and honours "stream alive, don't raise"."""
    assert _metadata_of(_circular()) == {"self": "<circular reference>"}


def test_a_shared_subtree_is_not_mistaken_for_a_cycle() -> None:
    """Cycle detection is per *path*, not a global seen-set. A DAG -- the same
    dict referenced from two places -- is not circular and must serialize in
    full, twice."""
    assert _metadata_of(_shared_but_acyclic()) == {"a": {"x": 1}, "b": {"x": 1}}


def test_depth_within_the_limit_is_reproduced_exactly() -> None:
    payload = _deep(10)
    assert _metadata_of(payload) == payload


def test_depth_beyond_the_limit_is_truncated_rather_than_raising() -> None:
    """The limit is *ours*, deliberately, and this is the correction to a claim
    an earlier revision of this file made.

    That revision asserted 3000 levels serialize, on the reasoning that
    "`json.dumps`'s C encoder has no Python recursion limit". Two things were
    wrong with it. `to_sse` passes `default=str`, which disqualifies the C
    encoder and selects the recursive pure-Python `_make_iterencode`. And how
    deep *that* can go is a property of the interpreter version: it handled
    ~14690 levels on CPython 3.14 locally and raised `RecursionError` at 3000 on
    a CPython 3.11 CI runner, because 3.12 decoupled pure-Python frames from the
    C stack. The assertion passed locally and failed on CI -- a host-environment
    assertion, not a test.

    A guarantee that every frame parses cannot be conditional on which Python is
    running it, so `_json_safe` truncates at its own `_MAX_DEPTH` and the frame
    is produced either way.
    """
    node: Any = _metadata_of(_deep(3000))
    hops = 0
    while isinstance(node, dict) and "n" in node:
        node = node["n"]
        hops += 1
    assert node == TOO_DEEP_MARKER
    assert hops < MAX_DEPTH


def test_the_truncation_boundary_is_deterministic_not_host_dependent() -> None:
    """Every payload deeper than the limit truncates at the same place, on any
    interpreter -- which is the whole point of pinning our own limit."""
    depths = [_depth_of(_metadata_of(_deep(n))) for n in (3000, 500, 200, 51)]
    assert len(set(depths)) == 1, depths


def test_totality_holds_under_a_constrained_recursion_limit() -> None:
    """The portable version of the depth guarantee.

    Rather than asserting what *this* interpreter's `json.dumps` can survive --
    the assertion that passed on 3.14 and failed on 3.11 -- constrain the stack
    deliberately and assert `to_sse` still produces a frame. This fails on any
    interpreter if the bound is ever removed, and passes on any interpreter
    while it is there.
    """
    original = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(200)
        frame = _frame(_deep(3000))
        frame.encode("utf-8")
        json.loads(_data_line(frame))
    finally:
        sys.setrecursionlimit(original)


def _depth_of(node: Any) -> int:
    hops = 0
    while isinstance(node, dict) and "n" in node:
        node = node["n"]
        hops += 1
    return hops


def test_tuples_become_lists_at_every_depth() -> None:
    assert _metadata_of({"xs": ("a", ("b", "c"))}) == {"xs": ["a", ["b", "c"]]}


# ----------------------------------------------------------------------
# Anti-vacuous
# ----------------------------------------------------------------------


def test_the_table_actually_exercises_both_halves() -> None:
    """A table that drifted to all-controls would make every parametrized case
    above pass while proving nothing."""
    labels = [lbl for lbl, _ in TABLE]
    assert len(labels) == len(set(labels))
    controls = [lbl for lbl in labels if lbl.startswith("CONTROL")]
    assert len(controls) >= 5
    assert len(labels) - len(controls) >= 12


def test_json_dumps_alone_would_still_fail_on_the_raw_inputs() -> None:
    """The rows are not hypothetical: unsanitized, `json.dumps` raises on the
    key rows and the cycle, which is what makes the sanitizer load-bearing."""
    with pytest.raises(TypeError, match="keys must be str"):
        json.dumps({(1, 2): "x"})
    with pytest.raises(ValueError, match="Circular reference"):
        json.dumps(_circular())
    # And the two that do NOT raise are the quiet ones this issue is about.
    assert json.dumps({1: "a", "1": "b"}) == '{"1": "a", "1": "b"}'  # duplicate name
    assert json.dumps({float("inf"): "x"}) == '{"Infinity": "x"}'
    assert math.isinf(float("inf"))
