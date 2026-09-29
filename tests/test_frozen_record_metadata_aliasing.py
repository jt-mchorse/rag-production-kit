"""A frozen record owns its container field (#227, D-022).

`frozen=True` prevents *rebinding* an attribute. It says nothing about the
object the attribute points at, so a `dict` field on a frozen record was
editable in place through any reference the caller still held — with no
`FrozenInstanceError` anywhere, because nothing was ever rebound.

All five of this package's frozen records with a container field were in that
state, and `portfolio-ops#71` triaged them here. Two things that sweep's
question ("is the field constructed from data a *caller* retains?") cannot see:

**Four containers were shared between records, across internal seams**, with no
caller-supplied dict involved at all:

    RetrievalResult.metadata -> Citation.metadata           (generator)
    RetrievalResult.metadata -> Candidate.metadata          (streaming)
    Candidate.metadata       -> ScoredCandidate.metadata    (reranker, 3 sites)
    RetrievalResult.metadata -> StreamEvent.payload["chunks"][i]["metadata"]
    RetrievalResult.ranks    -> StreamEvent.payload["chunks"][i]["ranks"]

The issue named the first and third. The fourth and fifth are the sharp ones:
`to_sse` serializes `payload`, so editing a `RetrievalResult` changed the frame
of an event that had **already been yielded**.

**And the copy's depth is the whole question.** `metadata` is `dict[str, Any]`;
`Any` proves nothing about the values, so `dict(...)` leaves every nested
container shared. `CostRecord.per_phase_ms` is the one row where a shallow copy
*is* complete, because its declared element type is `float` —
`test_the_shallow_copy_on_per_phase_ms_still_has_its_premise` locks that
premise, since widening the annotation is what would silently break it.

On the rows this file **clears**
-------------------------------

`Document.metadata`, `RetrievalResult.metadata`, `RetrievalResult.ranks` and
`PhaseTimings.{retrieving,reranking,generating,total}` are container fields on
**non-frozen** dataclasses. They make no immutability claim, so aliasing there
is not this defect. Pinned by name below so the next sweep reads a result rather
than re-deriving one, the way `llm-eval-harness#254` pinned its four cleared
rows.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag_kit import (
    Candidate,
    Citation,
    CostRecord,
    LexicalOverlapReranker,
    RetrievalResult,
    ScoredCandidate,
    StreamEvent,
    StreamingPipeline,
    TemplateGenerator,
)
from rag_kit.io_utils import copy_json_value
from rag_kit.streaming import to_sse

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "rag_kit"


def _citation(metadata: dict[str, Any]) -> Citation:
    return Citation(external_id="e", text="t", metadata=metadata)


def _candidate(metadata: dict[str, Any]) -> Candidate:
    return Candidate(external_id="e", text="t", metadata=metadata)


def _scored(metadata: dict[str, Any]) -> ScoredCandidate:
    return ScoredCandidate(
        external_id="e", text="t", metadata=metadata, rerank_score=1.0, rerank_rank=1
    )


def _event(payload: dict[str, Any]) -> StreamEvent:
    return StreamEvent(type="retrieving", payload=payload, elapsed_ms=1.0)


def _cost_record(per_phase_ms: dict[str, float]) -> CostRecord:
    return CostRecord(
        ts=0.0,
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


#: `(label, build, field_name)` for every frozen record with a container field.
#: Discovered by walking the package (see the population arms at the bottom),
#: then written down here so each row gets its own named failure.
ROWS = [
    ("Citation.metadata", _citation, "metadata"),
    ("Candidate.metadata", _candidate, "metadata"),
    ("ScoredCandidate.metadata", _scored, "metadata"),
    ("StreamEvent.payload", _event, "payload"),
    ("CostRecord.per_phase_ms", _cost_record, "per_phase_ms"),
]

#: The four rows whose declared element type is `Any`, i.e. the ones where the
#: copy has to be deep. `CostRecord.per_phase_ms` is deliberately absent — see
#: the premise lock below.
DEEP_ROWS = [row for row in ROWS if row[0] != "CostRecord.per_phase_ms"]


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        external_id="d1",
        text="the cat sat on the mat",
        metadata={"source": "corpus.md", "tags": {"topic": "animals"}},
        fused_score=0.9,
        ranks={"dense": 1},
        rerank_score=None,
        rerank_rank=None,
    )


# --------------------------------------------------------------------------
# One level: the caller's own reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "build", "field"), ROWS, ids=[r[0] for r in ROWS])
def test_the_constructor_does_not_alias_the_callers_container(
    label: str, build: Any, field: str
) -> None:
    """The row as `portfolio-ops#71` asks it: is the field the caller's object?"""
    supplied: dict[str, Any] = {"kept": 1.0}
    record = build(supplied)
    supplied["INJECTED"] = 2.0
    assert "INJECTED" not in getattr(record, field), (
        f"{label} is the caller's container; `frozen=True` stopped nothing "
        f"because nothing was rebound."
    )
    assert getattr(record, field)["kept"] == 1.0, f"{label} lost the data it was given"


@pytest.mark.parametrize(("label", "build", "field"), DEEP_ROWS, ids=[r[0] for r in DEEP_ROWS])
def test_the_copy_is_deep_over_containers(label: str, build: Any, field: str) -> None:
    """**The arm that rejects a shallow copy**, which is the whole finding.

    The defect *is* a shallow copy relationship, so an arm asking only "was it
    copied at all" is satisfied by the bug. `dict[str, Any]` proves nothing
    about the values; a nested `dict` under `dict(...)` is still the caller's.
    """
    supplied: dict[str, Any] = {"nested": {"k": "original"}, "listed": [{"k": "original"}]}
    record = build(supplied)
    supplied["nested"]["k"] = "MUTATED"
    supplied["listed"][0]["k"] = "MUTATED"
    stored = getattr(record, field)
    assert stored["nested"]["k"] == "original", (
        f"{label} was copied one level deep, so everything nested inside it is still the caller's."
    )
    assert stored["listed"][0]["k"] == "original", f"{label}'s copy does not recurse into lists."


def test_the_shallow_copy_on_per_phase_ms_still_has_its_premise() -> None:
    """`CostRecord.per_phase_ms` gets `dict(...)`, and this is why that is enough.

    A shallow copy is complete exactly when the element type is proved
    immutable. `Mapping[str, float]` proves it; `dict[str, Any]` does not. That
    premise lives in an annotation, so widening the annotation is what would
    silently turn a correct shallow copy into the defect this file exists for —
    and nothing else in the suite would say so.

    `Mapping` rather than `dict` is also why the copy is `dict(...)` and not
    `copy_json_value`: the latter passes a non-`dict` mapping straight back.
    """
    annotation = {f.name: f.type for f in dataclasses.fields(CostRecord)}["per_phase_ms"]
    assert str(annotation) == "Mapping[str, float]", (
        f"per_phase_ms is now annotated {annotation!r}. `dict(...)` in "
        f"`CostRecord.__post_init__` is only a complete copy while the element "
        f"type is immutable — route it through `copy_json_value` (and make it a "
        f"`dict`) if this widened."
    )


def test_build_and_the_bare_constructor_now_agree() -> None:
    """`CostRecord.build` has copied since it was written; the constructor was the
    unguarded half, and `CostRecord` is public in `__all__`."""
    from rag_kit.telemetry import ModelPrice, PriceTable

    prices = PriceTable({"m": ModelPrice(3.0, 15.0)})

    supplied = {"retrieving": 1.0}
    built = CostRecord.build(
        ts=0.0,
        query="q",
        model="m",
        retrieved_count=1,
        prompt_tokens=1,
        completion_tokens=1,
        total_latency_ms=1.0,
        per_phase_ms=supplied,
        price_table=prices,
    )
    direct = _cost_record(supplied)
    supplied["INJECTED"] = 2.0
    assert "INJECTED" not in built.per_phase_ms
    assert "INJECTED" not in direct.per_phase_ms, (
        "the factory copies and the bare constructor does not — the read path is "
        "stricter than the write path"
    )


# --------------------------------------------------------------------------
# Across the seams: no caller-supplied dict involved
# --------------------------------------------------------------------------


def test_a_citation_does_not_share_the_retrieval_result_it_cites() -> None:
    """Measured through `TemplateGenerator().generate(...)`, the ordinary path."""
    rr = _retrieval_result()
    answer = TemplateGenerator().generate("what sat on the mat?", [rr])
    citation = answer.citations[0]
    assert citation.metadata is not rr.metadata
    citation.metadata["source"] = "MUTATED_VIA_CITATION"
    assert rr.metadata["source"] == "corpus.md"


def test_a_scored_candidate_does_not_share_its_input_candidate() -> None:
    """Measured through `LexicalOverlapReranker().rerank(...)`."""
    candidate = _candidate({"source": "corpus.md", "tags": {"topic": "animals"}})
    scored = LexicalOverlapReranker().rerank("cat mat", [candidate])[0]
    assert scored.metadata is not candidate.metadata
    scored.metadata["tags"]["topic"] = "MUTATED"
    assert candidate.metadata["tags"]["topic"] == "animals", (
        "the nested dict survived the seam even though the outer one did not"
    )


def _retrieved_event(rr: RetrievalResult) -> StreamEvent:
    class _FakeRetriever:
        def search(self, query: str, k: int, *, reranker: Any = None) -> list[RetrievalResult]:
            return [rr]

    events = list(StreamingPipeline(_FakeRetriever()).run("cat mat", k=1))
    return next(e for e in events if e.type == "retrieved")


@pytest.mark.parametrize("field", ["metadata", "ranks"])
def test_the_retrieved_event_payload_does_not_share_the_retrieval_result(field: str) -> None:
    """`_chunk_to_event` put the live `metadata` and `ranks` dicts in the payload.

    Both, not just `metadata` — `ranks` is `dict[str, int]` on the same record
    and went into the same payload, and the issue named neither.
    """
    rr = _retrieval_result()
    chunk = _retrieved_event(rr).payload["chunks"][0]
    assert chunk[field] is not getattr(rr, field)


def test_an_already_yielded_frame_does_not_change_when_the_result_is_edited() -> None:
    """The consequence, stated as the wire sees it — the row with a published artifact.

    `StreamEvent` is frozen and already yielded; a consumer that buffers events
    and serializes later got a frame whose contents changed after it was
    emitted. No caller-supplied dict is involved anywhere in this arm: the only
    mutation is to a `RetrievalResult` the pipeline was handed.
    """
    rr = _retrieval_result()
    event = _retrieved_event(rr)
    before = to_sse(event)
    rr.metadata["source"] = "MUTATED_AFTER_THE_EVENT_WAS_YIELDED"
    rr.ranks["dense"] = 999
    assert to_sse(event) == before, (
        "the SSE frame for an already-yielded event changed when the retrieval "
        "result behind it was edited"
    )
    assert "MUTATED_AFTER_THE_EVENT_WAS_YIELDED" not in to_sse(event)


# --------------------------------------------------------------------------
# The copy itself: total, and faithful to the input's shape
# --------------------------------------------------------------------------


def test_a_cycle_is_copied_rather_than_exhausting_the_stack() -> None:
    """The recursive one-liner this started as never terminated here.

    `streaming._json_safe` was rewritten iteratively for exactly this and its
    docstring says why. Putting a recursive copy at `StreamEvent.__post_init__`
    reintroduces it one call *earlier* than the seam `_json_safe` protects —
    measured, 8 red arms in `tests/test_sse_frame_totality.py`.
    """
    payload: dict[str, Any] = {"self": None}
    payload["self"] = payload
    copied = copy_json_value(payload)
    assert copied is not payload
    assert copied["self"] is copied, "the cycle was not preserved as a cycle"


def test_a_cyclic_payload_still_reaches_the_wire_as_a_named_cycle() -> None:
    """A copier is not a sanitizer: `_json_safe` still owns the wire decision.

    Replacing the cycle with a marker inside the copy would move a wire-format
    decision into a record constructor, and would change what
    `tests/test_sse_frame_totality.py` pins.
    """
    payload: dict[str, Any] = {"self": None}
    payload["self"] = payload
    frame = to_sse(_event(payload))
    assert "circular" in frame.lower()


def test_deep_nesting_does_not_raise() -> None:
    """3000 levels, the depth this package's own totality table already uses."""
    deep: dict[str, Any] = {}
    node = deep
    for _ in range(3000):
        node["n"] = {}
        node = node["n"]
    copied = copy_json_value(deep)
    assert copied is not deep
    depth = 0
    node = copied
    while "n" in node:
        node = node["n"]
        depth += 1
    assert depth == 3000


def test_the_copy_is_total_under_a_constrained_recursion_limit() -> None:
    """Assert the outcome, not the road: a shallow limit must not change the answer."""
    payload: dict[str, Any] = {"a": [{"b": [{"c": 1}]}]}
    original = sys.getrecursionlimit()
    sys.setrecursionlimit(60)
    try:
        assert copy_json_value(payload) == payload
    finally:
        sys.setrecursionlimit(original)


def test_shared_substructure_stays_shared_in_the_copy() -> None:
    """The memo is not only a cycle guard.

    Two keys pointing at one dict still point at one dict afterwards — a fresh
    one. The recursive version expanded that into independent copies, which is
    both less faithful and, on a DAG, exponential.
    """
    inner: dict[str, Any] = {"k": "v"}
    payload = {"left": inner, "right": inner}
    copied = copy_json_value(payload)
    assert copied["left"] is copied["right"]
    assert copied["left"] is not inner


def test_a_mutable_container_inside_a_tuple_is_copied_too() -> None:
    """Was a pinned limitation; is now the rule (#229, D-023).

    The arm this replaces asserted the *sharing*, and its own failure message
    said: "the tuple case is now covered — update this arm and D-022 rather than
    deleting it, and say what happens to a namedtuple's class". Both halves are
    answered here and in
    `test_a_namedtuple_is_normalised_to_its_base_class_and_that_costs_nothing`.

    D-022 priced the tuple exclusion at a `namedtuple`'s class. That price is
    already paid by `streaming._new_container`, which flattens every tuple to a
    list before `json.dumps` is reached, so nothing a caller observes through
    this package was ever going to see the class.
    """
    inner: dict[str, Any] = {"k": "original"}
    record = _event({"pair": ("a", inner)})
    inner["k"] = "MUTATED"
    assert record.payload["pair"][1]["k"] == "original"
    assert isinstance(record.payload["pair"], tuple), (
        "a tuple must stay a tuple — the walk builds it as a list shell and "
        "freezes it, and forgetting the freeze would silently change the type "
        "of a field a caller reads back"
    )


def test_an_already_yielded_frame_does_not_change_when_a_tuple_nested_list_is_edited() -> None:
    """D-022's own headline harm, reproduced through the kind it skipped.

    Its rationale leads with "editing a `RetrievalResult` changed the SSE frame
    of an event that had already been yielded". At `ca0491b` that was still
    reachable with one tuple in the path, which is what #229 measured.
    """
    inner = ["b"]
    event = _event({"k": ("a", inner)})
    before = to_sse(event)
    inner.append("MUTATED-AFTER-YIELD")
    assert to_sse(event) == before


def test_a_namedtuple_is_normalised_to_its_base_class_and_that_costs_nothing() -> None:
    """The accepted cost, pinned as a decision so nobody rediscovers it as a surprise.

    The same test the docstring already applied to a `dict` subclass: equal to
    its base, identical JSON, so nothing observable through this package changes.
    A `namedtuple` passes it, and the wire seam had already decided the question.
    """
    point = collections.namedtuple("point", "x y")
    copied = copy_json_value({"p": point(1, 2)})
    assert copied["p"] == (1, 2)
    assert type(copied["p"]) is tuple
    assert json.dumps(copied) == json.dumps({"p": [1, 2]})
    # ... and the frame is what it always was.
    assert '"p": [1, 2]' in to_sse(_event({"p": point(1, 2)}))


def test_a_set_is_copied_and_a_frozenset_is_not() -> None:
    """The other diagonal of the rule, and the one the tuple reason never covered.

    A `set` is mutable, so sharing it is the defect; every element is hashable,
    so no mutable container is reachable through it and one level is the whole
    depth. A `frozenset` answers no to both questions. `_json_safe` renders
    either as `str(the_set)` via `default=`, so an aliased one changed an
    already-published frame exactly as the tuple case did.
    """
    tags = {"a"}
    event = _event({"tags": tags})
    before = to_sse(event)
    tags.add("MUTATED")
    assert to_sse(event) == before
    assert event.payload["tags"] == {"a"}
    assert type(event.payload["tags"]) is set

    shared = frozenset({"a"})
    copied = copy_json_value({"f": shared})
    assert copied["f"] is shared, (
        "a frozenset is immutable and cannot contain a mutable container, so "
        "copying it buys nothing; `isinstance(frozenset(), set)` is False, which "
        "is what keeps the set branch from catching it"
    )


def test_no_mutable_container_can_be_inside_a_set() -> None:
    """The reason the set branch does not walk, as a test rather than as prose.

    "One level is the whole depth for a `set`" is an argument, and an argument is
    not an arm — a neighbour that *does* walk into the set is **0 red** against
    every other test here, because for hashable elements it computes the same
    answer more slowly. It is redundant, not wrong, and no assertion about the
    copy's output can separate the two.

    What *is* falsifiable is the premise: Python refuses to put a mutable
    container in a set at all. Pinned here so the claim in `copy_json_value`'s
    table is checked rather than trusted, and so a future element kind that turns
    out to be both hashable and mutable fails loudly on this line.
    """
    # A tuple *containing* a mutable is unhashable for the same reason, which is
    # what makes the set branch safe rather than lucky: the tuple case cannot
    # hide inside the set case.
    for mutable in ({}, [], set(), bytearray(), ([],)):
        with pytest.raises(TypeError, match="unhashable"):
            set().add(mutable)


def test_a_cycle_through_a_tuple_is_copied_isomorphically() -> None:
    """A tuple cannot close a cycle alone, but it can sit inside one.

    `a = []; t = (a,); a.append(t)` is legal: the tuple holds an object that
    already existed, and the *list* closes the loop. That mutable link is what
    makes the deferred freeze work — the tuple's parents are patched after it is
    frozen, and a parent in a cycle is always mutable.
    """
    a: list[Any] = []
    a.append((a,))
    copied = copy_json_value(a)
    assert copied is not a
    assert isinstance(copied[0], tuple)
    assert copied[0][0] is copied


def test_a_tuple_directly_inside_a_tuple_is_frozen_child_first() -> None:
    """The arm that rejects a freeze in *discovery* order, and it needed adding.

    Every other tuple arm here happens to put a `dict` or `list` between the two
    tuples, and on that shape either order works — which is why a neighbour that
    froze parents first was **0 red** until this arm existed. Directly nested is
    the separating case: freezing the outer shell first captures the inner one
    while it is still a list, and patching the inner one afterwards writes into
    the pre-freeze shell that nothing points at any more.

    So the assertion is on the *type* two levels down, not on the values. The
    values are right under both orders.
    """
    inner: list[Any] = ["x"]
    copied = copy_json_value({"outer": ((inner,),)})
    assert isinstance(copied["outer"], tuple)
    assert isinstance(copied["outer"][0], tuple), (
        "the inner tuple is still a list — the freeze ran parents-first, so the "
        "outer tuple captured an unfrozen shell"
    )
    assert isinstance(copied["outer"][0][0], list)
    inner.append("MUTATED")
    assert copied["outer"][0][0] == ["x"]


def test_the_deferred_freeze_is_not_recursive() -> None:
    """The `RecursionError` the iterative walk exists to avoid stays avoided.

    D-022's recursive first draft put `tests/test_sse_frame_totality.py` 8 red on
    a 3000-deep payload. The freeze pass is a flat loop over a list of shells, so
    a deeply *tuple*-nested payload is total too — and that is the shape a
    plausible recursive freeze would fail on while every dict/list arm stayed
    green.
    """
    deep: Any = [1]
    for _ in range(3000):
        deep = (deep,)
        deep = {"n": deep}
    copied = copy_json_value(deep)
    assert isinstance(copied, dict)


def test_a_tuple_reached_twice_is_one_tuple_in_the_copy() -> None:
    """Sharing preservation has to survive the freeze, and the slots are why.

    Two keys pointing at one tuple still point at one tuple afterwards — a fresh
    one. A freeze that patched only the first slot it recorded would leave the
    second holding an unfrozen list, which is a *type* error a value-only
    assertion cannot see.
    """
    shared = ({"s": 1},)
    copied = copy_json_value({"a": shared, "b": shared})
    assert copied["a"] is copied["b"]
    assert copied["a"] is not shared
    assert isinstance(copied["b"], tuple)


# --------------------------------------------------------------------------
# The two container sets, derived from both functions rather than listed
# --------------------------------------------------------------------------

_BUILTIN_CONTAINERS = frozenset({"dict", "list", "tuple", "set", "frozenset"})


def _isinstance_container_kinds(module: str, function: str) -> frozenset[str]:
    """Every builtin container `function` tests for with `isinstance`.

    Derived from the AST rather than from a hand-written list, because a
    hand-written list is exactly how the one-kind difference between these two
    functions survived a review that quoted both of them.
    """
    tree = ast.parse((_PACKAGE / module).read_text(encoding="utf-8"))
    target = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function
    )
    kinds: set[str] = set()
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "isinstance" or len(node.args) != 2:
            continue
        probe = node.args[1]
        elements = probe.elts if isinstance(probe, (ast.Tuple, ast.List)) else [probe]
        for element in elements:
            text = ast.unparse(element)
            if text in _BUILTIN_CONTAINERS:
                kinds.add(text)
    return frozenset(kinds)


def test_the_copy_walks_at_least_every_kind_the_wire_seam_walks() -> None:
    """The invariant #229 exists for: a copy may not be narrower than what it feeds.

    `StreamEvent.__post_init__` copies the payload and `to_sse` serializes it, so
    the immutability claim on that record is void on exactly the difference
    between these two sets. At `ca0491b` the difference was `tuple`, and both
    functions' docstrings quoted each other while the difference sat between
    them.

    Derived from the source of both functions. A future `_new_container` that
    learned about another kind would trip this arm rather than opening the same
    gap again in the other direction.
    """
    copy_kinds = _isinstance_container_kinds("io_utils.py", "copy_json_value")
    wire_kinds = _isinstance_container_kinds("streaming.py", "_new_container")
    assert wire_kinds, "the walk found no isinstance container probe in _new_container"
    assert copy_kinds, "the walk found no isinstance container probe in copy_json_value"
    missing = sorted(wire_kinds - copy_kinds)
    assert not missing, (
        f"`_new_container` treats {missing} as containers and `copy_json_value` "
        f"does not, so a mutable object reachable through one of those kinds is "
        f"still the caller's when it reaches the wire (#229, D-023). "
        f"copy={sorted(copy_kinds)} wire={sorted(wire_kinds)}"
    )


def test_the_derivation_is_not_vacuous_and_names_the_kinds_it_found() -> None:
    """A pass over two empty sets is not a pass.

    Pins the derived sets by value as well, so a walk that silently stopped
    finding `isinstance` calls (a rename, a refactor to a match statement) fails
    loudly here instead of making the superset check trivially true. The
    asymmetry that remains is deliberate and the other direction: the copy knows
    about `set` and `frozenset`, which the wire seam hands to `default=`.
    """
    assert _isinstance_container_kinds("io_utils.py", "copy_json_value") == frozenset(
        {"dict", "list", "tuple", "set"}
    )
    assert _isinstance_container_kinds("streaming.py", "_new_container") == frozenset(
        {"dict", "list", "tuple"}
    )


# --------------------------------------------------------------------------
# The population
# --------------------------------------------------------------------------

_CONTAINER_HINTS = ("dict", "list", "set", "Mapping", "Sequence", "MutableMapping")


def _dataclasses_in_package() -> list[tuple[str, str, bool, list[tuple[str, str]]]]:
    """`(module, class, frozen, [(field, annotation)])` for every dataclass."""
    out = []
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            decorators = [ast.unparse(d) for d in node.decorator_list]
            if not any("dataclass" in d for d in decorators):
                continue
            fields = [
                (stmt.target.id, ast.unparse(stmt.annotation))
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and stmt.annotation is not None
                and isinstance(stmt.target, ast.Name)
            ]
            out.append((path.name, node.name, any("frozen=True" in d for d in decorators), fields))
    return out


def _is_container(annotation: str) -> bool:
    head = annotation.split("[", 1)[0].split(".")[-1]
    return head in _CONTAINER_HINTS


def test_every_frozen_record_with_a_container_field_copies_it() -> None:
    """Discover the population; do not trust the five the sweep listed.

    Keyed on what the defect needs — `frozen=True` (the claim) plus a
    container-typed field (the thing the claim does not cover) — so a sixth
    record added to this package is held to the rule without anyone updating a
    list. `tuple` is not a container hint: it is immutable, which is why
    `GeneratedAnswer.citations` and `RewriteResult.sub_queries` are not here.
    """
    offenders = []
    for module, cls, frozen, fields in _dataclasses_in_package():
        if not frozen:
            continue
        container_fields = [name for name, ann in fields if _is_container(ann)]
        if not container_fields:
            continue
        source = (_PACKAGE / module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        body = next(n.body for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
        post_init = "".join(
            ast.unparse(n)
            for n in body
            if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
        )
        for name in container_fields:
            if f"object.__setattr__(self, {name!r}" not in post_init:
                offenders.append(f"{module}:{cls}.{name}")
    assert not offenders, (
        f"these frozen records hold a container field they do not copy: "
        f"{offenders}. `frozen=True` stops a rebind and nothing else, so the "
        f"caller's object stays editable in place (#227)."
    )


def test_the_population_arm_found_the_five_rows() -> None:
    """A pass over an empty set is not a pass.

    Without this, a typo in the walk (matching no `ClassDef`, or a container
    hint that never fires) makes the arm above green over nothing.
    """
    found = {
        f"{cls}.{name}"
        for _, cls, frozen, fields in _dataclasses_in_package()
        if frozen
        for name, ann in fields
        if _is_container(ann)
    }
    assert found == {
        "Citation.metadata",
        "Candidate.metadata",
        "ScoredCandidate.metadata",
        "StreamEvent.payload",
        "CostRecord.per_phase_ms",
    }, f"the walk found {sorted(found)}; #227 triaged exactly five frozen rows"


def test_the_non_frozen_rows_are_cleared_by_name() -> None:
    """Seven container fields on four non-frozen dataclasses, recorded as a result.

    They make no immutability claim, so aliasing there is not this defect. Named
    here so the next `portfolio-ops#71`-style sweep reads a decision instead of
    re-deriving one — and so that freezing any of them later trips the arm
    above rather than passing quietly.
    """
    non_frozen = {
        f"{cls}.{name}"
        for _, cls, frozen, fields in _dataclasses_in_package()
        if not frozen
        for name, ann in fields
        if _is_container(ann)
    }
    assert non_frozen == {
        "Document.metadata",
        "RetrievalResult.metadata",
        "RetrievalResult.ranks",
        "PhaseTimings.retrieving",
        "PhaseTimings.reranking",
        "PhaseTimings.generating",
        "PhaseTimings.total",
    }, (
        f"the non-frozen container fields are now {sorted(non_frozen)}. If one of "
        f"these became `frozen=True`, it needs a copy and D-022's reasoning; if "
        f"one was added, decide it rather than widening this literal."
    )
