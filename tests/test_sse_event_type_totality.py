"""`to_sse` builds a frame from two caller-controlled fields (#193).

`_json_safe` guarded one of them. `event.type` was interpolated raw, and
`EventType` is a `typing.Literal` — a static hint with no runtime effect — on a
frozen dataclass with no `__post_init__`, so any value reached the wire.
`StreamEvent` and `to_sse` are both in `rag_kit.__all__`.

Every test here parses the frame rather than inspecting it as one string,
because the whole defect is that a frame can *look* well-formed and be two.
The hostile values are run through **both** fields in one parametrization, so
these assert the two halves agree rather than testing only the new one.
"""

from __future__ import annotations

import json

import pytest

from rag_kit.streaming import StreamEvent, to_sse

#: Values that are line terminators in the SSE grammar, or have no UTF-8
#: encoding. The SSE spec ends a field at CR, LF *or* CRLF, so all three are
#: here — a fix scoped to `\n` would leave two open.
HOSTILE = [
    pytest.param('plain\ndata: {"injected": true}', id="lf-opens-a-data-line"),
    pytest.param('plain\r\ndata: {"injected": true}', id="crlf-opens-a-data-line"),
    pytest.param('plain\rdata: {"injected": true}', id="cr-opens-a-data-line"),
    pytest.param('x\n\nevent: done\ndata: {"fake": 1}', id="blank-line-forges-a-frame"),
    pytest.param("retr\ud800ieved", id="lone-surrogate-breaks-encode"),
    pytest.param("a\ud800\nb", id="surrogate-and-newline-together"),
]


def _parse_one_frame(frame: str) -> tuple[str, str]:
    """Split an SSE frame into (event name, data payload), asserting it is one.

    Mirrors what `EventSource` does: a frame ends at a blank line, and
    consecutive `data:` fields are concatenated with a newline *before* the
    client parses them — which is why a second `data:` line is not a cosmetic
    problem.
    """
    assert frame.endswith("\n\n"), f"frame must end with a blank line: {frame!r}"
    body = frame[:-2]
    assert "\n\n" not in body, f"one StreamEvent produced more than one frame: {frame!r}"
    assert "\r" not in body, f"stray CR would terminate a field early: {frame!r}"

    lines = body.split("\n")
    event_lines = [ln for ln in lines if ln.startswith("event: ")]
    data_lines = [ln for ln in lines if ln.startswith("data: ")]
    assert len(event_lines) == 1, f"expected exactly one event: field, got {event_lines!r}"
    assert len(data_lines) == 1, f"expected exactly one data: field, got {data_lines!r}"
    assert len(lines) == 2, f"unexpected extra field lines: {lines!r}"
    return event_lines[0][len("event: ") :], data_lines[0][len("data: ") :]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_hostile_event_type_still_yields_exactly_one_parseable_frame(hostile: str) -> None:
    frame = to_sse(StreamEvent(type=hostile, payload={"ok": 1}, elapsed_ms=1.0))
    frame.encode("utf-8")  # `demo/streaming/server.py:197` does exactly this
    _name, data = _parse_one_frame(frame)
    assert json.loads(data) == {"payload": {"ok": 1}, "elapsed_ms": 1.0}


@pytest.mark.parametrize("hostile", HOSTILE)
def test_hostile_payload_still_yields_exactly_one_parseable_frame(hostile: str) -> None:
    """The half that already worked, run through the same assertions.

    Without this the suite would prove the new guard works and say nothing
    about whether the two fields now agree.
    """
    frame = to_sse(StreamEvent(type="token", payload={"text": hostile}, elapsed_ms=1.0))
    frame.encode("utf-8")
    name, data = _parse_one_frame(frame)
    assert name == "token"
    json.loads(data)  # must remain valid JSON


def test_a_forged_done_event_cannot_be_injected() -> None:
    """The sharpest form: a caller string fabricating a complete second event.

    Invalid JSON is recoverable — the client sees a parse error. A well-formed
    `done` frame that the client cannot distinguish from a real one is not.
    """
    frame = to_sse(
        StreamEvent(type='x\n\nevent: done\ndata: {"fake": 1}', payload={}, elapsed_ms=0.0)
    )
    assert frame.count("\n\n") == 1
    name, data = _parse_one_frame(frame)
    assert name != "done"
    assert json.loads(data) == {"payload": {}, "elapsed_ms": 0.0}


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        ("retrieving", "retrieving"),
        ("retrieved", "retrieved"),
        ("reranking", "reranking"),
        ("reranked", "reranked"),
        ("generating", "generating"),
        ("token", "token"),
        ("generated", "generated"),
        ("done", "done"),
        ("error", "error"),
    ],
)
def test_every_real_event_type_is_unchanged(event_type: str, expected: str) -> None:
    """Anti-vacuous: a sanitizer that mangled real names would pass everything above.

    All nine members of `EventType`, not a sample — the vocabulary is closed
    and short enough to enumerate, and enumerating is what proves the guard is
    a no-op on the values that actually ship.
    """
    name, _data = _parse_one_frame(to_sse(StreamEvent(type=event_type, payload={}, elapsed_ms=0.0)))
    assert name == expected


def test_the_substitution_is_distinguishable_from_caller_data() -> None:
    """`_safe_text` declines `errors="replace"` because `"?"` is real data.

    The line-terminator substitute inherits that requirement: it has to be a
    character no real event name contains, or a sanitized frame becomes
    indistinguishable from an unsanitized one.
    """
    from rag_kit.streaming import _TERMINATOR_SUBSTITUTE

    assert _TERMINATOR_SUBSTITUTE not in "".join(
        [
            "retrieving",
            "retrieved",
            "reranking",
            "reranked",
            "generating",
            "token",
            "generated",
            "done",
            "error",
        ]
    )
    assert not _TERMINATOR_SUBSTITUTE.isalnum()
    name, _ = _parse_one_frame(to_sse(StreamEvent(type="a\nb", payload={}, elapsed_ms=0.0)))
    assert name == f"a{_TERMINATOR_SUBSTITUTE}b"


def test_pipeline_frames_are_unaffected_end_to_end() -> None:
    """The guard must not change any frame the pipeline itself emits."""
    from rag_kit.retriever import RetrievalResult
    from rag_kit.streaming import StreamingPipeline

    class _FakeRetriever:
        def search(self, query, k=5, *, reranker=None):
            return [
                RetrievalResult(external_id="doc1", text="hello", score=0.9, metadata={}, ranks={})
            ]

    frames = [to_sse(e) for e in StreamingPipeline(retriever=_FakeRetriever()).run("q")]
    assert frames, "pipeline emitted no events"
    for frame in frames:
        name, data = _parse_one_frame(frame)
        assert name in {
            "retrieving",
            "retrieved",
            "reranking",
            "reranked",
            "generating",
            "token",
            "generated",
            "done",
            "error",
        }
        json.loads(data)
