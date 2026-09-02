"""The read side must refuse any two corpus ids a `[cite:...]` marker cannot
tell apart — whichever grammar feature makes them indistinguishable (#197).

#182 identified **two** ways an `external_id` fails to be citable back, and its
own comment says so: *"Two distinct causes, deliberately kept as two rules with
two messages."* Surrounding whitespace is defeated by the reader's `.strip()`;
a `]` is defeated by `_CITE_PATTERN` (`\\[cite:([^\\]]+)\\]`), which stops at the
first one.

It closed both at the *write* seam. `enforce_citations` also carries a read-side
backstop, because it "is handed rows from the database, not `Document` objects"
and a corpus indexed before that guard still has such rows on disk — and that
backstop was written as its own rule, "ids equal after `.strip()`", so it
covered exactly one of the two causes named in the same breath. Measured on a
corpus holding `doc` and `doc]1`::

    whitespace collision  ->  refused
    ']' collision         ->  ACCEPTED, cited 'doc', rendered 'doc's text

A claim sourced from `doc]1` rendered `doc`'s text as its citation, and the
resulting `Citation` is perfectly well-formed so nothing downstream can tell.

The fix is not a second hand-written rule — two on one side and one on the
other is what produced the gap. `_marker_readback` *runs* the grammar over a
synthesised marker, so this file's job splits in two:

1. the collision refusal itself, over both causes and a table of ids that must
   stay legal;
2. a **parity** arm pinning `Document.__post_init__`'s hand-written rules
   against that derived readback, so a future grammar change cannot reopen this
   quietly. That arm is the one that would have caught #197 the day it was
   introduced.
"""

from __future__ import annotations

import re

import pytest

from rag_kit.generator import CitationError, _marker_readback, enforce_citations
from rag_kit.indexer import Document
from rag_kit.retriever import RetrievalResult


def _chunk(external_id: str) -> RetrievalResult:
    return RetrievalResult(
        external_id=external_id,
        text=f"body of {external_id}",
        metadata={},
        fused_score=1.0,
        ranks={},
    )


def _answer(external_id: str) -> str:
    """A one-sentence answer citing `external_id` exactly as a generator would."""
    return f"The sky is blue [cite:{external_id}]."


# (label, id_a, id_b, value they collide on). Both causes #182 names, in the
# same table, because the point of the fix is that the read side stops
# distinguishing between them.
COLLIDING_PAIRS = [
    ("whitespace: leading", "doc1", " doc1", "doc1"),
    ("whitespace: trailing", "doc1", "doc1 ", "doc1"),
    ("whitespace: tab", "doc1", "\tdoc1", "doc1"),
    ("whitespace: newline", "doc1", "doc1\n", "doc1"),
    ("bracket: truncation", "doc", "doc]1", "doc"),
    ("bracket: truncation, two suffixes", "doc", "doc]2", "doc"),
    ("both causes at once", "doc", " doc]1", "doc"),
]


@pytest.mark.parametrize(
    ("label", "id_a", "id_b", "collides_on"),
    COLLIDING_PAIRS,
    ids=[row[0] for row in COLLIDING_PAIRS],
)
def test_indistinguishable_ids_are_refused(
    label: str, id_a: str, id_b: str, collides_on: str
) -> None:
    """A refusal is recoverable; a citation pointing at the wrong chunk is not."""
    retrieved = [_chunk(id_a), _chunk(id_b)]
    with pytest.raises(CitationError) as exc:
        enforce_citations(_answer(id_b), retrieved)
    assert exc.value.reason == "unparseable_output"
    message = str(exc.value)
    # Both ids and the value they collide on, matching the shape of the message
    # this replaced: an operator has to be able to find the two rows.
    assert repr(id_a) in message, message
    assert repr(id_b) in message, message
    assert repr(collides_on) in message, message


@pytest.mark.parametrize(
    ("label", "id_a", "id_b", "collides_on"),
    COLLIDING_PAIRS,
    ids=[row[0] for row in COLLIDING_PAIRS],
)
def test_the_wrong_chunks_text_is_never_rendered(
    label: str, id_a: str, id_b: str, collides_on: str
) -> None:
    """The harm, stated as the harm rather than as the guard.

    `test_indistinguishable_ids_are_refused` asserts an exception; this asserts
    the thing the exception exists to prevent — that no `Citation` ever comes
    back carrying `id_a`'s body for a claim that cited `id_b`. Before #197 this
    returned `Citation(external_id='doc', text='body of doc')` for a claim whose
    source was `doc]1`.
    """
    retrieved = [_chunk(id_a), _chunk(id_b)]
    try:
        citations = enforce_citations(_answer(id_b), retrieved)
    except CitationError:
        return  # refused, which is the whole point
    for c in citations:
        assert c.external_id != id_a or id_a == id_b, (
            f"claim cited {id_b!r} but got back {c.external_id!r} with text {c.text!r}"
        )


# Ids that must keep resolving. `external_id` is documented as caller-supplied
# (filename + chunk index, hash), so over-tightening this check is a real cost:
# it would refuse legitimate corpora to fix a problem they do not have.
LEGAL_IDS = ["doc1", "doc 1", "doc[1", "doc\n1", "doc.1", "docé1", "a/b/c.md#3", "a[cite:b"]


@pytest.mark.parametrize("external_id", LEGAL_IDS)
def test_ordinary_ids_still_resolve(external_id: str) -> None:
    citations = enforce_citations(_answer(external_id), [_chunk(external_id)])
    assert [c.external_id for c in citations] == [external_id]


def test_a_whole_corpus_of_legal_ids_is_not_a_collision() -> None:
    """All of them retrieved together — the check must not fire on a normal corpus."""
    retrieved = [_chunk(i) for i in LEGAL_IDS]
    citations = enforce_citations(_answer(LEGAL_IDS[0]), retrieved)
    assert [c.external_id for c in citations] == [LEGAL_IDS[0]]


def test_the_same_id_retrieved_twice_is_not_a_collision() -> None:
    """Duplicate rows are one document, not two indistinguishable ones.

    The check compares *distinct* ids; a retriever that returns the same chunk
    twice (a fusion tie, a caller concatenating two result sets) must not be
    turned into a refusal.
    """
    retrieved = [_chunk("doc1"), _chunk("doc1")]
    assert [c.external_id for c in enforce_citations(_answer("doc1"), retrieved)] == ["doc1"]


# --- the parity arm: the write seam and the grammar must agree ---------------


def test_every_id_the_write_seam_accepts_reads_back_unchanged() -> None:
    """`Document.__post_init__`'s two rules must match what the grammar does.

    This is the arm that makes the hand-written write-seam messages safe to
    keep. Those messages are good operator UX — they name *which* rule was
    broken — but a rule stated twice can be stated inconsistently, which is
    exactly how #197 happened one seam over. Here the write seam is checked
    against `_marker_readback`, which runs the real grammar: any id the seam
    lets through must survive a marker round-trip identically.

    The candidate set deliberately includes the shapes the write seam is
    *supposed* to reject, so this cannot pass by testing only clean ids.
    """
    candidates = [
        *LEGAL_IDS,
        " doc1",
        "doc1 ",
        "doc1\n",
        "\tdoc1",
        "   ",
        "doc]1",
        "]",
        "doc]",
        "]doc",
        "",
        "\u00a0doc1",  # NBSP, written as an escape so an editor or
        # formatter cannot silently normalise it away: it is whitespace to
        # `str.strip`, so the write seam must reject it too.
    ]
    accepted, rejected = [], []
    for external_id in candidates:
        try:
            Document(external_id=external_id, text="body")
        except ValueError:
            rejected.append(external_id)
        else:
            accepted.append(external_id)

    # Anti-vacuous: the candidate table has to exercise both verdicts, or the
    # loop below proves nothing.
    assert len(accepted) >= 5, accepted
    assert len(rejected) >= 5, rejected

    for external_id in accepted:
        assert _marker_readback(external_id) == external_id, (
            f"Document accepts {external_id!r} but a [cite:...] marker reads it back as "
            f"{_marker_readback(external_id)!r} -- the write seam's rules and the marker "
            "grammar have drifted apart (#197)"
        )


def test_the_readback_helper_uses_the_real_grammar() -> None:
    """Anti-vacuous: `_marker_readback` must not be a re-implementation.

    If someone replaces the helper's body with, say, `external_id.strip()`, every
    whitespace test above keeps passing and the `]` half silently reopens. So
    assert the helper tracks `_CITE_PATTERN` itself: patch the pattern to a
    grammar that terminates on `#` instead of `]`, and the helper's verdict must
    move with it.
    """
    import rag_kit.generator as generator

    original = generator._CITE_PATTERN
    assert _marker_readback("a#b") == "a#b"  # under the real grammar, `#` is ordinary
    try:
        generator._CITE_PATTERN = re.compile(r"\[cite:([^#]+)#")
        assert _marker_readback("a#b") == "a", (
            "_marker_readback did not follow _CITE_PATTERN -- it is restating the "
            "grammar rather than running it, which is the defect #197 fixed"
        )
    finally:
        generator._CITE_PATTERN = original
    assert generator._CITE_PATTERN is original
    assert _marker_readback("a#b") == "a#b"


def test_an_id_no_marker_can_name_is_not_treated_as_a_collision() -> None:
    """`_marker_readback` returns `None` when the grammar does not match at all.

    An empty id is the only such shape today (`[^\\]]+` needs one character).
    Two of them are not "indistinguishable ids" in the sense this check means —
    a citation aimed at either reads as dangling and the answer is refused,
    which is the safe direction. The check skips them rather than manufacturing
    a collision refusal with a misleading message.
    """
    assert _marker_readback("") is None
    retrieved = [_chunk(""), _chunk(""), _chunk("doc1")]
    citations = enforce_citations(_answer("doc1"), retrieved)
    assert [c.external_id for c in citations] == ["doc1"]
