"""An external_id the indexer accepts must be one the citation reader can resolve.

`enforce_citations` strips a `[cite: ...]` marker before looking it up, because
the Anthropic generator routinely emits padded markers (#88). That leniency is
load-bearing and can't be removed. It is only *safe*, though, while the corpus
holds no id that is changed by stripping — an assumption the code asserted in a
comment and nothing enforced.

Before #182, `Document` validated `external_id` for non-emptiness alone. Six of
ten shapes it accepted could not round-trip, and two ids differing only by
padding resolved a citation to the wrong chunk — which for a kit whose stated
guarantee is "every claim cites a chunk" is worse than a refusal, because the
resulting `Citation` looks perfectly well-formed.

The tables below are the measured tables from the issue. The *accepted* half
matters as much as the rejected half: `external_id` is documented as
caller-supplied (filename + chunk index, hash), so a fix that hardened into a
charset allowlist would reject legitimate corpora.
"""

from __future__ import annotations

import pytest

from rag_kit.generator import CitationError, enforce_citations
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


# Ids the reader cannot resolve, paired with the message fragment naming the
# rule each one breaks. Asserting on the fragment keeps the two causes distinct:
# whitespace is defeated by the reader's `.strip()`, `]` by the marker grammar.
UNRESOLVABLE = [
    (" doc1", "leading or trailing whitespace"),
    ("doc1 ", "leading or trailing whitespace"),
    ("doc1\n", "leading or trailing whitespace"),
    ("\tdoc1", "leading or trailing whitespace"),
    ("   ", "leading or trailing whitespace"),
    ("doc]1", "must not contain"),
]

# Ids that round-trip correctly and must keep working. Internal spaces,
# brackets, newlines, dots and non-ASCII are all fine — only the leading/
# trailing and `]` cases are defeated by the marker grammar.
RESOLVABLE = ["doc1", "doc 1", "doc[1", "doc\n1", "doc.1", "docé1", "a/b/c.md#3"]


@pytest.mark.parametrize(("external_id", "expected_rule"), UNRESOLVABLE)
def test_indexer_rejects_ids_the_reader_cannot_resolve(external_id, expected_rule):
    with pytest.raises(ValueError, match=expected_rule):
        Document(external_id=external_id, text="some text")


@pytest.mark.parametrize(("external_id", "expected_rule"), UNRESOLVABLE)
def test_the_rejected_ids_really_were_unresolvable(external_id, expected_rule):  # noqa: ARG001
    """The other half of the pair: show the read path genuinely fails on these.

    Without this, the guard above could be rejecting ids for no reason. The
    chunk is built directly as a `RetrievalResult`, bypassing `Document`, which
    is also how a row indexed before the guard reaches this function.
    """
    with pytest.raises(CitationError) as exc:
        enforce_citations(f"A claim [cite:{external_id}].", [_chunk(external_id)])
    assert exc.value.reason == "unparseable_output"


@pytest.mark.parametrize("external_id", RESOLVABLE)
def test_valid_ids_still_index_and_still_cite(external_id):
    """The guard must not become a charset allowlist."""
    Document(external_id=external_id, text="some text")

    citations = enforce_citations(f"A claim [cite:{external_id}].", [_chunk(external_id)])
    assert [c.external_id for c in citations] == [external_id]


def test_whitespace_only_id_is_rejected_by_the_strip_rule_not_the_empty_rule():
    """`not '   '` is False, which is exactly how this slipped past before."""
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        Document(external_id="   ", text="t")


def test_empty_id_still_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        Document(external_id="", text="t")


def test_bracket_rule_names_the_marker_grammar():
    with pytest.raises(ValueError, match=r"must not contain"):
        Document(external_id="doc]1", text="t")


def test_non_string_id_is_rejected_before_strip_is_called():
    with pytest.raises(ValueError, match="must be a string"):
        Document(external_id=None, text="t")  # type: ignore[arg-type]


def test_text_validation_is_unchanged():
    with pytest.raises(ValueError, match="text must be non-empty"):
        Document(external_id="doc1", text="")


# ----------------------------------------------------------------------
# The false-accept the strip comment used to rule out
# ----------------------------------------------------------------------


COLLIDING_PAIRS = [
    ("doc1", " doc1"),
    ("doc1", "doc1 "),
    (" doc1", "doc1 "),
    ("doc1", "doc1\n"),
]


@pytest.mark.parametrize(("first", "second"), COLLIDING_PAIRS)
def test_stripped_collision_refuses_rather_than_citing_the_wrong_chunk(first, second):
    """Measured before the fix: `[cite: doc1]` against a corpus holding both
    'doc1' and ' doc1' resolved to 'doc1' and rendered *that* chunk's body as
    the source of a claim grounded in the other one.

    These rows bypass `Document` on purpose — the write guard cannot reach a
    corpus already on disk, which is precisely the population this check covers.
    """
    corpus = [_chunk(first), _chunk(second)]
    with pytest.raises(CitationError) as exc:
        enforce_citations(f"A claim [cite:{second}].", corpus)
    assert "collide after stripping" in exc.value.detail
    assert exc.value.reason == "unparseable_output"


def test_collision_check_reports_both_offending_ids():
    corpus = [_chunk("doc1"), _chunk(" doc1")]
    with pytest.raises(CitationError) as exc:
        enforce_citations("A claim [cite:doc1].", corpus)
    assert "'doc1'" in exc.value.detail
    assert "' doc1'" in exc.value.detail


def test_the_same_id_twice_is_not_a_collision():
    """Two rows carrying the identical id are a duplicate, not an ambiguity.

    The check must compare the *pre-strip* ids, or a retrieval that returned the
    same chunk twice would be misreported as a collision.
    """
    corpus = [_chunk("doc1"), _chunk("doc1")]
    citations = enforce_citations("A claim [cite:doc1].", corpus)
    assert [c.external_id for c in citations] == ["doc1"]


def test_ordinary_corpus_raises_nothing():
    corpus = [_chunk("a"), _chunk("b"), _chunk("c")]
    citations = enforce_citations("One [cite:a]. Two [cite: b].", corpus)
    assert [c.external_id for c in citations] == ["a", "b"]


def test_padded_marker_against_a_clean_corpus_still_resolves():
    """#88's leniency is preserved — that fix must not be undone by this one."""
    corpus = [_chunk("doc1")]
    for marker in ("[cite: doc1]", "[cite:doc1 ]", "[cite:  doc1  ]"):
        citations = enforce_citations(f"A claim {marker}.", corpus)
        assert [c.external_id for c in citations] == ["doc1"]
