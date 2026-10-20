"""Retrieval is a function of the corpus, not of physical row order (#180).

Both channels of `Retriever._hybrid_search` ran `ORDER BY <non-unique key>
LIMIT n`. SQL leaves the choice among tied rows undefined, and Postgres settles
it by physical row order — so *which documents came back* changed whenever rows
moved on disk, which an ordinary `UPDATE`, `VACUUM` or reindex does.

Measured on Postgres 16 with six documents sharing one term profile, all at
`ts_rank_cd` 0.1, `LIMIT 3`:

    initial insert     -> doc-alpha,  doc-bravo,   doc-charlie
    one row rewritten  -> doc-bravo,  doc-charlie, doc-delta
    two more rewritten -> doc-delta,  doc-echo,    doc-foxtrot

Complete membership turnover — the first and third sets share no document.

`#40` gave RRF a doc-id tiebreak on a *different* axis: independence from the
order methods are supplied in. It cannot absorb an unstable input ranking,
because a different input rank changes the fused score itself. The last class
here pins that both properties now hold.

The `pg` marked test exercises the real database; the rest drive the same
orderings through a fake connection so the fusion consequence is covered on
every CI runner, including the ones with no Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_kit import HashEmbedder, Retriever
from rag_kit.fusion import reciprocal_rank_fusion

# The three orderings Postgres actually returned for one unchanged corpus.
LEX_ORDERINGS = [
    ["doc-alpha", "doc-bravo", "doc-charlie"],
    ["doc-bravo", "doc-charlie", "doc-delta"],
    ["doc-delta", "doc-echo", "doc-foxtrot"],
]
ALL_DOCS = ["doc-alpha", "doc-bravo", "doc-charlie", "doc-delta", "doc-echo", "doc-foxtrot"]


# ----------------------------------------------------------------------
# Fake connection that can return TIED distances, which the shared
# `_FakeConn` in test_retriever_rewriter.py deliberately cannot.
# ----------------------------------------------------------------------


@dataclass
class _TieConn:
    """Serves whatever row order it is handed, so a test can permute it.

    `dense_rows` carries an explicit distance per row, so a test can make the
    dense channel tie exactly rather than relying on an incidental ordering.
    """

    lexical_rows: list[tuple[str, str, dict[str, Any]]]
    dense_rows: list[tuple[str, str, dict[str, Any], float]]
    query_log: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def cursor(self) -> _TieCursor:
        return _TieCursor(self)


class _TieCursor:
    def __init__(self, conn: _TieConn) -> None:
        self._conn = conn
        self._buf: list[tuple[Any, ...]] = []

    def __enter__(self) -> _TieCursor:
        return self

    def __exit__(self, *exc) -> None:
        self._buf = []

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        upper = sql.upper()
        self._conn.query_log.append((sql, params))
        if "TS_RANK_CD" in upper:
            self._buf = list(self._conn.lexical_rows[: params[2]])
        elif "EMBEDDING <=>" in upper:
            self._buf = list(self._conn.dense_rows[: params[2]])
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected SQL: {sql}")

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._buf)


def _rows(ids: list[str]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(i, f"text for {i}", {}) for i in ids]


def _dense(ids: list[str], dists: list[float]) -> list[tuple[str, str, dict[str, Any], float]]:
    return [(i, f"text for {i}", {}, d) for i, d in zip(ids, dists, strict=True)]


class TestTheSqlCarriesATiebreak:
    """The lexical fix is in the SQL, so assert on the SQL the retriever sends."""

    def test_lexical_order_by_names_external_id(self) -> None:
        conn = _TieConn(_rows(LEX_ORDERINGS[0]), _dense(ALL_DOCS[:3], [0.1, 0.2, 0.3]))
        Retriever(conn, HashEmbedder()).search("refund policy", k=3)
        lexical_sql = next(s for s, _p in conn.query_log if "TS_RANK_CD" in s.upper())
        normalized = " ".join(lexical_sql.split()).upper()
        assert "ORDER BY TS_RANK_CD" in normalized
        assert "EXTERNAL_ID" in normalized[normalized.index("ORDER BY") :], (
            "the lexical ORDER BY must carry an external_id tiebreak; without it "
            "Postgres settles ties by physical row order"
        )

    def test_lexical_tiebreak_compares_code_points(self) -> None:
        """A bare `external_id ASC` sorts in the database's collation (#370).

        The dense and RRF tiebreaks sort in Python, which is code-point order,
        so the lexical one must say `COLLATE "C"` or the three disagree about
        what "ascending id" means on any non-C cluster.
        """
        conn = _TieConn(_rows(LEX_ORDERINGS[0]), _dense(ALL_DOCS[:3], [0.1, 0.2, 0.3]))
        Retriever(conn, HashEmbedder()).search("refund policy", k=3)
        lexical_sql = next(s for s, _p in conn.query_log if "TS_RANK_CD" in s.upper())
        normalized = " ".join(lexical_sql.split()).upper()
        order_by = normalized[normalized.index("ORDER BY") :]
        assert 'EXTERNAL_ID COLLATE "C" ASC' in order_by, (
            "the lexical tiebreak must compare code points like the dense and RRF "
            f"tiebreaks do; got {order_by!r}"
        )

    def test_dense_order_by_is_left_index_friendly(self) -> None:
        """The dense channel deliberately does NOT tiebreak in SQL.

        Appending `, external_id` there is what stops pgvector using an HNSW
        index-ordered scan. Determinism is imposed on the returned rows
        instead. Pinning this keeps the asymmetry deliberate.
        """
        conn = _TieConn(_rows(LEX_ORDERINGS[0]), _dense(ALL_DOCS[:3], [0.1, 0.2, 0.3]))
        Retriever(conn, HashEmbedder()).search("refund policy", k=3)
        dense_sql = next(s for s, _p in conn.query_log if "EMBEDDING <=>" in s.upper())
        normalized = " ".join(dense_sql.split()).upper()
        order_by = normalized[normalized.index("ORDER BY") :]
        assert "EXTERNAL_ID" not in order_by, (
            "the dense ORDER BY must stay a bare distance ordering so pgvector "
            "can use an index-ordered scan"
        )
        assert "AS DIST" in normalized, "the dense SELECT must expose the distance to sort on"


class TestDenseRankingIsRowOrderIndependent:
    def test_tied_distances_rank_by_external_id(self) -> None:
        ids = ["doc-echo", "doc-alpha", "doc-charlie"]
        results = set()
        for rotation in range(len(ids)):
            rotated = ids[rotation:] + ids[:rotation]
            conn = _TieConn(_rows([]), _dense(rotated, [0.25, 0.25, 0.25]))
            out = Retriever(conn, HashEmbedder()).search("refund policy", k=3)
            results.add(tuple(r.external_id for r in out))
        assert len(results) == 1, (
            f"dense ranking varied across row arrival order: {sorted(results)}"
        )
        assert next(iter(results)) == ("doc-alpha", "doc-charlie", "doc-echo")

    def test_distance_still_outranks_the_tiebreak(self) -> None:
        """Guards against sorting by id first.

        The nearest document has the lexicographically LAST id, so an id-first
        sort would rank it worst.
        """
        conn = _TieConn(
            _rows([]),
            _dense(["doc-zulu", "doc-alpha", "doc-bravo"], [0.01, 0.90, 0.95]),
        )
        out = Retriever(conn, HashEmbedder()).search("refund policy", k=3)
        assert [r.external_id for r in out][0] == "doc-zulu"

    def test_rows_the_database_separated_keep_their_order(self) -> None:
        # Python's sort is stable and distance leads the key, so a strictly
        # ordered dense result is passed through untouched.
        conn = _TieConn(
            _rows([]),
            _dense(["doc-charlie", "doc-bravo", "doc-alpha"], [0.1, 0.2, 0.3]),
        )
        out = Retriever(conn, HashEmbedder()).search("refund policy", k=3)
        assert [r.external_id for r in out] == ["doc-charlie", "doc-bravo", "doc-alpha"]


class TestFusionConsequence:
    """The measured propagation: an unstable input ranking survives RRF."""

    DENSE_FIXED = ["doc-charlie", "doc-alpha", "doc-echo"]

    def test_the_three_measured_orderings_fuse_three_different_ways(self) -> None:
        """Records WHY the SQL tiebreak is needed rather than an RRF change.

        This is the pre-fix behaviour and it is still true — RRF is working as
        designed. It is the *input* that must be stable, which is what the
        lexical ORDER BY now guarantees.
        """
        tops = {
            tuple(
                d
                for d, _s, _r in reciprocal_rank_fusion({"lexical": lex, "dense": self.DENSE_FIXED})
            )[:3]
            for lex in LEX_ORDERINGS
        }
        assert len(tops) == 3, (
            "the three orderings Postgres returned must still fuse three ways — "
            "if this collapses, the fusion layer changed and the SQL tiebreak's "
            "rationale needs rechecking"
        )

    def test_a_stable_input_fuses_one_way(self) -> None:
        stable = sorted(LEX_ORDERINGS[0])
        tops = {
            tuple(
                d
                for d, _s, _r in reciprocal_rank_fusion(
                    {"lexical": stable, "dense": self.DENSE_FIXED}
                )
            )
            for _ in range(5)
        }
        assert len(tops) == 1

    def test_issue_40s_property_still_holds(self) -> None:
        """Method-order independence is additive to this change, not replaced."""
        a = reciprocal_rank_fusion({"lexical": LEX_ORDERINGS[0], "dense": self.DENSE_FIXED})
        b = reciprocal_rank_fusion({"dense": self.DENSE_FIXED, "lexical": LEX_ORDERINGS[0]})
        assert a == b


@pytest.mark.pg
def test_lexical_membership_survives_row_rewrites(pg_conn) -> None:
    """The measured defect, against the real database.

    Six documents with one term profile tie at ts_rank_cd. Pre-fix, a no-op
    `UPDATE ... SET text = text` rotated the LIMIT-3 result set until it shared
    no document with the original.
    """
    # Seed through the repo's own Indexer rather than raw INSERTs: `tsv` is a
    # plain column populated by a database trigger, and going through the
    # supported write path is what the rest of the pg suite does.
    from rag_kit import Document, Indexer

    indexer = Indexer(pg_conn, HashEmbedder())
    indexer.clear()
    indexer.add_documents(
        Document(ext, "refund policy details for the account") for ext in ALL_DOCS
    )
    pg_conn.commit()

    sql = """
    SELECT external_id FROM documents
    WHERE tsv @@ plainto_tsquery('english', %s)
    ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC, external_id COLLATE "C" ASC
    LIMIT 3
    """

    def top3() -> list[str]:
        with pg_conn.cursor() as cur:
            cur.execute(sql, ("refund policy", "refund policy"))
            return [r[0] for r in cur.fetchall()]

    first = top3()
    assert len(first) == 3, "fixture must actually tie three rows into the limit"

    for victims in (("doc-alpha",), ("doc-bravo", "doc-charlie"), ("doc-foxtrot",)):
        with pg_conn.cursor() as cur:
            for ext in victims:
                cur.execute("UPDATE documents SET text = text WHERE external_id = %s", (ext,))
        pg_conn.commit()
        assert top3() == first, (
            "the retrieved set moved after a no-op UPDATE rewrote rows — this is "
            "the #180 defect, and it means the ORDER BY lost its tiebreak"
        )

    assert first == ["doc-alpha", "doc-bravo", "doc-charlie"]


# ----------------------------------------------------------------------
# #370: the lexical tiebreak sorted in the database's collation.
# ----------------------------------------------------------------------

# Ids that differ only in case and `-`/`_`, which is where a linguistic
# collation and code-point order part ways. Measured on PostgreSQL 17.6:
#
#   C                  DOC-G, Doc-B, Doc-D, doc-a, doc-c, doc-f, doc-h, doc_e
#   en_US (libc, ICU)  doc_e, doc-a, Doc-B, doc-c, Doc-D, doc-f, DOC-G, doc-h
#
# CI's `pgvector/pgvector:pg16` service initialises en_US.utf8, so the pre-fix
# SQL returned the second ordering there. Against a C database these tests
# cannot tell fixed from unfixed; they still pin the contract.
MIXED_IDS = ["doc-a", "Doc-B", "doc-c", "Doc-D", "doc_e", "doc-f", "DOC-G", "doc-h"]


def _seed_tied(conn, ids: list[str]) -> None:
    from rag_kit import Document, Indexer

    indexer = Indexer(conn, HashEmbedder())
    indexer.clear()
    # One text for every row: ts_rank_cd ties exactly, and so does the dense
    # distance, which makes the dense channel's (code-point) tiebreak decide
    # its order too.
    indexer.add_documents(Document(ext, "Refund policy boilerplate section.") for ext in ids)
    conn.commit()


def _lexical_ranking(results) -> list[str]:
    return [r.external_id for r in sorted(results, key=lambda r: r.ranks["lexical"])]


@pytest.mark.pg
def test_lexical_ties_rank_by_code_point(pg_conn) -> None:
    """The lexical channel's ranking among tied rows is `sorted(ids)` (#370)."""
    _seed_tied(pg_conn, MIXED_IDS)
    out = Retriever(pg_conn, HashEmbedder()).search("refund policy", k=len(MIXED_IDS))
    assert len(out) == len(MIXED_IDS)
    assert _lexical_ranking(out) == sorted(MIXED_IDS), (
        "the lexical tiebreak sorted in the database collation, not by code point"
    )


@pytest.mark.pg
def test_lexical_and_dense_agree_on_tied_rows(pg_conn) -> None:
    """With both channels tied, fusion must return the ids in code-point order.

    Pre-fix, on en_US the lexical channel said `doc_e` first and the dense
    channel `DOC-G`, so the fused top-2 was decided by the locale.
    """
    _seed_tied(pg_conn, MIXED_IDS)
    out = Retriever(pg_conn, HashEmbedder()).search("refund policy", k=2)
    assert [r.external_id for r in out] == sorted(MIXED_IDS)[:2]
    for r in out:
        assert r.ranks["lexical"] == r.ranks["dense"], r


@pytest.mark.pg
def test_ids_every_collation_agrees_on_still_rank_ascending(pg_conn) -> None:
    """Control: lowercase-alphabetic ids sort the same under C and en_US.

    Green before and after #370 -- it separates "the tiebreak is gone" from
    "the tiebreak uses the wrong collation".
    """
    _seed_tied(pg_conn, ALL_DOCS)
    out = Retriever(pg_conn, HashEmbedder()).search("refund policy", k=len(ALL_DOCS))
    assert _lexical_ranking(out) == ALL_DOCS
