"""Hermetic tests for evals/run_eval.py.

Exercises the in-memory retriever, the three suite scorers, and the JSON
write-and-read roundtrip. The composite comment poster is exercised in
``--dry-run`` mode (no token) and via a fake ``urlopen``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import run_eval
from rag_kit import GeneratedAnswer, Refusal


def test_tokens_lowercases_and_strips_punctuation():
    assert run_eval._tokens("Hello, world! 123") == ["hello", "world", "123"]
    assert run_eval._tokens("") == []


def test_retrieve_in_memory_ranks_overlapping_chunks_first():
    corpus = [
        run_eval._Chunk(external_id="match", text="refund window 14 days"),
        run_eval._Chunk(external_id="noise", text="completely unrelated content"),
    ]
    results = run_eval._retrieve_in_memory("refund window", corpus, k=2)
    assert results[0].external_id == "match"
    assert results[0].ranks == {"lex": 1, "dense": 1}
    assert results[1].external_id == "noise"


def test_retrieve_in_memory_respects_k():
    corpus = [run_eval._Chunk(external_id=f"d{i}", text=f"chunk {i}") for i in range(10)]
    out = run_eval._retrieve_in_memory("chunk", corpus, k=3)
    assert len(out) == 3


def test_score_faithfulness_pass_and_refusal():
    chunk = run_eval._retrieve_in_memory(
        "refund window", [run_eval._Chunk("doc-x", "refund window 14 days")], k=1
    )
    answer = GeneratedAnswer(
        text="Refund window 14 days [cite:doc-x].",
        citations=(),
        used_threshold=0.0,
        top_score=1.0,
    )
    score, _ = run_eval._score_faithfulness(answer, chunk)
    assert score == 1.0

    refusal = Refusal(
        reason="insufficient_context", detail="weak", used_threshold=0.0, top_score=0.0
    )
    s2, _ = run_eval._score_faithfulness(refusal, chunk)
    assert s2 == 0.0


def test_score_recall_at_5_partial_credit():
    retrieved = [
        run_eval._retrieve_in_memory("q", [run_eval._Chunk(f"d{i}", "x")], k=1)[0] for i in range(5)
    ]
    # Replace external ids manually so we can assert exact membership.
    from rag_kit.retriever import RetrievalResult

    retrieved = [
        RetrievalResult(
            external_id=f"d{i}",
            text="t",
            metadata={},
            fused_score=0.0,
            ranks={"lex": i + 1, "dense": i + 1},
        )
        for i in range(5)
    ]
    score, _ = run_eval._score_recall_at_5(retrieved, gold_ids=("d0", "d3"))
    assert score == 1.0
    score2, _ = run_eval._score_recall_at_5(retrieved, gold_ids=("d0", "d99"))
    assert score2 == pytest.approx(0.5)
    score3, _ = run_eval._score_recall_at_5(retrieved, gold_ids=())
    assert score3 == 0.0


def test_score_correctness_uses_content_tokens():
    answer = GeneratedAnswer(
        text="Pro customers get a 14 day refund window [cite:x].",
        citations=(),
        used_threshold=0.0,
        top_score=1.0,
    )
    score, _ = run_eval._score_correctness(
        answer, expected="Pro customers get a 14-day refund window."
    )
    assert score == 1.0


def test_run_all_suites_produces_expected_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "REPO_ROOT", run_eval.REPO_ROOT)  # unchanged but explicit
    runs = run_eval.run_all_suites()
    suites = [r.suite for r in runs]
    assert suites == ["faithfulness", "recall_at_5", "correctness"]
    for r in runs:
        assert r.rows  # at least one example
        result = r.to_run_result(dataset_version="rag-qa-v0.1")
        assert result["suite"] == r.suite
        assert result["n_rows"] == len(r.rows)
        assert 0.0 <= result["mean_score"] <= 1.0
        assert isinstance(result["rows"], list)


def test_write_runs_writes_one_json_per_suite(tmp_path):
    runs = run_eval.run_all_suites()
    out_dir = tmp_path / "current"
    paths = run_eval.write_runs(runs, out_dir, dataset_version="rag-qa-v0.1")
    assert set(paths.keys()) == {"faithfulness", "recall_at_5", "correctness"}
    for suite, p in paths.items():
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["suite"] == suite


def test_post_composite_comment_dry_run_prints_body(capsys):
    deltas = {"faithfulness": "## f", "recall_at_5": "## r", "correctness": "## c"}
    run_eval._post_composite_comment("owner/repo", 42, deltas, token=None)
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "rag-production-kit:eval-sticky" in out
    assert "## f" in out
    assert "## r" in out
    assert "## c" in out


def test_post_composite_comment_creates_when_no_existing(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class _ListResp:
        def __init__(self) -> None:
            self.status = 200

        def __enter__(self) -> _ListResp:
            return self

        def __exit__(self, *a) -> None:  # noqa: ANN401
            pass

        def read(self) -> bytes:
            return b"[]"

    class _CreateResp:
        def __init__(self) -> None:
            self.status = 201

        def __enter__(self) -> _CreateResp:
            return self

        def __exit__(self, *a) -> None:  # noqa: ANN401
            pass

        def read(self) -> bytes:
            return b'{"id": 99}'

    def _fake_urlopen(req):
        url = req.full_url
        method = req.get_method()
        body = req.data.decode() if req.data else None
        calls.append((f"{method} {url}", json.loads(body) if body else {}))
        if method == "GET":
            return _ListResp()
        return _CreateResp()

    monkeypatch.setattr(run_eval.urllib.request, "urlopen", _fake_urlopen)
    run_eval._post_composite_comment(
        "owner/repo",
        42,
        {"faithfulness": "## f", "recall_at_5": "## r", "correctness": "## c"},
        token="tkn",
    )
    methods = [c[0].split()[0] for c in calls]
    assert "GET" in methods
    assert "POST" in methods
    create_body = next(c[1] for c in calls if c[0].startswith("POST"))
    assert "rag-production-kit:eval-sticky" in create_body["body"]


def test_post_composite_comment_patches_existing(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class _ListResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return json.dumps(
                [
                    {"id": 7, "body": "older comment"},
                    {"id": 8, "body": "<!-- rag-production-kit:eval-sticky -->\nold"},
                ]
            ).encode()

    class _PatchResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return b'{"id": 8}'

    def _fake_urlopen(req):
        method = req.get_method()
        body = req.data.decode() if req.data else None
        calls.append((f"{method} {req.full_url}", json.loads(body) if body else {}))
        return _ListResp() if method == "GET" else _PatchResp()

    monkeypatch.setattr(run_eval.urllib.request, "urlopen", _fake_urlopen)
    run_eval._post_composite_comment(
        "owner/repo",
        42,
        {"faithfulness": "## f", "recall_at_5": "## r", "correctness": "## c"},
        token="tkn",
    )
    patch_call = next(c for c in calls if c[0].startswith("PATCH"))
    assert "/comments/8" in patch_call[0]


def test_main_writes_current_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path)
    rc = run_eval.main([])
    assert rc == 0
    assert (tmp_path / "faithfulness.json").exists()
    assert (tmp_path / "recall_at_5.json").exists()
    assert (tmp_path / "correctness.json").exists()


def test_committed_baselines_round_trip_to_valid_run_results():
    base = Path(__file__).resolve().parent.parent / "evals" / "baselines"
    for suite in ("faithfulness", "recall_at_5", "correctness"):
        data = json.loads((base / f"{suite}.json").read_text())
        assert data["suite"] == suite
        assert data["n_rows"] > 0
        assert 0.0 <= data["mean_score"] <= 1.0
        for row in data["rows"]:
            assert 0.0 <= row["score"] <= 1.0
