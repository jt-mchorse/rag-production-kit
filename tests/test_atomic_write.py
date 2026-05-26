"""Atomicity contract for `evals/run_eval.py::write_runs` (#44).

`Path.write_text` is not atomic: SIGINT/SIGTERM/disk-full/OOM between
the implicit `open(..., "w")` truncate and `close()` flush leaves the
destination zero-length or partial. The eval action's composite
sticky comment (`_post_composite_comment`) parses the three per-suite
JSONs that `write_runs` emits, so a half-written suite file corrupts
the PR comment or fails the workflow with a cryptic JSONDecodeError.

The fix routes `write_runs` through `rag_kit.io_utils.atomic_write_text`,
which writes to a sibling tempfile in the destination's parent
directory, fsyncs, then `os.replace`s.

Helper shape mirrors the helpers landed earlier in this session in
`llm-eval-harness#48`, `llm-cost-optimizer#42`, and
`prompt-regression-suite#39` — portfolio-wide uniformity is intentional.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import run_eval
from rag_kit import io_utils as io_mod
from rag_kit.io_utils import atomic_write_text

# ---------------------------------------------------------------------------
# Unit tests on the helper itself.
# ---------------------------------------------------------------------------


def test_atomic_write_text_happy_path(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    atomic_write_text(out, "hello\nworld\n")
    assert out.read_text(encoding="utf-8") == "hello\nworld\n"


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "x.json"
    assert not out.parent.exists()
    atomic_write_text(out, "{}")
    assert out.read_text(encoding="utf-8") == "{}"


def test_atomic_write_text_overwrites_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    out.write_text("STALE-CONTENT-MUST-NOT-SURVIVE", encoding="utf-8")
    atomic_write_text(out, "fresh")
    assert out.read_text(encoding="utf-8") == "fresh"


def test_atomic_write_text_replace_failure_leaves_destination_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "result.json"

    def boom(*_args, **_kwargs):
        raise OSError("simulated mid-rename failure")

    monkeypatch.setattr(io_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated mid-rename failure"):
        atomic_write_text(out, '{"k": "v"}')
    assert not out.exists()


def test_atomic_write_text_replace_failure_cleans_up_tmp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "artifacts" / "delta.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def boom(*_args, **_kwargs):
        raise OSError("simulated mid-rename failure")

    monkeypatch.setattr(io_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated mid-rename failure"):
        atomic_write_text(out, '{"k": "v"}')
    siblings = list(out.parent.iterdir())
    assert siblings == [], f"expected no temp leftovers in {out.parent}, got {siblings}"


def test_atomic_write_text_destination_unchanged_when_overwriting_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "existing.json"
    out.write_text('{"keep": true}', encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(io_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(out, '{"overwrite": true}')
    assert out.read_text(encoding="utf-8") == '{"keep": true}'


# ---------------------------------------------------------------------------
# Integration tests on write_runs.
# ---------------------------------------------------------------------------


def _three_suite_runs() -> list[run_eval._SuiteRun]:
    rows_faith = [run_eval._SuiteRow("ex1", 1.0, "faithful")]
    rows_recall = [run_eval._SuiteRow("ex1", 1.0, "recall_ok")]
    rows_correct = [run_eval._SuiteRow("ex1", 0.5, "partial")]
    sha = "deadbeef"
    return [
        run_eval._SuiteRun("faithfulness", rows_faith, sha),
        run_eval._SuiteRun("recall_at_5", rows_recall, sha),
        run_eval._SuiteRun("correctness", rows_correct, sha),
    ]


def test_write_runs_happy_path_produces_three_valid_suite_jsons(tmp_path: Path) -> None:
    """`write_runs` emits three valid JSON files, one per suite,
    deserializable to the eval_harness RunResult shape."""
    runs = _three_suite_runs()
    paths = run_eval.write_runs(runs, tmp_path / "results", dataset_version="rag-qa-v0.1")

    assert set(paths) == {"faithfulness", "recall_at_5", "correctness"}
    for suite, path in paths.items():
        body = path.read_text(encoding="utf-8")
        parsed = json.loads(body)
        assert parsed["suite"] == suite
        assert parsed["dataset_version"] == "rag-qa-v0.1"
        assert parsed["n_rows"] == 1


def test_write_runs_second_write_failure_leaves_partial_dir_no_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomicity invariant under multi-file `write_runs`.

    `write_runs` writes three files sequentially. Simulate a failure
    on the second `os.replace` — proves:
    (a) the **first** file is fully written (helper actually invoked
        `os.replace` and replaced atomically),
    (b) the **second** file's destination does not exist (helper never
        touched the destination before the rename),
    (c) the third file isn't reached (we raise before getting there),
    (d) no temp leftovers in `out_dir` from the failed second write.

    The cross-file invariant — that `out_dir` can end up with fresh
    suite #1 next to stale suite #2 — is a separate harm class (per
    the issue body) and out of scope for this PR.
    """
    out_dir = tmp_path / "results"

    # Pre-seed the second file with a stale value so we can prove the
    # invariant about overwrite-failure-preserving-existing.
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_recall_path = out_dir / "recall_at_5.json"
    stale_recall_path.write_text("STALE-RECALL", encoding="utf-8")

    real_replace = io_mod.os.replace
    calls = {"n": 0}

    def selective_boom(src, dst, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated rename failure on second write")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(io_mod.os, "replace", selective_boom)

    runs = _three_suite_runs()
    with pytest.raises(OSError, match="simulated rename failure on second write"):
        run_eval.write_runs(runs, out_dir, dataset_version="rag-qa-v0.1")

    # (a) First file fully replaced with valid JSON.
    first = out_dir / "faithfulness.json"
    assert first.exists()
    parsed_first = json.loads(first.read_text(encoding="utf-8"))
    assert parsed_first["suite"] == "faithfulness"
    assert parsed_first["dataset_version"] == "rag-qa-v0.1"

    # (b) Second file's pre-existing stale content untouched.
    assert stale_recall_path.read_text(encoding="utf-8") == "STALE-RECALL"

    # (c) Third file never written.
    assert not (out_dir / "correctness.json").exists()

    # (d) No temp leftovers next to the suite JSONs.
    leftovers = [p for p in out_dir.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"expected no temp leftovers, got {leftovers}"
