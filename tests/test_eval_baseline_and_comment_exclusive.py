"""`--write-baselines` and `--post-comment` cannot share a run (#231, D-024).

`evals/current/*.json` are committed. `--write-baselines` sends the run to
`evals/baselines/`, and `--post-comment` diffs `evals/current/` against
`evals/baselines/`, so together they posted the *last committed* current run
against the baseline this invocation had just overwritten. Measured in a
scratch copy with a visibly stale current: `current STALE-CO vs baseline ...
regressed 3 ... flagged 3`, at exit 0 -- regressions this run never produced,
as the PR's eval gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import run_eval

_EVAL_YML = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "eval.yml"


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    current, baselines = tmp_path / "current", tmp_path / "baselines"
    current.mkdir()
    baselines.mkdir()
    for suite in run_eval.SUITES:
        (baselines / f"{suite}.json").write_text('{"sentinel": true}', encoding="utf-8")
    monkeypatch.setattr(run_eval, "CURRENT_DIR", current)
    monkeypatch.setattr(run_eval, "BASELINES_DIR", baselines)
    return current, baselines


@pytest.fixture
def posts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    seen: list[dict[str, str]] = []

    def fake_post(repo: str, pr: int, deltas: dict[str, str], token: str | None) -> bool:
        seen.append(deltas)
        return True

    monkeypatch.setattr(run_eval, "_post_composite_comment", fake_post)
    return seen


def test_the_combination_exits_two_and_writes_and_posts_nothing(
    dirs: tuple[Path, Path], posts: list[dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    current, baselines = dirs
    rc = run_eval.main(["--write-baselines", "--post-comment", "--repo", "o/r", "--pr", "1"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--write-baselines and --post-comment cannot be combined" in err
    assert posts == []
    assert all(p.read_text(encoding="utf-8") == '{"sentinel": true}' for p in baselines.iterdir())
    assert list(current.iterdir()) == []


def test_the_combination_is_refused_before_the_suites_run(
    dirs: tuple[Path, Path], posts: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the other operator-input checks, not after a scoring pass."""

    def boom() -> list[object]:
        raise AssertionError("run_all_suites ran before the flag check")

    monkeypatch.setattr(run_eval, "run_all_suites", boom)
    assert run_eval.main(["--write-baselines", "--post-comment", "--repo", "o/r", "--pr", "1"]) == 2


def test_write_baselines_alone_still_writes_baselines(
    dirs: tuple[Path, Path], posts: list[dict[str, str]]
) -> None:
    _, baselines = dirs
    assert run_eval.main(["--write-baselines"]) == 0
    assert sorted(p.name for p in baselines.iterdir()) == [
        "correctness.json",
        "faithfulness.json",
        "recall_at_5.json",
    ]
    assert all(p.read_text(encoding="utf-8") != '{"sentinel": true}' for p in baselines.iterdir())
    assert posts == []


def test_post_comment_alone_still_posts(
    dirs: tuple[Path, Path], posts: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_eval.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(run_eval, "_diff_markdown", lambda cur, base: f"diff {cur.name}")
    rc = run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"])
    assert rc == 0
    assert len(posts) == 1
    assert posts[0]["faithfulness"] == "diff faithfulness.json"


def test_the_ci_invocations_do_not_combine_the_flags() -> None:
    """`eval.yml` runs the two as separate steps; pin that it still does."""
    text = _EVAL_YML.read_text(encoding="utf-8")
    assert "--post-comment" in text
    assert "--write-baselines" not in text
