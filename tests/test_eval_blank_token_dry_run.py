"""A blank `GITHUB_TOKEN` dry-runs `--post-comment` instead of posting (#316).

`.env.example` ships `GITHUB_TOKEN=` empty (#272), and its documented loader
(`set -a; . ./.env`) exports that as "". `main` decided the dry run with
`token is None`, so the script skipped it and called the API with
`Authorization: Bearer ` -- a 401 and exit 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import run_eval


@pytest.fixture
def harness_stubbed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_eval.shutil, "which", lambda _name: "/usr/bin/eval-harness")
    monkeypatch.setattr(run_eval, "_diff_markdown", lambda _cur, _base: "DELTA-MD")
    monkeypatch.setattr(run_eval, "CURRENT_DIR", tmp_path / "current")

    def no_network(*_a: object, **_k: object) -> object:
        raise AssertionError("the dry run opened a connection")

    monkeypatch.setattr(run_eval.urllib.request, "urlopen", no_network)


@pytest.mark.parametrize("value", [None, "", " ", "\n\t "], ids=["unset", "empty", "space", "ws"])
def test_a_blank_token_dry_runs(
    harness_stubbed: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", value)
    assert run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"]) == 0
    out = capsys.readouterr().out
    assert "dry-run: no GITHUB_TOKEN" in out
    assert "DELTA-MD" in out


def test_a_set_token_takes_the_post_path(
    harness_stubbed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Control: the post path is still reached, here the no-network stub.
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_real")
    seen: list[str | None] = []
    monkeypatch.setattr(
        run_eval,
        "_post_composite_comment",
        lambda _repo, _pr, _deltas, token: seen.append(token) or True,
    )
    assert run_eval.main(["--post-comment", "--repo", "o/r", "--pr", "1"]) == 0
    assert seen == ["ghp_real"]
