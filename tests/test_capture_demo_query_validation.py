"""``capture_demo.py --query`` is validated before STAGE 1 (#294).

STAGE 1 hands ``--query`` to ``StreamingPipeline.run``, whose empty-query
guard sits above the generator's ``try``. Before #294 an empty ``--query``
printed the STAGE 1 banner and then died with a raw ``ValueError`` traceback
at exit 1, while ``--pause-seconds`` beside it was already a parse-time usage
error at exit 2. The pre-check uses ``run``'s rule exactly. Since #424 that
rule is ``rag_kit.text.is_blank``, so a whitespace-only query is a usage error
too, as ``run`` and the SSE server now refuse it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "capture_demo.py"


def _run(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pause-seconds=0",
            "--skip-server-cheatsheet",
            "--skip-nextjs-cheatsheet",
            *flags,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


@pytest.mark.parametrize(
    "flags", [("--query=",), ("--query", ""), ("--query", "   "), ("--query", "\u3000\u200b")]
)
def test_empty_query_is_a_usage_error_before_stage_1(flags: tuple[str, ...]) -> None:
    proc = _run(*flags)
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-400:])
    assert "--query must be non-empty" in proc.stderr
    assert "Traceback" not in proc.stderr
    # Rejected before STAGE 1 starts, like --pause-seconds.
    assert "STAGE 1" not in proc.stdout


@pytest.mark.parametrize("query", ["  postgres tuning  ", "postgres tuning"])
def test_non_empty_query_still_runs(query: str) -> None:
    # Padding is not blank: `StreamingPipeline.run` streams a padded query and
    # so does the SSE server, so the pre-check must not be stricter (#424).
    proc = _run("--query", query)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "STAGE 1" in proc.stdout
    assert "'done']" in proc.stdout


def test_pre_check_agrees_with_the_pipeline_guard() -> None:
    # The two guards are one rule: whatever `run` rejects as a query, the
    # script rejects at parse time, and nothing else.
    sys.path.insert(0, str(REPO_ROOT))
    from rag_kit.streaming import StreamingPipeline

    class _Never:
        def search(self, query, k=5, *, reranker=None):
            raise AssertionError("not reached")

    import scripts.capture_demo as cd

    for blank in ("", "   ", "\u3000\u200b"):
        with pytest.raises(ValueError, match="query must be non-empty"):
            next(StreamingPipeline(_Never()).run(blank, k=1))
        with pytest.raises(SystemExit) as exc:
            cd.main(["--pause-seconds=0", "--query", blank])
        assert exc.value.code == 2
    assert next(StreamingPipeline(_Never()).run(" x ", k=1)).type == "retrieving"
