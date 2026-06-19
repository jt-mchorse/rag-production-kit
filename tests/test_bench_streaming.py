"""Lock the `--out PATH` sink on `scripts/bench_streaming.py`.

Issue #60 plumbed `PhaseTimings.dump_summary_json` (PR #59 / issue #58)
into the streaming bench script as `--out PATH` so a CI step (or a
local operator wanting a hermetic artifact) can capture per-phase
p50/p95 without parsing the table back out of stdout.

The pattern is sibling to the `validate --out` propagation that landed
across `llm-eval-harness#66`, `chunking-strategies-lab#45`,
`prompt-regression-suite#59`, `embedding-model-shootout#55`. Sink-parity
across the portfolio's measurement entrypoints.

These tests exercise the real CLI via `subprocess.run` so argparse,
`main()` wiring, and the atomic-write helper are all covered end-to-end
(unit tests for `dump_summary_json` itself live in `test_streaming.py`
behind the `#58` banner).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_MODULE = "scripts.bench_streaming"
_PHASES = ("retrieving", "reranking", "generating", "total")


def _run_bench(*args: str, cwd: Path = _REPO_ROOT) -> subprocess.CompletedProcess[str]:
    """Invoke `python -m scripts.bench_streaming` with the given args.

    Returns the completed process so callers can assert on stdout /
    stderr / returncode. Captures both streams as text so the test
    body doesn't deal with bytes plumbing.
    """
    return subprocess.run(
        [sys.executable, "-m", _SCRIPT_MODULE, *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def test_bench_streaming_out_writes_file_with_per_phase_shape(tmp_path: Path) -> None:
    """`--out PATH` writes a JSON file whose top-level keys are the four
    phases (`retrieving`, `reranking`, `generating`, `total`) with the
    `n` / `p50_ms` / `p95_ms` shape produced by `PhaseTimings.summary`.

    A tiny `--n 3` is enough to populate the runtime keys without
    burning bench-runner time in CI.
    """
    out = tmp_path / "sink" / "streaming.json"
    _run_bench("--n", "3", "--out", str(out))

    assert out.exists(), "the --out file must exist after the bench finishes"
    body = out.read_text(encoding="utf-8")
    assert body.endswith("\n"), "atomic-write helper writes a trailing newline"
    payload = json.loads(body)
    assert set(payload) == set(_PHASES)
    for phase in _PHASES:
        entry = payload[phase]
        assert {"n", "p50_ms", "p95_ms"}.issubset(entry), (
            f"phase {phase!r} payload missing n / p50_ms / p95_ms keys: {entry}"
        )
    # The runtime phases must have observed work; `total` is the wrapper
    # over each iteration so it always records.
    assert payload["retrieving"]["n"] == 3
    assert payload["total"]["n"] == 3


def test_bench_streaming_out_does_not_suppress_stdout(tmp_path: Path) -> None:
    """`--out PATH` is a sink, not a replacement — the stdout table
    must still render so an operator running the bench locally sees
    the numbers in their terminal.
    """
    out = tmp_path / "streaming.json"
    proc = _run_bench("--n", "3", "--out", str(out))
    assert "Streaming pipeline benchmark" in proc.stdout, proc.stdout
    assert "retrieving" in proc.stdout
    assert "total" in proc.stdout


def test_bench_streaming_without_out_writes_no_file(tmp_path: Path) -> None:
    """The default invocation must not create any artifact in the cwd
    or anywhere else — `--out` is the explicit opt-in. Cassette this
    against the tmp_path being clean before AND after.
    """
    before = set(tmp_path.iterdir())
    proc = _run_bench("--n", "3", cwd=tmp_path)
    after = set(tmp_path.iterdir())
    assert before == after, (
        f"bench without --out wrote unexpected files to cwd: "
        f"{sorted(p.name for p in after - before)}"
    )
    # Stdout shape unchanged from prior bench-only behavior.
    assert "Streaming pipeline benchmark" in proc.stdout


def test_bench_streaming_out_creates_parent_dirs(tmp_path: Path) -> None:
    """`dump_summary_json` delegates to `atomic_write_text` which does
    `parent.mkdir(parents=True)`. Confirm the CLI inherits that — an
    operator shouldn't have to `mkdir -p artifacts/` before the run.
    """
    out = tmp_path / "deeply" / "nested" / "sink" / "streaming.json"
    _run_bench("--n", "3", "--out", str(out))
    assert out.exists()
    assert out.parent.is_dir()


def test_bench_streaming_out_overwrites_atomically(tmp_path: Path) -> None:
    """Two successive bench runs to the same `--out` PATH must leave a
    valid self-contained JSON document (not the concatenation, not a
    stale half-written file), and no `.tmp` leftovers from the
    atomic-write helper.

    Wall-clock p50/p95 timings vary run-to-run, so this test asserts
    *shape* not *byte equality* — the second payload parses cleanly
    on its own (concatenation would produce invalid JSON) and the
    parent directory has no atomic-write leftovers. Mirrors
    `test_phase_timings_dump_summary_json_overwrites_atomically` in
    `test_streaming.py` — CLI-side coverage so a future refactor that
    bypasses `dump_summary_json` regresses loudly.
    """
    out = tmp_path / "streaming.json"
    _run_bench("--n", "3", "--out", str(out))
    body1 = out.read_text(encoding="utf-8")
    _run_bench("--n", "3", "--out", str(out))
    body2 = out.read_text(encoding="utf-8")
    # Second run must yield a single valid JSON document, not a stream
    # of two concatenated ones (which is what would happen if the
    # writer were appending instead of replacing).
    payload2 = json.loads(body2)
    assert set(payload2) == set(_PHASES)
    # The second write should not preserve any unique substring of the
    # first beyond the JSON document boundary — same shape, fresh data.
    assert body2.startswith("{") and body2.endswith("}\n")
    assert body2.count("{") == body1.count("{"), (
        "second JSON document has a different brace count than the first; "
        "either the writer is no longer atomic or the payload shape "
        "drifted between runs."
    )
    leftovers_tmp = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    leftovers_dot = [p.name for p in tmp_path.iterdir() if p.name.startswith(".streaming.json.")]
    assert leftovers_tmp == [], leftovers_tmp
    assert leftovers_dot == [], leftovers_dot


def test_bench_streaming_out_help_text_mentions_dump_summary_json() -> None:
    """`--help` must explain what `--out` writes so a `--help`-only
    reader knows the artifact shape without diving into the source.

    Loose substring match — drop the docstring entirely and CI fails
    with a specific pointer back to issue #60 / PR sibling.
    """
    proc = subprocess.run(
        [sys.executable, "-m", _SCRIPT_MODULE, "--help"],
        cwd=str(_REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--out" in proc.stdout
    # Either "dump_summary_json" or "PhaseTimings" or both must appear
    # so the reader sees the explicit linkage to the writer in code.
    help_lower = proc.stdout.lower()
    assert "dump_summary_json" in help_lower or "phasetimings" in help_lower, (
        "--out help text should reference the writer it routes through "
        "(dump_summary_json or PhaseTimings) so the artifact shape is "
        "discoverable from --help alone."
    )


@pytest.mark.parametrize("phase", _PHASES)
def test_bench_streaming_out_phase_keys_complete(phase: str, tmp_path: Path) -> None:
    """Parametrized lock: every one of the four canonical phases must
    appear in the on-disk artifact. Drop a phase from `PhaseTimings`
    and this fingerprints which one is missing.
    """
    out = tmp_path / "streaming.json"
    _run_bench("--n", "3", "--out", str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert phase in payload, (
        f"on-disk JSON sink is missing phase key {phase!r}. Expected the "
        f"four canonical phases {sorted(_PHASES)}; got {sorted(payload)}."
    )
