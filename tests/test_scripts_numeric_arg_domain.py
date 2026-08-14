"""Value-domain guards on the ``scripts/`` numeric CLI arguments (#176).

``scripts/`` is the entry-point directory this repo's validation waves have
historically enumerated last — the same blind spot that produced #114
(``bench_streaming --n/--k``) and #172. Two flags were still accepting values
their own code cannot use:

- ``capture_demo.py --pause-seconds`` had no validation at all. This was the
  **portfolio's last unguarded capture_demo pause**; the other four
  (llm-cost-optimizer, llm-eval-harness, nextjs-streaming-ai-patterns,
  prompt-regression-suite) all guard it, llm-eval-harness#198 being the
  identical fix to the identical script.
- ``telemetry_dashboard.py --port`` had no range check and surfaced an
  out-of-range value as a raw ``OverflowError`` from ``bind()`` at **exit 1**,
  while the sibling scripts already exit 2 with a flag-named message.

Every assertion below is anchored to the **measured pre-fix outcome** recorded
in #176, not merely to "a ValueError is raised now".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _load_capture_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if "capture_demo" in sys.modules:
        del sys.modules["capture_demo"]
    import capture_demo  # noqa: WPS433 — dynamic import is the point here.

    return capture_demo


def _run(script: str, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *flags],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


# ----------------------------------------------------------------------
# capture_demo.py --pause-seconds
#
# NOTE: every case below uses the `--pause-seconds=VALUE` form, not the
# space-separated form. argparse treats a bare `-inf` / `-1` as a *flag*
# because it starts with a dash, so `--pause-seconds -inf` fails with
# "expected one argument" — a different error that would make these tests
# pass for the wrong reason.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_fragment"),
    [
        # The quiet half. `_pause` guards `if seconds > 0`, and none of these
        # is > 0, so pre-fix every stage printed and the script exited 0
        # having taken no pause at all — an unusable recording reported as a
        # clean run. The inter-stage pauses are the script's only reason to
        # exist.
        ("nan", "must be finite"),
        ("-1", "must be >= 0"),
        ("-2.5", "must be >= 0"),
        ("-inf", "must be finite"),
        # The loud half. Pre-fix `time.sleep(inf)` raised OverflowError
        # ("timestamp out of range for platform time_t") at the *first*
        # `_pause` — after STAGE 1 had already run, costing the operator a
        # partial capture.
        ("inf", "must be finite"),
        # Looks like a finite, in-range literal. `float("1e400")` is `inf`,
        # so it reaches the same OverflowError. This is the row that argues
        # for a finiteness check rather than a magnitude check.
        ("1e400", "must be finite"),
        ("-1e400", "must be finite"),
    ],
)
def test_unusable_pause_seconds_is_rejected_at_parse_time(
    value: str, expected_fragment: str
) -> None:
    proc = _run(
        "capture_demo.py",
        f"--pause-seconds={value}",
        "--skip-server-cheatsheet",
        "--skip-nextjs-cheatsheet",
    )
    assert proc.returncode == 2, f"expected usage exit 2, got {proc.returncode}"
    assert "--pause-seconds" in proc.stderr
    assert expected_fragment in proc.stderr
    # Rejected *before* STAGE 1 — the whole point of validating at parse time
    # rather than at the first `_pause`. Pre-fix, `inf` printed this banner
    # and ran the streaming preview before blowing up.
    assert "STAGE 1" not in proc.stdout


@pytest.mark.parametrize("value", ["0", "0.0", "-0.0", "0.25"])
def test_valid_pause_seconds_still_runs(value: str) -> None:
    # `0` is the documented CI value and must not be caught by the new guard.
    # `-0.0` is accepted too: it compares equal to 0, so it is not "< 0", and
    # `_pause` treats it exactly like `0`. Kept explicit so a future guard
    # rewritten with a sign test rather than a comparison trips this test.
    proc = _run(
        "capture_demo.py",
        f"--pause-seconds={value}",
        "--skip-server-cheatsheet",
        "--skip-nextjs-cheatsheet",
    )
    assert proc.returncode == 0, proc.stderr
    assert "STAGE 1" in proc.stdout


def test_validate_pause_seconds_rejects_bool_for_programmatic_callers() -> None:
    # `bool` subclasses `int`, so `True` would otherwise sail through as 1.0.
    # Unreachable through argparse (`type=float` on "True" fails first), but
    # `main(argv=...)` is callable programmatically and the smoke tests do
    # exactly that. Same exclusion as llm-eval-harness#198.
    capture_demo = _load_capture_module()
    assert capture_demo._validate_pause_seconds(True) is not None
    assert capture_demo._validate_pause_seconds(False) is not None
    assert "must be a number" in capture_demo._validate_pause_seconds(True)
    # A plain int is fine — it is a usable number of seconds.
    assert capture_demo._validate_pause_seconds(0) is None
    assert capture_demo._validate_pause_seconds(3) is None


# ----------------------------------------------------------------------
# telemetry_dashboard.py --port
# ----------------------------------------------------------------------


@pytest.mark.parametrize("port", ["-1", "65536", "99999", "-70000"])
def test_out_of_range_port_is_a_usage_error_not_an_overflow_traceback(port: str) -> None:
    # Pre-fix: rc=1 with `OverflowError: bind(): port must be 0-65535.`
    # raised from inside ThreadingHTTPServer — the wrong exit code for a
    # usage error, and a diagnostic pointing at the socket layer instead of
    # at the flag the operator typed.
    proc = _run("telemetry_dashboard.py", f"--port={port}")
    assert proc.returncode == 2, f"expected usage exit 2, got {proc.returncode}"
    assert "--port" in proc.stderr
    assert "0-65535" in proc.stderr
    assert "OverflowError" not in proc.stderr
    assert "Traceback" not in proc.stderr


def test_port_guard_matches_the_sibling_scripts_exit_code_contract() -> None:
    # The reference implementations this fix ports from. If either of these
    # regresses, the claim that --port now "matches the sibling scripts" is
    # no longer true and this test says so.
    for script, flag in (("bench_streaming.py", "--n=0"), ("bench_rewriter.py", "--k=-3")):
        proc = _run(script, flag)
        assert proc.returncode == 2, f"{script} {flag} -> {proc.returncode}"
        assert "must be positive" in proc.stderr
