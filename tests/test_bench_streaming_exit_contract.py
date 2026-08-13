"""Exit-code contract for `scripts/bench_streaming.py` (#172).

This file exists because of a two-part defect. The `--out` write seam was
bare, so an unwritable target escaped as a raw traceback at exit 1 — and
`main()` was `-> None` under a bare `main()` module guard, so a returned exit
code would have been *discarded*. Fixing only the seam would have produced a
script that reads correctly at the call site and still exits 0 on every I/O
failure.

That is why every exit-code assertion here goes through a **subprocess** and
reads `returncode`. Asserting on `main`'s return value would pass on a tree
where the module guard is still throwing it away — i.e. it would test the one
thing that was already fine and miss the one that wasn't.

`tests/test_bench_streaming.py` covers the `--out` happy paths (file written,
parent dirs created, stdout not suppressed, atomic overwrite, `--help` text).
Its `_run_bench` helper passes `check=True`, so it cannot be reused for
failure cases; the local helper below uses `check=False`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_MODULE = "scripts.bench_streaming"

# Every entry point in the repo, as (module path, relative file).
_ENTRY_POINTS = (
    ("scripts.bench_streaming", "scripts/bench_streaming.py"),
    ("scripts.bench_rewriter", "scripts/bench_rewriter.py"),
    ("scripts.telemetry_dashboard", "scripts/telemetry_dashboard.py"),
    ("evals.run_eval", "evals/run_eval.py"),
)


def _run_bench(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the real CLI, tolerating a non-zero exit."""
    return subprocess.run(
        [sys.executable, "-m", _SCRIPT_MODULE, *args],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


# ----------------------------------------------------------------------
# A. The --out write seam
# ----------------------------------------------------------------------


def test_out_under_a_file_parent_exits_2(tmp_path: Path) -> None:
    """`atomic_write_text` makes parent dirs; a *file* in the parent chain
    raises FileExistsError, which used to escape as a traceback."""
    parent = tmp_path / "a_file"
    parent.write_text("", encoding="utf-8")

    proc = _run_bench("--n", "3", "--out", str(parent / "sub.json"))

    assert proc.returncode == 2, (
        f"an unwritable --out must exit 2, not {proc.returncode}; stderr:\n{proc.stderr}"
    )
    assert "Traceback" not in proc.stderr, f"must not surface as a traceback:\n{proc.stderr}"
    assert "::error::" in proc.stderr, f"expected a clean ::error:: line:\n{proc.stderr}"


def test_out_that_is_a_directory_exits_2(tmp_path: Path) -> None:
    """The temp file is written fine and `os.replace` onto a directory raises
    IsADirectoryError — a different exception from the case above, reached
    through a different part of the helper."""
    target = tmp_path / "a_dir"
    target.mkdir()

    proc = _run_bench("--n", "3", "--out", str(target))

    assert proc.returncode == 2, (
        f"a directory --out must exit 2, not {proc.returncode}; stderr:\n{proc.stderr}"
    )
    assert "Traceback" not in proc.stderr, f"must not surface as a traceback:\n{proc.stderr}"
    assert str(target) in proc.stderr, f"the error must name the path:\n{proc.stderr}"


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX directory permissions; chmod is a no-op on Windows"
)
def test_out_under_a_read_only_parent_exits_2(tmp_path: Path) -> None:
    """PermissionError, raised from `tempfile` before the temp file exists."""
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        proc = _run_bench("--n", "3", "--out", str(parent / "x.json"))
    finally:
        parent.chmod(0o700)

    assert proc.returncode == 2, (
        f"a read-only --out parent must exit 2, not {proc.returncode}; stderr:\n{proc.stderr}"
    )
    assert "Traceback" not in proc.stderr, f"must not surface as a traceback:\n{proc.stderr}"


def test_the_bench_still_prints_its_table_before_the_write_fails(tmp_path: Path) -> None:
    """The write is the last thing `main` does, and it must stay that way.

    All `--n` queries have already been measured by the time the sink is
    written, so a failed write must not discard the run's output — the
    operator can still read the numbers off stdout. If a future refactor moves
    the write earlier, this fails.
    """
    target = tmp_path / "a_dir"
    target.mkdir()

    proc = _run_bench("--n", "3", "--out", str(target))

    assert proc.returncode == 2
    assert "Streaming pipeline benchmark" in proc.stdout, (
        f"the stdout table must survive a failed --out write; stdout:\n{proc.stdout}"
    )
    for phase in ("retrieving", "reranking", "generating", "total"):
        assert phase in proc.stdout, f"expected the {phase!r} row on stdout:\n{proc.stdout}"


def test_successful_run_still_exits_0(tmp_path: Path) -> None:
    """Adding the return plumbing must not change a valid run."""
    out = tmp_path / "summary.json"
    proc = _run_bench("--n", "3", "--out", str(out))

    assert proc.returncode == 0, f"a valid run must exit 0; stderr:\n{proc.stderr}"
    assert out.exists()


def test_successful_run_without_out_still_exits_0() -> None:
    proc = _run_bench("--n", "3")
    assert proc.returncode == 0, f"a valid run must exit 0; stderr:\n{proc.stderr}"


# ----------------------------------------------------------------------
# B. The plumbing that made A un-fixable on its own
# ----------------------------------------------------------------------


def _parse(rel_path: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _module_level_main(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("no module-level `main` found")


@pytest.mark.parametrize(("module", "rel_path"), _ENTRY_POINTS, ids=[m for m, _ in _ENTRY_POINTS])
def test_entry_point_main_accepts_argv_and_returns_int(module: str, rel_path: str) -> None:
    """Every entry point agrees on `main(argv) -> int`.

    Parsed from the AST rather than grepped, so a docstring or comment
    *describing* the shape can't satisfy the check — a comment asking the next
    author to keep these in sync is exactly what produced the gap this issue
    fixes, so the lock derives the property instead of restating it.
    """
    fn = _module_level_main(_parse(rel_path))

    arg_names = [a.arg for a in fn.args.args]
    assert "argv" in arg_names, (
        f"{rel_path}: main must accept `argv` so it is driveable in-process; got {arg_names}"
    )

    assert fn.returns is not None, f"{rel_path}: main must annotate a return type"
    returns = ast.unparse(fn.returns)
    assert returns == "int", (
        f"{rel_path}: main must return int so an exit code can reach the "
        f"process; annotated {returns!r}"
    )


@pytest.mark.parametrize(("module", "rel_path"), _ENTRY_POINTS, ids=[m for m, _ in _ENTRY_POINTS])
def test_entry_point_module_guard_propagates_the_return_value(module: str, rel_path: str) -> None:
    """The `if __name__ == "__main__":` block must pass `main()`'s value to the
    process — `raise SystemExit(main())` or `sys.exit(main())`.

    A bare `main()` discards the return value, so the process exits 0 no matter
    what `main` reports. That is the defect this test exists for: without it, a
    `return 2` added to `bench_streaming` would have been silently inert.
    `evals/run_eval.py` uses `sys.exit(...)` and the scripts use
    `raise SystemExit(...)`; both propagate, so both are accepted.
    """
    tree = _parse(rel_path)
    guards = [n for n in tree.body if isinstance(n, ast.If)]
    assert guards, f"{rel_path}: no module-level `if` block found"

    body = [ast.unparse(stmt) for g in guards for stmt in g.body]
    propagating = [s for s in body if s in ("raise SystemExit(main())", "sys.exit(main())")]
    assert propagating, (
        f"{rel_path}: the __main__ guard must propagate main()'s return value "
        f"via `raise SystemExit(main())` or `sys.exit(main())`; found {body}. "
        "A bare `main()` discards the exit code."
    )


def test_a_bare_main_call_really_does_discard_the_exit_code(tmp_path: Path) -> None:
    """Demonstrate the mechanism the lock above protects against.

    Without this, `test_entry_point_module_guard_propagates_the_return_value`
    is just a style rule. It isn't — a bare `main()` genuinely swallows a
    non-zero code, which is why fixing the write seam alone would have been a
    no-op.
    """
    bare = tmp_path / "bare.py"
    bare.write_text(
        "def main():\n    return 2\n\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    propagating = tmp_path / "propagating.py"
    propagating.write_text(
        "def main():\n    return 2\n\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n",
        encoding="utf-8",
    )

    bare_rc = subprocess.run([sys.executable, str(bare)], check=False).returncode
    prop_rc = subprocess.run([sys.executable, str(propagating)], check=False).returncode

    assert bare_rc == 0, "a bare main() call discards the return value — the defect"
    assert prop_rc == 2, "SystemExit(main()) propagates it — the fix"
