"""Atomic write helper.

`Path.write_text` is not atomic: SIGINT/SIGTERM/disk-full/OOM between
the implicit `open(..., "w")` truncate and `close()` flush leaves the
destination zero-length or partial. The eval action's composite
sticky comment (`evals/run_eval.py::_post_composite_comment`) parses
the three per-suite JSONs that `write_runs` emits, so a half-written
suite file corrupts the PR comment or fails the workflow with a
cryptic `JSONDecodeError`.

Pattern matches `llm-eval-harness/eval_harness/cli.py::_atomic_write_text`
(#48 there), `llm-cost-optimizer/scripts/_io.py::atomic_write_text`
(#42 there), and `prompt-regression-suite/prompt_regression/io.py::atomic_write_text`
(#39 there). Portfolio-wide uniformity is intentional.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str) -> None:
    # Write to a sibling temp file in the destination's parent
    # directory, fsync, then `os.replace` (atomic on POSIX within the
    # same filesystem). Same-directory placement guarantees same
    # filesystem so the rename cannot fall back to a copy. On any
    # exception between the temp write and the rename, the temp is
    # unlinked.
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
