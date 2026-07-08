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

# Cap the target basename's contribution to the temp filename. The temp name
# is `.<base>.<random>.tmp`; the affixes add ~20 bytes, so prepending a full
# basename that is itself near NAME_MAX (255 on ext4/APFS) overflows the limit
# and the write fails with `OSError: [Errno 63] File name too long` — even
# though a plain `Path.write_text` of that same target succeeds (#128, sibling
# of mcp-server-cookbook#96). The base in the temp name is cosmetic
# (`ls`-ability); uniqueness comes from `NamedTemporaryFile`'s random component,
# so truncating it is safe. Budget is in BYTES (NAME_MAX is a byte limit) and we
# trim on a char boundary so multibyte names are never split mid-codepoint.
_MAX_TEMP_BASE_BYTES = 200


def _cap_base_for_temp(base: str) -> str:
    if len(base.encode("utf-8")) <= _MAX_TEMP_BASE_BYTES:
        return base
    out = base
    while out and len(out.encode("utf-8")) > _MAX_TEMP_BASE_BYTES:
        out = out[:-1]
    return out


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
            prefix=f".{_cap_base_for_temp(target.name)}.",
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
