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


def _name_bytes(base: str) -> int:
    """Length of *base* in the bytes the filesystem actually sees.

    `os.fsencode`, not `base.encode("utf-8")` (#199). Both halves of the
    comment above are true and the old implementation still counted the wrong
    bytes: NAME_MAX limits the bytes handed to the kernel, which is
    `os.fsencode` — `sys.getfilesystemencoding()` together with
    `sys.getfilesystemencodeerrors()`, i.e. `surrogateescape` on POSIX.

    That handler is why the distinction bites rather than being pedantry. A
    path byte that is not valid UTF-8 arrives in Python as a lone surrogate in
    `U+DC80..U+DCFF`, and strict `str.encode("utf-8")` refuses to encode it —
    so `_cap_base_for_temp` used to raise `UnicodeEncodeError` on a destination
    the OS can name, *before* reaching the length question. `sys.argv` decodes
    with the same handler, so `bench_streaming --out $'bench\\xff.json'` is
    enough to produce one.

    `UnicodeEncodeError` is a `ValueError`, so it is not an `OSError` and
    neither write-seam guard catches it. `scripts/bench_streaming.py` lists
    what it is guarding — "a file parent (FileExistsError), a directory target
    (IsADirectoryError), a read-only parent (PermissionError)" — three members
    of one class, where the population is *ways an operator-supplied `--out`
    can be unusable*. An unencodable name is a fourth, and it produced exactly
    the outcome that comment describes: the full benchmark table printed, then
    a traceback at exit 1.

    `os.fsencode` never raises: `surrogateescape` on POSIX, `surrogatepass` on
    Windows, so every `str` a `Path` can hold round-trips. For a name that is
    valid UTF-8 it returns exactly the old number, so the budget is unchanged
    for every name that worked before.
    """
    return len(os.fsencode(base))


def _cap_base_for_temp(base: str) -> str:
    if _name_bytes(base) <= _MAX_TEMP_BASE_BYTES:
        return base
    out = base
    while out and _name_bytes(out) > _MAX_TEMP_BASE_BYTES:
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
