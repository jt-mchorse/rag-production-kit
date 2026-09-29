"""Atomic write helper, and the copy a frozen record needs for a free-form field.

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
from typing import Any

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


def copy_json_value(value: Any) -> Any:
    """Copy a JSON value deeply over *containers*, leaving everything else alone.

    The copy a frozen record needs for a free-form JSON field, and the one
    ``dict(...)`` is not (#227).

    ``frozen=True`` prevents *rebinding* an attribute. It says nothing about the
    object the attribute points at, so a ``dict`` field on a frozen record is
    editable in place through any reference the caller still holds, with no
    ``FrozenInstanceError`` anywhere, because nothing is ever rebound. All five
    of this package's frozen records with a container field were in that state:
    ``Citation.metadata``, ``Candidate.metadata``, ``ScoredCandidate.metadata``,
    ``StreamEvent.payload`` and ``CostRecord.per_phase_ms``.

    **Not ``dict(...)``, which is the trap this function exists to avoid.** The
    defect *is* a shallow copy relationship, so a fix that re-spells one would
    pass any arm that only asks "was it copied at all". ``metadata`` is declared
    ``dict[str, Any]``; ``Any`` proves nothing about the values, and a nested
    ``dict`` under a shallow copy is still the caller's.

    **Deep over containers rather than ``copy.deepcopy``**, on purpose. The
    contract on these fields is JSON, and every non-container JSON value --
    ``str``, ``int``, ``float``, ``bool``, ``None`` -- is already immutable, so
    copying them buys nothing. ``deepcopy`` would additionally recurse into
    whatever a direct constructor happened to store there, changing the failure
    mode for out-of-contract input (a ``deepcopy`` of an open file handle
    raises, at a boundary whose job is to copy metadata) without making any
    in-contract case safer.

    **``dict`` and ``list`` only, and that is a decision with a measured cost.**
    A mutable container nested inside a ``tuple`` stays shared. The alternative
    is rebuilding the tuple, which loses a ``namedtuple``'s class -- the
    measured reason ``llm-eval-harness``' D-027 drew the same line -- and this
    package's write seam is *lenient* where that one is strict: ``_json_safe``
    coerces a tuple to a JSON array rather than refusing the record, so the
    limitation is real here rather than unreachable. It is pinned by name in
    ``tests/test_frozen_record_metadata_aliasing.py`` so it reads as a decision
    and not as an oversight.

    A ``dict`` or ``list`` **subclass** is normalised to its base type. Both
    compare equal to their base and serialize to identical JSON, so nothing a
    caller can observe through this package changes. Rebuilding via
    ``type(value)(...)`` would preserve the class for the well-behaved ones and
    raise for any subclass with a different ``__init__`` signature -- a strictly
    worse trade at a boundary whose contract is JSON.

    **Iterative, with a memo, and that is not a style choice.** The recursive
    one-liner is what this function was first written as, and this package's own
    ``tests/test_sse_frame_totality.py`` went **8 red** on it: a payload nested
    3000 levels deep raised ``RecursionError``, and a *circular* payload
    recursed forever. ``streaming._json_safe`` had already been rewritten
    iteratively for exactly this, and its docstring says why -- a helper added
    to guarantee frame validity must not blow the stack before ``json.dumps``
    is reached. Putting a recursive copy at ``StreamEvent.__post_init__`` would
    have reintroduced that on the seam ``_json_safe`` protects, one call
    earlier.

    A cycle is **preserved**, not replaced with a marker. This is a copier, not
    a sanitizer: the result is isomorphic to the input, so ``_json_safe`` still
    sees the back-reference and still names it ``<circular reference>`` on the
    wire, and ``_MAX_DEPTH`` still truncates at the same boundary. Replacing it
    here would move a wire-format decision into a record constructor.

    Duplicated from ``llm-eval-harness``' ``eval_harness/io_utils.py`` rather
    than shared, for the reason D-021 already records for ``comparison.py``:
    the two are separate distributions with no dependency between them. Note
    that the sibling there **is** recursive; it is fed ``json.loads`` output,
    which cannot contain a cycle, but ``Example`` is exported and constructible
    directly (``llm-eval-harness#259``).
    """
    if not isinstance(value, (dict, list)):
        return value
    root: Any = {} if isinstance(value, dict) else []
    # id(src) -> its copy. Two jobs: it terminates on a cycle, and it preserves
    # the input's *sharing* structure, so two fields pointing at one dict still
    # point at one dict afterwards -- a fresh one.
    memo: dict[int, Any] = {id(value): root}
    # Every source container stays referenced while the walk runs, so CPython
    # cannot recycle an `id` out from under `memo`.
    keep: list[Any] = [value]
    stack: list[tuple[Any, Any]] = [(value, root)]
    while stack:
        src, dst = stack.pop()
        items = src.items() if isinstance(src, dict) else enumerate(src)
        for key, child in items:
            if isinstance(child, (dict, list)):
                copied = memo.get(id(child))
                if copied is None:
                    copied = {} if isinstance(child, dict) else []
                    memo[id(child)] = copied
                    keep.append(child)
                    stack.append((child, copied))
            else:
                copied = child
            if isinstance(dst, dict):
                dst[key] = copied
            else:
                dst.append(copied)
    return root
