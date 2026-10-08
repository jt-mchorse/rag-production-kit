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
import secrets
import stat
from pathlib import Path
from typing import Any

# Cap the target basename's contribution to the temp filename. The temp name
# is `.<base>.<random>.tmp`; the affixes add ~20 bytes, so prepending a full
# basename that is itself near NAME_MAX (255 on ext4/APFS) overflows the limit
# and the write fails with `OSError: [Errno 63] File name too long` — even
# though a plain `Path.write_text` of that same target succeeds (#128, sibling
# of mcp-server-cookbook#96). The base in the temp name is cosmetic
# (`ls`-ability); uniqueness comes from the random component `_open_temp` adds,
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


# Retries for a temp-name collision. Eight hex chars is 2**32 names per
# base, so a single collision is already vanishingly rare.
_TEMP_NAME_ATTEMPTS = 100


def _open_temp(target: Path) -> tuple[int, Path]:
    """Create `.<base>.<random>.tmp` beside *target*, honouring the umask.

    `tempfile.NamedTemporaryFile` / `mkstemp` always create **0600**, whatever
    the umask, and `os.replace` carries that mode onto the target, so every
    file this helper wrote was owner-only and an overwrite demoted an existing
    0644 file to 0600 -- which the `Path.write_text` it replaced never did
    (#247, portfolio-ops#81). Creating with `0o666` lets the *kernel* apply the
    umask. Reading the umask instead (`os.umask(0); os.umask(old)`) would set a
    process-wide umask of 0 for every other thread in between.
    """
    base = _cap_base_for_temp(target.name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(_TEMP_NAME_ATTEMPTS):
        candidate = target.parent / f".{base}.{secrets.token_hex(4)}.tmp"
        try:
            return os.open(candidate, flags, 0o666), candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"no usable temporary name beside {target}")


def _resolve_symlinked_target(target: Path) -> Path:
    """The file a write to *target* lands in: through a symlink, as `write_text` does (#296).

    `os.replace` renames onto the LINK, not the file it points at. So a
    symlinked destination became a regular file and the linked file kept its
    old contents, while the `Path.write_text` this helper replaced writes
    through the link. The mode copy below already followed the link
    (`os.stat`), so the linked file's mode was copied onto a file that then
    replaced the link instead (python-async-llm-pipelines#157).

    Resolving here also places the temp file beside the RESOLVED file, which
    keeps the rename on one filesystem when the link points at a different
    one, and gives `_open_temp` the resolved basename to cap. A dangling link
    resolves to the path it names, and the write creates that file, as
    `write_text` would. With a link loop, non-strict `realpath` returns the
    path unresolved, and the mode copy's `os.stat` raises `OSError` (ELOOP),
    which is the error the write-seam guards already translate. A plain path
    comes back unchanged.
    """
    if not target.is_symlink():
        return target
    return Path(os.path.realpath(target))


def atomic_write_text(path: str | Path, text: str) -> None:
    # Write to a sibling temp file in the destination's parent
    # directory, fsync, then `os.replace` (atomic on POSIX within the
    # same filesystem). Same-directory placement guarantees same
    # filesystem so the rename cannot fall back to a copy. On any
    # exception between the temp write and the rename, the temp is
    # unlinked.
    target = _resolve_symlinked_target(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        fd, tmp_path = _open_temp(target)
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        # An overwrite keeps the existing file's mode, as `write_text` did:
        # without this the rename would swap a 0644 target for a temp created
        # at `0o666 & ~umask` (#247). A missing target is the new-file case.
        with contextlib.suppress(FileNotFoundError):
            os.chmod(tmp_path, stat.S_IMODE(os.stat(target).st_mode))
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

    **The container set is derived, not listed, and that is the D-023 amendment
    to D-022.** The rule is two questions per kind, and a *list* of kinds got the
    second one wrong:

    ==========  ===============  =======================  ====================
    kind        mutable itself?  can reach a mutable?     action
    ==========  ===============  =======================  ====================
    ``dict``    yes              yes                      rebuild and walk
    ``list``    yes              yes                      rebuild and walk
    ``tuple``   **no**           **yes**                  walk, then freeze
    ``set``     yes              **no**                   rebuild, do not walk
    frozenset   no               no                       return unchanged
    ==========  ===============  =======================  ====================

    D-022 walked the two kinds that answer *yes* to both and missed both
    diagonals. An **immutable container can still reach a mutable one** -- that
    is the whole of the ``tuple`` case, and a rule phrased as "copy the mutable
    containers" cannot express it. A ``set`` is the mirror: mutable, so sharing
    it is the defect, but every element is hashable and therefore no mutable
    container is reachable *through* it, so one level is the whole depth.
    ``frozenset`` answers no twice and is returned as-is.

    **The reason D-022 gave for excluding ``tuple`` was a cost this package
    already pays.** It priced the loss of a ``namedtuple``'s class, citing
    ``llm-eval-harness``' D-027 -- and then observed, in the same paragraph, that
    "this package's write seam is *lenient* where that one is strict:
    ``_json_safe`` coerces a tuple to a JSON array rather than refusing the
    record". Both halves are true. The leniency is what makes the limitation
    *reachable*, and it is the same leniency that makes the price *zero*:
    ``streaming._new_container`` has been flattening every tuple to a list all
    along, under the comment ``# tuple -> list, matching what json.dumps does
    anyway``. A ``namedtuple``'s class is gone before any consumer sees the
    payload. Measured: ``namedtuple("P", "x y")(1, 2)`` reaches the wire as
    ``[1, 2]``, compares equal to ``(1, 2)``, and produces byte-identical
    ``json.dumps`` output -- which is exactly the test the next paragraph already
    applies to a ``dict`` subclass.

    The harm was D-022's own headline, through the kind it skipped: a
    ``StreamEvent`` built with ``payload={"k": ("a", inner_list)}`` published one
    SSE frame, and appending to ``inner_list`` afterwards changed the frame of an
    event that had already been yielded.

    **A copy may not walk fewer kinds than the seam it feeds.** That is the
    invariant ``tests/test_frozen_record_metadata_aliasing.py`` now derives from
    both functions' ASTs, rather than pinning either list by hand. The
    immutability claim is void on exactly the difference between the two sets,
    and a hand-written list is how a one-kind difference survived.

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

    **A tuple is built as a list and frozen at the end, and a cycle is why that
    is the only order that works.** A ``dict`` or ``list`` copy is a shell filled
    in place, which is what lets the memo close a cycle -- the shell exists
    before its contents do. A tuple cannot be filled in place, so it is walked
    into a list shell and converted afterwards, in **reverse discovery order**: a
    tuple nested directly inside another is always discovered later, so every
    child is already final when its parent freezes. The slots pointing at each
    shell are recorded so its parents learn the final object.

    That handles a cycle *through* a tuple, which is reachable --
    ``a = []; t = (a,); a.append(t)`` -- even though a tuple can never close one
    by itself, because a tuple cannot contain an object that did not already
    exist. The mutable link in such a cycle is what makes the patch possible:
    ``copy_json_value(a)`` returns ``r`` with ``r[0] == (r,)`` and
    ``r[0][0] is r``. The freeze pass is a flat loop over a list, so the
    ``RecursionError`` the iterative walk exists to avoid is not reintroduced.

    Duplicated from ``llm-eval-harness``' ``eval_harness/io_utils.py`` rather
    than shared, for the reason D-021 already records for ``comparison.py``:
    the two are separate distributions with no dependency between them. Note
    that the sibling there **is** recursive; it is fed ``json.loads`` output,
    which cannot contain a cycle, but ``Example`` is exported and constructible
    directly (``llm-eval-harness#259``).
    """
    # A `set` is mutable, so sharing it is the defect -- but every element is
    # hashable, so no mutable container is reachable through it and one level is
    # the whole depth. A `frozenset` answers no to both questions and falls
    # through to the identity return below, as does any non-container.
    # `isinstance(frozenset(), set)` is False: the two are siblings, not
    # subclasses, so this branch does not catch it by accident.
    if isinstance(value, set):
        return set(value)
    if not isinstance(value, (dict, list, tuple)):
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
    # List shells standing in for tuples, in discovery order, and every
    # `(container, key)` slot holding each one. See the docstring: a tuple cannot
    # be filled in place, so it is frozen after the walk, children first.
    tuple_shells: list[Any] = [root] if isinstance(value, tuple) else []
    slots: dict[int, list[tuple[Any, Any]]] = {}
    while stack:
        src, dst = stack.pop()
        items = src.items() if isinstance(src, dict) else enumerate(src)
        for key, child in items:
            copied: Any
            if isinstance(child, set):
                copied = set(child)
            elif isinstance(child, (dict, list, tuple)):
                copied = memo.get(id(child))
                if copied is None:
                    copied = {} if isinstance(child, dict) else []
                    memo[id(child)] = copied
                    keep.append(child)
                    stack.append((child, copied))
                    if isinstance(child, tuple):
                        tuple_shells.append(copied)
                if isinstance(child, tuple):
                    # Recorded on every *reference*, not only on discovery: a
                    # tuple reached twice has two slots to patch, and the memo
                    # hit above is exactly the second one.
                    slots.setdefault(id(copied), []).append((dst, key))
            else:
                copied = child
            if isinstance(dst, dict):
                dst[key] = copied
            else:
                dst.append(copied)
    for shell in reversed(tuple_shells):
        frozen = tuple(shell)
        for container, key in slots.get(id(shell), ()):
            # `container` may itself be an unfrozen tuple shell; it is still a
            # list at this point, and it freezes later with the final value in
            # place. Subscript assignment works for both a dict key and a list
            # index, and the index is the key because children were appended in
            # `enumerate` order.
            container[key] = frozen
        if shell is root:
            root = frozen
    return root


def refuse_bare_string(name: str, value: Any) -> None:
    """Raise `ValueError` if `value` is a bare `str`/`bytes` (#253).

    For a parameter that takes a collection of ids. A `str` *is* a
    `Sequence[str]` and an `Iterable[str]`, so the annotation admits it, mypy
    accepts it, and `list("doc-7")` is five well-formed ids -- nothing after the
    coercion can tell. Only the bare-string shape is refused; every other
    collection behaves exactly as before.
    """
    if isinstance(value, (str, bytes, bytearray)):
        fix = f"pass [{value!r}]" if isinstance(value, str) else "decode it to str ids first"
        raise ValueError(
            f"{name} must be a sequence of ids, not a bare {type(value).__name__}: "
            f"{value!r} would be split into its characters -- {fix}"
        )
