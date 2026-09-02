"""The temp-name byte budget is measured in the bytes the filesystem sees (#199).

`_cap_base_for_temp` exists so a destination basename near NAME_MAX does not
overflow the limit once the temp affixes are prepended (#128). Its comment says
the budget is in bytes because NAME_MAX is a byte limit — true — and the old
implementation counted `str.encode("utf-8")` under the strict error handler,
which is a different set of bytes from the ones the kernel is handed.

The gap is reachable, not pedantic, because POSIX path bytes and `sys.argv`
both decode through `surrogateescape`: a byte that is not valid UTF-8 becomes a
lone surrogate in `U+DC80..U+DCFF`, which strict UTF-8 encoding refuses. So
`bench_streaming --out $'bench\\xff.json'` made the cap raise
`UnicodeEncodeError` before it ever reached the length question.

`UnicodeEncodeError` is a `ValueError`, so neither write-seam guard catches it.
`scripts/bench_streaming.py` lists what it guards against — "a file parent
(`FileExistsError`), a directory target (`IsADirectoryError`), a read-only
parent (`PermissionError`)" — three members of one class, where the population
is *ways an operator-supplied `--out` can be unusable*. An unencodable name is a
fourth, and it produced precisely the outcome that comment describes: the full
benchmark table printed, then a traceback at exit 1.

These tests are written so the *host* never decides the verdict. The
`_cap_base_for_temp` cases are pure-function and hold everywhere. The seam
cases assert a property true on both a byte-transparent filesystem (ext4, where
the write succeeds) and a UTF-8-validating one (APFS, which returns `EILSEQ`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_kit import io_utils as io_mod
from rag_kit.io_utils import _MAX_TEMP_BASE_BYTES, _cap_base_for_temp, atomic_write_text

_REPO_ROOT = Path(__file__).resolve().parents[1]

# A lone low surrogate is what `surrogateescape` produces for the raw byte
# 0xFF. Built from its codepoint rather than written literally so the character
# cannot be mangled by an editor or a copy-paste round trip.
SURROGATE = chr(0xDCFF)


def _fs_len(text: str) -> int:
    """The byte length the kernel sees. Never raises; that is the whole point."""
    return len(os.fsencode(text))


# ---------------------------------------------------------------------------
# The variant table. Axes: length (fits / overflows) x encoding class
# (pure ASCII / multibyte UTF-8 / surrogate-bearing / mixed).
# ---------------------------------------------------------------------------

NAME_VARIANTS = [
    ("ascii-short", "streaming.json"),
    ("ascii-at-budget", "a" * _MAX_TEMP_BASE_BYTES),
    ("ascii-long", "a" * 250),
    # "é" is 2 bytes in UTF-8, so 150 of them is 300 bytes: over budget in
    # bytes while well under it in characters.
    ("multibyte-short", "strëaming.json"),
    ("multibyte-long", "é" * 150),
    # Each surrogate is exactly one byte under `os.fsencode` — the byte the
    # name actually came from.
    ("surrogate-short", "bench" + SURROGATE + ".json"),
    ("surrogate-long", SURROGATE * 250),
    ("mixed-long", "é" * 50 + SURROGATE * 150),
    ("surrogate-only", SURROGATE),
    ("mixed-at-boundary", "a" * (_MAX_TEMP_BASE_BYTES - 1) + SURROGATE),
]


@pytest.mark.parametrize(("label", "base"), NAME_VARIANTS, ids=[v[0] for v in NAME_VARIANTS])
def test_cap_base_for_temp_never_raises_and_stays_within_budget(label: str, base: str) -> None:
    """Every name a `Path` can hold gets a capped answer, not an exception.

    Strict-UTF-8 measurement raised `UnicodeEncodeError` for the surrogate-
    bearing rows before it could answer the length question at all.
    """
    capped = _cap_base_for_temp(base)

    assert _fs_len(capped) <= _MAX_TEMP_BASE_BYTES, f"{label}: over budget"
    assert capped == base[: len(capped)], (
        f"{label}: the capped name must be a character-boundary prefix of the "
        "original — trimming happens by character so no codepoint is split"
    )
    if _fs_len(base) <= _MAX_TEMP_BASE_BYTES:
        assert capped == base, f"{label}: a name within budget must be returned unchanged"
    else:
        # Maximality: one more character would have gone over. Without this the
        # test would also pass for a cap that returns "" for everything.
        assert len(capped) < len(base)
        assert _fs_len(base[: len(capped) + 1]) > _MAX_TEMP_BASE_BYTES, (
            f"{label}: the cap trimmed further than the budget required"
        )


def test_cap_base_for_temp_agrees_with_the_old_measurement_on_encodable_names() -> None:
    """Switching the measurement must not move the budget for names that worked.

    `os.fsencode` and `str.encode("utf-8")` return the same bytes for every
    string that is valid UTF-8, so every previously-passing name is unaffected;
    the change is confined to the names the old call refused outright.
    """
    for _label, base in NAME_VARIANTS:
        try:
            strict = len(base.encode("utf-8"))
        except UnicodeEncodeError:
            continue  # the population the old measurement could not count at all
        assert _fs_len(base) == strict


def test_name_bytes_never_raises_on_a_surrogate() -> None:
    """The measurement helper itself is total over `str`.

    `os.fsencode` uses `surrogateescape` on POSIX and `surrogatepass` on
    Windows, so it round-trips every string a `Path` can carry.
    """
    assert io_mod._name_bytes("out" + SURROGATE + ".json") == len(b"out\xff.json")


# ---------------------------------------------------------------------------
# The seams. The exception *class* is the contract every caller is written
# against, so that is what gets asserted.
# ---------------------------------------------------------------------------


def test_atomic_write_text_unencodable_target_name_fails_as_oserror_if_at_all(
    tmp_path: Path,
) -> None:
    """A destination name the filesystem cannot represent is an OS-level
    problem, and must surface as one.

    Deliberately not asserted as "succeeds" or as "raises": ext4 accepts any
    non-NUL byte in a name and the write goes through, while APFS validates
    UTF-8 and returns `EILSEQ`. Both are correct, and both are `OSError` or
    nothing — which is what a plain `Path.write_text` of the same target does,
    and the only class either write-seam guard catches.
    """
    target = tmp_path / ("bench" + SURROGATE + ".json")

    try:
        atomic_write_text(target, '{"ok": true}')
    except UnicodeEncodeError as e:  # pragma: no cover - the bug this closes
        pytest.fail(
            "atomic_write_text raised UnicodeEncodeError for an unencodable "
            f"destination *name*: {e!r}. The content was pure ASCII."
        )
    except OSError:
        # The filesystem refused the name. Nothing was left behind.
        assert list(tmp_path.iterdir()) == []
        return

    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


def test_atomic_write_text_long_unencodable_target_name_is_capped_not_refused(
    tmp_path: Path,
) -> None:
    """The long-name and the unencodable-name axes compose.

    This is the row that needs both halves of the fix: the fast-path check has
    to survive the surrogate to discover the name is over budget, and the trim
    loop has to survive it on every iteration.
    """
    target = tmp_path / (SURROGATE * 250)

    try:
        atomic_write_text(target, "x")
    except UnicodeEncodeError as e:  # pragma: no cover - the bug this closes
        pytest.fail(f"cap raised on a long unencodable name: {e!r}")
    except OSError:
        assert list(tmp_path.iterdir()) == []


def test_bench_streaming_unencodable_out_honors_the_exit_code_contract(tmp_path: Path) -> None:
    """The argv road, end to end, through a real subprocess.

    `--out` is operator input and `sys.argv` decodes with `surrogateescape`, so
    passing a `str` holding a lone surrogate through `subprocess` puts the raw
    byte on the command line and gets it back as the same surrogate inside the
    child — which is exactly how a real shell `$'bench\\xff.json'` reaches the
    cap.

    Asserted as "the process exits with a code, and if nothing was written that
    code is 2" rather than a fixed code, because whether the write succeeds is
    the filesystem's call, not this test's. What must never happen on any host
    is the traceback: the bench prints its whole table first, so a crash here
    is a complete, successful-looking benchmark followed by a stack (#172).
    """
    out = tmp_path / ("bench" + SURROGATE + ".json")

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_streaming", "--n", "3", "--out", str(out)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert "UnicodeEncodeError" not in proc.stderr, (
        "an unencodable --out must not escape as a traceback:\n" + proc.stderr
    )
    assert "Traceback" not in proc.stderr, proc.stderr
    if not out.exists():
        assert proc.returncode == 2, (
            "an unusable --out is an I/O error: exit 2, not the success range"
        )
        assert "::error::failed to write" in proc.stderr
