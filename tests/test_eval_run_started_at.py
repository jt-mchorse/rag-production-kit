"""``started_at`` is a measurement, not a literal (#221, D-020).

`_SuiteRun.to_run_result` wrote `"2026-05-16T00:00:00Z"` unconditionally, for
every run ever produced, since the file was created. The determinism reading
does not survive the record's own contents: `git_sha` is read from
`git rev-parse HEAD` and `run_id` is `sha256(suite|git_sha)`, so both move on
every commit. Freezing only the timestamp produced a record that contradicts
itself -- a run asserted to have started 2026-05-16 against a commit created
months later.

The module docstring claims this shape matches `eval_harness.runner.RunResult`.
Upstream's `run_suite` resolves the same field as `started_at or utc_now_iso()`
with a caller override "so tests can pin them"; this field was the one place the
declared parity was false.

Every arm here pins a property of the *payload*, never of the host clock -- an
assertion that the stamp is "about now" would be a clock test, which is the
standing rule in this portfolio. The one arm that touches the real clock asserts
only that two stamps taken from a monotonically advancing injected clock differ,
and it injects the clock.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest

from evals import run_eval

# The exact literal the writer shipped. Named so the arm below reads as "this
# specific value is gone", not "some string changed".
FROZEN_LITERAL = "2026-05-16T00:00:00Z"

# `eval_harness.runs` sorts every listing with a lexicographic string compare on
# this column, so the spelling is load-bearing, not cosmetic: `isoformat()`
# renders `+00:00`, which does not sort against a `Z` suffix.
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _run(**kwargs) -> dict:
    run = run_eval._SuiteRun(suite="correctness", rows=[], git_sha="a" * 40)
    return run.to_run_result(dataset_version="rag-qa-v0.1", **kwargs)


def test_the_frozen_literal_is_gone() -> None:
    """The defect itself, pinned by value.

    Red against the pre-#221 writer. This is deliberately an assertion about
    one specific string rather than a general property: the general properties
    below are all satisfied *by the literal too* -- it is a well-formed ISO-8601
    `Z` timestamp, it round-trips, and it is stable. Only naming it separates
    the two.
    """
    assert _run()["started_at"] != FROZEN_LITERAL
    assert run_eval._utc_now_iso() != FROZEN_LITERAL


def test_an_injected_stamp_reaches_the_payload_verbatim() -> None:
    """The override exists and is not reformatted on the way through."""
    assert _run(started_at="2019-03-04T05:06:07Z")["started_at"] == "2019-03-04T05:06:07Z"


def test_two_different_stamps_give_two_different_payloads() -> None:
    """The field varies with the run, which is the whole claim.

    Against the pre-#221 writer both of these are `"2026-05-16T00:00:00Z"` --
    the arm that says the value is a *function of the run* rather than of the
    source file.
    """
    early = _run(started_at="2026-01-01T00:00:00Z")
    late = _run(started_at="2026-12-31T23:59:59Z")
    assert early["started_at"] != late["started_at"]
    # And every other field is untouched, so this change moves exactly one
    # thing. Green on both trees by construction; it is what rejects a
    # neighbour that also perturbs `run_id` or `git_sha`.
    assert {k: v for k, v in early.items() if k != "started_at"} == {
        k: v for k, v in late.items() if k != "started_at"
    }


def test_the_default_is_the_clock_not_a_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """The *default* path reads a clock, verified by replacing the clock.

    Not "is the stamp close to now" -- that is a host-environment assertion. The
    injected clock makes it a statement about the code.
    """
    stamps = iter(
        [
            datetime(2001, 2, 3, 4, 5, 6, tzinfo=UTC),
            datetime(2002, 3, 4, 5, 6, 7, tzinfo=UTC),
        ]
    )

    class _Clock:
        @staticmethod
        def now(tz=None):  # noqa: ANN001, ANN205 - stdlib signature
            return next(stamps)

    monkeypatch.setattr(run_eval, "datetime", _Clock)
    assert _run()["started_at"] == "2001-02-03T04:05:06Z"
    assert _run()["started_at"] == "2002-03-04T05:06:07Z"


def test_the_stamp_is_the_lexicographically_sortable_Z_form() -> None:
    """`+00:00` would not sort against `Z`, and the store sorts on this column.

    `eval_harness.runs.latest_run_id_for_suite` is
    `ORDER BY started_at DESC LIMIT 1` -- a string compare, by its own
    docstring's reasoning ("the ISO-8601 format is lexicographically sortable so
    a string compare suffices"). A `datetime.isoformat()` stamp is a correct
    timestamp that sorts wrongly against the existing rows.
    """
    stamp = run_eval._utc_now_iso()
    assert ISO_Z.match(stamp), stamp
    assert "+" not in stamp
    # It parses back to the instant it names.
    assert datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) <= datetime.now(UTC)


def test_one_run_stamps_all_three_suites_identically(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_runs` resolves the stamp once (#221).

    One invocation of this script is one eval run. Letting each suite stamp
    itself would, at this field's one-second resolution, *usually agree anyway*
    -- and occasionally split a single run's three artifacts across a second
    boundary, which is the kind of intermittent difference nobody diagnoses.

    That "usually agrees" is why the clock is injected here rather than read.
    The first draft of this arm called the real clock and asserted
    `len(stamps) == 1`; the neighbour that drops the threading and lets each
    suite stamp itself was built and run, and it **passed** -- three
    sub-millisecond calls land in the same second. The arm was green for the
    wrong reason and proved nothing. A clock that advances one second per read
    makes the two versions differ every time: one stamp when the stamp is
    resolved once, three when it is not.
    """
    ticks = iter([datetime(2026, 1, 1, 0, 0, second, tzinfo=UTC) for second in range(10)])

    class _TickingClock:
        @staticmethod
        def now(tz=None):  # noqa: ANN001, ANN205 - stdlib signature
            return next(ticks)

    monkeypatch.setattr(run_eval, "datetime", _TickingClock)
    runs = run_eval.run_all_suites()
    paths = run_eval.write_runs(runs, tmp_path / "out", dataset_version="rag-qa-v0.1")
    stamps = {json.loads(p.read_text())["started_at"] for p in paths.values()}
    assert len(paths) == 3
    assert stamps == {"2026-01-01T00:00:00Z"}, stamps
    assert FROZEN_LITERAL not in stamps


def test_the_shape_still_matches_eval_harness_RunResult() -> None:
    """The parity the module docstring claims, asserted rather than asserted-in-prose.

    Skipped rather than failed when the `[eval]` extra is absent, because this
    module is importable without it. Green on both trees -- the pre-#221 writer
    had the right *key set* and the wrong *value*, which is precisely why a
    shape check never caught this and a value check had to.
    """
    runner = pytest.importorskip("eval_harness.runner")
    import dataclasses

    expected = {f.name for f in dataclasses.fields(runner.RunResult)}
    assert set(_run()) == expected
