"""Validation sweep on the three sites called out as deferred in PR #41.

`Generator.max_chunks` (`rag_kit/generator.py:184`) and `HashEmbedder.dim`
(`rag_kit/embedder.py:54`) extend the established positive-int contract
(`runs.list_runs.limit` in `llm-eval-harness#42`, k-validators in #41 of
this repo, batch_size in `embedding-model-shootout#34`).

`PhaseTimings.percentile.p` (`rag_kit/streaming.py:119`) adds a narrower
finiteness/type check: NaN previously slipped both clamp branches and
crashed `int(NaN)` deep in interpolation; `True`/`False` silently
coerced to 1/0 percentiles. Out-of-range FINITE values continue to clamp
per the pre-existing contract (test_streaming.py documents this as a
"match numpy's well-behaved default"), so the validator is type+NaN
only, not bounded-range.
"""

from __future__ import annotations

import math

import pytest

from rag_kit.embedder import HashEmbedder
from rag_kit.generator import TemplateGenerator
from rag_kit.streaming import PhaseTimings

# ----------------------------------------------------------------------
# Generator.max_chunks — positive int construction contract
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        True,  # bool: silently bound self.max_chunks=True, sliced [:True] → 1 chunk.
        False,  # bool: would slice [:False] → 0 chunks.
        0,  # zero: meaningless top-k.
        -1,  # negative.
        0.5,  # non-int float.
        1.0,  # whole float, still not int.
        2.5,  # would surface as TypeError deep in generate.
        math.nan,
        math.inf,
        -math.inf,
        None,
        "3",
    ],
)
def test_generator_rejects_non_positive_int_max_chunks(bad_value):
    with pytest.raises(ValueError, match="max_chunks must be a positive integer"):
        TemplateGenerator(max_chunks=bad_value)


def test_generator_error_message_includes_repr_of_bad_value():
    with pytest.raises(ValueError, match="True"):
        TemplateGenerator(max_chunks=True)
    with pytest.raises(ValueError, match="0.5"):
        TemplateGenerator(max_chunks=0.5)


@pytest.mark.parametrize("good_value", [1, 2, 3, 10, 1000])
def test_generator_accepts_positive_int_max_chunks(good_value):
    g = TemplateGenerator(max_chunks=good_value)
    assert g.max_chunks == good_value


# ----------------------------------------------------------------------
# HashEmbedder.dim — positive int construction contract, then % 8
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        True,  # bool=1: also fails % 8 — wrong error message before this fix.
        False,  # bool=0: was caught by sign-only but masked the type bug.
        0,
        -8,  # negative multiple of 8 (only this fix catches it cleanly).
        -32,
        8.0,  # whole float, multiple of 8: silently bound to self.dim=8.0
        # → range(8.0) TypeError deep in embed().
        16.0,
        0.5,
        math.nan,
        math.inf,
        -math.inf,
        None,
        "8",
    ],
)
def test_hash_embedder_rejects_non_positive_int_dim(bad_value):
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        HashEmbedder(dim=bad_value)


@pytest.mark.parametrize("good_value", [8, 16, 32, 256, 1024])
def test_hash_embedder_accepts_positive_multiple_of_8_dim(good_value):
    e = HashEmbedder(dim=good_value)
    assert e.dim == good_value


def test_hash_embedder_multiple_of_8_check_only_fires_on_real_int():
    # After the type contract, the multiple-of-8 check sees only ints.
    # A non-multiple-of-8 int still raises the multiple-of-8 message.
    with pytest.raises(ValueError, match="multiple of 8"):
        HashEmbedder(dim=10)
    # 1 is a real int, positive, but not a multiple of 8 — so the second
    # check fires with the correct message (was wrongly fired by `True`
    # before the fix, since `True == 1` slipped sign-only).
    with pytest.raises(ValueError, match="multiple of 8"):
        HashEmbedder(dim=1)


# ----------------------------------------------------------------------
# PhaseTimings.percentile.p — bounded-float contract with finiteness
# ----------------------------------------------------------------------


@pytest.fixture
def populated_timings():
    t = PhaseTimings()
    for ms in (1.0, 2.0, 3.0, 4.0, 5.0):
        t.record("retrieving", ms)
    return t


@pytest.mark.parametrize(
    "bad_p",
    [
        math.nan,  # slipped both p <= 0 and p >= 100 branches → int(NaN) raised.
        True,  # bool-is-int: silently treated as 1st percentile.
        False,  # bool-is-int: silently returned values[0].
        None,
        "50",
        [],
        (1,),
        {"p": 50},
    ],
)
def test_percentile_rejects_nan_or_wrong_type_p(populated_timings, bad_p):
    # Out-of-range finite values continue to clamp (`-5` → values[0],
    # `110` → values[-1]) per the existing contract — see
    # `test_phase_timings_percentile_clamps_edges` in test_streaming.py.
    # `inf` and `-inf` also clamp cleanly via the existing >=100 / <=0
    # branches. Only NaN and non-numeric types are genuinely broken.
    with pytest.raises(ValueError, match="p must be a finite number"):
        populated_timings.percentile("retrieving", bad_p)


@pytest.mark.parametrize("good_p", [0, 0.0, 1, 50, 50.0, 95, 99.99, 100, 100.0])
def test_percentile_accepts_in_range_finite_p(populated_timings, good_p):
    result = populated_timings.percentile("retrieving", good_p)
    assert result is not None
    assert isinstance(result, float)


def test_percentile_boundary_zero_returns_min(populated_timings):
    assert populated_timings.percentile("retrieving", 0) == 1.0


def test_percentile_boundary_hundred_returns_max(populated_timings):
    assert populated_timings.percentile("retrieving", 100) == 5.0


def test_percentile_preserves_clamp_contract_for_out_of_range_finites(populated_timings):
    # Explicit pin: out-of-range finites clamp; this is the contract
    # `test_phase_timings_percentile_clamps_edges` documents elsewhere.
    assert populated_timings.percentile("retrieving", -5) == 1.0
    assert populated_timings.percentile("retrieving", 110) == 5.0
    # `inf` and `-inf` are finite-looking enough for clamp purposes here.
    assert populated_timings.percentile("retrieving", math.inf) == 5.0
    assert populated_timings.percentile("retrieving", -math.inf) == 1.0


def test_percentile_empty_phase_still_validates_p_first():
    # Validator runs before the empty-phase short-circuit, so callers get
    # consistent errors regardless of recording state.
    empty = PhaseTimings()
    with pytest.raises(ValueError, match="p must be a finite number"):
        empty.percentile("retrieving", math.nan)
    # A valid p on an empty phase still returns None (existing contract).
    assert empty.percentile("retrieving", 50) is None
