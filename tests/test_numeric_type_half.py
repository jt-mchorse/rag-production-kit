"""The type half of the refusal-gate guards (#380).

`_validate_threshold` (#78), the `_top_score` score gate (#372) and
`LexicalOverlapReranker(length_penalty=...)` (#75) checked finiteness only.
`math.isfinite(True)` is True, so a `bool` passed as 1.0/0.0 -- `threshold=True`
refused a 0.5 top score -- and a str or `None` escaped as a bare TypeError
instead of the ValueError every sibling guard raises. #303/#362 closed the same
split for telemetry and `PhaseTimings`.

Any other `numbers.Real` is accepted and coerced to `float`:
sentence-transformers' `CrossEncoder.predict` returns `numpy.float32`, which
is not a `float` subclass, and it is the usual custom reranker.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from rag_kit.generator import (
    AnthropicGenerator,
    GeneratedAnswer,
    Refusal,
    TemplateGenerator,
    _top_score,
)
from rag_kit.reranker import LexicalOverlapReranker
from rag_kit.retriever import RetrievalResult

_BAD = [True, False, "0.5", None]
_BAD_IDS = ["true", "false", "str", "none"]


def _np_float32(value: float) -> object:
    np = pytest.importorskip("numpy")
    return np.float32(value)


def _result(*, fused: object = 0.5, rerank: object = None) -> RetrievalResult:
    return RetrievalResult(
        external_id="a",
        text="Refunds are accepted within a 30 day window.",
        metadata={"source": "test"},
        fused_score=fused,  # type: ignore[arg-type]
        ranks={"lexical": 1, "dense": 1},
        rerank_score=rerank,  # type: ignore[arg-type]
        rerank_rank=None if rerank is None else 1,
    )


class _NeverCalledClient:
    """The threshold guard runs before any API call; reaching one is a failure."""

    class _Messages:
        def create(self, **_kwargs: object) -> object:
            raise AssertionError("threshold guard should fire before the API call")

    def __init__(self) -> None:
        self.messages = self._Messages()


class TestThresholdType:
    @pytest.mark.parametrize("threshold", _BAD, ids=_BAD_IDS)
    def test_template_rejects_non_number(self, threshold: object) -> None:
        with pytest.raises(ValueError, match=r"threshold must be a finite number; got "):
            TemplateGenerator().generate("refund window", [_result()], threshold=threshold)  # type: ignore[arg-type]

    @pytest.mark.parametrize("threshold", _BAD, ids=_BAD_IDS)
    def test_anthropic_rejects_non_number(self, threshold: object) -> None:
        gen = AnthropicGenerator(client=_NeverCalledClient())
        with pytest.raises(ValueError, match=r"threshold must be a finite number; got "):
            gen.generate("refund window", [_result()], threshold=threshold)  # type: ignore[arg-type]

    # Controls: a real float or int floor keeps its verdict.

    def test_control_float_threshold_answers(self) -> None:
        result = TemplateGenerator().generate("refund window", [_result()], threshold=0.3)
        assert isinstance(result, GeneratedAnswer)

    def test_control_int_threshold_refuses(self) -> None:
        result = TemplateGenerator().generate("refund window", [_result()], threshold=1)
        assert isinstance(result, Refusal)
        assert result.used_threshold == 1

    # Other reals are accepted and stored as float.

    def test_fraction_threshold_stored_as_float(self) -> None:
        # Also pins the refusal detail: `format(Fraction, ".4f")` raises on 3.11.
        result = TemplateGenerator().generate(
            "refund window", [_result()], threshold=Fraction(3, 4)
        )
        assert isinstance(result, Refusal)
        assert type(result.used_threshold) is float
        assert result.used_threshold == 0.75
        assert "threshold=0.7500" in result.detail

    def test_numpy_float32_threshold_stored_as_float(self) -> None:
        result = TemplateGenerator().generate(
            "refund window", [_result()], threshold=_np_float32(0.75)
        )
        assert isinstance(result, Refusal)
        assert type(result.used_threshold) is float
        assert result.used_threshold == 0.75

    def test_int_threshold_too_large_for_a_double_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"threshold must be a finite number"):
            TemplateGenerator().generate("refund window", [_result()], threshold=10**400)


class TestScoreType:
    @pytest.mark.parametrize("score", _BAD[:3], ids=_BAD_IDS[:3])
    def test_rerank_score_rejected(self, score: object) -> None:
        # `None` is not an arm here: it means "no rerank score", falling back
        # to fused_score by design.
        with pytest.raises(ValueError, match=r"'a' has a non-numeric rerank_score"):
            TemplateGenerator().generate("refund window", [_result(rerank=score)], threshold=0.1)

    @pytest.mark.parametrize("score", _BAD, ids=_BAD_IDS)
    def test_fused_score_rejected(self, score: object) -> None:
        with pytest.raises(ValueError, match=r"'a' has a non-numeric fused_score"):
            _top_score([_result(fused=score)])

    # Controls: real scores, float and int, still pass the gate.

    @pytest.mark.parametrize("score", [0.5, 2], ids=["float", "int"])
    def test_control_numeric_rerank_score(self, score: float) -> None:
        assert _top_score([_result(rerank=score)]) == score

    def test_control_none_rerank_falls_back_to_fused(self) -> None:
        assert _top_score([_result(fused=0.25, rerank=None)]) == 0.25

    # Other reals are accepted and come back as float.

    def test_fraction_rerank_score_is_float(self) -> None:
        top = _top_score([_result(rerank=Fraction(1, 4))])
        assert type(top) is float
        assert top == 0.25

    def test_numpy_float32_rerank_score_is_float(self) -> None:
        # The CrossEncoder.predict shape: a float32 per chunk.
        top = _top_score([_result(rerank=_np_float32(0.25)), _result(rerank=_np_float32(0.5))])
        assert type(top) is float
        assert top == 0.5

    def test_numpy_float32_score_refusal_detail_renders(self) -> None:
        result = TemplateGenerator().generate(
            "refund window", [_result(rerank=_np_float32(0.25))], threshold=0.5
        )
        assert isinstance(result, Refusal)
        assert type(result.top_score) is float
        assert "top_score=0.2500 below threshold=0.5000" in result.detail

    def test_numpy_float32_nan_score_still_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"non-finite rerank_score \(nan\)"):
            _top_score([_result(rerank=_np_float32(float("nan")))])


class TestLengthPenaltyType:
    @pytest.mark.parametrize("penalty", _BAD, ids=_BAD_IDS)
    def test_rejects_non_number(self, penalty: object) -> None:
        with pytest.raises(ValueError, match=r"length_penalty must be a finite number >= 0\.0"):
            LexicalOverlapReranker(length_penalty=penalty)  # type: ignore[arg-type]

    @pytest.mark.parametrize("penalty", [0.001, 0], ids=["float", "int"])
    def test_control_numeric_accepted(self, penalty: float) -> None:
        assert LexicalOverlapReranker(length_penalty=penalty).length_penalty == penalty

    def test_fraction_stored_as_float(self) -> None:
        stored = LexicalOverlapReranker(length_penalty=Fraction(1, 1000)).length_penalty
        assert type(stored) is float
        assert stored == 0.001

    def test_numpy_float32_stored_as_float(self) -> None:
        stored = LexicalOverlapReranker(length_penalty=_np_float32(0.5)).length_penalty
        assert type(stored) is float
        assert stored == 0.5
