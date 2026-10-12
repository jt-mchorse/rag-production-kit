"""A sentence ending terminator, closer, terminator keeps its marker (#314).

#258 put the marker before a terminator+closer tail and #262 extended that to a
terminator run, but both anchored the tail on closers *ending* the sentence.
`up?".` and `FAQ.).` end in a terminator after the closer, so the tail matched
nothing, the fallback stripped only the final `.`, and the marker landed after
`?"`, where `split_sentences` cuts the claim from it. A fully grounded chunk was
refused as `unparseable_output`: 5,213 of 60,000 random draws on `main`.
"""

from __future__ import annotations

import itertools
import random

import pytest

from rag_kit.generator import _CLOSERS, _TERMINATORS, GeneratedAnswer, TemplateGenerator
from rag_kit.retriever import RetrievalResult


def _answer(text: str):
    r = RetrievalResult(
        external_id="doc1", text=text, metadata={}, fused_score=1.0, ranks={"dense": 1}
    )
    return TemplateGenerator(max_chunks=3).generate("q", [r], threshold=0.0)


@pytest.mark.parametrize(
    ("text", "first"),
    [
        (
            'The manual asks "Is the server up?". Restart it if not.',
            'The manual asks "Is the server up [cite:doc1]?".',
        ),
        ("Use the flag (see the FAQ.). It is safe.", "Use the flag (see the FAQ [cite:doc1].)."),
        ('The error reads "Stop!". Check the log.', 'The error reads "Stop [cite:doc1]!".'),
    ],
)
def test_the_reported_shapes_are_answered(text: str, first: str) -> None:
    out = _answer(text)
    assert isinstance(out, GeneratedAnswer), out
    assert out.text.startswith(f"Per the retrieved context, {first} ")


def test_a_closer_before_the_only_terminator_is_unchanged() -> None:
    # Control: no terminator before the closer, so the old path, byte for byte.
    out = _answer('He said "hi". Fine.')
    assert isinstance(out, GeneratedAnswer), out
    assert out.text == (
        'Per the retrieved context, He said "hi" [cite:doc1]. '
        "Per the retrieved context, Fine [cite:doc1]."
    )


# Representative members of each set, ASCII and not, built from code points.
_T = [".", "?", "!", chr(0x3002), chr(0x0964)]
_C = ['"', ")", "]", chr(0x201D), chr(0x300D)]


@pytest.mark.parametrize(
    "tail",
    ["".join(p) for p in itertools.product(_T, _C, _T)]
    + ["".join(p) for p in itertools.product(_T, _C, _C, _T)][:60],
)
def test_terminator_closer_terminator_round_trips(tail: str) -> None:
    out = _answer(f"Alpha beta{tail} Gamma delta.")
    assert isinstance(out, GeneratedAnswer), (tail, out)


def test_a_seeded_fuzz_never_refuses_a_single_grounded_chunk() -> None:
    # The hunt's fuzz, seeded and shrunk: random short sentences over the
    # splitter's own terminator and closer sets.
    rng = random.Random(314)
    alphabet = list("ab ") + list(_TERMINATORS) + list(_CLOSERS) + ["Dr.", "1"]
    refused = []
    for _ in range(4000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        if not any(c.isalnum() for c in text) or "[" in text:
            continue
        if not isinstance(_answer(text), GeneratedAnswer):
            refused.append(text)
    assert not refused, refused[:5]
