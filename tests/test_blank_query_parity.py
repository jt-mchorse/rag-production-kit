"""The generators and the Next.js demo refuse a blank query too (#426).

#424 put `is_blank` on every entry point that already refused `""`. Two never
did, measured on `3cb15e8` (fake client, one chunk, threshold 0.0):

    AnthropicGenerator.generate(q) for "", "   ", "\\u200b", "\\ufeff\\u3000"
                                      1 messages.create call, GeneratedAnswer
    TemplateGenerator.generate(q)     same four               GeneratedAnswer
    Next.js GET /api/stream, no q / q= / q=%20%20 / q=%E2%80%8B / q=%E3%80%80
                                      200, 276 SSE events each

The demo's TypeScript `isBlank` (`demo/nextjs/lib/blank.ts`) is a port of
`rag_kit.text.is_blank`. Both are checked against one row table,
`demo/nextjs/test/blank-rows.json`: vitest runs it through `isBlank`, and this
file runs it through `is_blank` and checks its whitespace list against the
`str.isspace` catalog, so the port cannot drift without one side going red.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag_kit.generator import AnthropicGenerator, GeneratedAnswer, TemplateGenerator
from rag_kit.retriever import RetrievalResult
from rag_kit.text import is_blank
from tests.test_non_str_query_type_check import _ALL_NON_STR, _FakeAnthropic

_TABLE_PATH = Path(__file__).resolve().parents[1] / "demo" / "nextjs" / "test" / "blank-rows.json"
_TABLE: dict[str, Any] = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
_ROWS: dict[str, tuple[str, bool]] = {
    r["name"]: ("".join(map(chr, r["codepoints"])), r["blank"]) for r in _TABLE["rows"]
}
_BLANK = sorted(n for n, (_, blank) in _ROWS.items() if blank)
_NOT_BLANK = sorted(n for n, (_, blank) in _ROWS.items() if not blank)

_CHUNK = RetrievalResult(
    external_id="a",
    text="Paris is the capital of France.",
    metadata={},
    fused_score=0.9,
    ranks={"dense": 1},
)


# ---------------------------------------------------------------------------
# The shared table
# ---------------------------------------------------------------------------


def test_the_table_has_both_kinds_of_row_and_unique_names() -> None:
    assert len(_ROWS) == len(_TABLE["rows"])
    assert len(_BLANK) >= 10
    assert len(_NOT_BLANK) >= 5


def test_the_whitespace_list_the_port_spells_out_is_the_isspace_catalog() -> None:
    # `lib/blank.ts` lists these codepoints instead of using JS `\s`; vitest
    # checks its list equals this one.
    catalog = [cp for cp in range(sys.maxunicode + 1) if chr(cp).isspace()]
    assert _TABLE["py_isspace_codepoints"] == catalog


@pytest.mark.parametrize("name", sorted(_ROWS))
def test_is_blank_agrees_with_the_shared_row(name: str) -> None:
    text, blank = _ROWS[name]
    assert is_blank(text) is blank


def test_the_table_holds_the_rows_js_whitespace_gets_wrong() -> None:
    # JS `\s` lacks U+001C-U+001F and U+0085; a port written as `\s` would
    # pass a table without these rows.
    blank_cps = {ord(c) for n in _BLANK for c in _ROWS[n][0]}
    assert {0x1C, 0x85} <= blank_cps


# ---------------------------------------------------------------------------
# AnthropicGenerator.generate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _BLANK)
def test_anthropic_generator_refuses_a_blank_query_before_the_api(name: str) -> None:
    client = _FakeAnthropic()
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        AnthropicGenerator(client=client).generate(_ROWS[name][0], [_CHUNK], threshold=0.0)
    assert client.messages.calls == []


@pytest.mark.parametrize("retrieved", [[], [_CHUNK]], ids=["no-chunks", "below-threshold"])
def test_anthropic_generator_blank_check_precedes_the_refusal_paths(
    retrieved: list[RetrievalResult],
) -> None:
    # A blank query is a caller error whatever was retrieved, as in `run`.
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        AnthropicGenerator(client=_FakeAnthropic()).generate("  ", retrieved, threshold=0.99)


def test_anthropic_generator_type_check_still_comes_first() -> None:
    with pytest.raises(ValueError, match=r"^query must be a string; got bytes$"):
        AnthropicGenerator(client=_FakeAnthropic()).generate(b"", [_CHUNK], threshold=0.0)


@pytest.mark.parametrize("name", _NOT_BLANK)
def test_control_anthropic_generator_still_calls_the_api_for_a_non_blank_query(
    name: str,
) -> None:
    client = _FakeAnthropic()
    with pytest.raises(RuntimeError, match="the fake client was called"):
        AnthropicGenerator(client=client).generate(_ROWS[name][0], [_CHUNK], threshold=0.0)
    assert len(client.messages.calls) == 1


# ---------------------------------------------------------------------------
# TemplateGenerator.generate: free, but the same protocol rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _BLANK)
def test_template_generator_refuses_a_blank_query(name: str) -> None:
    with pytest.raises(ValueError, match=r"^query must be non-empty$"):
        TemplateGenerator().generate(_ROWS[name][0], [_CHUNK], threshold=0.0)


@pytest.mark.parametrize("name", sorted(_ALL_NON_STR))
def test_template_generator_refuses_a_non_str_query(name: str) -> None:
    with pytest.raises(ValueError, match=r"^query must be a string; got "):
        TemplateGenerator().generate(_ALL_NON_STR[name], [_CHUNK], threshold=0.0)


@pytest.mark.parametrize("name", _NOT_BLANK)
def test_control_template_generator_answers_a_non_blank_query(name: str) -> None:
    out = TemplateGenerator().generate(_ROWS[name][0], [_CHUNK], threshold=0.0)
    assert isinstance(out, GeneratedAnswer)
