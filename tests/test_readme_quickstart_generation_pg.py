"""The README's Quickstart search, fed to its generation call, answers (#327).

#237 fixed the snippet's `threshold=0.05` and locked the default against
*best-case* fused results -- rank 1 in both channels, built by the test. The
list the README actually hands the generator came from
`retr.search("refund window", k=3)` over its three documents, and none of them
holds both `refund` and `window`: `plainto_tsquery` ANDs its terms, so the
lexical channel returned nothing and every result was dense-only. One channel
tops out at `1/61 = 0.0164`, below the `0.02` default, and the documented call
refused (`top_score=0.0164 below threshold=0.0200`) under a comment promising
`[cite:doc-1]`.

So this runs the README's own code, not a stand-in for its output: the
indexing fence and the generation fence are executed verbatim, and the search
between them is the call the indexing fence makes, with its own arguments.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from rag_kit import GeneratedAnswer, Retriever

_README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def _python_fence(marker: str) -> str:
    blocks = [b for b in re.findall(r"```python\n(.*?)```", _README, re.S) if marker in b]
    assert len(blocks) == 1, f"expected one README python fence containing {marker!r}"
    return blocks[0]


def _search_call_args(source: str) -> tuple[list[Any], dict[str, Any]]:
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "search"
    ]
    assert len(calls) == 1, "expected one `.search(...)` call in the Quickstart fence"
    call = calls[0]
    args = [ast.literal_eval(a) for a in call.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    return args, kwargs


@pytest.mark.pg
def test_the_readme_quickstart_search_answers_at_the_documented_threshold(pg_conn) -> None:
    index_fence = _python_fence("indexer.add_documents(")
    namespace: dict[str, Any] = {}
    exec(compile(index_fence, "README.md:quickstart", "exec"), namespace)

    args, kwargs = _search_call_args(index_fence)
    retrieved = Retriever(pg_conn, namespace["embedder"]).search(*args, **kwargs)
    assert retrieved, "the README's search returned nothing"

    namespace["retrieved"] = retrieved
    exec(compile(_python_fence("gen.generate("), "README.md:generation", "exec"), namespace)
    out = namespace["out"]
    assert isinstance(out, GeneratedAnswer), (
        f"the README's generation call refuses on the README's own search "
        f"{args}{kwargs}: {out} -- ranks {[r.ranks for r in retrieved]}"
    )
    assert any("lexical" in r.ranks for r in retrieved), (
        "no result came from the lexical channel; a dense-only fused score tops "
        "out at 1/(k+1), below the default threshold"
    )


def test_the_readme_one_channel_ceiling_note_holds() -> None:
    """The README's note says a one-channel chunk cannot reach the default and
    a two-channel one can. If the default or `DEFAULT_K` moves, so must it."""
    from rag_kit.fusion import DEFAULT_K
    from rag_kit.generator import _DEFAULT_THRESHOLD

    assert 1 / (DEFAULT_K + 1) < _DEFAULT_THRESHOLD < 2 / (DEFAULT_K + 1)
    assert f"1/{DEFAULT_K + 1} ~= {1 / (DEFAULT_K + 1):.3f}" in _README
