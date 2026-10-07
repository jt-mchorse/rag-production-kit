"""Word tokenization shared by the dep-free lexical paths (#285).

The lexical reranker and the hermetic eval retriever each tokenized with an
ASCII-only class (`[A-Za-z0-9]+` and `[a-z0-9]+`). For any non-Latin text that
silently did nothing: a Russian chunk containing every word of the query
`Python ошибка импорта` matched only `python`, and the length penalty then ranked
an unrelated, shorter English chunk first; `café crème` became `caf`, `cr`,
`me`. llm-eval-harness fixed the same class in its drift tokenizer (#108, #314).

ASCII text takes the old pattern, so every ASCII token, and every committed
eval number built from one, is unchanged. Other text is NFC-normalized and split
into runs of alphanumerics plus the combining marks on them (Python's `re` has
no `\\p{M}`, and `str.isalnum()` excludes marks, which is how a Devanagari vowel
sign would otherwise split its word).
"""

from __future__ import annotations

import re
import unicodedata

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")


def word_tokens(text: str) -> list[str]:
    """Lowercased word tokens of ``text``; underscores and punctuation separate."""
    if text.isascii():
        return _ASCII_TOKEN_RE.findall(text.lower())
    out: list[str] = []
    current: list[str] = []
    for ch in unicodedata.normalize("NFC", text).lower():
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out
