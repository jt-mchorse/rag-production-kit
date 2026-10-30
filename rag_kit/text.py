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
import sys
import unicodedata

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")

# The sentence terminators: ONE set for every site that reads or writes a
# sentence end -- `generator`'s splitter, its template writer's strip and
# tail, and `rewriter`'s then-split and strip (#301). A curated subset of
# Unicode's Sentence_Terminal, one row per script, the same set
# chunking-strategies-lab settled on in csl#240:
#   ASCII  . ! ?          Devanagari  । ॥        Arabic/Urdu  ؟ ۔
#   Ethiopic  ። ፧        Armenian  ։ ՜ ՞      Myanmar  ။    Khmer  ។ ៕
#   CJK  。！？ ｡ (halfwidth)                    general  … ‼ ⁇ ⁈ ⁉
# The splitter knew only `.!?…。！？؟`, so a Hindi, Urdu, Amharic, Armenian,
# Burmese or Khmer answer was ONE sentence and an uncited claim passed on its
# neighbour's marker. The Greek question mark (U+037E) is left out: it is
# canonically `;` (NFC maps it there), and a semicolon ends no sentence in any
# other script, so the two cannot be told apart.
SENTENCE_TERMINATORS = ".!?…。！？｡؟۔।॥።፧։՜՞။។៕‼⁇⁈⁉"

#: Closing punctuation that may follow a terminator inside the sentence it ends
#: (`."`, `.)`, `."**`, `。」`), as the generator's splitter reads it (#161, #262).
#: One set for every split site, so the rewriter's Then-split cannot drift from
#: the generator again (#320).
SENTENCE_CLOSERS = "\"'”’»›)]}*_`~」』）】〕〗〙〛〉》＂＇］｝"

#: Every format character (category ``Cf``: U+200B, U+FEFF, U+2060, U+00AD, ...),
#: the non-whitespace half of `is_blank`'s rule. Enumerated from the
#: `unicodedata` catalog rather than hand-listed, because Python's `re` has no
#: `\\p{Cf}` and a list written from memory misses members. The splitters read
#: one beside a terminator, closer or gap as they read nothing at all (#431).
FORMAT_CHARACTERS = "".join(
    ch for ch in map(chr, range(sys.maxunicode + 1)) if unicodedata.category(ch) == "Cf"
)


def is_blank(text: str) -> bool:
    """True when ``text`` has nothing to search for: empty, or only blank characters.

    The one definition behind every "query must be non-empty" check (#424). A
    blank character is whitespace by `str.isspace` (the set `str.strip()` and
    `re`'s `\\s` also use: the same 29 codepoints, U+00A0 and U+3000 among
    them) or a format character, category ``Cf`` (U+200B, U+FEFF, U+2060 and
    167 more), which `str.strip()` leaves alone. None of those 199 codepoints
    yields a `word_tokens` token, so a blank query is one the lexical channel
    cannot match and whose embedding encodes nothing: the same request as ``""``.
    """
    return all(ch.isspace() or unicodedata.category(ch) == "Cf" for ch in text)


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
