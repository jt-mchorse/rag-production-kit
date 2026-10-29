/**
 * The demo's port of `rag_kit.text.is_blank` (#426): a query is blank when
 * every character is whitespace by Python's `str.isspace` or a format
 * character (Unicode category Cf). The empty string is blank.
 *
 * The whitespace half is spelled out codepoint by codepoint rather than
 * written as JS `\s`, because the two sets differ: `\s` lacks U+001C-U+001F
 * and U+0085, which `str.isspace` counts, and includes U+FEFF, which Python
 * counts as Cf instead (so it is blank either way). The Cf half is
 * `\p{Cf}`, the same Unicode property Python reads.
 *
 * `test/blank-rows.json` is the row table both definitions are checked
 * against: vitest runs it through `isBlank` and pytest
 * (`tests/test_blank_query_parity.py`) runs it through `is_blank`, and checks
 * `PY_ISSPACE_CODEPOINTS` against the `str.isspace` catalog, so the two
 * cannot drift apart silently.
 */

/** The 29 codepoints Python's `str.isspace` accepts (the same set as `strip()` and `re`'s `\s`). */
export const PY_ISSPACE_CODEPOINTS: readonly number[] = [
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0,
  0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
  0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
];

const PY_ISSPACE = new Set<number>(PY_ISSPACE_CODEPOINTS);
const CF = /^\p{Cf}$/u;

export function isBlank(text: string): boolean {
  // `for...of` walks codepoints, so an astral Cf (U+E0001) is one character,
  // as it is in Python, not two lone surrogates.
  for (const ch of text) {
    if (!PY_ISSPACE.has(ch.codePointAt(0)!) && !CF.test(ch)) return false;
  }
  return true;
}
