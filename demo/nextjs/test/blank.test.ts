import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { GET } from "../app/api/stream/route";
import { PY_ISSPACE_CODEPOINTS, isBlank } from "../lib/blank";

// The row table `tests/test_blank_query_parity.py` also runs through
// `rag_kit.text.is_blank` (#426): one set of rows, two definitions.
interface Row {
  readonly name: string;
  readonly codepoints: readonly number[];
  readonly blank: boolean;
}
const TABLE = JSON.parse(
  readFileSync(new URL("./blank-rows.json", import.meta.url), "utf8"),
) as { py_isspace_codepoints: number[]; rows: Row[] };

const text = (row: Row): string => String.fromCodePoint(...row.codepoints);

describe("isBlank — the port of rag_kit.text.is_blank", () => {
  it("lists exactly the whitespace codepoints the Python side checks against str.isspace", () => {
    expect([...PY_ISSPACE_CODEPOINTS]).toEqual(TABLE.py_isspace_codepoints);
  });

  it.each(TABLE.rows.map((r) => [r.name, r] as const))("row %s", (_name, row) => {
    expect(isBlank(text(row))).toBe(row.blank);
  });

  it.each(PY_ISSPACE_CODEPOINTS.map((cp) => [cp.toString(16), cp] as const))(
    "whitespace U+%s is blank",
    (_hex, cp) => {
      expect(isBlank(String.fromCodePoint(cp))).toBe(true);
    },
  );

  it("is not JS \\s: the rows where the two disagree are in the table", () => {
    // The plausible wrong port, `/^[\s\p{Cf}]*$/u`, refuses none of these.
    const jsWrong = /^[\s\p{Cf}]*$/u;
    const differ = TABLE.rows.filter((r) => jsWrong.test(text(r)) !== r.blank).map((r) => r.name);
    expect(differ).toEqual(["file-separator", "next-line", "mixed"]);
  });
});

describe("GET /api/stream — a blank q is a 400 before the stream", () => {
  const get = (query: string) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    GET(new Request(`http://localhost/api/stream?${query}`) as any);

  it.each([
    ["missing", "delay=0"],
    ["empty", "q=&delay=0"],
    ...TABLE.rows
      .filter((r) => r.blank && r.codepoints.length > 0)
      .map((r) => [r.name, `q=${encodeURIComponent(text(r))}&delay=0`]),
  ])("%s q -> 400 missing q", async (_name, query) => {
    const res = await get(query);
    expect(res.status).toBe(400);
    expect(res.headers.get("content-type")).toMatch(/^text\/plain/);
    expect(await res.text()).toBe("missing q");
  });

  it.each(
    TABLE.rows
      .filter((r) => !r.blank && !r.name.includes("surrogate"))
      .map((r) => [r.name, `q=${encodeURIComponent(text(r))}&delay=0`]),
  )("non-blank %s q still streams", async (_name, query) => {
    const res = await get(query);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("text/event-stream");
    const blob = await res.text();
    expect(blob.trimEnd().split("\n\n").at(-1)).toMatch(/^event: done\n/);
  });
});
