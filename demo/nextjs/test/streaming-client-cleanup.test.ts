/**
 * Streaming-client cleanup lock (issue #124).
 *
 * The demo client owns an `AbortController`, hands its signal to `fetch`, and
 * reads the SSE response in a `while (true) { reader.read() }` loop. If it does
 * not abort that controller when the component unmounts, navigating away
 * mid-stream leaks the reader (setState on a detached component) and holds the
 * connection open until the server finishes on its own — a browser `fetch` is
 * NOT auto-aborted when its initiating component unmounts.
 *
 * This is a source-level lock in the idiom of this demo's node/vitest suite
 * (no jsdom / testing-library — the tests read committed source and exercise
 * `lib/` logic). It discovers every `components/*.tsx` that owns an
 * `AbortController` and asserts each imports `useEffect` and aborts on unmount,
 * so a future component (or a refactor of this one) can't silently reintroduce
 * the leak. Mirrors the sibling lock in nextjs-streaming-ai-patterns#78.
 */
import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const COMPONENTS_DIR = resolve(__dirname, "..", "components");

// A `useEffect` unmount cleanup that aborts a controller, e.g.
//   return () => abortRef.current?.abort();
// Comments are stripped first so an explanatory comment inside the cleanup
// body can't push `.abort()` outside the proximity window.
const UNMOUNT_ABORT = /return\s*\(\s*\)\s*=>[\s\S]{0,160}?\.abort\s*\(/;

function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/[^\n]*/g, "");
}

function listAbortControllerComponents(): string[] {
  return readdirSync(COMPONENTS_DIR)
    .filter((f) => f.endsWith(".tsx"))
    .filter((f) =>
      readFileSync(resolve(COMPONENTS_DIR, f), "utf-8").includes(
        "new AbortController(",
      ),
    )
    .sort();
}

describe("streaming-client cleanup (#124)", () => {
  const clients = listAbortControllerComponents();

  it("finds the AbortController-owning client component(s) (sanity)", () => {
    expect(
      clients,
      `expected demo-client.tsx among the AbortController-owning components, got ${JSON.stringify(clients)}`,
    ).toContain("demo-client.tsx");
  });

  it.each(listAbortControllerComponents())(
    "%s imports useEffect (required for an unmount cleanup)",
    (filename) => {
      const src = readFileSync(resolve(COMPONENTS_DIR, filename), "utf-8");
      expect(
        /import\s*\{[^}]*\buseEffect\b[^}]*\}\s*from\s*["']react["']/.test(src),
        `${filename} owns an AbortController but does not import useEffect, ` +
          "so it cannot register an unmount cleanup to abort the fetch.",
      ).toBe(true);
    },
  );

  it.each(listAbortControllerComponents())(
    "%s aborts its AbortController on unmount",
    (filename) => {
      const src = stripComments(
        readFileSync(resolve(COMPONENTS_DIR, filename), "utf-8"),
      );
      expect(
        UNMOUNT_ABORT.test(src),
        `${filename} owns an AbortController but has no unmount cleanup that ` +
          "calls .abort(). Navigating away mid-stream would leak the fetch/" +
          "reader (setState on a detached component) and hold the connection " +
          "open. Add `return () => abortRef.current?.abort()` to a useEffect.",
      ).toBe(true);
    },
  );
});
