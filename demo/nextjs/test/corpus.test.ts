import { describe, expect, it } from "vitest";

import { CORPUS, retrieve } from "../lib/corpus";
import { buildAnswer, tokenize } from "../lib/streamer";

describe("retrieve", () => {
  it("returns chunks with non-negative scores for a relevant query", () => {
    const r = retrieve("postgres tuning", 4);
    expect(r.length).toBeGreaterThan(0);
    expect(r.length).toBeLessThanOrEqual(4);
    // The top result should be the pg-tuning chunk for that query.
    expect(r[0].chunk.id).toBe("pg-tuning");
    expect(r[0].score).toBeGreaterThan(0);
  });

  it("falls back to a deterministic set when the query has zero overlap", () => {
    const r = retrieve("zzz-unrelated-xxx", 4);
    expect(r.length).toBe(4);
    // Fallback is the first k chunks in corpus order.
    expect(r.map((x) => x.chunk.id)).toEqual(CORPUS.slice(0, 4).map((c) => c.id));
  });

  it("results are stable across repeated calls (determinism)", () => {
    const a = retrieve("rrf hnsw", 4);
    const b = retrieve("rrf hnsw", 4);
    expect(a.map((r) => r.chunk.id)).toEqual(b.map((r) => r.chunk.id));
    expect(a.map((r) => r.score)).toEqual(b.map((r) => r.score));
  });
});

describe("buildAnswer + tokenize", () => {
  it("emits one citation token per retrieved chunk", () => {
    const r = retrieve("citation enforcement rerank", 4);
    const answer = buildAnswer(r);
    const tokens = tokenize(answer);
    const citationTokens = tokens.filter((t) => /^\[\d+\]$/.test(t));
    expect(citationTokens.length).toBe(r.length);
  });

  it("citation tokens are 1..N in order", () => {
    const r = retrieve("anything", 4);
    const tokens = tokenize(buildAnswer(r));
    const citations = tokens
      .filter((t) => /^\[\d+\]$/.test(t))
      .map((t) => Number.parseInt(t.slice(1, -1), 10));
    expect(citations).toEqual(citations.slice().sort((a, b) => a - b));
    expect(citations[0]).toBe(1);
    expect(citations[citations.length - 1]).toBe(r.length);
  });

  it("reassembled tokens equal the original answer string", () => {
    const r = retrieve("postgres rerank hnsw", 4);
    const answer = buildAnswer(r);
    const tokens = tokenize(answer);
    expect(tokens.join("")).toBe(answer);
  });
});
