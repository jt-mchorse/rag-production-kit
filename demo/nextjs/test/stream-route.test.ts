import { describe, expect, it } from "vitest";

import { GET } from "../app/api/stream/route";

async function readAllText(res: Response): Promise<string> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let out = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    out += decoder.decode(value, { stream: true });
  }
  out += decoder.decode();
  return out;
}

interface ParsedFrame {
  readonly event: string;
  readonly payload: Record<string, unknown>;
  readonly elapsed_ms: number;
}

function parseSSE(blob: string): ParsedFrame[] {
  return blob
    .split("\n\n")
    .filter((s) => s.trim().length > 0)
    .map((frame) => {
      let event = "";
      let dataLine = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice("event: ".length);
        else if (line.startsWith("data: ")) dataLine = line.slice("data: ".length);
      }
      const parsed = JSON.parse(dataLine);
      return {
        event,
        payload: parsed.payload as Record<string, unknown>,
        elapsed_ms: parsed.elapsed_ms as number,
      };
    });
}

function makeReq(q: string): Request {
  // `delay=0` strips the hand-tuned UI delays so the suite is fast.
  return new Request(
    `http://localhost/api/stream?q=${encodeURIComponent(q)}&delay=0`,
  );
}

describe("GET /api/stream — protocol shape", () => {
  it("emits the seven canonical phase events plus token frames", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await GET(makeReq("postgres reranking") as any);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("text/event-stream");

    const blob = await readAllText(res);
    const frames = parseSSE(blob);

    const eventNames = frames.map((f) => f.event);
    expect(eventNames).toContain("retrieving");
    expect(eventNames).toContain("retrieved");
    expect(eventNames).toContain("reranking");
    expect(eventNames).toContain("reranked");
    expect(eventNames).toContain("generating");
    expect(eventNames).toContain("generated");
    expect(eventNames).toContain("done");

    // Phase events appear in protocol order.
    const order = [
      "retrieving",
      "retrieved",
      "reranking",
      "reranked",
      "generating",
      "generated",
      "done",
    ];
    let cursor = 0;
    for (const f of frames) {
      const idx = order.indexOf(f.event);
      if (idx === -1) continue; // token / error / etc.
      expect(idx).toBeGreaterThanOrEqual(cursor);
      cursor = idx;
    }

    // `done` is the last frame.
    expect(frames[frames.length - 1].event).toBe("done");
  });

  it("retrieved payload carries chunks with 1-indexed citation_index", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await GET(makeReq("hnsw rerank") as any);
    const frames = parseSSE(await readAllText(res));
    const retrieved = frames.find((f) => f.event === "retrieved")!;
    const chunks = retrieved.payload.chunks as Array<{
      citation_index: number;
      id: string;
      text: string;
      score: number;
    }>;
    expect(chunks.length).toBeGreaterThan(0);
    chunks.forEach((c, i) => {
      expect(c.citation_index).toBe(i + 1);
      expect(typeof c.id).toBe("string");
      expect(typeof c.text).toBe("string");
      expect(typeof c.score).toBe("number");
    });
  });

  it("citation tokens reference valid chunk indices", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await GET(makeReq("citations rrf") as any);
    const frames = parseSSE(await readAllText(res));
    const retrieved = frames.find((f) => f.event === "retrieved")!;
    const chunkCount = (retrieved.payload.chunks as unknown[]).length;
    const citations = frames
      .filter((f) => f.event === "token" && f.payload.is_citation === true)
      .map((f) => f.payload.citation_index as number);
    expect(citations.length).toBeGreaterThan(0);
    for (const ci of citations) {
      expect(ci).toBeGreaterThanOrEqual(1);
      expect(ci).toBeLessThanOrEqual(chunkCount);
    }
  });

  it("non-citation token frames reassemble to a non-empty string", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await GET(makeReq("streaming") as any);
    const frames = parseSSE(await readAllText(res));
    const tokenTexts = frames
      .filter((f) => f.event === "token")
      .map((f) => String(f.payload.text));
    const joined = tokenTexts.join("");
    expect(joined.length).toBeGreaterThan(10);
  });

  it("done payload carries a total_ms integer", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await GET(makeReq("hello") as any);
    const frames = parseSSE(await readAllText(res));
    const done = frames[frames.length - 1];
    expect(done.event).toBe("done");
    expect(typeof done.payload.total_ms).toBe("number");
    expect(Number.isInteger(done.payload.total_ms)).toBe(true);
  });
});

describe("GET /api/stream — edge cases", () => {
  it("returns chunks even when the query has zero overlap", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await GET(makeReq("zzz-unrelated-xxx") as any);
    const frames = parseSSE(await readAllText(res));
    const retrieved = frames.find((f) => f.event === "retrieved")!;
    expect((retrieved.payload.chunks as unknown[]).length).toBeGreaterThan(0);
  });

  it("treats a missing q as empty and still completes", async () => {
    const req = new Request("http://localhost/api/stream?delay=0");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await GET(req as any);
    const frames = parseSSE(await readAllText(res));
    expect(frames[frames.length - 1].event).toBe("done");
  });
});
