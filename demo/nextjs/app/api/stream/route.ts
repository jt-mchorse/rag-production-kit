import type { NextRequest } from "next/server";

import { retrieve } from "@/lib/corpus";
import { streamAnswer } from "@/lib/streamer";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/stream?q=<query>
 *
 * Emits Server-Sent Events that mirror the protocol the Python demo
 * (`demo/streaming/server.py`) already speaks. Phase event types:
 *
 *   event: retrieving        — phase start
 *   event: retrieved         — phase end, payload includes chunks + ms
 *   event: reranking         — phase start (always emitted in the demo)
 *   event: reranked          — phase end, payload includes ms
 *   event: generating        — phase start
 *   event: token             — per-token delta with optional citation
 *   event: generated         — full answer + phase ms
 *   event: done              — total elapsed ms
 *
 * The demo's phases sleep deterministically (small fixed delays) so a
 * visitor sees each phase card materialize, then the answer streams in
 * with inline `[N]` citation chips that link to the chunks panel.
 */
export async function GET(req: NextRequest): Promise<Response> {
  const url = new URL(req.url);
  const q = url.searchParams.get("q") ?? "";
  // Per-token streaming delay. Defaults to 12ms (a hand-tuned value that
  // gives a "watching it type" feel without being slow). Tests pass
  // `delay=0` to skip the wait; the demo UI never passes the flag.
  const delayRaw = url.searchParams.get("delay");
  const tokenDelayMs =
    delayRaw !== null && /^\d+$/.test(delayRaw) ? Number.parseInt(delayRaw, 10) : 12;
  const phaseDelay = tokenDelayMs === 0 ? 0 : undefined;

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const t0 = performance.now();
      const send = (eventName: string, payload: unknown) => {
        const data = JSON.stringify({
          payload,
          elapsed_ms: Math.round(performance.now() - t0),
        });
        controller.enqueue(
          encoder.encode(`event: ${eventName}\ndata: ${data}\n\n`),
        );
      };

      try {
        send("retrieving", { query: q, k: 4 });
        if (phaseDelay !== 0) await sleep(60);
        const results = retrieve(q, 4);
        send("retrieved", {
          chunks: results.map((r, i) => ({
            citation_index: i + 1,
            id: r.chunk.id,
            text: r.chunk.text,
            score: Number(r.score.toFixed(4)),
          })),
          n: results.length,
        });

        send("reranking", { n: results.length });
        if (phaseDelay !== 0) await sleep(40);
        send("reranked", { n: results.length });

        send("generating", { n: results.length });
        let fullText = "";
        for await (const ev of streamAnswer(results, tokenDelayMs)) {
          fullText += ev.text;
          send("token", {
            text: ev.text,
            is_citation: ev.is_citation,
            citation_index: ev.citation_index ?? null,
          });
        }
        send("generated", { text: fullText });
        send("done", { total_ms: Math.round(performance.now() - t0) });
      } catch (err) {
        const reason = err instanceof Error ? err.message : String(err);
        send("error", { reason });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
