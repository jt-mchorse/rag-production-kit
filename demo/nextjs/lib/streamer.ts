// Deterministic answer streamer for the Next.js demo.
//
// Given a list of retrieved chunks, weaves a short prose answer that
// references each retrieved chunk by its 1-indexed citation slot. The
// citation tokens look like `[1]` and are emitted as their own chunks
// so the client can detect them mid-stream and render them as chips.
//
// This is *not* a real model call — it's a fixture-driven token
// stream. The protocol is what the demo teaches; the answer text is a
// stand-in. An operator who wants a real Anthropic answer can wire
// the SDK call in here behind `ANTHROPIC_API_KEY`.

import type { RetrievalResult } from "./corpus";

interface AnswerTemplate {
  readonly intro: string;
  readonly perChunk: string;
  readonly outro: string;
}

const TEMPLATES: ReadonlyArray<AnswerTemplate> = [
  {
    intro: "Here is a short answer grounded in the retrieved chunks.",
    perChunk: "The chunk on {topic} says: {snippet} [{n}].",
    outro: "Each statement above is bounded by exactly one chunk's content.",
  },
];

const TOPIC_FROM_ID: Record<string, string> = {
  "pg-tuning": "Postgres tuning",
  rrf: "reciprocal rank fusion",
  hnsw: "HNSW index parameters",
  rerank: "cross-encoder reranking",
  cite: "citation enforcement",
  stream: "server-sent streaming",
};

/**
 * Build the answer text for a given retrieval result list.
 * Citation tokens `[1]`, `[2]`, ... are 1-indexed against the input order.
 */
export function buildAnswer(results: ReadonlyArray<RetrievalResult>): string {
  if (results.length === 0) return "No chunks retrieved; nothing to ground on.";
  const tpl = TEMPLATES[0];
  const perChunkParts = results.map((r, i) => {
    const topic = TOPIC_FROM_ID[r.chunk.id] ?? r.chunk.id;
    const snippet = firstSentence(r.chunk.text);
    return tpl.perChunk
      .replace("{topic}", topic)
      .replace("{snippet}", snippet)
      .replace("{n}", String(i + 1));
  });
  return `${tpl.intro}\n\n${perChunkParts.join(" ")}\n\n${tpl.outro}`;
}

function firstSentence(s: string): string {
  // Trim to roughly one sentence so the answer is readable.
  const idx = s.indexOf(".");
  if (idx === -1) return s.trim();
  return s.slice(0, idx + 1).trim();
}

/**
 * Tokenize the answer into stream-ready chunks. Citation tokens
 * (`[1]`, `[2]`, ...) are emitted as their own chunk so the client
 * detects them with a simple equality check, not a regex on
 * partially-decoded text.
 */
export function tokenize(answer: string): string[] {
  // Split into citation tokens and plain text segments.
  const out: string[] = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(answer)) !== null) {
    if (m.index > last) {
      const slice = answer.slice(last, m.index);
      out.push(...splitTextIntoTokens(slice));
    }
    out.push(m[0]);
    last = m.index + m[0].length;
  }
  if (last < answer.length) {
    out.push(...splitTextIntoTokens(answer.slice(last)));
  }
  return out;
}

function splitTextIntoTokens(s: string): string[] {
  // Word-like tokens, but keep adjacent whitespace attached to each word
  // so reassembly is `tokens.join("")` and not `tokens.join(" ")`.
  return s.split(/(\s+)/).filter((t) => t.length > 0);
}

export interface DeltaEvent {
  readonly kind: "delta";
  readonly text: string;
  readonly is_citation: boolean;
  readonly citation_index?: number;
}

export async function* streamAnswer(
  results: ReadonlyArray<RetrievalResult>,
  delayMs: number = 12,
): AsyncGenerator<DeltaEvent, void, unknown> {
  const answer = buildAnswer(results);
  for (const tok of tokenize(answer)) {
    const m = tok.match(/^\[(\d+)\]$/);
    if (m) {
      const n = Number.parseInt(m[1], 10);
      yield {
        kind: "delta",
        text: tok,
        is_citation: true,
        citation_index: n,
      };
    } else {
      yield { kind: "delta", text: tok, is_citation: false };
    }
    if (delayMs > 0) await sleep(delayMs);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
