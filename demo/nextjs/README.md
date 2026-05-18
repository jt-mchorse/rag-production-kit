# Next.js demo frontend

A minimal Next.js 15 + React 19 app that demonstrates the
`rag-production-kit` streaming protocol with inline footnote
citations and a retrieved-chunks panel. Stands alongside
`demo/streaming/server.py` — same wire format, different layer.

## Run

```bash
cd demo/nextjs
npm install
npm run dev          # → http://localhost:3000
```

No Postgres. No Anthropic key. No Python backend. The corpus + answer
streamer are deterministic in-process fixtures so a fresh clone
demonstrates the pattern offline.

## What you see

- A query box at the top.
- A row of pipeline phase pills (`retrieve` · `rerank` · `generate` · `done`)
  that go from neutral → active → done in real time as the SSE stream
  walks through them.
- The streamed answer on the left, with each `[N]` citation rendered as
  a clickable chip. Hover a chip — the matching chunk highlights on the
  right; click — it scrolls into view.
- The retrieved chunks panel on the right, each card numbered to match
  its citation index, with the source chunk id and retrieval score
  visible.

## Protocol

The route handler at [`app/api/stream/route.ts`](app/api/stream/route.ts)
re-emits the SSE event protocol the Python demo (`demo/streaming/server.py`)
already speaks. One frame is `event: <type>\ndata: <json>\n\n` where
`<json>` is `{ "payload": { ... }, "elapsed_ms": <int> }`.

| Event         | When            | Payload                                                |
|---------------|-----------------|--------------------------------------------------------|
| `retrieving`  | phase start     | `{ query, k }`                                         |
| `retrieved`   | phase end       | `{ chunks: [{ citation_index, id, text, score }] }`    |
| `reranking`   | phase start     | `{ n }`                                                |
| `reranked`    | phase end       | `{ n }`                                                |
| `generating`  | phase start     | `{ n }`                                                |
| `token`       | per delta       | `{ text, is_citation, citation_index }`                |
| `generated`   | phase end       | `{ text }`                                             |
| `done`        | pipeline finished | `{ total_ms }`                                       |

Citation tokens are emitted as their own `token` frame with
`is_citation: true` so the client doesn't have to regex on
partially-decoded text.

## Tests

```bash
npm test          # 13 tests · corpus + streamer + SSE protocol shape
npm run typecheck
npm run build
```

The route tests pass `?delay=0` so the suite runs in under a second.

## Swapping in a real backend

The fixture-driven demo is intentional — the protocol is what this
teaches, not the model. To wire a real answer:

1. Replace `lib/streamer.ts`'s `streamAnswer` with an Anthropic SDK
   call that yields one `DeltaEvent` per token-stream chunk; emit
   citation events whenever the model writes `[N]`.
2. Replace `lib/corpus.ts`'s `retrieve` with a call out to a Python
   service running the real `Retriever` from the kit (over HTTP or
   a subprocess), or port the lexical-overlap fixture to a server-side
   pgvector call.
3. The client component (`components/demo-client.tsx`) and the route's
   event sequence do not change.

## Why these decisions

D-016 in [`MEMORY/core_decisions_human.md`](../../MEMORY/core_decisions_human.md)
records: the Next.js demo re-emits the SSE protocol the Python demo
already defined, rather than inventing a parallel wire format. Same
posture as D-005 (one protocol across layers).
