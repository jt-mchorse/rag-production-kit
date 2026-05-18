// In-memory corpus + deterministic retriever for the Next.js demo.
//
// The corpus mirrors the one in `demo/streaming/server.py`'s Python
// demo so the two demos teach the same lesson with the same vocabulary.
// Real production wires `Retriever(conn, embedder)` from the Python
// kit; here we hard-code a small set so the demo runs offline.

export interface Chunk {
  readonly id: string;
  readonly text: string;
}

export const CORPUS: ReadonlyArray<Chunk> = [
  {
    id: "pg-tuning",
    text:
      "Postgres tuning for production: shared_buffers should be ~25% of RAM, work_mem sized per parallel query, effective_cache_size set high to encourage index plans.",
  },
  {
    id: "rrf",
    text:
      "Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) combines lexical and dense retrieval channels by summing 1/(rank + k) across each channel's per-document rank.",
  },
  {
    id: "hnsw",
    text:
      "HNSW index parameters: M controls graph degree (higher = better recall, more memory). ef_search at query time trades latency for recall; ef_construction controls build quality.",
  },
  {
    id: "rerank",
    text:
      "Cross-encoder reranking after first-stage retrieval improves NDCG@5 by 0.05–0.12 on technical-docs corpora, at the cost of ~80 ms per query for a small reranker.",
  },
  {
    id: "cite",
    text:
      "Citation enforcement is the load-bearing posture for grounded RAG: every assertion in the answer must reference a chunk id, and the model refuses when context is weak.",
  },
  {
    id: "stream",
    text:
      "Server-Sent Events let the browser render intermediate pipeline phases — retrieving, reranking, generating — so a user sees progress before the first answer token lands.",
  },
];

export interface RetrievalResult {
  readonly chunk: Chunk;
  readonly score: number;
}

/**
 * Deterministic retriever. Ranks chunks by lexical overlap of
 * normalized query tokens against chunk tokens; ties broken by
 * chunk-id alphabetical. Returns top-`k` results (default 4).
 */
export function retrieve(query: string, k: number = 4): RetrievalResult[] {
  const qTokens = normalize(query);
  const scored = CORPUS.map((chunk) => {
    const cTokens = normalize(chunk.text);
    const qset = new Set(qTokens);
    let overlap = 0;
    for (const t of cTokens) if (qset.has(t)) overlap += 1;
    const score = qset.size === 0 ? 0 : overlap / qset.size;
    return { chunk, score };
  });
  scored.sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score;
    return a.chunk.id < b.chunk.id ? -1 : 1;
  });
  // If no overlap at all, return the first k chunks in id order so the
  // demo always has something to render and citations always resolve.
  if (scored.every((r) => r.score === 0)) {
    return CORPUS.slice(0, k).map((chunk, i) => ({
      chunk,
      score: 1 / (i + 1),
    }));
  }
  return scored.slice(0, k);
}

function normalize(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s_-]+/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}
