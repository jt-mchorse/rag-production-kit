"use client";

import { useCallback, useRef, useState } from "react";

interface Chunk {
  readonly citation_index: number;
  readonly id: string;
  readonly text: string;
  readonly score: number;
}

type Phase = "retrieving" | "retrieved" | "reranking" | "reranked" | "generating" | "generated" | "done";

const PHASES: Array<{ key: Phase; label: string }> = [
  { key: "retrieving", label: "retrieve" },
  { key: "reranking", label: "rerank" },
  { key: "generating", label: "generate" },
  { key: "done", label: "done" },
];

type AnswerPart =
  | { kind: "text"; text: string }
  | { kind: "citation"; index: number };

const DEFAULT_QUERY = "postgres tuning and reranking";

export function DemoClient() {
  const [q, setQ] = useState(DEFAULT_QUERY);
  const [running, setRunning] = useState(false);
  const [reachedPhase, setReachedPhase] = useState<Phase | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [answer, setAnswer] = useState<AnswerPart[]>([]);
  const [hoverCitation, setHoverCitation] = useState<number | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (running) return;
      setRunning(true);
      setReachedPhase(null);
      setChunks([]);
      setAnswer([]);
      setTotalMs(null);

      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const resp = await fetch(`/api/stream?q=${encodeURIComponent(q)}`, {
          signal: controller.signal,
        });
        if (!resp.ok || !resp.body) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) handleFrame(frame);
        }
      } catch (err) {
        // Surface as text inline; this is a demo, not a production handler.
        // eslint-disable-next-line no-console
        console.error(err);
      } finally {
        setRunning(false);
        abortRef.current = null;
      }
    },
    [q, running],
  );

  function handleFrame(raw: string): void {
    let evName: string | null = null;
    let dataLine = "";
    for (const line of raw.split("\n")) {
      if (line.startsWith("event: ")) evName = line.slice("event: ".length);
      else if (line.startsWith("data: ")) dataLine = line.slice("data: ".length);
    }
    if (!evName || !dataLine) return;
    let frame: { payload: unknown; elapsed_ms?: number };
    try {
      frame = JSON.parse(dataLine);
    } catch {
      return;
    }
    const payload = (frame.payload ?? {}) as Record<string, unknown>;

    if (
      evName === "retrieving" ||
      evName === "retrieved" ||
      evName === "reranking" ||
      evName === "reranked" ||
      evName === "generating" ||
      evName === "generated" ||
      evName === "done"
    ) {
      setReachedPhase(evName);
    }

    if (evName === "retrieved" && Array.isArray(payload.chunks)) {
      setChunks(payload.chunks as Chunk[]);
    }
    if (evName === "token") {
      const text = String(payload.text ?? "");
      const isCite = Boolean(payload.is_citation);
      const idx = payload.citation_index;
      setAnswer((prev) => {
        if (isCite && typeof idx === "number") {
          return [...prev, { kind: "citation", index: idx }];
        }
        // Merge consecutive text deltas so React doesn't churn over many tiny nodes.
        const last = prev[prev.length - 1];
        if (last && last.kind === "text") {
          return [...prev.slice(0, -1), { kind: "text", text: last.text + text }];
        }
        return [...prev, { kind: "text", text }];
      });
    }
    if (evName === "done" && typeof payload.total_ms === "number") {
      setTotalMs(payload.total_ms);
    }
  }

  return (
    <>
      <form className="q" onSubmit={onSubmit}>
        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask about the retrieved chunks…"
          disabled={running}
          aria-label="Query"
        />
        <button type="submit" disabled={running}>
          {running ? "running…" : "ask"}
        </button>
      </form>

      <div className="phases" aria-label="Pipeline phases">
        {PHASES.map(({ key, label }) => {
          const state = phaseState(key, reachedPhase);
          return (
            <span key={key} className={`phase-pill ${state}`}>
              <span className="dot" />
              {label}
              {key === "done" && totalMs !== null ? (
                <span style={{ marginLeft: 4 }}>· {totalMs} ms</span>
              ) : null}
            </span>
          );
        })}
      </div>

      <div className="grid">
        <section className="col-card">
          <h2>Answer</h2>
          <div className="answer">
            {answer.length === 0 && !running ? (
              <span style={{ color: "var(--muted)" }}>
                Ask a question. The retrieval, rerank, and generate phases stream
                in as cards above; the answer renders inline below with footnote
                citations linking to the chunks panel on the right.
              </span>
            ) : null}
            {answer.map((part, i) => {
              if (part.kind === "text") return <span key={i}>{part.text}</span>;
              const isHover = hoverCitation === part.index;
              return (
                <a
                  key={i}
                  href={`#chunk-${part.index}`}
                  className={`cite ${isHover ? "peer-hover" : ""}`}
                  onMouseEnter={() => setHoverCitation(part.index)}
                  onMouseLeave={() => setHoverCitation(null)}
                  onClick={(e) => {
                    e.preventDefault();
                    const el = document.getElementById(`chunk-${part.index}`);
                    if (el) {
                      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
                      setHoverCitation(part.index);
                      window.setTimeout(() => setHoverCitation(null), 1200);
                    }
                  }}
                  title={`See chunk #${part.index}`}
                >
                  {part.index}
                </a>
              );
            })}
            {running && reachedPhase !== "done" ? (
              <span className="cursor" />
            ) : null}
          </div>
        </section>

        <section className="col-card">
          <h2>Retrieved chunks · {chunks.length}</h2>
          <div className="chunks">
            {chunks.length === 0 ? (
              <span style={{ color: "var(--muted)", fontSize: 13 }}>
                Chunks will appear here after the retrieve phase.
              </span>
            ) : (
              chunks.map((c) => {
                const isHover = hoverCitation === c.citation_index;
                return (
                  <div
                    key={c.citation_index}
                    id={`chunk-${c.citation_index}`}
                    className={`chunk ${isHover ? "peer-hover" : ""}`}
                    onMouseEnter={() => setHoverCitation(c.citation_index)}
                    onMouseLeave={() => setHoverCitation(null)}
                  >
                    <div className="head">
                      <span className="num">{c.citation_index}</span>
                      <span className="id">
                        {c.id} · score {c.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="body">{c.text}</div>
                  </div>
                );
              })
            )}
          </div>
        </section>
      </div>
    </>
  );
}

function phaseState(key: Phase, reached: Phase | null): "active" | "done" | "" {
  if (reached === null) return "";
  const order: Phase[] = [
    "retrieving",
    "retrieved",
    "reranking",
    "reranked",
    "generating",
    "generated",
    "done",
  ];
  const ri = order.indexOf(reached);
  // Map each pill to the "phase end" event that completes it.
  const completedBy: Record<Phase, Phase> = {
    retrieving: "retrieved",
    retrieved: "retrieved",
    reranking: "reranked",
    reranked: "reranked",
    generating: "generated",
    generated: "generated",
    done: "done",
  };
  const startedBy: Record<Phase, Phase> = {
    retrieving: "retrieving",
    retrieved: "retrieving",
    reranking: "reranking",
    reranked: "reranking",
    generating: "generating",
    generated: "generating",
    done: "done",
  };
  const startIdx = order.indexOf(startedBy[key]);
  const doneIdx = order.indexOf(completedBy[key]);
  if (ri >= doneIdx) return "done";
  if (ri >= startIdx) return "active";
  return "";
}
