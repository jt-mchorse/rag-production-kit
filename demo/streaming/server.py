"""Stdlib-only SSE demo server for the streaming pipeline.

What this is: a single-file HTTP server that exposes `/stream?q=...` as
Server-Sent Events, plus `/` serving `index.html` and `/app.js`. The
point is to demonstrate that the streaming layer works end-to-end with
zero web-framework deps — production would wrap `StreamingPipeline` in
FastAPI or Starlette and stream the same frames out a real ASGI server.

The demo uses an in-memory `FakeRetriever` so it runs without Postgres.
Swap in `from rag_kit import Retriever; Retriever(conn, embedder)` and
the rest of the file is unchanged.

Run:
    python -m demo.streaming.server          # listens on :8765
    open http://localhost:8765/

The page wires `EventSource('/stream?q=postgres+tuning')` and renders
each phase event as a card with the wall-clock elapsed time.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# The demo lives inside the repo, so a sibling import works whether the
# package is installed or the user is running from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_kit.reranker import (  # noqa: E402
    Candidate,
    LexicalOverlapReranker,
    ScoredCandidate,
)
from rag_kit.retriever import RetrievalResult  # noqa: E402
from rag_kit.streaming import StreamingPipeline, to_sse  # noqa: E402

_DEMO_DIR = Path(__file__).resolve().parent


# A tiny in-memory corpus so the demo is self-contained. Keep this list
# short and varied enough that the reranker has something to reorder.
_CORPUS: list[tuple[str, str]] = [
    ("pg-tuning", "Postgres tuning for production: shared_buffers, work_mem, effective_cache_size."),
    ("rrf", "Reciprocal rank fusion combines lexical and dense retrieval channels."),
    ("hnsw", "HNSW index parameters: M controls graph degree, ef_search trades latency for recall."),
    ("rerank", "Cross-encoder reranking improves NDCG at the cost of latency."),
    ("cite", "Citation enforcement: every claim should reference a chunk by id."),
    ("stream", "Server-sent events let a browser render intermediate pipeline phases."),
    ("evals", "Faithfulness, recall@k, and answer correctness are the three RAG eval axes."),
    ("cost", "Token cost telemetry: prompt + completion tokens × per-1M-token price."),
]


class FakeRetriever:
    """In-memory retriever for the demo. Sleeps to make the phase event visible."""

    def __init__(self, corpus: list[tuple[str, str]], sleep_ms: float = 60.0) -> None:
        self._corpus = corpus
        self._sleep_ms = sleep_ms

    def search(self, query: str, k: int = 5, *, reranker=None) -> list[RetrievalResult]:
        time.sleep(self._sleep_ms / 1000.0)
        query_terms = {t.lower() for t in query.split() if t}
        scored: list[tuple[str, str, float]] = []
        for ext_id, text in self._corpus:
            overlap = sum(1 for t in text.lower().split() if t.strip(".,") in query_terms)
            scored.append((ext_id, text, float(overlap) + 0.01))
        scored.sort(key=lambda r: r[2], reverse=True)
        out: list[RetrievalResult] = []
        for i, (ext_id, text, sc) in enumerate(scored[:k]):
            out.append(
                RetrievalResult(
                    external_id=ext_id,
                    text=text,
                    metadata={},
                    fused_score=sc,
                    ranks={"lexical": i + 1, "dense": i + 1},
                )
            )
        return out


class SlowReranker:
    """Wraps LexicalOverlapReranker with a small sleep so the rerank phase is observable in the demo."""

    def __init__(self, sleep_ms: float = 80.0) -> None:
        self._inner = LexicalOverlapReranker(length_penalty=0.0)
        self._sleep_ms = sleep_ms

    def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> list[ScoredCandidate]:
        time.sleep(self._sleep_ms / 1000.0)
        return self._inner.rerank(query, candidates)


def _stub_token_stream(query: str, retrieved: Sequence[RetrievalResult]) -> Iterable[str]:
    # Tokenizes a short response using the top retrieved chunk's text so
    # the demo "answer" is grounded in something visible above it. Real
    # token streams come from the issue-#4 generator once it lands.
    if not retrieved:
        text = "I don't have context to answer that."
    else:
        text = f"Based on the top chunk ({retrieved[0].external_id}): {retrieved[0].text}"
    for tok in text.split(" "):
        time.sleep(0.025)
        yield tok + " "


class Handler(BaseHTTPRequestHandler):
    # Quiet down the default per-request logging — the demo is interactive.
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_file(_DEMO_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._serve_file(_DEMO_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if parsed.path == "/stream":
            self._serve_stream(parse_qs(parsed.query))
            return
        self.send_error(404)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self, qs: dict[str, list[str]]) -> None:
        query = (qs.get("q") or [""])[0]
        if not query:
            self.send_error(400, "missing q")
            return
        try:
            k = int((qs.get("k") or ["3"])[0])
        except ValueError:
            k = 3

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        # Disable proxy / browser buffering so events arrive frame-by-frame.
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        pipe = StreamingPipeline(
            FakeRetriever(_CORPUS),
            reranker=SlowReranker(),
            token_stream=_stub_token_stream,
        )
        for event in pipe.run(query, k=k):
            frame = to_sse(event).encode("utf-8")
            try:
                self.wfile.write(frame)
                self.wfile.flush()
            except BrokenPipeError:
                # Client disconnected mid-stream — fine, just stop.
                return


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"streaming demo on http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
