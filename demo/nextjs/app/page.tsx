import { DemoClient } from "@/components/demo-client";

export default function Home() {
  return (
    <main>
      <header className="app-header">
        <h1>rag-production-kit · streaming demo</h1>
        <p className="sub">
          A query box, a streamed answer with inline footnote citations, and the
          retrieved chunks alongside. The retrieval, rerank, and generate phases
          tick across the top in real time; each <span className="cite-inline">[N]</span>{" "}
          chip in the answer links to the matching chunk on the right. The
          backend is the same Server-Sent Events protocol the Python demo
          speaks — the answer text and corpus are deterministic, so the demo
          runs without Postgres, Anthropic, or any external service.
        </p>
      </header>

      <DemoClient />

      <footer className="app-footer">
        Same wire format as <code>demo/streaming/server.py</code>; this is the
        TypeScript client + a Next.js 15 App Router route handler. Source under
        <code> demo/nextjs/</code>.
      </footer>
    </main>
  );
}
