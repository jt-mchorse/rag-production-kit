import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "rag-production-kit — Next.js demo",
  description:
    "Streaming answer with inline citation chips + retrieved-chunks panel.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
