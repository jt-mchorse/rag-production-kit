"""Eval suite for rag-production-kit.

Three metrics over a versioned synthetic golden set:

- `faithfulness`: every generated answer's claim is grounded in a retrieved chunk.
- `recall_at_5`: did the retriever surface the gold chunk in the top-5.
- `correctness`: the fraction of the expected output's content tokens the
  generated answer contains (a deterministic overlap proxy, not a semantic
  judge, #411).

Runs hermetically against an in-memory corpus so CI needs no Postgres
or API key. There is no real-LLM or pgvector backend: every run is this
fixture (#414).
"""
