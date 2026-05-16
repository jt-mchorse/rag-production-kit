"""Eval suite for rag-production-kit.

Three metrics over a versioned synthetic golden set:

- `faithfulness`: every generated answer's claim is grounded in a retrieved chunk.
- `recall_at_5`: did the retriever surface the gold chunk in the top-5.
- `correctness`: did the generated answer match the expected output (semantic).

Runs hermetically against an in-memory corpus so CI needs no Postgres
or API key. Real-LLM runs are operator-triggered with `RAG_EVAL_GENERATOR=anthropic`.
"""
