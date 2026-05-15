"""Embedder protocol + a deterministic, dependency-free reference impl.

The retrieval path doesn't care which model produced an embedding — it
just needs a `.embed(text) -> list[float]` callable. That's the
`Embedder` Protocol. Downstream consumers plug in Anthropic, Voyage,
Cohere, BGE, etc. — see embedding-model-shootout for comparisons.

`HashEmbedder` is the package's dependency-free reference. It produces
deterministic 64-d vectors from a SHA-256 hash of the input text and
exists so:

  - the package is importable without an API key,
  - tests are hermetic in CI,
  - the schema (infra/postgres/init.sql vector(64)) has a default that
    works out of the box.

It is not a meaningful embedding model. Real-quality retrieval requires
swapping in a real embedder.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol, runtime_checkable

EMBEDDING_DIM = 64
"""Default embedding dimensionality, matched to the SQL schema vector(64)."""


@runtime_checkable
class Embedder(Protocol):
    """Anything with an .embed(text) -> list[float] is an embedder.

    The protocol is intentionally tiny — concrete embedders carry their
    own configuration (model id, batch size, retry policy) and expose
    only the call shape the indexer/retriever need.
    """

    def embed(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


class HashEmbedder:
    """Deterministic SHA-256-derived embedding. Not for production use.

    Properties retained for retrieval semantics:
      - same text → same vector (cache hits, idempotent re-indexing)
      - unit-length output (cosine similarity reduces to dot product)
      - distinct texts → distinct vectors with overwhelming probability
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % 8 != 0:
            # SHA-256 gives 32 bytes; we want clean multiples for the struct expansion.
            raise ValueError(f"dim must be a multiple of 8, got {dim}")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        # Generate enough bytes to fill `dim` floats by re-hashing until full.
        # 1 float = 4 bytes (we read big-endian uint32 → normalize to [-1, 1]).
        needed_bytes = self.dim * 4
        buf = bytearray()
        counter = 0
        seed = text.encode("utf-8")
        while len(buf) < needed_bytes:
            h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            buf.extend(h)
            counter += 1
        floats: list[float] = []
        for i in range(self.dim):
            (u32,) = struct.unpack(">I", bytes(buf[i * 4 : i * 4 + 4]))
            floats.append((u32 / 0xFFFFFFFF) * 2.0 - 1.0)  # → [-1, 1]
        norm = math.sqrt(sum(x * x for x in floats))
        if norm == 0:  # extraordinarily unlikely; keep the invariant clean anyway
            return floats
        return [x / norm for x in floats]
