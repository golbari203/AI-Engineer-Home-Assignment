"""
Embeddings.

Why a local model
-----------------
Anthropic does not offer an embeddings endpoint, so using a hosted embedder
would mean a second provider, a second key in the README and a second bill.
A small local model removes all three: setup is one pip install, embedding
cost is zero, and the reviewers do not need an account anywhere else.

bge-small-en-v1.5 is 384 dimensions and ~130MB. On a corpus of 421 chunks the
whole encode takes a few seconds on CPU, and query encoding is a few
milliseconds, which is well inside the latency budget for a chat turn.

Storage
-------
Vectors go into MySQL as JSON beside a FULLTEXT index on the same table, so
the two halves of hybrid retrieval live in one store and cannot drift apart.
At 421 chunks, scoring every vector in NumPy takes about a millisecond, so
there is no index to maintain. Past roughly 100k chunks this becomes the wrong
answer and an ANN index is needed; that threshold is far outside this brief.

The offline fallback
--------------------
HashingEmbedder produces deterministic vectors with no model download. It
exists so the test suite and a sandboxed CI run without network access, and so
the pipeline can be exercised end to end before anyone downloads weights. It
has no semantic ability whatsoever: "margin pressure" and "profitability
concerns" are unrelated to it. Never use it to serve answers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384

# bge models are trained with an instruction prefix on the query side only.
# Omitting it costs a few points of retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        ...


class LocalEmbedder:
    """sentence-transformers, normalised so dot product is cosine similarity."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(list(texts), batch_size=self.batch_size,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)


class HashingEmbedder:
    """Deterministic, offline, and semantically useless. For tests only."""

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def get_embedder(offline: bool = False) -> Embedder:
    if offline:
        return HashingEmbedder()
    try:
        return LocalEmbedder()
    except Exception as exc:                             # noqa: BLE001
        raise RuntimeError(
            f"could not load {DEFAULT_MODEL} ({exc}). Install "
            "sentence-transformers and allow the model download, or pass "
            "offline=True for a non-semantic fallback suitable only for tests."
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_chunks(chunks, embedder: Embedder) -> np.ndarray:
    """Embed the prefixed text, so the ticker and period are part of the vector."""
    return embedder.encode([c.text for c in chunks])


def to_json(vector: np.ndarray) -> str:
    return json.dumps([round(float(x), 6) for x in vector])


def from_json(blob: str) -> np.ndarray:
    return np.asarray(json.loads(blob), dtype=np.float32)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Both sides are normalised, so a dot product is the cosine."""
    return matrix @ query


def top_k(query: np.ndarray, matrix: np.ndarray, k: int = 30) -> list[int]:
    scores = cosine_scores(query, matrix)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return list(idx[np.argsort(-scores[idx])])
