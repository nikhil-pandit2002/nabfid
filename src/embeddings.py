"""
embeddings.py — the local embedding model, wrapped in one place.

Keeping all embedding calls behind this module means the knowledge base stays
portable (open-source model, runs on CPU, no API) and the model can be swapped
by changing config.EMBED_MODEL only.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, BGE_QUERY_PREFIX


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the model once and reuse it (first call downloads ~130 MB)."""
    return SentenceTransformer(EMBED_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing. Normalized for cosine similarity."""
    vecs = _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user query. BGE wants the retrieval instruction prefix on queries."""
    vec = _model().encode(
        BGE_QUERY_PREFIX + text, normalize_embeddings=True
    )
    return vec.tolist()
