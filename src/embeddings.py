"""
embeddings.py — the local embedding model, wrapped in one place.

Keeping all embedding calls behind this module means the knowledge base stays
portable (open-source model, runs on CPU, no API) and the model can be swapped
by changing config.EMBED_MODEL only.

Runtime: ONNX (fastembed), not PyTorch. Same model — BAAI/bge-small-en-v1.5,
same weights, same 384-dim vectors — but torch + sentence-transformers cost
~410 MB of RSS before a single model even loads, which pushed the whole app past
the ~1 GB memory ceiling of free hosts; the container was OOM-killed mid-query
and the app appeared to hang forever. onnxruntime does the same maths in a
fraction of the memory. It is also strictly more portable for the on-premise
move later: ONNX runs anywhere, with no torch to install.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from config import EMBED_MODEL, BGE_QUERY_PREFIX, MODELS_DIR


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Load the model once and reuse it.

    cache_dir=MODELS_DIR: the weights ship inside the repo, so this loads from
    disk and never downloads (see config.MODELS_DIR).
    threads=1: the free host has 1-2 vCPUs, so extra onnxruntime worker threads
    buy no speed but each keeps its own memory arena.
    """
    return TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(MODELS_DIR),
                         threads=1)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing. Normalized for cosine similarity."""
    # fastembed streams vectors in input order, already L2-normalized for BGE.
    return [v.tolist() for v in _model().embed(texts, batch_size=32)]


def embed_query(text: str) -> list[float]:
    """Embed a user query. BGE wants the retrieval instruction prefix on queries.

    The prefix is prepended here rather than using fastembed's query_embed() so
    the query side stays exactly what it was under sentence-transformers:
    passages are embedded with no prefix, only the query carries one.
    """
    return next(iter(_model().embed([BGE_QUERY_PREFIX + text]))).tolist()
