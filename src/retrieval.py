"""
retrieval.py — the retrieval half of the live query pipeline.

    user question
      -> hybrid retrieval (vector + BM25, merged)
      -> rerank (cross-encoder) to the best few
      -> amendment / version awareness (annotate + optional scope filter)
      -> return chunks ready for grounded generation

Each returned item is a dict with the chunk text + its citation/version metadata,
so the generator and the UI can cite circular + date + page without any extra
lookups.
"""

from __future__ import annotations

import pickle
from functools import lru_cache

import chromadb

from config import (
    BM25_PICKLE, CHROMA_COLLECTION, CHROMA_DIR, RERANK_MODEL,
)
from embeddings import embed_query
from build_index import tokenize


# --- Lazy singletons -------------------------------------------------------
@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


@lru_cache(maxsize=1)
def _bm25():
    with BM25_PICKLE.open("rb") as fh:
        return pickle.load(fh)


# Reranker batch size. This is the memory knob that matters: onnxruntime sizes
# its arena from the largest activation tensor it sees, i.e. batch x sequence.
# Measured over the real chunks (40 candidates, ~1.7k chars each):
#
#   batch=8, truncated to 1000 chars : 409 MB, hit@8 26/28, MRR 0.798
#   batch=8, full length             : 2539 MB  (arena blows up, never released)
#   batch=1, full length             : 244 MB, hit@8 28/28, MRR 0.905
#
# So scoring one pair at a time uses the LEAST memory AND scores best, because
# the cross-encoder sees the whole passage instead of a truncated head. It costs
# a few seconds per query, which is dwarfed by the LLM call anyway. That is why
# there is no truncation here: batch=1 is what keeps memory flat.
RERANK_BATCH = 1


@lru_cache(maxsize=1)
def _reranker():
    """The cross-encoder that reorders the candidate pool.

    ONNX (fastembed), not sentence-transformers/torch — same ms-marco-MiniLM-L-6
    weights, but without dragging torch into the process. See embeddings.py: the
    torch stack alone cost ~410 MB and pushed the app past the free host's ~1 GB
    ceiling, which OOM-killed the container mid-query.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=RERANK_MODEL, threads=1)


# --- Retrieval stages ------------------------------------------------------
def _vector_hits(query: str, k: int) -> dict[str, dict]:
    res = _collection().query(query_embeddings=[embed_query(query)], n_results=k)
    hits = {}
    for cid, doc, md in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
        hits[cid] = {"chunk_id": cid, "text": doc, **md}
    return hits


def _bm25_hits(query: str, k: int) -> dict[str, dict]:
    store = _bm25()
    scores = store["bm25"].get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    hits = {}
    for i in order:
        if scores[i] <= 0:
            continue
        cid = store["chunk_ids"][i]
        hits[cid] = {"chunk_id": cid, "text": store["texts"][i], **store["metas"][i]}
    return hits


def retrieve(query: str, *, top_k: int = 8, pool: int = 20,
             scope_doc_ids: set[str] | None = None) -> list[dict]:
    """Full retrieval: hybrid recall -> rerank -> top_k.

    scope_doc_ids: if given, restrict results to those doc_ids (used by the
    document-scoped chat — pass the viewed doc + its amendment chain).

    pool=20 gives the reranker a wide candidate set; top_k=8 feeds enough chunks
    to the LLM that a narrow fact buried in a long direction still lands in
    context (the "right document, wrong chunk" recall problem). Cheap now that
    the reranker is a small/fast model.
    """
    # 1. Hybrid recall: union of vector + keyword candidates.
    merged = {**_vector_hits(query, pool), **_bm25_hits(query, pool)}
    candidates = list(merged.values())

    # Optional scope filter (document-scoped chat).
    if scope_doc_ids is not None:
        candidates = [c for c in candidates if c["doc_id"] in scope_doc_ids]
    if not candidates:
        return []

    # 2. Rerank the pool with a cross-encoder (query, passage) scorer.
    scores = _reranker().rerank(
        query, [c["text"] for c in candidates], batch_size=RERANK_BATCH,
    )
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    return candidates[:top_k]


# --- Amendment / version helpers (read the SQLite metadata store) -----------
def amendments_of(doc_id: str) -> list[str]:
    """doc_ids of amendments whose `amends` points at doc_id."""
    import sqlite3
    from config import METADATA_DB
    conn = sqlite3.connect(METADATA_DB)
    try:
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE amends = ?", (doc_id,)
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def scope_for_document(doc_id: str, doc_type: str, amends: str) -> set[str]:
    """The doc_ids a document-scoped chat should search.

    Viewing a master direction -> it + all its amendments.
    Viewing an amendment       -> it + its parent master.
    """
    scope = {doc_id}
    if doc_type == "amendment" and amends:
        scope.add(amends)
    else:
        scope.update(amendments_of(doc_id))
    return scope
