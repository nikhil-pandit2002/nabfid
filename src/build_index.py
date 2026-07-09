"""
build_index.py — Stage 3 of the ingestion pipeline: build the knowledge base.

From data/chunks.jsonl + data/manifest.csv it builds the three stores the query
pipeline reads (CLAUDE.md architecture):

  1. Vector index  (Chroma, index/chroma)      — semantic retrieval
  2. Keyword index (BM25, index/bm25.pkl)       — exact terms / circular numbers
  3. Metadata store (SQLite, data/metadata.sqlite) — dates, divisions, amendment
     chains; powers version filtering + browse-by-division

Re-running rebuilds all three from scratch (idempotent).

Run:  python src/build_index.py
"""

from __future__ import annotations

import csv
import json
import pickle
import re
import sqlite3
import sys

import chromadb
from rank_bm25 import BM25Okapi

from config import (
    BM25_PICKLE, CHROMA_COLLECTION, CHROMA_DIR, CHUNKS_JSONL,
    INDEX_DIR, MANIFEST_CSV, METADATA_DB,
)
from embeddings import embed_passages

# Metadata stored alongside each vector / keyword entry (everything a retrieval
# hit needs for citation + version filtering, minus the chunk text itself).
META_FIELDS = [
    "chunk_id", "doc_id", "division", "doc_type", "title", "circular_no",
    "issue_date", "applicable_from", "consolidated_as_of", "amends",
    "page_start", "page_end", "section_ref", "char_count",
]


def load_chunks() -> list[dict]:
    if not CHUNKS_JSONL.exists():
        sys.exit(f"ERROR: {CHUNKS_JSONL} not found — run build_chunks.py first")
    with CHUNKS_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def tokenize(text: str) -> list[str]:
    """Simple keyword tokenizer for BM25 (lowercased alphanumeric runs)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def meta_of(chunk: dict) -> dict:
    """Chroma/BM25 metadata: only str/int values, no None."""
    return {k: chunk.get(k, "") for k in META_FIELDS}


def build_vector_index(chunks: list[dict]) -> None:
    print(f"[1/3] Vector index: embedding {len(chunks)} chunks with BGE-small...")
    embeddings = embed_passages([c["text"] for c in chunks])

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh collection each run so re-indexing never leaves stale vectors.
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(
        CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    # Add in batches (Chroma caps batch size).
    B = 500
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        coll.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings[i:i + B],
            documents=[c["text"] for c in batch],
            metadatas=[meta_of(c) for c in batch],
        )
    print(f"      -> {coll.count()} vectors in Chroma ({CHROMA_DIR.name}/)")


def build_bm25(chunks: list[dict]) -> None:
    print(f"[2/3] Keyword index: tokenizing {len(chunks)} chunks for BM25...")
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metas": [meta_of(c) for c in chunks],
    }
    with BM25_PICKLE.open("wb") as fh:
        pickle.dump(payload, fh)
    print(f"      -> BM25 index written to {BM25_PICKLE.name}")


def build_sqlite() -> None:
    print("[3/3] Metadata store: loading manifest into SQLite...")
    with MANIFEST_CSV.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    cols = list(rows[0].keys())

    conn = sqlite3.connect(METADATA_DB)
    try:
        conn.execute("DROP TABLE IF EXISTS documents")
        col_defs = ", ".join(
            f'"{c}" {"INTEGER" if c == "page_count" else "TEXT"}' for c in cols
        )
        conn.execute(f"CREATE TABLE documents ({col_defs})")
        placeholders = ", ".join("?" for _ in cols)
        conn.executemany(
            f'INSERT INTO documents VALUES ({placeholders})',
            [[r[c] for c in cols] for r in rows],
        )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()
    print(f"      -> {n} documents in {METADATA_DB.name} (table: documents)")


def main() -> int:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    build_vector_index(chunks)
    build_bm25(chunks)
    build_sqlite()
    print("\nKnowledge base built. Ready for retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
