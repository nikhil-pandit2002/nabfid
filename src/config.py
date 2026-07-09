"""
config.py — central paths and model names for the RBI Compliance Assistant.

Everything that another module might need to locate or configure lives here, so
switching a model or moving a directory is a one-line change, not a hunt through
the codebase (CLAUDE.md: swappable model layer, portable code).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Directories -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load secrets/config from the project .env (gitignored). Safe if absent.
load_dotenv(PROJECT_ROOT / ".env")
SOURCE_DIR = PROJECT_ROOT / "AIFI latest"     # source PDFs (read-only)
DATA_DIR = PROJECT_ROOT / "data"              # derived data (manifest, chunks, sqlite)
INDEX_DIR = PROJECT_ROOT / "index"            # vector + keyword indexes

# --- Derived artifacts -----------------------------------------------------
MANIFEST_CSV = DATA_DIR / "manifest.csv"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
PAGES_JSONL = DATA_DIR / "pages.jsonl"        # exact per-page PDF text (citations)
METADATA_DB = DATA_DIR / "metadata.sqlite"    # manifest loaded into SQLite

CHROMA_DIR = INDEX_DIR / "chroma"             # persistent Chroma store
CHROMA_COLLECTION = "rbi_chunks"
BM25_PICKLE = INDEX_DIR / "bm25.pkl"          # keyword index

# --- Embedding model (local, open-source, CPU) -----------------------------
# BGE-small-en-v1.5: strong English retrieval, ~130 MB, fast on CPU. Portable
# to on-premise unchanged (not tied to any API).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE recommends prefixing the QUERY (not the passages) with this instruction
# for short-query -> passage retrieval. Passages are embedded as-is.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Reranker (local cross-encoder, CPU) -----------------------------------
# Scores (query, passage) pairs to reorder the shortlist. We use a SMALL cross-
# encoder (22M params) so reranking stays ~1-2s on CPU. The larger
# bge-reranker-base (278M) was ~1.8s PER pair -> ~58s per query on CPU, which
# made the app unusable. MiniLM-L-6 is the standard fast reranker and keeps
# retrieval quality high. (Swapping the reranker needs no re-indexing — it runs
# at query time only, independent of the stored embeddings.)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- LLM adapter (swappable: Gemini now -> local Ollama later) --------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")   # "gemini" | "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Prototype access gate (not real auth; keeps the URL from being open) ----
ACCESS_CODE = os.getenv("ACCESS_CODE", "nabfid")

