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

# The corpus is split by REGULATED ENTITY: the same RBI topic (capital adequacy,
# credit risk, ...) is issued separately for AIFIs and for Commercial Banks, and
# the two must never be mixed — an AIFI answer must not cite a Commercial Bank
# rule, and an amendment must only ever attach to a parent in its own entity.
# Each source root maps to one entity; folders inside a root are divisions.
#
# doc_id namespacing: AIFI doc_ids are UNPREFIXED and must stay that way — they
# key the 34 hand-reviewed explanation guides, the baked citations and the
# committed vector index. Commercial Bank ids carry the "cb-" prefix, which also
# resolves the 11 real filename collisions between the two corpora (e.g. both
# have "prudential-norms-on-capital-adequacy-amendment-directions-2026").
ENTITY_AIFI = "AIFI"
ENTITY_COMMERCIAL_BANK = "Commercial Bank"
# Some RBI directions are addressed to ALL regulated entities at once — e.g. the
# Trade Relief Measures Directions, 2025 apply to "(i) Commercial Banks ...
# (iv) All-India Financial Institutions". These bind NaBFID directly, so they
# must surface in the AIFI scope as well as the Commercial Bank one; see
# docstore.doc_ids_for_entity, which folds them into every entity scope.
ENTITY_ALL_RES = "All Regulated Entities"

SOURCE_ROOTS = [
    # (entity, folder, doc_id prefix)
    (ENTITY_AIFI, PROJECT_ROOT / "AIFI latest", ""),
    (ENTITY_COMMERCIAL_BANK, PROJECT_ROOT / "commercial bank", "cb-"),
    (ENTITY_ALL_RES, PROJECT_ROOT / "all regulated entities", "re-"),
]
DATA_DIR = PROJECT_ROOT / "data"              # derived data (manifest, chunks, sqlite)
INDEX_DIR = PROJECT_ROOT / "index"            # vector + keyword indexes

# The ONNX embedding + reranker models, shipped WITH the repo rather than
# downloaded at runtime. Free hosts hand you a fresh container on every restart,
# so a runtime download re-runs every time — and HuggingFace throttles anonymous
# downloads from shared cloud IPs to a crawl (observed: stuck at 20%, app never
# started). Shipping them makes startup deterministic and offline-capable, which
# the on-premise NaBFID server will need anyway (no internet to HuggingFace).
MODELS_DIR = PROJECT_ROOT / "models"

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
# (Xenova/... is the ONNX port of the same cross-encoder/ms-marco-MiniLM-L-6-v2
# weights — identical model, run by onnxruntime instead of torch. See
# embeddings.py for why the whole stack moved off torch.)
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# --- LLM adapter (swappable: Gemini now -> local Ollama later) --------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")   # "gemini" | "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Prototype access gate (not real auth; keeps the URL from being open) ----
ACCESS_CODE = os.getenv("ACCESS_CODE", "nabfid")

# --- Persistent storage for audit log + saved chats -------------------------
# Empty by default -> local SQLite/JSON files (data/), fine for on-premise or
# local dev where disk is durable. On free hosts (Heroku dynos, Streamlit
# Community Cloud) the local filesystem is wiped on every restart/reboot, so
# set DATABASE_URL (a free hosted Postgres, e.g. Neon/Supabase) there instead —
# audit.py and conversations.py both switch to it automatically when present.
DATABASE_URL = os.getenv("DATABASE_URL", "")

