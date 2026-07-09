"""
conversations.py — persistent chat history for the Chatbot surface.

Claude-style saved conversations: every message exchange is written to disk so
the user can leave, come back (even next day / after a restart), reopen a past
conversation from the sidebar, and continue it. One JSON file per conversation
under data/conversations/ (same per-file convention as data/explanations/), plus
a small _index.json so the sidebar can list conversations without reading every
full file (message bodies + their source snippets can get large).

A conversation file:
    {
      "id": "20260707-101530-a1b2c3",
      "title": "What is the minimum CRAR ...",   # from the first user message
      "created_utc": "...", "updated_utc": "...",
      "messages": [ {role, content, sources?}, ... ]   # same shape as the
    }                                                    # in-memory chat history
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from config import DATA_DIR

CONV_DIR = DATA_DIR / "conversations"
_INDEX = CONV_DIR / "_index.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(conv_id: str):
    return CONV_DIR / f"{conv_id}.json"


def new_id() -> str:
    """A sortable, unique conversation id."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def derive_title(messages: list[dict]) -> str:
    """Conversation title = first user message, trimmed (like Claude)."""
    for m in messages:
        if m.get("role") == "user":
            text = " ".join(str(m.get("content", "")).split())
            if text:
                return text[:48] + "…" if len(text) > 48 else text
    return "New chat"


# --------------------------------------------------------------------------
# Lightweight index (id -> {title, updated_utc, n}) for fast sidebar listing.
# --------------------------------------------------------------------------
def _load_index() -> dict:
    if _INDEX.exists():
        try:
            return json.loads(_INDEX.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_index(idx: dict) -> None:
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")


def list_all() -> list[dict]:
    """All conversations as [{id, title, updated_utc, n}], newest first."""
    idx = _load_index()
    items = [{"id": cid, **meta} for cid, meta in idx.items()]
    items.sort(key=lambda x: x.get("updated_utc", ""), reverse=True)
    return items


def load(conv_id: str) -> dict | None:
    path = _path(conv_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save(conv_id: str, title: str, messages: list[dict]) -> None:
    """Write the conversation and refresh its index entry."""
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    now = _now()
    path = _path(conv_id)
    created = now
    if path.exists():
        try:
            created = json.loads(path.read_text(encoding="utf-8")).get("created_utc", now)
        except (json.JSONDecodeError, OSError):
            pass
    data = {
        "id": conv_id,
        "title": title,
        "created_utc": created,
        "updated_utc": now,
        "messages": messages,
    }
    # default=str keeps us safe if a retrieved source carries a stray non-JSON
    # value (e.g. a numpy score) — rendering never uses it anyway.
    path.write_text(
        json.dumps(data, ensure_ascii=False, default=str, indent=1),
        encoding="utf-8",
    )
    idx = _load_index()
    idx[conv_id] = {
        "title": title,
        "updated_utc": now,
        "n": sum(1 for m in messages if m.get("role") == "user"),
    }
    _save_index(idx)


def delete(conv_id: str) -> None:
    path = _path(conv_id)
    if path.exists():
        path.unlink()
    idx = _load_index()
    idx.pop(conv_id, None)
    _save_index(idx)
