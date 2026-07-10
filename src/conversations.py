"""
conversations.py — persistent chat history for the Chatbot surface.

Claude-style saved conversations: every message exchange is saved so the user
can leave, come back (even next day / after a restart), reopen a past
conversation from the sidebar, and continue it.

Storage: local JSON files under data/conversations/ by default (one file per
conversation, plus a small _index.json for fast sidebar listing). When
DATABASE_URL is set (a free hosted Postgres, e.g. Neon), conversations are
stored there instead — needed on Heroku/Streamlit Community Cloud, where local
disk writes are lost on every dyno restart/reboot.

A conversation record:
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

from config import DATA_DIR, DATABASE_URL

CONV_DIR = DATA_DIR / "conversations"
_INDEX = CONV_DIR / "_index.json"

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    created_utc  TEXT NOT NULL,
    updated_utc  TEXT NOT NULL,
    messages     JSONB NOT NULL
)
"""


def _conn_pg():
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_PG)
    conn.commit()
    return conn


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
# Postgres backend (used when DATABASE_URL is set)
# --------------------------------------------------------------------------
def _list_all_pg() -> list[dict]:
    conn = _conn_pg()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, updated_utc, messages FROM conversations "
                "ORDER BY updated_utc DESC"
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "updated_utc": r[2],
                "n": sum(1 for m in r[3] if m.get("role") == "user"),
            }
            for r in rows
        ]
    finally:
        conn.close()


def _load_pg(conv_id: str) -> dict | None:
    conn = _conn_pg()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, created_utc, updated_utc, messages "
                "FROM conversations WHERE id = %s",
                (conv_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "title": row[1], "created_utc": row[2],
            "updated_utc": row[3], "messages": row[4],
        }
    finally:
        conn.close()


def _save_pg(conv_id: str, title: str, messages: list[dict]) -> None:
    from psycopg2.extras import Json
    now = _now()
    conn = _conn_pg()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT created_utc FROM conversations WHERE id = %s", (conv_id,)
            )
            row = cur.fetchone()
            created = row[0] if row else now
            cur.execute(
                "INSERT INTO conversations (id, title, created_utc, updated_utc, messages) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, "
                "updated_utc = EXCLUDED.updated_utc, messages = EXCLUDED.messages",
                (
                    conv_id, title, created, now,
                    Json(messages, dumps=lambda o: json.dumps(o, ensure_ascii=False, default=str)),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _delete_pg(conv_id: str) -> None:
    conn = _conn_pg()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Local file backend (dev default, no DATABASE_URL)
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


# --------------------------------------------------------------------------
# Public API — dispatches to Postgres or local files
# --------------------------------------------------------------------------
def list_all() -> list[dict]:
    """All conversations as [{id, title, updated_utc, n}], newest first."""
    if DATABASE_URL:
        return _list_all_pg()
    idx = _load_index()
    items = [{"id": cid, **meta} for cid, meta in idx.items()]
    items.sort(key=lambda x: x.get("updated_utc", ""), reverse=True)
    return items


def load(conv_id: str) -> dict | None:
    if DATABASE_URL:
        return _load_pg(conv_id)
    path = _path(conv_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save(conv_id: str, title: str, messages: list[dict]) -> None:
    """Write the conversation and refresh its index entry."""
    if DATABASE_URL:
        _save_pg(conv_id, title, messages)
        return
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
    if DATABASE_URL:
        _delete_pg(conv_id)
        return
    path = _path(conv_id)
    if path.exists():
        path.unlink()
    idx = _load_index()
    idx.pop(conv_id, None)
    _save_index(idx)
