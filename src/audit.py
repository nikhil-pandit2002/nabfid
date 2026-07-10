"""
audit.py — append-only audit log (CLAUDE.md cross-cutting requirement).

Every question answered by the assistant is logged with its answer, the sources
used, and a timestamp, so usage is traceable for governance / RBI review.

Storage: local SQLite (data/metadata.sqlite) by default. When DATABASE_URL is
set (a free hosted Postgres, e.g. Neon), audit rows go there instead — needed
on Heroku/Streamlit Community Cloud, where the local filesystem is wiped on
every dyno restart/reboot and would otherwise silently lose the audit trail.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from config import DATABASE_URL, METADATA_DB

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc     TEXT NOT NULL,
    surface    TEXT NOT NULL,   -- 'chatbot' | 'document_chat'
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    abstained  INTEGER NOT NULL,
    sources    TEXT NOT NULL,   -- JSON: [{circular_no, issue_date, page_start, division}]
    scope_doc  TEXT             -- doc_id when the chat was document-scoped
)
"""

_SCHEMA_PG = _SCHEMA_SQLITE.replace(
    "id         INTEGER PRIMARY KEY AUTOINCREMENT,", "id         SERIAL PRIMARY KEY,"
)


def _sources_json(result: dict) -> str:
    src = [
        {
            "circular_no": c.get("circular_no", ""),
            "issue_date": c.get("issue_date", ""),
            "page_start": c.get("page_start", ""),
            "division": c.get("division", ""),
            # quote-verified exact citation page (see query.answer); verified
            # False = quote not found on any page, page fell back to page_start
            "cite_page": c.get("cite_page", c.get("page_start", "")),
            "cite_verified": bool(c.get("cite_verified")),
        }
        for c in result.get("sources", [])
    ]
    return json.dumps(src, ensure_ascii=False)


def _conn_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(METADATA_DB)
    conn.execute(_SCHEMA_SQLITE)
    return conn


def _conn_pg():
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_PG)
    conn.commit()
    return conn


def log(surface: str, question: str, result: dict, scope_doc: str | None = None) -> None:
    """Record one Q&A. `result` is the dict returned by query.answer()."""
    row = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        surface,
        question,
        result.get("answer", ""),
        int(bool(result.get("abstained"))),
        _sources_json(result),
        scope_doc,
    )
    if DATABASE_URL:
        conn = _conn_pg()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_log "
                    "(ts_utc, surface, question, answer, abstained, sources, scope_doc) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    row,
                )
            conn.commit()
        finally:
            conn.close()
        return
    conn = _conn_sqlite()
    try:
        conn.execute(
            "INSERT INTO audit_log "
            "(ts_utc, surface, question, answer, abstained, sources, scope_doc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        conn.commit()
    finally:
        conn.close()


def recent(limit: int = 50) -> list[dict]:
    """Most recent audit rows (for an admin view later)."""
    if DATABASE_URL:
        conn = _conn_pg()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, ts_utc, surface, question, answer, abstained, "
                    "sources, scope_doc FROM audit_log ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()
    conn = _conn_sqlite()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
