"""
audit.py — append-only audit log (CLAUDE.md cross-cutting requirement).

Every question answered by the assistant is logged with its answer, the sources
used, and a timestamp, so usage is traceable for governance / RBI review. Stored
in the same SQLite file as the metadata, in its own table.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from config import METADATA_DB

_SCHEMA = """
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


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(METADATA_DB)
    conn.execute(_SCHEMA)
    return conn


def log(surface: str, question: str, result: dict, scope_doc: str | None = None) -> None:
    """Record one Q&A. `result` is the dict returned by query.answer()."""
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
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO audit_log "
            "(ts_utc, surface, question, answer, abstained, sources, scope_doc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                surface,
                question,
                result.get("answer", ""),
                int(bool(result.get("abstained"))),
                json.dumps(src, ensure_ascii=False),
                scope_doc,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def recent(limit: int = 50) -> list[dict]:
    """Most recent audit rows (for an admin view later)."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
