"""
docstore.py — read-side helpers over the SQLite metadata store.

Powers Browse-by-division and the Document explanation view: list divisions,
list documents, fetch one document, and compute a human-readable status
(in force / amended / superseded) from the amendment chain.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from config import METADATA_DB


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _amend_map() -> dict[str, list[str]]:
    """parent doc_id -> [amendment doc_ids that amend it]."""
    out: dict[str, list[str]] = {}
    for r in _rows("SELECT doc_id, amends FROM documents WHERE amends != ''"):
        out.setdefault(r["amends"], []).append(r["doc_id"])
    return out


def divisions() -> list[dict]:
    """Each division with its document count, alphabetical."""
    return _rows(
        "SELECT division, COUNT(*) AS n FROM documents "
        "GROUP BY division ORDER BY division"
    )


def documents_in(division: str) -> list[dict]:
    """Documents in a division, master directions first then by date."""
    docs = _rows(
        "SELECT * FROM documents WHERE division = ? "
        "ORDER BY (doc_type != 'master_direction'), issue_date",
        (division,),
    )
    for d in docs:
        d["status"] = status_of(d)
    return docs


def get_document(doc_id: str) -> dict | None:
    rows = _rows("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    if not rows:
        return None
    doc = rows[0]
    doc["status"] = status_of(doc)
    return doc


def amendments_of(doc_id: str) -> list[dict]:
    """Full rows of amendments that amend doc_id, by issue date."""
    ids = _amend_map().get(doc_id, [])
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    return _rows(
        f"SELECT * FROM documents WHERE doc_id IN ({placeholders}) ORDER BY issue_date",
        tuple(ids),
    )


def status_of(doc: dict) -> str:
    """Human-readable status for the Browse list."""
    if doc["doc_type"] == "amendment":
        eff = doc.get("applicable_from") or doc.get("issue_date") or "?"
        return f"Amendment · effective {eff}"
    n = len(_amend_map().get(doc["doc_id"], []))
    if n:
        return f"In force · amended ({n})"
    return "In force"
