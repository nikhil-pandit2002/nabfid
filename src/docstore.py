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


def entities() -> list[dict]:
    """Each regulated entity (AIFI, Commercial Bank) with its document count.

    The corpus holds the SAME RBI topics issued separately for each kind of
    institution, so the entity is the top level of Browse — and the two must
    never be conflated in an answer.
    """
    return _rows(
        "SELECT entity, COUNT(*) AS n FROM documents "
        "WHERE entity != '' GROUP BY entity ORDER BY entity"
    )


def divisions(entity: str | None = None) -> list[dict]:
    """Each division with its document count, alphabetical.

    entity: restrict to one regulated entity. Omit for the whole corpus (the
    same division name can exist under both entities, so callers that show a
    flat list should expect them merged).
    """
    if entity:
        return _rows(
            "SELECT division, COUNT(*) AS n FROM documents "
            "WHERE entity = ? GROUP BY division ORDER BY division",
            (entity,),
        )
    return _rows(
        "SELECT division, COUNT(*) AS n FROM documents "
        "GROUP BY division ORDER BY division"
    )


def documents_in(division: str, entity: str | None = None) -> list[dict]:
    """Documents in a division, master directions first then by date."""
    if entity:
        docs = _rows(
            "SELECT * FROM documents WHERE division = ? AND entity = ? "
            "ORDER BY (doc_type != 'master_direction'), issue_date",
            (division, entity),
        )
    else:
        docs = _rows(
            "SELECT * FROM documents WHERE division = ? "
            "ORDER BY (doc_type != 'master_direction'), issue_date",
            (division,),
        )
    for d in docs:
        d["status"] = status_of(d)
    return docs


ENTITY_ALL_RES = "All Regulated Entities"


def doc_ids_for_entity(entity: str) -> set[str]:
    """All doc_ids in scope for one entity — used so an AIFI question is never
    answered from Commercial Bank rules.

    Cross-entity directions (those addressed to ALL regulated entities, e.g.
    Trade Relief Measures 2025, which applies to Commercial Banks AND
    All-India Financial Institutions alike) are folded into EVERY scope: they
    genuinely bind NaBFID, so excluding them from the AIFI scope would hide a
    rule that actually applies.
    """
    return {r["doc_id"] for r in
            _rows("SELECT doc_id FROM documents WHERE entity = ? OR entity = ?",
                  (entity, ENTITY_ALL_RES))}


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
