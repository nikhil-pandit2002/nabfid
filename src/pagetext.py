"""
pagetext.py — ground-truth per-page PDF text, for verifying citations.

Chunks (chunks.jsonl) are the retrieval unit, but 74% of them span more than
one page, so a chunk's page_start is NOT a reliable citation page. This module
gives every citation checker the same ground truth: the exact text of each
physical page (data/pages.jsonl, built by build_pages.py), plus one shared
normalizer so "₹1,500 crore" in a guide matches "1500" on the page.

Used by:
  * bake_guide_citations.py — offline: anchor-match guide lines to pages.
  * eval_citations.py       — offline: verify every baked citation.
  * app.py / query flow     — runtime: locate the exact page of a verbatim
    quote inside a retrieved chunk (locate_page).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from config import PAGES_JSONL

_WS_RE = re.compile(r"\s+")
_COMMA_NUM_RE = re.compile(r"(?<=\d),(?=\d)")


def norm(text: str) -> str:
    """Normalize text for matching: lowercase, collapse whitespace, join
    comma-grouped digits (1,500 -> 1500), drop currency/percent symbols."""
    t = text.lower()
    t = _COMMA_NUM_RE.sub("", t)
    t = t.replace("₹", " ").replace("%", " ").replace(" ", " ")
    return _WS_RE.sub(" ", t).strip()


@lru_cache(maxsize=1)
def _pages_by_doc() -> dict[str, list[tuple[int, str]]]:
    """doc_id -> ordered [(page_no, normalized_text), ...]."""
    out: dict[str, list[tuple[int, str]]] = {}
    if not PAGES_JSONL.exists():
        return out
    with PAGES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out.setdefault(row["doc_id"], []).append(
                (row["page"], norm(row["text"]))
            )
    for pages in out.values():
        pages.sort(key=lambda p: p[0])
    return out


def pages_for(doc_id: str) -> list[tuple[int, str]]:
    """Ordered (page_no, normalized page text) for a document ([] if absent)."""
    return _pages_by_doc().get(doc_id, [])


def _find(doc_id: str, needle: str,
          page_lo: int | None = None, page_hi: int | None = None) -> int | None:
    """First page (searching a range first if given) whose text contains needle."""
    if not needle:
        return None
    pages = pages_for(doc_id)
    if not pages:
        return None
    in_range = [(p, t) for p, t in pages
                if (page_lo is None or p >= page_lo)
                and (page_hi is None or p <= page_hi)]
    for scope in (in_range, pages):
        for p, t in scope:
            if needle in t:
                return p
    return None


def locate_page(doc_id: str, chunk: dict, quote: str) -> tuple[int, bool]:
    """Exact page of a verbatim quote from a retrieved chunk.

    Searches the chunk's own page span first (the quote should live there),
    then the whole document; if the full quote isn't found (minor extraction
    differences), retries with its longest 8-word window. Returns
    (page, verified): on total miss, falls back to the chunk's page_start
    with verified=False so callers can log it.
    """
    lo, hi = chunk.get("page_start"), chunk.get("page_end")
    q = norm(quote)
    page = _find(doc_id, q, lo, hi)
    if page is None and len(q.split()) > 8:
        words = q.split()
        for start in range(0, len(words) - 7):
            page = _find(doc_id, " ".join(words[start:start + 8]), lo, hi)
            if page is not None:
                break
    if page is not None:
        return page, True
    return (lo or 1), False
