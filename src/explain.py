"""
explain.py — content for the Document explanation view.

Produces, for a selected document:
  * key_points()   — the important provisions in plain language, CITED (grounded
                     on the document's own chunks).
  * what_changed()  — for amendments, the version story / diff (grounded on the
                     amendment's explicit change-instructions).
  * implications()  — practical so-what for NaBFID. This is the ONE place general
                     knowledge is allowed, so it is clearly labelled and told not
                     to contradict or add to the regulatory substance.

All regulatory substance stays grounded; only implications() is framing.
"""

from __future__ import annotations

import json
from functools import lru_cache

from config import CHUNKS_JSONL, DATA_DIR
from llm import generate

# Persistent on-disk store for generated explanations. Generating an explanation
# calls the LLM (costs money/tokens), so we do it ONCE and save it here — the app
# then reads the file with no LLM call, and it survives restarts / next day.
EXPLANATIONS_DIR = DATA_DIR / "explanations"


def _saved_path(doc_id: str):
    return EXPLANATIONS_DIR / f"{doc_id}.json"


def load_saved(doc_id: str) -> dict:
    """Return the saved {key_points, what_changed, implications} for a doc, or {}."""
    p = _saved_path(doc_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_section(doc_id: str, kind: str, text: str) -> None:
    """Persist one explanation section to disk (merges with any existing)."""
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_saved(doc_id)
    data[kind] = text
    _saved_path(doc_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

# Map-reduce sizing. A document is split into ordered batches of <= BATCH_CHARS,
# each explained in detail (the "map"), then the parts are concatenated in
# document order (the "reduce" is a plain join — no lossy re-summarisation, so no
# detail is dropped). Most directions are one batch; the 97-page KYC is ~2 and
# the 360-page Capital Adequacy is ~5. MAX_BATCHES caps runaway cost/latency.
BATCH_CHARS = 130000        # ~32k tokens of source per map call
MAX_BATCHES = 6             # covers even the 360-page direction end-to-end


@lru_cache(maxsize=1)
def _chunks_by_doc() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with CHUNKS_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            out.setdefault(c["doc_id"], []).append(c)
    return out


def _render(chunks: list[dict]) -> str:
    """SOURCES block from a list of chunks (page + section tagged)."""
    return "\n\n".join(
        f"[p{c['page_start']} | {c['section_ref'] or '-'}]\n{c['text']}"
        for c in chunks
    )


def _batches(doc_id: str) -> list[list[dict]]:
    """Split a document's chunks into ordered batches under BATCH_CHARS."""
    batches: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for c in _chunks_by_doc().get(doc_id, []):
        if cur and size + c["char_count"] > BATCH_CHARS:
            batches.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += c["char_count"]
    if cur:
        batches.append(cur)
    return batches


def _context(doc_id: str, budget: int = BATCH_CHARS) -> str:
    """First batch of a document's text (used by the short single-call helpers)."""
    return _render(_batches(doc_id)[0]) if _batches(doc_id) else ""


def _cite(doc: dict) -> str:
    return f"{doc['circular_no']}, dated {doc['issue_date']}"


def _is_own_rulebook(doc: dict) -> bool:
    """True when the direction is addressed to NaBFID's own class (AIFIs).

    NaBFID is an AIFI, so the AIFI directions bind it directly. The Commercial
    Banks directions are addressed to banks — BUT they are not simply "someone
    else's rules": the AIFI directions incorporate parts of them by reference,
    which makes those parts binding on an AIFI too. Real examples in this corpus:

      RBI/DOR/2025-26/323 (AIFI Responsible Business Conduct), para on
      disclosures: "An AIFI SHALL provide a Key Fact Statement (KFS), as per
      instructions contained in the RBI (Commercial Banks - Responsible
      Business Conduct) Directions, 2025."

      RBI/DOR/2025-26/321 (AIFI Capital Adequacy), p.41/p.237: "An AIFI shall
      refer to RBI (Commercial Banks - Credit Risk Management) Directions, 2025
      which cover provision on unhedged foreign currency exposures."

    So the guides must not claim a Commercial Banks direction is irrelevant to
    NaBFID, nor that it binds NaBFID wholesale. See _KP_ENTITY_NOTE.

    "All Regulated Entities" counts as NaBFID's own rulebook: those directions
    name All-India Financial Institutions in their own applicability clause
    (e.g. Trade Relief Measures 2025), so they bind NaBFID directly and must be
    explained as such — not framed as somebody else's rules.
    """
    return (doc.get("entity") or "AIFI") in ("AIFI", "All Regulated Entities")


_KP_ENTITY_NOTE = (
    " NOTE ON APPLICABILITY — read carefully, the reader is NaBFID (an AIFI), "
    "and this direction is addressed to COMMERCIAL BANKS. Describe its "
    "requirements as obligations on BANKS (say 'a bank shall...', not 'NaBFID "
    "shall...'). However, do NOT state or imply that it is irrelevant to "
    "NaBFID: the AIFI directions incorporate parts of the Commercial Banks "
    "directions by reference (for example an AIFI must follow the Commercial "
    "Banks instructions for the Key Fact Statement, and must refer to them for "
    "provisioning on unhedged foreign currency exposures), so specific "
    "provisions here can bind an AIFI. Where the text does not itself settle "
    "applicability to AIFIs, say plainly that it depends on whether an AIFI "
    "direction cross-refers to that provision, and that this must be confirmed "
    "with the compliance team. Never invent such a cross-reference."
)

_KP_SYSTEM = (
    "You explain RBI directions for NaBFID compliance staff. Use ONLY the "
    "provided document text — do not add rules from outside knowledge, and do not "
    "omit material provisions. Produce a THOROUGH, well-structured explanation "
    "that walks through every chapter / major section IN ORDER. For each section "
    "use a bold markdown heading, then bullet points capturing all its key "
    "provisions: scope and applicability, definitions, limits and thresholds "
    "(give the exact numbers/percentages), obligations and prohibitions, "
    "timelines and effective dates, reporting requirements, and any exceptions or "
    "provisos. Be precise and complete — a compliance officer should be able to "
    "rely on this without opening the PDF. Write in clear plain language."
)


def key_points(doc: dict) -> str:
    """Detailed, section-by-section explanation of the WHOLE document, cited.

    Long directions are processed in ordered batches (map) and the parts are
    concatenated (reduce), so coverage runs end-to-end even for the 97/360-page
    directions. Grounded on the document text only.
    """
    batches = _batches(doc["doc_id"])
    if not batches:
        return "_No extracted text available for this document._"

    truncated = len(batches) > MAX_BATCHES
    batches = batches[:MAX_BATCHES]
    multi = len(batches) > 1
    cite = _cite(doc)

    # A Commercial Banks direction does not bind NaBFID; say so up front so the
    # guide never reads as an obligation on the reader's own institution.
    base_system = _KP_SYSTEM if _is_own_rulebook(doc) else _KP_SYSTEM + _KP_ENTITY_NOTE

    parts: list[str] = []
    for i, batch in enumerate(batches, start=1):
        first_pg, last_pg = batch[0]["page_start"], batch[-1]["page_start"]
        if multi:
            system = (base_system + f" This is PART {i} of {len(batches)} of the "
                      "document (a continuous slice, roughly pages "
                      f"{first_pg}-{last_pg}). Explain ONLY the sections in this "
                      "part. Do NOT write an overall introduction or conclusion "
                      "for the whole document — start directly at the first "
                      "section present in this part and continue in order.")
            head = f"\n\n---\n\n### (continued — pages {first_pg}–{last_pg})\n\n" if i > 1 else ""
        else:
            system = base_system
            head = ""
        prompt = (f"DOCUMENT: {doc['title']} ({cite})\n\n"
                  f"TEXT:\n{_render(batch)}\n\n"
                  "Write the detailed section-by-section key compliance points, "
                  f"citing ({cite}) where a specific figure or rule is stated.")
        parts.append(head + generate(system, prompt, max_tokens=4500))

    out = "".join(parts)
    if truncated:
        out += ("\n\n---\n_Note: this direction is exceptionally long; coverage "
                f"shown for approximately the first {MAX_BATCHES} sections' worth "
                "of pages. Consult the full PDF for the remainder._")
    return out


def what_changed(doc: dict) -> str:
    """For an amendment: what it inserts/substitutes/deletes + effective date."""
    if doc["doc_type"] != "amendment":
        return ""
    ctx = _context(doc["doc_id"])
    system = (
        "You explain what an RBI Amendment Direction changes. Use ONLY the "
        "provided amendment text. List each change as: the paragraph/chapter "
        "affected, whether it is inserted / substituted / deleted, and the effect "
        "in plain language. State the effective date if given. Do not invent "
        "changes. Cite the amendment "
        f"({_cite(doc)})."
        + ("" if _is_own_rulebook(doc) else
           " This amendment sits in the COMMERCIAL BANKS rulebook, while the "
           "reader is NaBFID (an AIFI): describe the changes as changes for "
           "banks, but do not say they are irrelevant to NaBFID — AIFI "
           "directions incorporate parts of the Commercial Banks directions by "
           "reference, so a change here can flow through to an AIFI.")
    )
    prompt = (f"AMENDMENT: {doc['title']} ({_cite(doc)})\n"
              f"Effective (from manifest): {doc.get('applicable_from') or 'see text'}\n\n"
              f"AMENDMENT TEXT:\n{ctx}\n\nList what changed.")
    return generate(system, prompt)


def implications(doc: dict) -> str:
    """NaBFID-context framing. General knowledge allowed but clearly bounded."""
    own = _is_own_rulebook(doc)
    system = (
        "You help NaBFID staff (a Development Finance Institution doing long-tenor "
        "infrastructure lending, no retail deposits) understand why an RBI "
        "direction matters in practice. You MAY use general banking knowledge for "
        "framing and a short illustrative example, but you MUST NOT state any new "
        "rule, limit, or obligation as fact, and must not contradict the "
        "direction. Keep it to 2-4 sentences plus at most one clearly-labelled "
        "'Illustrative example'. This is explanatory framing, not legal advice."
        + ("" if own else
           " IMPORTANT: this direction is addressed to COMMERCIAL BANKS, not to "
           "AIFIs. Explain why a NaBFID reader still cares: (a) it is the "
           "benchmark the AIFI rules are compared against, and (b) AIFI "
           "directions incorporate parts of the Commercial Banks directions by "
           "reference, so some provisions can bind NaBFID. Do not assert that "
           "any specific provision binds NaBFID unless the text says so — say it "
           "must be checked against the AIFI direction and confirmed with "
           "compliance.")
    )
    ask = ("Briefly explain why this matters for NaBFID and give one short "
           "illustrative example relevant to infrastructure lending."
           if own else
           "Briefly explain why a NaBFID (AIFI) reader should care about this "
           "Commercial Banks direction — as a comparison benchmark and because "
           "AIFI directions may cross-refer to it — and give one short "
           "illustrative example relevant to infrastructure lending.")
    prompt = (f"DIRECTION: {doc['title']} ({_cite(doc)}), "
              f"division: {doc['division']}.\n\n" + ask)
    return generate(system, prompt, temperature=0.2)
