"""
compare.py — side-by-side comparison of the AIFI and Commercial Bank rulebooks.

RBI issues the same topics separately for each kind of institution, usually with
different numbers. Staff want to see those differences directly ("what is the
single-counterparty limit for us vs for a bank?"), and reading two 400-page
directions to find out is exactly the work this tool exists to remove.

How it stays honest, which matters more here than anywhere else in the app:

  * Retrieval runs TWICE, once scoped to each entity, so neither side can be
    answered out of the other's rulebook.
  * The two source sets are numbered in ONE sequence and each header states its
    entity, so the model cites normally and every claim resolves back to a
    document that actually belongs to that side. The existing quote-verified
    page pinning (query._split_blocks / pagetext.locate_page) is reused
    unchanged.
  * Silence is reported, not filled: if one rulebook does not address a point,
    the answer must say so rather than infer it from the other side. An
    invented symmetry would be worse than no comparison at all.
"""

from __future__ import annotations

import re

from llm import generate
from retrieval import retrieve
import docstore
import pagetext
from query import _split_blocks, ABSTAIN

ENTITY_A = "AIFI"
ENTITY_B = "Commercial Bank"

SYSTEM_PROMPT = f"""You are the RBI Compliance Assistant for NaBFID (an AIFI). \
You are comparing how a topic is regulated for TWO kinds of institution.

You are given SOURCES from two rulebooks:
  * "{ENTITY_A}" sources — the All India Financial Institutions directions (these \
bind NaBFID).
  * "{ENTITY_B}" sources — the Commercial Banks directions (these bind banks; \
they matter to NaBFID as a benchmark, and because AIFI directions sometimes \
incorporate parts of them by reference).

Rules — follow strictly:

1. Use ONLY the provided SOURCES. Never state a rule from outside knowledge.
2. Cite with the bracketed SOURCE NUMBER at the end of every factual statement, \
e.g. "... shall be 20 per cent.[3]". A statement about the AIFI position must \
cite an {ENTITY_A} source; a statement about the Commercial Bank position must \
cite a {ENTITY_B} source. Never support one side's claim with the other side's \
document.
3. If a rulebook does not address the point, write exactly "Not addressed in the \
provided {ENTITY_A} sources" (or {ENTITY_B}) for that cell. NEVER infer one \
side's rule from the other. Asymmetry is a real and useful finding.
4. If NEITHER rulebook addresses the question, reply with exactly this sentence \
and nothing else: "{ABSTAIN}"
5. Amendments override the base direction. Mention an effective date when a \
change is not yet in force.
6. You MUST finish with the ===CITES=== block specified at the end. It is parsed \
by the application to pin each citation to the exact PDF page, so an answer \
without it is incomplete. Include one line for every source number you cited.

Be CRISP. The whole answer is a verdict line and a table — no essay. Output \
exactly these three parts and nothing else:

FIRST LINE — the verdict, in bold, one of:
**⚠️ Different** — <max 12 words naming the key difference>
**✅ Same** — <max 12 words, e.g. "both require 9% CRAR, 5.5% CET1">
**◑ Partly different** — <max 12 words>

SECOND: the table. AT MOST 5 rows — only the aspects that matter most, and \
prefer rows where the two sides actually DIFFER (skip minor details that are \
identical). Each cell must be a short phrase with the key figure, roughly 12 \
words max — never a sentence. Numbers, not prose.

| Aspect | {ENTITY_A} (NaBFID) | {ENTITY_B} |
|---|---|---|
| <aspect> | <figure/rule, ≤12 words>[n] | <figure/rule, ≤12 words>[n] |

Do NOT add a "why it differs", "what to watch", background, or summary section. \
Do NOT restate the table in prose. Do NOT include a row where both cells say the \
same thing unless it is a headline number the reader expects to see. If a figure \
needs a caveat, put it in the cell in brackets — never as a paragraph.

Then append the machine block described below.

===CITES===
1: "a verbatim quote of at most 20 words copied exactly from SOURCE 1"
One line per source number you actually cited; the quote must be an exact, \
contiguous excerpt of that source. Nothing else in the block."""


def _block(chunks: list[dict], start: int, entity: str) -> str:
    out = []
    for i, c in enumerate(chunks, start=start):
        head = (f"[{i}] ({entity}) {c['circular_no']}, dated {c['issue_date']} "
                f"| {c['doc_type']} | page {c['page_start']} "
                f"| {c['section_ref'] or '-'}")
        if c.get("applicable_from"):
            head += f" | effective {c['applicable_from']}"
        out.append(head + "\n" + c["text"])
    return "\n\n".join(out)


def compare(question: str, *, per_side: int = 10) -> dict:
    """Retrieve each rulebook separately, then compare them in one grounded pass.

    per_side caps how many chunks each rulebook contributes, so neither side can
    crowd the other out of the context window (the Commercial Banks corpus is
    larger, and an unbalanced context quietly biases the comparison). It is set
    generously: a comparison fails in a particularly misleading way when one
    side's limit simply was not retrieved — the answer then reads "not addressed
    in the ... sources" for a rule that does exist, which looks like a finding
    rather than a gap.
    """
    a = retrieve(question, top_k=per_side,
                 scope_doc_ids=docstore.doc_ids_for_entity(ENTITY_A))
    b = retrieve(question, top_k=per_side,
                 scope_doc_ids=docstore.doc_ids_for_entity(ENTITY_B))
    if not a and not b:
        return {"answer": ABSTAIN, "sources": [], "abstained": True,
                "n_a": 0, "n_b": 0}

    prompt = (
        f"SOURCES — {ENTITY_A} rulebook:\n{_block(a, 1, ENTITY_A)}\n\n"
        f"SOURCES — {ENTITY_B} rulebook:\n{_block(b, len(a) + 1, ENTITY_B)}\n\n"
        f"QUESTION: {question}\n\n"
        f"Compare how the {ENTITY_A} and {ENTITY_B} directions treat this. Cite "
        f"every statement with its bracketed source number, keep each side's "
        f"claims on its own sources, and say plainly when a rulebook does not "
        f"address the point. Finish with the ===CITES=== block."
    )
    text = generate(SYSTEM_PROMPT, prompt, max_tokens=4000)
    text, _example, quotes = _split_blocks(text)
    if text.strip().startswith(ABSTAIN[:30]):
        return {"answer": text, "sources": [], "abstained": True,
                "n_a": len(a), "n_b": len(b)}

    # Same quote-verified page pinning as the chat surface.
    sources = []
    for i, c in enumerate(a + b, start=1):
        c = dict(c)
        q = quotes.get(i)
        if q:
            page, ok = pagetext.locate_page(c["doc_id"], c, q)
            c["cite_page"], c["cite_quote"], c["cite_verified"] = page, q, ok
        else:
            c["cite_page"], c["cite_verified"] = c["page_start"], False
        sources.append(c)
    return {"answer": text, "sources": sources, "abstained": False,
            "n_a": len(a), "n_b": len(b)}


def comparable_topics() -> list[str]:
    """Topics that exist as a master direction in BOTH rulebooks.

    Only these can be compared honestly; for anything else one side would be
    empty, which the UI says outright instead of pretending to compare.
    """
    def topic(title: str) -> str:
        for clause in ("All India Financial Institutions", "Commercial Banks",
                       "Commercial Bank"):
            if clause in title:
                tail = title.split(clause, 1)[1].replace("(AIFIs)", "").lstrip(" -")
                return re.sub(r"\s+", " ", tail.split(")", 1)[0]).strip().lower()
        return ""

    per_entity: dict[str, set[str]] = {}
    for ent in (ENTITY_A, ENTITY_B):
        got = set()
        for div in docstore.divisions(ent):
            for d in docstore.documents_in(div["division"], ent):
                if d["doc_type"] == "master_direction":
                    t = topic(d["title"])
                    if t:
                        got.add(t)
        per_entity[ent] = got
    return sorted(per_entity[ENTITY_A] & per_entity[ENTITY_B])
