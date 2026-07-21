"""
build_crossrefs.py — find where one direction points at another, offline.

RBI directions cross-refer constantly, and the cross-ENTITY ones matter most to
NaBFID: the AIFI rulebook incorporates parts of the Commercial Banks rulebook by
reference, which makes those parts binding on an AIFI. Real examples:

    AIFI Credit Facilities, p.12:  "An AIFI SHALL provide a Key Fact Statement
    (KFS), as per instructions contained in the RBI (Commercial Banks -
    Responsible Business Conduct) Directions, 2025."

    AIFI Capital Adequacy, p.41:  "An AIFI shall refer to RBI (Commercial Banks
    - Credit Risk Management) Directions, 2025 which cover provision on
    unhedged foreign currency exposures."

Reading only the AIFI direction, you would miss the rule. So the app shows these
links both ways — what a document refers to, and what refers to it.

Detection is deterministic (a title pattern in the real page text, resolved
against the manifest) and every hit keeps its page number and the sentence it
came from, so a compliance officer can judge whether a reference is a binding
incorporation or just a borrowed definition. References we cannot resolve are
recorded too: "referenced but not in this corpus" is a finding, not a silent gap.

Run:  python src/build_crossrefs.py      ->  data/crossrefs.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict

from config import DATA_DIR, MANIFEST_CSV, PAGES_JSONL

OUT_JSON = DATA_DIR / "crossrefs.json"

# "Reserve Bank of India (<subject>) [<qualifier>] Directions, <year>".
# The qualifier catches amendment forms: ") - Second Amendment Directions, 2026".
# The subject may itself contain one bracketed aside — real titles read
# "(All India Financial Institutions (AIFIs) - Prudential Norms ...)" — so the
# subject group allows a single nested level. Matching only [^)] truncates at
# "(AIFIs" and loses the topic entirely.
REF_RE = re.compile(
    r"Reserve Bank of India\s*\(\s*((?:[^()]|\([^()]*\)){4,120}?)\)"
    r"\s*([^,]{0,45}?)Directions,\s*(\d{4})",
    re.IGNORECASE,
)

ENTITY_CLAUSES = [
    ("AIFI", "all india financial institutions"),
    ("Commercial Bank", "commercial banks"),
    ("Commercial Bank", "commercial bank"),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def split_ref(subject: str) -> tuple[str | None, str]:
    """'Commercial Banks - Credit Risk Management' -> ('Commercial Bank',
    'credit risk management'). Returns (None, topic) when the reference carries
    no entity clause (directions addressed to all regulated entities)."""
    n = norm(subject).replace("aifis", " ")
    for entity, clause in ENTITY_CLAUSES:
        if n.startswith(clause):
            return entity, re.sub(r"\s+", " ", n[len(clause):]).strip(" -")
    return None, re.sub(r"\s+", " ", n).strip(" -")


def load_manifest() -> list[dict]:
    with MANIFEST_CSV.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def doc_topic(row: dict) -> str:
    """The regulated subject of a manifest row, normalized like split_ref.

    Parsed with the SAME pattern used on references, so both sides of a match
    are derived identically. (Hand-rolling this split on ") " truncated titles
    at the nested "(AIFIs)" aside and produced an empty topic, which made every
    AIFI master unmatchable and silently resolved AIFI references to the
    Commercial Bank document of the same name.)
    """
    m = REF_RE.search(row["title"])
    if not m:
        return norm(row["title"])
    _, topic = split_ref(m.group(1))
    return topic


def main() -> int:
    rows = load_manifest()
    by_id = {r["doc_id"]: r for r in rows}

    # Resolve targets against MASTER DIRECTIONS only: a reference names the
    # rulebook, not one of its amendments.
    masters: list[tuple[str, str, str]] = []   # (doc_id, entity, topic)
    for r in rows:
        if r["doc_type"] != "master_direction":
            continue
        masters.append((r["doc_id"], r["entity"], doc_topic(r)))

    def resolve(entity: str | None, topic: str) -> str | None:
        if len(topic) < 6:
            return None
        cands = [(d, e, t) for d, e, t in masters if t and (topic in t or t in topic)]
        if entity:
            same = [c for c in cands if c[1] == entity]
            if same:
                cands = same
        if not cands:
            return None
        # Prefer the closest topic match, then the longest (most specific) title.
        cands.sort(key=lambda c: (abs(len(c[2]) - len(topic)), -len(c[2])))
        return cands[0][0]

    out_refs: dict[str, dict[str, dict]] = defaultdict(dict)
    unresolved: dict[str, set[str]] = defaultdict(set)

    with PAGES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            src = o["doc_id"]
            if src not in by_id:
                continue
            text = " ".join(o["text"].split())
            for m in REF_RE.finditer(text):
                entity, topic = split_ref(m.group(1))
                target = resolve(entity, topic)
                if not target:
                    unresolved[m.group(1).strip()[:80]].add(src)
                    continue
                # Skip self-references and an amendment pointing at its own
                # parent — that relationship is already the `amends` chain.
                if target == src or target == by_id[src].get("amends"):
                    continue
                # Keep the sentence CONTAINING the reference, so a human can
                # judge whether it is a binding incorporation ("an AIFI shall
                # ... as per") or just a borrowed definition ("X means X as
                # defined under"). The sentence must span the match itself.
                lo = text.rfind(". ", 0, m.start())
                lo = 0 if lo == -1 else lo + 2
                hi = text.find(". ", m.end())
                hi = len(text) if hi == -1 else hi + 1
                quote = text[max(lo, m.start() - 260):min(hi, m.end() + 200)].strip()
                prev = out_refs[src].get(target)
                if prev is None or o["page"] < prev["page"]:
                    out_refs[src][target] = {"page": o["page"],
                                             "quote": quote[:320]}

    # Reverse index: "who points at me".
    back: dict[str, list[dict]] = defaultdict(list)
    forward: dict[str, list[dict]] = defaultdict(list)
    n_cross = 0
    for src, targets in out_refs.items():
        for tgt, info in sorted(targets.items(), key=lambda kv: kv[1]["page"]):
            cross = by_id[src]["entity"] != by_id[tgt]["entity"]
            n_cross += int(cross)
            forward[src].append({"doc_id": tgt, "page": info["page"],
                                 "quote": info["quote"], "cross_entity": cross})
            back[tgt].append({"doc_id": src, "page": info["page"],
                              "quote": info["quote"], "cross_entity": cross})

    payload = {
        "references": forward,          # doc -> what it points at
        "referenced_by": back,          # doc -> what points at it
        "unresolved": {k: sorted(v) for k, v in sorted(
            unresolved.items(), key=lambda kv: -len(kv[1]))},
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    total = sum(len(v) for v in forward.values())
    print(f"Resolved {total} references across {len(forward)} documents "
          f"({n_cross} cross-entity).")
    print(f"Unresolved (referenced but not in this corpus): {len(unresolved)}")
    for subj, srcs in list(payload["unresolved"].items())[:8]:
        print(f"   {len(srcs)}x  {subj[:70]}")
    print(f"\nWritten -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
