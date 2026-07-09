"""
eval_citations.py — verify every baked guide citation against page text.

The contract behind the guides' [n] links: a citation may only point at a page
that verifiably contains the cited fact. This checker re-derives each line's
anchors independently and confirms the CITED page still passes the confidence
gate — run it after re-baking guides, re-chunking, or swapping a PDF.

Three citation classes, three checks:
  * auto     — page must still pass the anchor gate (hard FAIL otherwise)
  * opener   — section-opening summary lines must cite p.1 (title/scope page)
  * override — human-reviewed pages (data/citation_overrides.json); reported
               with a weak anchor check (warn, not fail — a human chose them)

Run:  python src/eval_citations.py     (exit 1 on any hard failure)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_guide_citations import (EXPLANATIONS_DIR, OVERRIDES_JSON, SECTIONS,
                                  chapter_page_ranges, extract_anchors,
                                  iter_guide_lines, line_hash, page_doc_freq,
                                  passes_gate, score_page)
import pagetext


def main() -> int:
    overrides = {}
    if OVERRIDES_JSON.exists():
        overrides = json.loads(OVERRIDES_JSON.read_text(encoding="utf-8"))

    n_auto = n_opener = n_override = 0
    fails: list[str] = []
    warns: list[str] = []

    for path in sorted(EXPLANATIONS_DIR.glob("*.json")):
        doc_id = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        citations = data.get("citations") or {}
        pages = pagetext.pages_for(doc_id)
        if not pages:
            fails.append(f"{doc_id}: no page text (run build_pages.py)")
            continue
        doc_pages = dict(pages)
        df = page_doc_freq(pages)
        doc_over = overrides.get(doc_id, {})

        for section in SECTIONS:
            text = data.get(section)
            if not text:
                continue
            sec_cites = citations.get(section, {})
            first = True
            for _raw, citable, heading in iter_guide_lines(text):
                if citable is None:
                    continue
                is_first = first
                first = False
                h = line_hash(citable)
                cited = sec_cites.get(h)
                if not cited:
                    continue                     # uncited-by-design lines
                if h in doc_over.get(section, {}):
                    n_override += 1              # human-verified; weak check
                    anchors = extract_anchors(citable)
                    ok = any(score_page(doc_pages.get(p, ""), anchors, df,
                                        len(pages))[0] > 0 for p in cited)
                    if not ok:
                        warns.append(f"{doc_id}/{section} p.{cited} (override, "
                                     f"no anchor overlap): {citable[:70]}")
                    continue
                if is_first and heading is None and not citable.startswith(("-", "|", ">")):
                    n_opener += 1
                    if cited != [1]:
                        fails.append(f"{doc_id}/{section}: opener cites {cited},"
                                     f" expected [1]")
                    continue
                n_auto += 1
                ok = False
                for p in cited:
                    score, hits = score_page(doc_pages.get(p, ""), anchors=extract_anchors(citable),
                                             df=df, n_pages=len(pages))
                    if score > 0 and passes_gate(hits):
                        ok = True
                        break
                if not ok:
                    fails.append(f"{doc_id}/{section} p.{cited}: {citable[:70]}")

    print(f"auto-verified: {n_auto} | openers->p.1: {n_opener} "
          f"| human overrides: {n_override}")
    print(f"HARD FAILURES: {len(fails)} | override warnings: {len(warns)}")
    for f in fails:
        print("  FAIL", f)
    for w in warns:
        print("  warn", w)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
