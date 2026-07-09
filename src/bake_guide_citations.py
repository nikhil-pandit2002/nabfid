"""
bake_guide_citations.py — attach VERIFIED page citations to the baked guides.

For every citable line of every guide section (key_points / what_changed /
implications in data/explanations/*.json), find the PDF page(s) that provably
contain that line's facts, and store them in the same JSON under "citations".
The app then renders only these baked, verified citations — no render-time
guessing (the old fuzzy matcher put ~1 in 3 word-only lines on a wrong page).

How a line earns a citation (all matching against data/pages.jsonl ground
truth, normalized by pagetext.norm):

  anchors of a line
    * numbers        — digit tokens; also number+unit bigrams ("90 days")
    * bold phrases   — **multi-word terms** matched as verbatim substrings
    * rare words     — words on <=20% of the doc's pages (kills "interest",
                       "management"; keeps "behavioural", "crar")
    Anchor value is discounted by page-frequency, so a token that appears
    everywhere ("2025", paragraph numbers) contributes ~nothing.

  candidate pages
    * if the line sits under a "### Chapter N" heading and the doc's chunks
      carry chapter section_refs, that chapter's page range (padded) is
      searched first; whole doc as fallback.

  confidence gate (below it -> NO citation, listed in the report)
    * needs a numeric/phrase anchor hit plus a corroborating hit, or two
      independent phrase hits; and the top page must beat the runner-up
      (adjacent runner-up tolerated — same provision crossing a page break).

Hand-review loop: data/citation_report.md lists every decision with the page
snippet behind it. Corrections go into data/citation_overrides.json
({doc_id: {section: {line_hash: [pages]}}}) which is merged LAST on every
run — so re-baking never clobbers a human fix.

Run:  python src/build_pages.py   (first, once)
      python src/bake_guide_citations.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR
import explain
import pagetext

EXPLANATIONS_DIR = DATA_DIR / "explanations"
OVERRIDES_JSON = DATA_DIR / "citation_overrides.json"
REPORT_MD = DATA_DIR / "citation_report.md"

SECTIONS = ("key_points", "what_changed", "implications")

# --- line filtering (must mirror app.py's render skip rules exactly) --------
_SEP_ROW_RE = re.compile(r"^[\s|:-]+$")
_HEADING_RE = re.compile(r"^#{1,4}\s+(.*)$")
# Some guides write chapter headings as bold-only lines instead of "###":
# "**Chapter IIIA – Regulatory Restrictions**". Structure, not a claim.
_BOLD_HEADING_RE = re.compile(
    r"^\*\*(chapter|part|annex)[^*]*\*\*[:.]?\s*(\(.*\))?\s*$", re.IGNORECASE)
_ROMAN_RE = re.compile(r"chapter[s]?\s+([ivxlcdm]+(?:\s*[–\-]\s*[ivxlcdm]+)?)\b",
                       re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
# also catches paragraph refs like "17a" / "126b" (strong locator anchors)
_NUM_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?[a-z]?\b")
_WORD_RE = re.compile(r"[a-z]{4,}")

_STOP = frozenset("""
the and for that with this from are was were has have had not but its their
they them then than which will shall may can under over within into upon such
other any all each both more most also only same these those there here when
where what how why been being does done used using must should would could
about after before between during without across whether against including
directions direction circular reserve bank india aifi aifis shall provided
paragraph paragraphs chapter""".split())

_UNIT_WORDS = frozenset(
    "days months years crore lakh cent bps percent basis times day month year".split())


def line_hash(line: str) -> str:
    """Stable key for a guide line (the renderer computes the same)."""
    return hashlib.sha1(line.strip().encode("utf-8")).hexdigest()[:12]


def iter_guide_lines(text: str):
    """Yield (raw_line, citable_stripped_or_None, chapter_heading_or_None) for
    EVERY line of a guide. The single source of truth for which lines carry
    citations — the offline baker and the app's renderer both walk through
    here, so their skip rules can never drift apart."""
    lines = text.splitlines()
    heading: str | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = _HEADING_RE.match(line)
        if m:
            # "##" is the document title, not a chapter — skip it but leave
            # heading unset so the opener paragraph is recognisable below.
            if line.lstrip().startswith("###"):
                heading = m.group(1)
            yield line, None, heading
            continue
        if _BOLD_HEADING_RE.match(stripped):
            heading = stripped.strip("*:. ")
            yield line, None, heading
            continue
        next_is_sep = (i + 1 < len(lines)
                       and _SEP_ROW_RE.match(lines[i + 1].strip() or ""))
        if (not stripped or _SEP_ROW_RE.match(stripped) or next_is_sep
                or stripped.startswith("*(") or stripped.startswith("*Illustrative")
                or stripped.startswith("_Verify")):
            yield line, None, heading
            continue
        yield line, stripped, heading


def citable_lines(text: str) -> list[tuple[str, str | None]]:
    """(citable line, chapter heading) pairs — the baker's working view."""
    return [(citable, heading) for _raw, citable, heading in iter_guide_lines(text)
            if citable is not None]


# --- chapter -> page range (from chunk section_refs, where they exist) ------
def chapter_page_ranges(doc_id: str) -> dict[str, tuple[int, int]]:
    """roman numeral -> (min page_start, max page_end) over that chapter's chunks."""
    ranges: dict[str, tuple[int, int]] = {}
    for c in explain._chunks_by_doc().get(doc_id, []):
        ref = (c.get("section_ref") or "")
        m = _ROMAN_RE.search(ref.lower())
        if not m:
            continue
        key = re.split(r"[–\-]", m.group(1))[0].strip()
        lo, hi = ranges.get(key, (c["page_start"], c["page_end"]))
        ranges[key] = (min(lo, c["page_start"]), max(hi, c["page_end"]))
    return ranges


def heading_range(heading: str | None,
                  ranges: dict[str, tuple[int, int]]) -> tuple[int, int] | None:
    """Page range for a guide heading like 'Chapter IV — ...' or 'Chapter I–II'."""
    if not heading or not ranges:
        return None
    m = _ROMAN_RE.search(heading.lower())
    if not m:
        return None
    parts = [p.strip() for p in re.split(r"[–\-]", m.group(1))]
    spans = [ranges[p] for p in parts if p in ranges]
    if not spans:
        return None
    lo = min(s[0] for s in spans)
    hi = max(s[1] for s in spans)
    return (max(1, lo - 1), hi + 1)          # pad one page each side


# --- anchors -----------------------------------------------------------------
def page_doc_freq(pages: list[tuple[int, str]]) -> Counter:
    """token -> number of pages it appears on (words and numbers alike)."""
    df: Counter = Counter()
    for _pg, text in pages:
        tokens = set(_WORD_RE.findall(text)) | set(_NUM_TOKEN_RE.findall(text))
        df.update(tokens)
    return df


def extract_anchors(line: str) -> dict:
    """The verifiable pieces of a guide line."""
    n = pagetext.norm(line)
    plain = re.sub(r"[*_`\[\]]", "", n)       # strip markdown emphasis chars
    numbers = _NUM_TOKEN_RE.findall(plain)
    # number+unit bigrams ("90 days", "15 crore") — strong, phrase-matched
    bigrams = re.findall(
        r"\b(\d+(?:\.\d+)?\s+(?:" + "|".join(_UNIT_WORDS) + r"))\b", plain)
    phrases = []
    for ph in _BOLD_RE.findall(line):
        phn = pagetext.norm(re.sub(r"[*_`]", "", ph))
        if len(phn.split()) >= 2 and len(phn) >= 8:
            phrases.append(phn)
    words = [w for w in set(_WORD_RE.findall(plain)) if w not in _STOP]
    return {"numbers": numbers, "bigrams": bigrams,
            "phrases": phrases, "words": words}


def score_page(text: str, anchors: dict, df: Counter, n_pages: int) -> tuple[float, dict]:
    """Anchor-hit score of one page, with the detail of what hit."""
    rare_cut = max(2, int(0.20 * n_pages))
    hits = {"numbers": [], "bigrams": [], "phrases": [], "words": []}
    score = 0.0
    for num in set(anchors["numbers"]):
        if df.get(num, 0) > rare_cut:          # page numbers, years — worthless
            continue
        if re.search(rf"\b{re.escape(num)}\b", text):
            hits["numbers"].append(num)
            score += 2.0 + min(len(num), 5)    # longer digit strings are rarer
    for bg in set(anchors["bigrams"]):
        if bg in text:
            hits["bigrams"].append(bg)
            score += 6.0
    for ph in set(anchors["phrases"]):
        if ph in text:
            hits["phrases"].append(ph)
            score += 5.0
    for w in anchors["words"]:
        if df.get(w, 0) <= rare_cut and re.search(rf"\b{re.escape(w)}\b", text):
            hits["words"].append(w)
            score += 1.0
    return score, hits


def passes_gate(hits: dict) -> bool:
    strong = len(hits["numbers"]) + len(hits["bigrams"]) + len(hits["phrases"])
    corroborating = strong + len(hits["words"])
    return ((strong >= 1 and corroborating >= 2)
            or len(hits["phrases"]) >= 2
            # no numbers at all, but several independent rare words landing on
            # the same page is strong evidence too (words are DF-filtered)
            or len(hits["words"]) >= 3)


def cite_line(line: str, heading: str | None, pages: list[tuple[int, str]],
              df: Counter, ranges: dict) -> tuple[list[int], dict]:
    """Choose verified page(s) for one line. Returns ([], detail) if unsure."""
    anchors = extract_anchors(line)
    span = heading_range(heading, ranges)
    scopes = ([p for p in pages if span[0] <= p[0] <= span[1]], pages) if span \
        else (pages,)
    for scope in scopes:
        scored = []
        for pg, text in scope:
            s, hits = score_page(text, anchors, df, len(pages))
            if s > 0:
                scored.append((s, pg, hits))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored:
            continue
        top_s, top_pg, top_hits = scored[0]
        if not passes_gate(top_hits):
            continue
        # margin rule: runner-up must be clearly weaker or an adjacent page
        if len(scored) > 1:
            run_s, run_pg, run_hits = scored[1]
            if run_s >= top_s and abs(run_pg - top_pg) > 1:
                continue                        # ambiguous — leave for review
        chosen = [top_pg]
        # optional second page: passes gate AND contributes a NEW number
        for s2, pg2, h2 in scored[1:3]:
            if pg2 in chosen or not passes_gate(h2):
                continue
            if set(h2["numbers"]) - set(top_hits["numbers"]):
                chosen.append(pg2)
                break
        return sorted(chosen), {"hits": top_hits, "score": top_s,
                                "scoped": scope is not pages}
    return [], {"hits": None, "score": 0, "anchors": anchors}


def snippet(doc_pages: dict[int, str], page: int, hits: dict) -> str:
    """±120 chars of page text around the strongest anchor that hit."""
    text = doc_pages.get(page, "")
    for kind in ("bigrams", "phrases", "numbers", "words"):
        for a in hits.get(kind) or []:
            i = text.find(a)
            if i >= 0:
                lo, hi = max(0, i - 120), i + len(a) + 120
                return ("…" + text[lo:hi].replace("\n", " ") + "…")
    return text[:200].replace("\n", " ")


def main() -> int:
    overrides = {}
    if OVERRIDES_JSON.exists():
        overrides = json.loads(OVERRIDES_JSON.read_text(encoding="utf-8"))

    report: list[str] = ["# Citation report — verified page citations for the baked guides\n"]
    tot = cited = uncited = overridden = 0

    for path in sorted(EXPLANATIONS_DIR.glob("*.json")):
        doc_id = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        pages = pagetext.pages_for(doc_id)
        if not pages:
            report.append(f"\n## {doc_id}\n\n**NO PAGE TEXT — skipped**\n")
            continue
        doc_pages = dict(pages)
        df = page_doc_freq(pages)
        ranges = chapter_page_ranges(doc_id)
        doc_over = overrides.get(doc_id, {})

        citations: dict[str, dict[str, list[int]]] = {}
        report.append(f"\n## {doc_id}\n")
        for section in SECTIONS:
            text = data.get(section)
            if not text:
                continue
            sec_map: dict[str, list[int]] = {}
            report.append(f"\n### {section}\n")
            first_citable = True
            for line, heading in citable_lines(text):
                h = line_hash(line)
                tot += 1
                is_first = first_citable
                first_citable = False
                if h in doc_over.get(section, {}):
                    sec_map[h] = doc_over[section][h]
                    overridden += 1
                    report.append(f"- **p.{sec_map[h]} (OVERRIDE)** — {line[:100]}")
                    continue
                # The section's opener paragraph (first citable line, prose,
                # before any chapter heading) summarises the WHOLE document —
                # anchor-matching it can hook a random content-rich page. The
                # title page states the document's scope, so it cites p.1.
                if is_first and heading is None and not line.startswith(("-", "|", ">")):
                    sec_map[h] = [1]
                    cited += 1
                    report.append(f"- **p.[1]** (opener -> title page) — {line[:100]}")
                    continue
                pgs, detail = cite_line(line, heading, pages, df, ranges)
                if pgs:
                    sec_map[h] = pgs
                    cited += 1
                    sn = snippet(doc_pages, pgs[0], detail["hits"])
                    report.append(
                        f"- **p.{pgs}** (score {detail['score']:.0f}"
                        f"{', chapter-scoped' if detail.get('scoped') else ''}) — "
                        f"{line[:100]}\n  - `{sn[:240]}`")
                else:
                    uncited += 1
                    report.append(f"- **UNCITED** — {line[:100]}")
            if sec_map:
                citations[section] = sec_map
        data["citations"] = citations
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    header = (f"\nTotal lines: {tot} | cited: {cited} | overridden: {overridden} "
              f"| UNCITED: {uncited} ({100 * uncited // max(1, tot)}%)\n")
    report.insert(1, header)
    REPORT_MD.write_text("\n".join(report), encoding="utf-8")
    print(header)
    print(f"Report -> {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
