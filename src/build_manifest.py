"""
build_manifest.py — Stage 1 of the RBI Compliance Assistant ingestion pipeline.

Walks the source PDF corpus and produces data/manifest.csv, one row per document.
The manifest is the single source of truth for metadata (dates, divisions,
amendment chains) that the rest of the system reads from.

Design rules (from CLAUDE.md):
  * The source PDFs in "AIFI latest/" are NEVER modified. We only read them.
  * High-confidence fields (circular_no, issue_date, title, doc_type) are
    auto-filled from the PDF's first page.
  * Uncertain fields (amends, applicable_from, consolidated_as_of) are
    auto-SUGGESTED, and the row is flagged needs_verification=yes with a note,
    so a human reviews rather than trusting a silent guess.

Run:  python src/build_manifest.py
Output: data/manifest.csv  (open in Excel; encoded utf-8-sig so it opens clean)
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber

# --- Paths -----------------------------------------------------------------
# Resolve relative to this file so the script works regardless of the
# directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "AIFI latest"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = OUTPUT_DIR / "manifest.csv"

# Manifest columns, in order. The first block matches CLAUDE.md's spec; the
# trailing helper columns support review + later pipeline stages.
COLUMNS = [
    "doc_id",
    "division",
    "doc_type",
    "title",
    "circular_no",
    "issue_date",
    "applicable_from",
    "amends",
    "consolidated_as_of",
    "file_path",
    "source_url",
    # --- helper columns (not in the original spec, useful for us) ---
    "dept_ref",
    "page_count",
    "needs_verification",
    "notes",
]

MONTHS = (
    "January|February|March|April|May|June|July|"
    "August|September|October|November|December"
)
# A date like "November 28, 2025" or "April 1, 2026" (day may be un-padded).
DATE_RE = re.compile(rf"({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})")


# --- Small text helpers ----------------------------------------------------
def normalize(text: str) -> str:
    """Collapse whitespace and fix the odd dash / quote encodings RBI PDFs use."""
    if not text:
        return ""
    # Various unicode dashes + the PDF "replacement char" all become a plain "-".
    for ch in ("–", "—", "‒", "‐", "�"):
        text = text.replace(ch, "-")
    text = text.replace("’", "'").replace("‘", "'")
    # Collapse all runs of whitespace (incl. newlines) to single spaces.
    return re.sub(r"\s+", " ", text).strip()


def to_iso(match: re.Match | None) -> str:
    """Turn a DATE_RE match into ISO yyyy-mm-dd, or '' if it can't be parsed."""
    if not match:
        return ""
    raw = f"{match.group(1)} {int(match.group(2))}, {match.group(3)}"
    try:
        return datetime.strptime(raw, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return ""


def slugify(text: str) -> str:
    """Filename stem -> stable, unique, readable doc_id."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug


# --- Field extractors (each reads the normalized page-1 text) --------------
def extract_circular_no(text: str) -> str:
    """The RBI reference like 'RBI/2026-27/74' or 'RBI/DOR/2025-26/325'."""
    m = re.search(r"RBI/(?:[A-Za-z]+/)?\d{4}-\d{2,4}/\d+", text)
    return m.group(0) if m else ""


def extract_dept_ref(text: str) -> str:
    """The department reference, e.g. 'DOR.CRE.REC.244/07-02-007/2025-26'.

    Kept for citations. Format varies (dots vs spaces), so we match loosely.
    """
    m = re.search(
        r"D[oO]R[.\s][A-Za-z().]+REC[.\s]?(?:No\.?\s*)?\d+[/\d.\-]+(?:\s/[\d\-]+)?",
        text,
    )
    return m.group(0).strip() if m else ""


def extract_title(text: str) -> str:
    """Full official title, from 'Reserve Bank of India (All India ...' up to
    the first 'Directions, <year>'. Excludes the '(Updated as on ...)' suffix,
    which we capture separately as consolidated_as_of."""
    m = re.search(
        r"Reserve Bank of India \(All India Financial Institutions.*?Directions,\s*\d{4}",
        text,
    )
    return m.group(0).strip() if m else ""


def detect_doc_type(title: str) -> str:
    """One of: amendment, master_direction, circular, notification.

    This corpus is entirely master directions + their amendments, but we keep
    the logic extensible for circulars/notifications added later.
    """
    low = title.lower()
    if "amendment" in low:
        return "amendment"
    if "directions" in low or "master" in low:
        return "master_direction"
    if "circular" in low:
        return "circular"
    if "notification" in low:
        return "notification"
    return "master_direction"


def extract_consolidated_as_of(text: str) -> str:
    """From the '(Updated as on <date>)' suffix present on consolidated
    master directions. Returns ISO date or ''."""
    m = re.search(r"[Uu]pdated as on\s+" + DATE_RE.pattern, text)
    if not m:
        return ""
    # Re-run DATE_RE on just the matched span to reuse the ISO parser.
    return to_iso(DATE_RE.search(m.group(0)))


def extract_applicable_from(text: str) -> tuple[str, str]:
    """Effective date. Returns (iso_date_or_empty, note).

    RBI phrases this many ways. We handle:
      * explicit future date  -> that date
      * 'immediate' / 'date of issue' / 'upon issuance' -> flag, caller uses issue_date
      * 'placed on the official website'                -> flag, caller uses issue_date
    """
    low = text.lower()

    # 1. "Immediate commencement" phrasing takes PRECEDENCE. This IS the
    #    commencement, and checking it first avoids accidentally grabbing a
    #    later transitional "with effect from <date>" clause that appears
    #    deeper in a master direction's Chapter I. In these cases the effective
    #    date equals the issue date; the caller fills that in and keeps the flag.
    if "immediate effect" in low or "immediately upon issuance" in low:
        return "", "effective immediately -> equals issue_date; confirm"
    if "date of issue" in low or "date of the issue" in low:
        return "", "effective from date of issue -> equals issue_date; confirm"
    if "placed on the official website" in low:
        return "", "effective when placed on RBI website -> ~issue_date; confirm"

    # 2. Otherwise look for an explicit commencement date. Reliable for
    #    amendments, which state a specific "come into force from <date>".
    m = re.search(
        r"come into (?:force|effect)[^.]{0,80}?" + DATE_RE.pattern, text, re.IGNORECASE
    )
    if not m:
        m = re.search(r"with effect from\s+" + DATE_RE.pattern, text, re.IGNORECASE)
    if m:
        iso = to_iso(DATE_RE.search(m.group(0)))
        if iso:
            return iso, ""

    return "", "could not detect commencement date; check document"


def extract_topic(title: str) -> str:
    """The regulated subject, normalized for matching amendments to their parent.

    'Reserve Bank of India (All India Financial Institutions (AIFIs) - Prudential
    Norms on Capital Adequacy) Second Amendment Directions, 2026'
        -> 'prudential norms on capital adequacy'
    """
    after = title.split("All India Financial Institutions", 1)
    if len(after) < 2:
        return ""
    tail = after[1]
    tail = tail.replace("(AIFIs)", "")  # drop the nested abbreviation
    tail = tail.lstrip(" -")            # drop the leading dash/spaces
    # Topic is everything up to the ")" that closes the RBI(...) title clause.
    topic = tail.split(")", 1)[0]
    # Normalize: lowercase, strip punctuation, collapse spaces.
    topic = re.sub(r"[^a-z0-9]+", " ", topic.lower()).strip()
    return topic


# --- Per-file processing ---------------------------------------------------
def process_pdf(path: Path) -> dict:
    """Read one PDF's first pages and build a manifest row (pre amends-linking)."""
    rel_path = path.relative_to(PROJECT_ROOT).as_posix()
    division = path.parent.name
    notes: list[str] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        raw_p1 = pdf.pages[0].extract_text() or ""

        # Title/circular/issue-date all live on page 1.
        text = normalize(raw_p1)
        title = extract_title(text)
        doc_type = detect_doc_type(title)

        # The commencement ("come into force/effect") clause can sit deeper:
        #  * amendments are short — scan the whole document (the effective date
        #    is the single most important field, so we accept the extra reads);
        #  * master directions are long and full of transitional dates, so we
        #    only scan the first few pages (Chapter I - Preliminary) to avoid
        #    grabbing an unrelated "with effect from" from the body.
        if doc_type == "amendment":
            scan_pages = range(page_count)
        else:
            scan_pages = range(min(4, page_count))
        commence_text = normalize(
            " ".join(pdf.pages[i].extract_text() or "" for i in scan_pages)
        )

    if not title:
        notes.append("title not auto-detected")
    circular_no = extract_circular_no(text)
    if not circular_no:
        notes.append("circular_no not auto-detected")

    issue_date = to_iso(DATE_RE.search(text))  # first date on page 1 = issue date
    if not issue_date:
        notes.append("issue_date not auto-detected")

    consolidated_as_of = extract_consolidated_as_of(text)

    applicable_from, appl_note = extract_applicable_from(commence_text)
    if not applicable_from:
        # Fall back to issue_date for the "immediate/on issue" cases, but keep
        # the flag so a human confirms the interpretation.
        if "issue_date" in appl_note and issue_date:
            applicable_from = issue_date
        notes.append(appl_note)

    return {
        "doc_id": slugify(path.stem),
        "division": division,
        "doc_type": doc_type,
        "title": title,
        "circular_no": circular_no,
        "issue_date": issue_date,
        "applicable_from": applicable_from,
        "amends": "",               # filled in the second pass
        "consolidated_as_of": consolidated_as_of,
        "file_path": rel_path,
        "source_url": "",           # filled later (RBI deep-links)
        "dept_ref": extract_dept_ref(text),
        "page_count": page_count,
        "needs_verification": "",   # decided after amends-linking
        "notes": "; ".join(n for n in notes if n),
        # transient helper, dropped before writing CSV:
        "_topic": extract_topic(title),
    }


def link_amendments(rows: list[dict]) -> None:
    """Second pass: point each amendment's `amends` at its parent master
    direction, matched by regulated topic (falls back to same division)."""
    masters = [r for r in rows if r["doc_type"] == "master_direction"]

    for row in rows:
        if row["doc_type"] != "amendment":
            continue

        topic = row["_topic"]
        candidates = [m for m in masters if m["_topic"] == topic]

        note = ""
        if len(candidates) == 1:
            row["amends"] = candidates[0]["doc_id"]
        elif len(candidates) > 1:
            # Prefer one in the same division folder.
            same_div = [m for m in candidates if m["division"] == row["division"]]
            if len(same_div) == 1:
                row["amends"] = same_div[0]["doc_id"]
                note = "parent matched by topic + division (multiple topic matches)"
            else:
                note = f"AMBIGUOUS parent: {len(candidates)} topic matches — verify"
        else:
            note = "NO parent master direction found by topic — verify"

        if note:
            row["notes"] = "; ".join(x for x in (row["notes"], note) if x)


def main() -> int:
    if not SOURCE_DIR.exists():
        print(f"ERROR: source folder not found: {SOURCE_DIR}", file=sys.stderr)
        return 1

    pdf_paths = sorted(SOURCE_DIR.rglob("*.pdf"))
    if not pdf_paths:
        print(f"ERROR: no PDFs found under {SOURCE_DIR}", file=sys.stderr)
        return 1

    print(f"Found {len(pdf_paths)} PDFs under '{SOURCE_DIR.name}'. Parsing...\n")

    rows: list[dict] = []
    for path in pdf_paths:
        try:
            rows.append(process_pdf(path))
        except Exception as exc:  # keep going; flag the failure in the manifest
            print(f"  ! failed to parse {path.name}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "doc_id": slugify(path.stem),
                    "division": path.parent.name,
                    "doc_type": "",
                    "title": "",
                    "circular_no": "",
                    "issue_date": "",
                    "applicable_from": "",
                    "amends": "",
                    "consolidated_as_of": "",
                    "file_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "source_url": "",
                    "dept_ref": "",
                    "page_count": "",
                    "needs_verification": "",
                    "notes": f"PARSE ERROR: {exc}",
                    "_topic": "",
                }
            )

    link_amendments(rows)

    # Decide the verification flag: every amendment (amends chain must be
    # human-checked per CLAUDE.md) plus any row that couldn't be fully parsed.
    for row in rows:
        needs = row["doc_type"] == "amendment" or bool(row["notes"])
        row["needs_verification"] = "yes" if needs else "no"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # --- Console summary so the run is self-explaining ---
    n_master = sum(r["doc_type"] == "master_direction" for r in rows)
    n_amend = sum(r["doc_type"] == "amendment" for r in rows)
    n_flag = sum(r["needs_verification"] == "yes" for r in rows)
    print(f"Parsed {len(rows)} docs: {n_master} master directions, {n_amend} amendments.")
    print(f"Flagged for human verification: {n_flag}")
    print(f"\nManifest written to: {OUTPUT_CSV.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
