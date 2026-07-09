"""
build_chunks.py — Stage 2 of the RBI Compliance Assistant ingestion pipeline.

Turns each source PDF into retrieval-ready chunks, written to data/chunks.jsonl
(one JSON object per line). Every chunk carries enough metadata to (a) be
retrieved and version-filtered, and (b) be cited back to an exact page +
section so the frontend can deep-link into the PDF preview.

Design rules (from CLAUDE.md + NOTES.md):
  * Source PDFs are never modified — read only.
  * Chunk by SECTION / paragraph, not blind fixed-size: we start a new chunk at
    chapter boundaries and split long sections at paragraph boundaries.
  * Every chunk records page_start, page_end and a section_ref (chapter path),
    plus the document's manifest metadata (dates, amendment chain) denormalized
    onto the chunk so retrieval is self-contained.
  * Running headers (the repeated title) and the RBI address/telephone footer
    are stripped so they don't pollute chunk text.

Run:  python src/build_chunks.py   (reads data/manifest.csv, writes data/chunks.jsonl)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = PROJECT_ROOT / "data" / "manifest.csv"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "chunks.jsonl"

# Chunk sizing (characters). ~1600 chars ≈ 380 tokens — comfortable for BGE/E5.
MAX_CHARS = 1600      # soft cap: once exceeded, split at the next paragraph start
HARD_CAP = 2600       # hard cap: force a split even mid-section (runaway paragraph)
MIN_CHARS = 60        # drop pure-noise fragments shorter than this

# --- Structural markers in RBI directions -----------------------------------
CHAPTER_RE = re.compile(r"^(chapter|part)\s+[-\divxlc]+\b", re.IGNORECASE)
# A new paragraph/section start: "1.", "5A.", "126A.", "A.", "(i)", "(a)", "(1)"
PARA_START_RE = re.compile(r"^(\d+[A-Z]?\.|\([ivxlca-z0-9]+\)|[A-Z]\.)\s")

# Lines that are pure boilerplate noise (whole-line match): bare bank name,
# rule lines, page numbers.
JUNK_RE = re.compile(r"^(reserve bank of india|_+|-+|\d{1,4}|page\s+\d+.*)$", re.IGNORECASE)
# Footer/header tells found ANYWHERE in the line. RBI PDFs prepend the Hindi
# address on the same line as the English one, so we match by substring rather
# than anchoring to the line start.
FOOTER_SUBSTR = ("www.rbi.org.in", "tel no", "fax no", "department of regulation",
                 "table of content")
# Footer address block carries the Mumbai pincode; a reliable tell.
PINCODE_RE = re.compile(r"400\s?001")
# A run of dot leaders ("......") marks a Table-of-Contents entry, not body text.
DOT_LEADER_RE = re.compile(r"\.{4,}")
# Devanagari block — the bilingual header/footer; strip so it doesn't add noise.
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]+")


def normalize(text: str) -> str:
    """Fix the odd dash/quote encodings and drop Hindi; keep line structure."""
    if not text:
        return ""
    for ch in ("–", "—", "‒", "‐", "�"):
        text = text.replace(ch, "-")
    text = text.replace("’", "'").replace("‘", "'")
    return DEVANAGARI_RE.sub("", text)


def collapse(text: str) -> str:
    """Whitespace-collapse a single line for comparison/output."""
    return re.sub(r"\s+", " ", text).strip()


def find_running_lines(pages: list[str], n_pages: int) -> set[str]:
    """Detect lines repeated across many pages (running header/footer)."""
    counts: Counter[str] = Counter()
    for page in pages:
        seen = {collapse(ln) for ln in page.splitlines() if collapse(ln)}
        counts.update(seen)
    # Appears on a large share of pages -> boilerplate. Guard for short docs.
    threshold = max(3, int(0.4 * n_pages))
    return {ln for ln, c in counts.items() if c >= threshold and len(ln) > 3}


def is_junk(line: str, running: set[str], title: str) -> bool:
    c = collapse(line)
    if not c:
        return True
    if c in running:
        return True
    if title and c.lower() == title.lower():
        return True
    low = c.lower()
    if any(s in low for s in FOOTER_SUBSTR):
        return True
    if PINCODE_RE.search(c):
        return True
    if DOT_LEADER_RE.search(c):        # table-of-contents entry
        return True
    if JUNK_RE.match(c):
        return True
    return False


def clean_lines(page_text: str, running: set[str], title: str) -> list[str]:
    """Return the meaningful lines of a page, boilerplate removed.

    Also drops the *head* of a wrapped Table-of-Contents entry: TOC lines wrap,
    with the page-number half carrying the dot leaders. We already drop the
    dotted continuation; here we also drop any line whose next line is a dotted
    continuation, so a wrapped heading like "Chapter V - Filing ... (other than"
    isn't mistaken for a real body heading. Body text is never followed by a
    dot-leader line, so this is safe.
    """
    raw_lines = [collapse(r) for r in page_text.splitlines()]
    out = []
    for j, c in enumerate(raw_lines):
        if is_junk(c, running, title):
            continue
        nxt = raw_lines[j + 1] if j + 1 < len(raw_lines) else ""
        if DOT_LEADER_RE.search(nxt):   # this line is a wrapped TOC head
            continue
        out.append(c)
    return out


def flush(buf: list[str], page_start: int, page_end: int, chapter: str) -> dict | None:
    """Turn the current line buffer into a chunk record (text + provenance)."""
    text = " ".join(buf).strip()
    if len(text) < MIN_CHARS:
        return None
    return {
        "text": text,
        "page_start": page_start,
        "page_end": page_end,
        "section_ref": chapter,
        "char_count": len(text),
    }


def chunk_document(pdf, title: str) -> list[dict]:
    """Section-aware chunking of one open PDF. Returns list of chunk dicts."""
    raw_pages = [normalize(p.extract_text() or "") for p in pdf.pages]
    running = find_running_lines(raw_pages, len(raw_pages))

    # Flatten to a sequence of (page_no, line) with boilerplate stripped.
    flat: list[tuple[int, str]] = []
    for page_no, page_text in enumerate(raw_pages, start=1):
        for line in clean_lines(page_text, running, title):
            flat.append((page_no, line))

    chunks: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    page_start = page_end = None
    chapter = ""  # running chapter/part heading = section_ref

    for i, (page_no, line) in enumerate(flat):
        is_chapter = bool(CHAPTER_RE.match(line))

        # Chapter boundary: close the current chunk before starting the new one.
        if is_chapter and buf:
            rec = flush(buf, page_start, page_end, chapter)
            if rec:
                chunks.append(rec)
            buf, buf_len, page_start = [], 0, None
        if is_chapter:
            chapter = line

        if page_start is None:
            page_start = page_no
        page_end = page_no
        buf.append(line)
        buf_len += len(line) + 1

        # Decide whether to split after this line.
        next_starts_para = (
            i + 1 < len(flat)
            and bool(PARA_START_RE.match(flat[i + 1][1]) or CHAPTER_RE.match(flat[i + 1][1]))
        )
        if (buf_len >= MAX_CHARS and next_starts_para) or buf_len >= HARD_CAP:
            rec = flush(buf, page_start, page_end, chapter)
            if rec:
                chunks.append(rec)
            buf, buf_len, page_start = [], 0, None

    if buf:
        rec = flush(buf, page_start, page_end, chapter)
        if rec:
            chunks.append(rec)
    return chunks


def load_manifest() -> list[dict]:
    with MANIFEST_CSV.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# Manifest fields copied onto every chunk so retrieval results are self-contained
# (citations + version filtering need no join back to the manifest).
CARRIED_FIELDS = [
    "doc_id", "division", "doc_type", "title", "circular_no",
    "issue_date", "applicable_from", "consolidated_as_of", "amends",
]


def main() -> int:
    if not MANIFEST_CSV.exists():
        print(f"ERROR: manifest not found at {MANIFEST_CSV} — run build_manifest.py first",
              file=sys.stderr)
        return 1

    rows = load_manifest()
    print(f"Chunking {len(rows)} documents from the manifest...\n")

    all_chunks: list[dict] = []
    per_doc: list[tuple[str, int]] = []

    for row in rows:
        pdf_path = PROJECT_ROOT / row["file_path"]
        if not pdf_path.exists():
            print(f"  ! missing file, skipped: {row['file_path']}", file=sys.stderr)
            continue

        with pdfplumber.open(pdf_path) as pdf:
            doc_chunks = chunk_document(pdf, row["title"])

        meta = {k: row[k] for k in CARRIED_FIELDS}
        for idx, ch in enumerate(doc_chunks):
            ch.update(meta)
            ch["chunk_id"] = f"{row['doc_id']}::{idx:04d}"
            all_chunks.append(ch)
        per_doc.append((row["doc_id"], len(doc_chunks)))

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as fh:
        for ch in all_chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # --- Console summary ---
    sizes = [c["char_count"] for c in all_chunks]
    avg = sum(sizes) // len(sizes) if sizes else 0
    print(f"Wrote {len(all_chunks)} chunks from {len(per_doc)} docs -> "
          f"{OUTPUT_JSONL.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Chunk size chars: min {min(sizes)}, avg {avg}, max {max(sizes)}\n")
    print("Chunks per document (a few):")
    for doc_id, n in sorted(per_doc, key=lambda x: -x[1])[:5]:
        print(f"  {n:4}  {doc_id}")
    fewest = min(per_doc, key=lambda x: x[1])
    print(f"  ...\n  {fewest[1]:4}  {fewest[0]}  (fewest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
