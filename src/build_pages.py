"""
build_pages.py — extract the exact text of every PDF page -> data/pages.jsonl.

Ground truth for citations: a citation may only point at a page whose text
verifiably contains the cited fact (see pagetext.py). Chunks can't serve that
purpose because 74% of them span multiple pages. One row per physical page:

    {"doc_id": ..., "page": 1-based physical page, "text": raw page text}

Page numbers match both build_chunks.py (pdfplumber, enumerate(start=1)) and
the browser PDF viewer's "#page=N" fragment — all physical and 1-based.

Run:  python src/build_pages.py          (seconds; re-run after adding PDFs)
"""

from __future__ import annotations

import csv
import json
import sys

import fitz  # PyMuPDF — already a dependency (PDF page rendering in the app)

from config import MANIFEST_CSV, PAGES_JSONL, PROJECT_ROOT


def load_manifest() -> list[dict]:
    with MANIFEST_CSV.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    if not MANIFEST_CSV.exists():
        print(f"ERROR: manifest not found at {MANIFEST_CSV}", file=sys.stderr)
        return 1

    rows = load_manifest()
    print(f"Extracting page text for {len(rows)} documents...\n")

    n_pages = 0
    empty: list[tuple[str, int]] = []   # (doc_id, page) with no/near-no text
    with PAGES_JSONL.open("w", encoding="utf-8") as out:
        for row in rows:
            pdf_path = PROJECT_ROOT / row["file_path"]
            if not pdf_path.exists():
                print(f"  ! missing file, skipped: {row['file_path']}",
                      file=sys.stderr)
                continue
            with fitz.open(pdf_path) as pdf:
                doc_pages = pdf.page_count
                for i, page in enumerate(pdf, start=1):
                    text = page.get_text()
                    if len(text.strip()) < 40:      # likely scanned / blank
                        empty.append((row["doc_id"], i))
                    out.write(json.dumps(
                        {"doc_id": row["doc_id"], "page": i, "text": text},
                        ensure_ascii=False) + "\n")
                    n_pages += 1
            print(f"  {row['doc_id'][:60]:60s} {doc_pages:4d} pages")

    print(f"\nWrote {n_pages} pages -> {PAGES_JSONL}")
    if empty:
        print(f"\nWARNING — {len(empty)} pages with little/no text "
              f"(scanned? citations can't be page-verified there):")
        for doc_id, pg in empty:
            print(f"  {doc_id}  p.{pg}")
    else:
        print("All pages have a text layer — every page is verifiable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
