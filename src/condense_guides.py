"""
condense_guides.py — bring verbose generated guides into the house style.

build_explanations.py extracts thoroughly: for a 400-page direction it walks
every chapter in order and can emit ~1,900 lines. That is faithful but unusable
as a "Key points" tab, and it hurts citations too — restated prose produces many
lines with no distinctive anchor for bake_guide_citations to pin to a page.

The 34 AIFI guides are the reference style: ~25-40 lines, a one-line purpose,
"### Chapter" headings, dense bullets with the exact numbers kept in bold. This
script rewrites over-long key_points into that shape, using a real AIFI guide as
the exemplar. It is a REDUCE step over text we already generated — it never
reads the PDF, so it cannot introduce a rule that was not already extracted.

Guides at or under THRESHOLD_LINES are left untouched (so the hand-reviewed AIFI
set is never rewritten). The originals are backed up first.

Run:  python src/condense_guides.py [--threshold 60] [--dry-run] [--only <doc_id>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from config import DATA_DIR
from llm import generate, LLMError

EXPLANATIONS_DIR = DATA_DIR / "explanations"
BACKUP_DIR = DATA_DIR / "explanations_verbose_backup"

# Guides longer than this get condensed. The AIFI set runs 15-40 lines, and the
# longest legitimate one is ~45, so 60 leaves headroom and only catches the
# runaway generated ones.
THRESHOLD_LINES = 60

# A real AIFI guide, used as a few-shot style exemplar.
EXEMPLAR_DOC_ID = "concentration-risk-management-directions-2025"

SYSTEM = """You rewrite an over-long RBI compliance guide into a tight house-style guide.

You are given a THOROUGH but far too long explanation of one RBI direction. \
Compress it to roughly 25-45 lines while keeping everything a compliance officer \
needs.

STRICT RULES:
- Use ONLY facts present in the input. Never add a rule, limit, threshold, date \
or obligation that is not there. This is a compression task, not a writing task.
- KEEP EVERY NUMBER: percentages, ratios, amounts, day-counts, thresholds and \
effective dates must survive verbatim. Losing a number is the worst failure.
- Preserve the applicability framing exactly as the input states it (who the \
direction binds, and any caution about whether it applies to NaBFID).
- Merge repetition ruthlessly: drop restated definitions, table-of-contents \
padding, and sentences that only paraphrase a heading.

FORMAT (match it exactly):
## <Topic> — <Directions, YYYY> — Detailed Compliance Guide
*(<circular no.>, dated <YYYY-MM-DD>)*

<ONE sentence saying what the direction governs and why it exists.>

### Chapter <N> — <short chapter name>
- **<Label>:** dense bullet, exact numbers in **bold**.
- ...

Use "### Chapter ..." headings in document order. Bullets may nest one level. \
Bold the key term at the start of a bullet and every figure. No closing summary, \
no preamble, no meta commentary about the rewrite."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _exemplar() -> str:
    p = EXPLANATIONS_DIR / f"{EXEMPLAR_DOC_ID}.json"
    if not p.exists():
        return ""
    return (_load(p).get("key_points") or "")[:5000]


def condense(text: str, exemplar: str) -> str:
    prompt = (
        "Here is a guide in the target house style, for reference only — do NOT "
        "copy its content:\n\n<<<EXEMPLAR\n" + exemplar + "\nEXEMPLAR\n\n"
        "Now rewrite the following guide into that same style. Keep every "
        "number and the applicability framing; cut the length to roughly 25-45 "
        "lines.\n\n<<<GUIDE\n" + text + "\nGUIDE"
    )
    # Near-zero temperature: this is compression, not composition.
    return generate(SYSTEM, prompt, max_tokens=4000).strip()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=THRESHOLD_LINES)
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be condensed; make no changes.")
    ap.add_argument("--only", help="Condense a single doc_id (for spot checks).")
    args = ap.parse_args(argv[1:])

    paths = sorted(EXPLANATIONS_DIR.glob("*.json"))
    todo = []
    for p in paths:
        if args.only and p.stem != args.only:
            continue
        kp = (_load(p).get("key_points") or "")
        n = len(kp.splitlines())
        if n > args.threshold:
            todo.append((p, n, len(kp)))

    if not todo:
        print("Nothing over the threshold — all guides are already concise.")
        return 0

    print(f"{len(todo)} guide(s) over {args.threshold} lines:")
    for p, n, c in sorted(todo, key=lambda x: -x[1])[:10]:
        print(f"  {n:5d} lines  {c/1024:7.1f} KB  {p.stem[:52]}")
    if len(todo) > 10:
        print(f"  ... and {len(todo)-10} more")
    if args.dry_run:
        return 0

    exemplar = _exemplar()
    if not exemplar:
        print("ERROR: style exemplar not found — aborting.", file=sys.stderr)
        return 1

    # Back up before overwriting: the verbose text is the only copy of that
    # extraction, and re-generating it costs another full pass over the PDFs.
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for i, (p, n, _c) in enumerate(todo, start=1):
        bak = BACKUP_DIR / p.name
        if not bak.exists():
            shutil.copyfile(p, bak)
        data = _load(p)
        try:
            new = condense(data["key_points"], exemplar)
        except LLMError as exc:
            print(f"  {i:2}/{len(todo)} [ERR] {p.stem[:44]}: {exc}", file=sys.stderr)
            failed += 1
            continue
        if len(new.splitlines()) > n:          # never make it worse
            print(f"  {i:2}/{len(todo)} [skip, grew] {p.stem[:44]}")
            continue
        data["key_points"] = new
        # Baked citations are keyed by a hash of each line, so rewriting the
        # lines invalidates them. Drop them here and re-run
        # bake_guide_citations.py afterwards.
        data.pop("citations", None)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        done += 1
        print(f"  {i:2}/{len(todo)} {n:5d} -> {len(new.splitlines()):4d} lines  "
              f"{p.stem[:48]}", flush=True)
        time.sleep(0.3)

    print(f"\nCondensed {done}, failed {failed}. Originals in {BACKUP_DIR.name}/")
    print("NEXT: python src/bake_guide_citations.py   (citations were dropped)")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
