"""
build_explanations.py — pre-bake all document explanations to disk (offline).

Generates key_points (+ what_changed for amendments + implications) for every
document and saves them under data/explanations/<doc_id>.json. Once baked, the
app serves them from disk with NO LLM call — instant, free, and surviving
restarts / next-day use. Only the chatbot uses the LLM at runtime.

This uses the configured LLM (see .env), so running it spends API credit ONCE
(~$1.5 on Haiku for the full corpus). After that, explanations are free forever.
Re-running skips docs already baked (unless --force).

Run:  python src/build_explanations.py [--force] [--only key_points]
"""

from __future__ import annotations

import argparse
import sys
import time

from config import LLM_MODEL
import docstore
import explain
from llm import LLMError

KINDS = ("key_points", "what_changed", "implications")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Pre-bake document explanations.")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if already saved.")
    ap.add_argument("--only", choices=KINDS, help="Bake only one kind of section.")
    args = ap.parse_args(argv[1:])

    docs = [d for div in docstore.divisions()
            for d in docstore.documents_in(div["division"])]
    kinds = (args.only,) if args.only else KINDS
    gen = {"key_points": explain.key_points,
           "what_changed": explain.what_changed,
           "implications": explain.implications}

    print(f"Baking explanations for {len(docs)} docs on {LLM_MODEL}...\n", flush=True)
    made = skipped = failed = 0
    for n, doc in enumerate(docs, start=1):
        saved = explain.load_saved(doc["doc_id"])
        for kind in kinds:
            # what_changed only applies to amendments.
            if kind == "what_changed" and doc["doc_type"] != "amendment":
                continue
            if saved.get(kind) and not args.force:
                skipped += 1
                continue
            try:
                text = gen[kind](doc)
                explain.save_section(doc["doc_id"], kind, text)
                made += 1
                print(f"  {n:2}/{len(docs)} [{kind}] {doc['doc_id'][:45]}", flush=True)
            except LLMError as exc:
                failed += 1
                print(f"  {n:2}/{len(docs)} [ERR {kind}] {doc['doc_id'][:40]}: "
                      f"{str(exc)[:50]}", flush=True)
            time.sleep(0.3)

    print(f"\nDone. generated={made}, skipped(existing)={skipped}, failed={failed}")
    print(f"Saved under: {explain.EXPLANATIONS_DIR.relative_to(explain.DATA_DIR.parent).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
