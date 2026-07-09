"""
eval_runner.py — measure the assistant against data/eval_set.jsonl.

Scores the three things that matter (CLAUDE.md quality bar):
  * Retrieval    — does the expected source document appear in what we retrieve?
  * Groundedness — for answered questions, are the expected facts present AND is
                   a source cited? (proxy for "answer supported by cited text")
  * Abstention   — for out-of-corpus questions, does it correctly refuse?

Writes data/eval_results.csv and prints a summary. Run:
    python src/eval_runner.py
"""

from __future__ import annotations

import csv
import json
import os
import time

# The eval uses whatever provider/model is configured in .env (LLM_PROVIDER +
# LLM_MODEL). Set EVAL_MODEL to override just the model for an eval run without
# touching .env (set before importing config so it wins — config's load_dotenv
# does not override an already-set env var).
if os.getenv("EVAL_MODEL"):
    os.environ["LLM_MODEL"] = os.environ["EVAL_MODEL"]

from config import DATA_DIR, LLM_MODEL  # noqa: E402
from retrieval import retrieve  # noqa: E402
from query import answer  # noqa: E402
from llm import LLMError  # noqa: E402

# Small pause between questions. Sized for free-tier per-minute limits; paid
# Anthropic limits are generous, so keep it short.
THROTTLE_S = float(os.getenv("EVAL_THROTTLE", "0.5"))

EVAL_SET = DATA_DIR / "eval_set.jsonl"
RESULTS_CSV = DATA_DIR / "eval_results.csv"

import re  # noqa: E402


def _norm(s: str) -> str:
    """Normalize for tolerant term matching: 'per cent'/'percent' -> '%', and
    drop whitespace so '20 per cent', '20%', '20 %' all compare equal."""
    s = s.lower().replace("per cent", "%").replace("percent", "%")
    return re.sub(r"\s+", "", s)


def load_eval() -> list[dict]:
    with EVAL_SET.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def score_one(item: dict, retrieval_only: bool = False) -> dict:
    q = item["question"]
    should_abstain = item["should_abstain"]

    # Retrieve ONCE and reuse for both the retrieval metric and generation
    # (avoids a second, CPU-heavy rerank pass per question).
    chunks = retrieve(q)
    retrieved_docs = {c["doc_id"] for c in chunks}
    retrieval_hit = bool(set(item["expected_docs"]) & retrieved_docs)

    # Retrieval-only mode skips the LLM entirely (free + instant). Useful when the
    # free-tier generation quota is exhausted — retrieval is the #1 quality metric.
    if retrieval_only:
        return {
            "id": item["id"], "should_abstain": should_abstain, "abstained": "",
            "retrieval_hit": retrieval_hit if not should_abstain else "",
            "terms_ok": "", "terms_found": "", "cited": "",
            "passed": retrieval_hit if not should_abstain else "",
            "question": q, "expected_docs": "|".join(item["expected_docs"]),
            "retrieved_or_cited": "|".join(sorted(retrieved_docs))[:120],
        }

    # Generation (reuses the chunks above).
    res = answer(q, prefetched=chunks)
    ans = res["answer"]
    abstained = res["abstained"]

    cited_docs = {c["doc_id"] for c in res["sources"]}
    ans_norm = _norm(ans)
    terms_present = [t for t in item["expected_terms"] if _norm(t) in ans_norm]
    terms_ok = len(terms_present) == len(item["expected_terms"])
    cited = bool(res["sources"])  # answered with sources shown

    if should_abstain:
        passed = abstained
    else:
        passed = (not abstained) and retrieval_hit and terms_ok and cited

    return {
        "id": item["id"],
        "should_abstain": should_abstain,
        "abstained": abstained,
        "retrieval_hit": retrieval_hit if not should_abstain else "",
        "terms_ok": terms_ok if not should_abstain else "",
        "terms_found": f"{len(terms_present)}/{len(item['expected_terms'])}"
                       if not should_abstain else "",
        "cited": cited if not should_abstain else "",
        "passed": passed,
        "question": q,
        "expected_docs": "|".join(item["expected_docs"]),
        "retrieved_or_cited": "|".join(sorted(cited_docs))[:120],
    }


def _write(rows: list[dict], path) -> None:
    """(Re)write the results CSV. Called after every question so a run that is
    interrupted or throttled still leaves a usable partial scorecard."""
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate the RBI assistant.")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="Measure retrieval only (no LLM calls; free + instant).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N questions (to stay within quota).")
    args = ap.parse_args(argv[1:])

    items = load_eval()
    if args.limit:
        items = items[:args.limit]
    mode = "retrieval-only" if args.retrieval_only else f"full ({LLM_MODEL})"
    out = RESULTS_CSV.with_name(
        "eval_results_retrieval.csv" if args.retrieval_only else "eval_results.csv")
    print(f"Running {len(items)} questions — mode: {mode}\n", flush=True)

    rows = []
    for n, it in enumerate(items, start=1):
        try:
            r = score_one(it, retrieval_only=args.retrieval_only)
        except LLMError as exc:
            r = {"id": it["id"], "should_abstain": it["should_abstain"],
                 "abstained": "", "retrieval_hit": "", "terms_ok": "",
                 "terms_found": "", "cited": "", "passed": False,
                 "question": it["question"], "expected_docs": "|".join(it["expected_docs"]),
                 "retrieved_or_cited": f"LLM_ERROR: {str(exc)[:80]}"}
            print(f"  {n:2}/{len(items)} [ERR ] {it['id']}  {str(exc)[:55]}", flush=True)
        else:
            mark = "PASS" if r["passed"] is True else ("—" if r["passed"] == "" else "FAIL")
            extra = ("abstained" if r["abstained"] else
                     f"ret={r['retrieval_hit']} terms={r['terms_found']}")
            print(f"  {n:2}/{len(items)} [{mark:4}] {r['id']}  {extra}", flush=True)
        rows.append(r)
        _write(rows, out)                      # persist after every question
        if not args.retrieval_only:
            time.sleep(THROTTLE_S)

    # --- Summary ---
    grounded = [r for r in rows if not r["should_abstain"]]
    abstain = [r for r in rows if r["should_abstain"]]
    ret = sum(1 for r in grounded if r["retrieval_hit"] is True)

    def pct(n, d): return f"{n}/{d} ({100*n//d if d else 0}%)"

    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY — mode: {mode}")
    print("=" * 60)
    print(f"Grounded questions:        {len(grounded)}")
    print(f"  Retrieval hit@6:         {pct(ret, len(grounded))}")
    if not args.retrieval_only:
        terms = sum(1 for r in grounded if r["terms_ok"] is True)
        gp = sum(1 for r in grounded if r["passed"] is True)
        ap_ = sum(1 for r in abstain if r["passed"] is True)
        errs = sum(1 for r in rows if str(r["retrieved_or_cited"]).startswith("LLM_ERROR"))
        print(f"  Expected facts present:  {pct(terms, len(grounded))}")
        print(f"  Fully correct:           {pct(gp, len(grounded))}")
        print(f"Abstention questions:      {len(abstain)}")
        print(f"  Correctly abstained:     {pct(ap_, len(abstain))}")
        if errs:
            print(f"  (LLM errors, not scored:  {errs})")
    print(f"\nResults -> {out.relative_to(DATA_DIR.parent).as_posix()}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
