"""
query.py — the generation half of the live query pipeline (grounded answers).

    retrieve() -> build a SOURCES block -> LLM answers ONLY from it -> answer
    with citations + the source snippets shown + abstention when not found.

This is the heart of the "grounded, cited, abstains" behaviour from CLAUDE.md.
Run from the terminal to test end-to-end:

    python src/query.py "What is the single counterparty exposure limit?"
"""

from __future__ import annotations

import re
import sys
import textwrap

from llm import generate, LLMError
from retrieval import retrieve
import pagetext

# The exact abstention sentence. The model is told to use it verbatim, so we can
# detect it and so the user sees consistent, honest "I don't know" behaviour.
ABSTAIN = "I could not find this in the circulars I have."

SYSTEM_PROMPT = f"""You are the RBI Compliance Assistant for NaBFID (a Development \
Finance Institution). You help staff understand RBI circulars and directions.

You answer ONLY using the SOURCES provided in the user message (excerpts from RBI \
directions/amendments). Follow these rules strictly:

1. Use ONLY the SOURCES for any regulatory statement. Never use outside knowledge \
for what a rule says.
2. Cite using the bracketed SOURCE NUMBER(S) from the SOURCES block, placed at the \
END of the sentence or clause they support, like this: "... shall be at least 9 per \
cent.[1]". To cite several sources for one statement, list them together like \
"[1][3]" or "[1, 3]". Use ONLY these bracketed numbers as citations — do NOT write \
the circular number, date, or page inside the sentence (the numbered source list \
carries those details). Every factual/regulatory statement must carry at least one \
bracketed citation, placed at the end of that sentence.
3. If the SOURCES do not contain the answer, reply with EXACTLY this sentence and \
nothing else: "{ABSTAIN}" Do not guess or fill gaps from memory.
4. Amendments override the base direction for any conflicting provision. If a \
source is an amendment, its text prevails. State the version story when relevant, \
e.g. "Per the [master direction], as amended by [amendment] effective <date>: ...".
5. Mention the effective date when it matters. If a change is announced but its \
effective date is in the future, say so explicitly.
6. Give a detailed, well-structured answer, not a one-liner. Start with a short \
direct answer, then elaborate: explain the relevant provision in plain language, \
and include the definitions, thresholds, conditions, exceptions, and effective \
dates that the SOURCES contain on the point. Use short headings or bullet points \
where it helps readability. Elaborate ONLY with what is in the SOURCES — never \
add an obligation, limit, or condition that is not there, and do not pad with \
generic filler.
7. After the answer, add a plain-language illustrative example, introduced by a \
line containing exactly ===EXAMPLE=== and then the example itself.
Purpose: let a NON-TECHNICAL reader see how the rule plays out in practice at an \
AIFI. NaBFID context to use: a Development Finance Institution — it takes no \
retail deposits, raises money by issuing bonds, and lends long-tenor (15-20 year) \
to large infrastructure projects (roads, ports, power, renewables).
Rules for the example (these are strict):
- It may ONLY illustrate rules you already stated in the answer above (which came \
from the SOURCES). NEVER introduce an obligation, limit, threshold, condition, \
rate or date that is not in the SOURCES.
- Make it concrete: a short "Suppose NaBFID ..." scenario, walked through step by \
step, ending with what the AIFI must actually do. Any amounts you invent must be \
clearly hypothetical and must be consistent with the thresholds in the SOURCES.
- Simple language, no jargon; expand any technical term you must use.
- Do NOT put bracketed citations in the example — it is explanatory framing, not \
regulatory text.
- If you replied with the abstention sentence, or the point does not lend itself \
to a meaningful example, omit this block entirely.
8. Finally, append a machine-parsed block (hidden from the reader) in EXACTLY \
this format:
===CITES===
1: "a short quote of at most 20 words copied VERBATIM from SOURCE 1's text"
3: "..."
One line per source number you actually cited in the answer; the quote must be \
an exact, contiguous excerpt of that source (it is used to locate the exact PDF \
page). Nothing else goes in the block. If you replied with the abstention \
sentence, do not add this block.

Order of your reply: the cited answer, then ===EXAMPLE=== (if any), then \
===CITES===.
"""

# Parses one line of the ===CITES=== block: `3: "verbatim quote"` (any quote style).
_CITE_LINE_RE = re.compile(r'^\s*(\d+)\s*[:.\-]\s*["“](.+?)["”]?\s*$')


def _split_blocks(text: str) -> tuple[str, str, dict[int, str]]:
    """Split the raw model reply into its three parts.

    The model replies with: the cited answer, then an optional ===EXAMPLE===
    block (plain-language framing), then a machine-only ===CITES=== block of
    verbatim quotes. Returns (answer, example, {source_no: quote}). The example
    is kept SEPARATE from the answer so the app can render it in a distinct
    callout — CLAUDE.md requires illustrative framing to be visually
    distinguished from the cited regulatory substance.
    """
    body, _, cites_block = text.partition("===CITES===")
    quotes: dict[int, str] = {}
    for line in cites_block.splitlines():
        m = _CITE_LINE_RE.match(line)
        if m:
            quotes[int(m.group(1))] = m.group(2).strip()
    answer_text, _, example = body.partition("===EXAMPLE===")
    return answer_text.strip(), example.strip(), quotes


def build_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as a numbered SOURCES block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = (f"[{i}] {c['circular_no']}, dated {c['issue_date']} "
                  f"| {c['division']} | {c['doc_type']} "
                  f"| page {c['page_start']} | {c['section_ref'] or '-'}")
        eff = c.get("applicable_from")
        if eff:
            header += f" | effective {eff}"
        blocks.append(header + "\n" + c["text"])
    return "\n\n".join(blocks)


def answer(question: str, *, scope_doc_ids: set[str] | None = None,
           prefetched: list[dict] | None = None) -> dict:
    """Run retrieval + grounded generation. Returns answer + sources + flags.

    prefetched: reuse already-retrieved chunks (skips retrieval) — lets callers
    like the eval runner retrieve once and avoid a second rerank pass.
    """
    chunks = prefetched if prefetched is not None else retrieve(
        question, scope_doc_ids=scope_doc_ids)
    if not chunks:
        return {"answer": ABSTAIN, "sources": [], "abstained": True, "example": ""}

    prompt = (f"SOURCES:\n{build_context(chunks)}\n\n"
              f"QUESTION: {question}\n\n"
              f"Answer using only the SOURCES above. Be detailed and "
              f"well-structured, end each supported sentence with its "
              f"bracketed source number(s), e.g. [1] or [1][3]. Then give the "
              f"===EXAMPLE=== block (a plain-language NaBFID/AIFI scenario "
              f"illustrating only what you stated above), and finish with the "
              f"===CITES=== block of verbatim quotes.")
    # Detailed answers + an illustrative example need room. (Thinking is off in
    # the adapter, so the whole budget goes to the visible reply.)
    text = generate(SYSTEM_PROMPT, prompt, max_tokens=4500)
    text, example, quotes = _split_blocks(text)
    abstained = text.strip().startswith(ABSTAIN[:30])
    if abstained:
        # Never attach an example to an abstention — there is nothing grounded
        # to illustrate, and a "helpful" example would be exactly the invented
        # content abstention exists to prevent.
        return {"answer": text, "sources": [], "abstained": True, "example": ""}

    # Pin each cited source to the EXACT page inside its chunk by locating the
    # model's verbatim quote in the per-page PDF text (chunks span pages, so
    # page_start alone is wrong ~a third of the time). Verified=False marks
    # quotes we could not find (kept at page_start; visible in the audit log).
    sources = []
    for i, c in enumerate(chunks, start=1):
        c = dict(c)
        quote = quotes.get(i)
        if quote:
            page, verified = pagetext.locate_page(c["doc_id"], c, quote)
            c["cite_page"], c["cite_quote"], c["cite_verified"] = page, quote, verified
        else:
            c["cite_page"], c["cite_verified"] = c["page_start"], False
        sources.append(c)
    return {"answer": text, "sources": sources, "abstained": False,
            "example": example}


# Standing human-in-the-loop note appended by the app (never by the model).
VERIFY_NOTE = "Verify with the compliance team before acting."


def _print_cli(result: dict) -> None:
    print("\n" + "=" * 78)
    print(textwrap.fill(result["answer"], width=78))
    print("=" * 78)
    if result.get("example"):
        print("\nILLUSTRATIVE EXAMPLE (explanatory framing, not regulatory text):")
        print(textwrap.fill(result["example"], width=78))
    if result["sources"]:
        print("\nSOURCES USED:")
        for i, c in enumerate(result["sources"], start=1):
            print(f"  [{i}] {c['circular_no']} ({c['issue_date']}) — "
                  f"{c['division']}, page {c['page_start']}, {c['section_ref'] or '-'}")
            snippet = " ".join(c["text"].split())[:180]
            print(f"      \"{snippet}...\"")
    print(f"\n{VERIFY_NOTE}\n")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('Usage: python src/query.py "your question"')
        return 1
    question = " ".join(argv[1:])
    try:
        _print_cli(answer(question))
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
