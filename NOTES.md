# NOTES.md — design decisions & things to defend

Plain-English record of non-obvious choices, so every part of the system can be
explained and defended. Newest stage at the bottom.

## Project layout

```
RBI Compliance Assistant/
  AIFI latest/            source PDFs — THE source of truth, never modified, gitignored
  src/
    build_manifest.py     Stage 1: PDF corpus -> data/manifest.csv
  data/
    manifest.csv          generated metadata (one row per document); gitignored
  requirements.txt        Python deps (grows per stage)
  .env / .env.example     API keys (real .env never committed)
  .gitignore
  CLAUDE.md               full project brief
  NOTES.md                this file
```

Derived data (manifest, later the vector/keyword indexes and SQLite) lives
outside `AIFI latest/` and is regenerable, so it is all gitignored. The source
PDFs are gitignored too (regulatory documents; keep a separate backup).

## Stage 1 — manifest generator (`src/build_manifest.py`)

**What it does:** reads each PDF's first page(s), fills the manifest columns from
CLAUDE.md, and writes `data/manifest.csv` (encoded `utf-8-sig` so it opens
cleanly in Excel).

**Extraction confidence (34 docs, 19 master directions + 15 amendments):**

| Field | Confidence | How |
|---|---|---|
| circular_no | 100% | `RBI/...` reference, top of page 1 |
| issue_date | 100% | first "Month DD, YYYY" date on page 1 |
| title | 100% | "Reserve Bank of India (All India ... ) Directions, YYYY" |
| doc_type | 100% | "Amendment" in title -> amendment, else master_direction |
| dept_ref | 100% | "DOR.XXX.REC..." department reference (helper column) |
| amends | 15/15 amendments linked | topic match (see below) |
| consolidated_as_of | 5/5 present | "(Updated as on <date>)" title suffix |
| applicable_from | 32/34 | commencement clause (see below) |

**Key decisions:**

1. **`doc_id` = slug of the (now-standardized) filename.** Stable, unique, and
   human-readable. We cleaned all filenames/folders first, so this is safe.

2. **`amends` is linked by *regulated topic*, not by the "Please refer to" line.**
   An amendment's own title names the direction it modifies (e.g. "... Credit
   Risk Management) Second Amendment ..."). We extract that topic and match it to
   the master direction with the same topic. This deliberately ignores the
   "Please refer to ..." sentence, because at least one amendment (Credit Risk
   Second Amendment) refers to a *sibling* division's amendment in that line —
   matching on it would mislink. Topic-matching got all 15 correct, including
   that one. Still flagged for human verification per CLAUDE.md.

3. **Commencement date: "immediate" phrasing wins over an explicit date.**
   Master directions say "come into effect immediately upon issuance" (= issue
   date) but ALSO contain later transitional "with effect from <date>" clauses.
   We check the immediate/upon-issuance/date-of-issue phrasing FIRST and only
   look for an explicit date if none is present — otherwise the regex would grab
   a transitional date (this actually happened to the Credit Risk master and was
   fixed). Amendments have no "immediate" phrase, so they resolve to their real
   explicit effective date.

4. **Scan scope differs by doc type.** Amendments are short, so we scan the whole
   document for the commencement clause (the effective date is the most important
   field, and short docs make full scans cheap). Master directions are long and
   full of transitional dates, so we scan only the first 4 pages (Chapter I -
   Preliminary) to avoid noise.

5. **`needs_verification` is conservative.** Set to `yes` for every amendment
   (amendment chains must be human-checked per CLAUDE.md) and for any row whose
   commencement date was interpreted (the "confirm" notes) or not found. Better a
   human glances at a correct row than trusts a silent wrong one.

**Needs human verification before we rely on it:**
- All 15 `amends` links (auto-suggested; spot-check the chain).
- 2 master directions have no auto-detected `applicable_from` (Credit Facilities,
  Responsible Business Conduct) — commencement clause sits past page 4. These
  masters are effectively in force from issuance (2025-11-28); confirm.
- The "effective immediately/on website" masters: `applicable_from` was set equal
  to `issue_date`; confirm that reading.
- `consolidated_as_of` present on only 5 masters — correct as far as the files
  show, but confirm the other masters genuinely have no consolidated-as-of date.
- `source_url` is intentionally empty (RBI deep-links filled in a later pass).

## Stage 2 — chunker (`src/build_chunks.py`)

**What it does:** reads each PDF, strips boilerplate, splits into section-aware
chunks, and writes `data/chunks.jsonl` (one JSON per line). 34 docs -> **1,094
chunks**, avg ~1,750 chars (≈ 400 tokens). Every chunk carries the manifest's
citation + version fields (doc_id, circular_no, dates, amends) so retrieval
results are self-contained.

**Per chunk:** `text`, `page_start`, `page_end`, `section_ref` (chapter/part
heading), `char_count`, `chunk_id` (`<doc_id>::0007`), plus the carried metadata.
Page numbers are what let the frontend deep-link a cited point to its PDF page.

**Key decisions:**

1. **Section-aware, not fixed-size.** New chunk at every Chapter/Part boundary;
   long sections split at a paragraph start once past ~1,600 chars (hard cap
   2,600 to break a runaway paragraph). Keeps each chunk within one section so
   citations are precise.

2. **Boilerplate stripping — three RBI-specific gotchas handled:**
   - *Running header/footer:* the repeated title and the RBI address/telephone
     footer are detected (lines recurring on >40% of pages) and dropped.
   - *Bilingual text:* RBI prepends the Hindi (Devanagari) header/footer on the
     same line as the English, so footers are matched by SUBSTRING (not anchored
     to line start) and Devanagari characters are stripped outright.
   - *Table of Contents:* TOC entries have dot-leaders ("......"); we drop dotted
     lines AND the wrapped head-line that precedes one (so a wrapped TOC heading
     isn't mistaken for a real chapter heading — this bug polluted `section_ref`
     until fixed).
   Verified: the only residual "Fax No" / "Department of Regulation" hits are
   genuine body content (a UAPA fax number; a reporting instruction), correctly
   NOT filtered.

**Known minor items (deferred, low value):**
- A few lone heading-only chunks (~60 chars, e.g. a "PART C ..." heading). Harmless.
- Chunk overlap is currently zero; add a small paragraph overlap later if
  retrieval recall needs it.

## Stage 3 — knowledge base (`src/build_index.py`, `config.py`, `embeddings.py`)

**What it does:** builds the three stores the query pipeline reads.

1. **Vector index — Chroma** (`index/chroma/`): 1,094 chunks embedded with
   **BGE-small-en-v1.5** (local, ~130 MB, CPU, cosine space). Model access is
   centralized in `embeddings.py`; the model id lives in `config.py`, so swapping
   it is a one-line change. Passages embedded as-is; queries get BGE's retrieval
   instruction prefix.
2. **Keyword index — BM25** (`index/bm25.pkl`): rank-bm25 over the same chunks,
   for exact terms (circular numbers, section refs) that semantics can miss.
3. **Metadata store — SQLite** (`data/metadata.sqlite`, table `documents`): the
   manifest loaded row-per-document, for version filtering + browse-by-division.

Re-running rebuilds all three from scratch (idempotent — vector collection is
dropped and recreated).

**Sanity-checked** end to end: semantic queries return the correct division +
chapter + page (e.g. "single counterparty exposure limit" -> Concentration Risk
Mgmt, Ch II Large Exposures; "calamity relief effective date" -> Resolution of
Stressed Assets amendment, Ch VI-A), and BM25 resolves an exact circular number.

**Environment note:** a pre-existing `langchain-community` in this Python
environment pins a newer `requests` than we have. We don't use langchain; the
warning is harmless for this project.

## Stage 4 — query pipeline (`llm.py`, `retrieval.py`, `query.py`)

The live path: **question -> hybrid retrieval -> rerank -> grounded, cited answer**.
This is the working end-to-end slice (CLAUDE.md's thin-slice goal).

- **`llm.py`** — the single swappable LLM adapter. ALL generation goes through
  `generate()`. Backend is Gemini (via the current `google-genai` SDK); switching
  to a local Ollama model later is a change here + `.env` only. Retries with
  exponential backoff on 429 / transient errors (free Flash tier is rate-limited).
- **`retrieval.py`** — hybrid recall (Chroma vectors ∪ BM25, ~20 each) ->
  cross-encoder rerank (`bge-reranker-base`, local) -> top 6. Also
  `scope_for_document()` for the document-scoped chat (master -> it + its
  amendments; amendment -> it + its parent).
- **`query.py`** — builds a numbered SOURCES block from the retrieved chunks and
  a strict system prompt (answer only from sources; cite circular + date on every
  claim; amendments override; surface effective dates; abstain with a fixed
  sentence if not found). Temperature 0. Returns answer + sources + `abstained`.

**Model choice:** `.env` uses **gemini-2.5-flash**. The older `gemini-2.0-flash`
returned 429 RESOURCE_EXHAUSTED (no free quota on this key); 2.5-flash / flash-lite
work. Change `LLM_MODEL` in `.env` to swap.

**Verified behaviours (terminal, `python src/query.py "..."`):**
- Grounded + cited: single-counterparty exposure limit answered from the
  Concentration Risk direction, every claim tagged with the circular + date.
- Abstention: an out-of-corpus question returns exactly the abstain sentence.
- Amendment precedence: calamity-relief question pulled BOTH relevant amendments,
  stated the version story, and surfaced the future effective date (July 1, 2026).

**Minor refinement (deferred):** the model over-cites (repeats the citation on
every bullet). Cosmetic; tighten via the prompt later.

## Stage 5 — Streamlit frontend (`app.py`, `docstore.py`, `audit.py`, `explain.py`)

Full four-surface app behind an access gate, with the standing banner + audit log.
Run: `streamlit run app.py` (or the `.claude/launch.json` config). Access code in
`.env` (`ACCESS_CODE`, default `nabfid-demo`).

- **Chatbot** — wraps `query.answer()`; chat UI, sources expander, verify note.
- **Browse by division** — `docstore.py` reads the SQLite store; lists divisions ->
  documents with status (In force / amended (N) / Amendment · effective DATE);
  "Open" jumps to the explanation view.
- **Document explanation** — tabs: Key points (grounded, cited via `explain.py`),
  What changed (amendments only), Implications (labelled general-knowledge
  callout), PDF preview (inline base64 iframe + download), and a chat scoped to
  the circular + its amendment chain (`retrieval.scope_for_document`).
- **Audit log** — `audit.py` writes every Q&A (question, answer, sources,
  abstained, timestamp, scope) to an `audit_log` table; admin page shows recent.

**Decisions / gotchas:**
- LLM-backed explanation content is wrapped in `st.cache_data` keyed by doc_id so
  Gemini isn't re-called on every rerun.
- Navigation: `nav` is a widget-bound key, so it can't be written after the radio
  is created. The "Open" button stashes `_goto` and `main()` applies it before the
  radio instantiates (fixed a StreamlitAPIException).
- PDF preview uses an inline base64 iframe + a download button. Works for the
  prototype; if a browser blocks the data-URI iframe, swap to `streamlit-pdf-viewer`.
- Harmless console noise: Streamlit's internal wavesurfer component logs
  "Container not found"; unrelated to this app (no audio used).

## Performance — reranker latency fix (important)

**Symptom:** every query took ~67 seconds — unusable for real staff.

**Diagnosis (timed each stage of one query):**
- Vector search: 0.10s · BM25: 0.01s · LLM answer: ~8-10s — all fine.
- **Reranker: 58s** — the cross-encoder `BAAI/bge-reranker-base` (278M params)
  scored ~32 (query, passage) pairs on CPU at ~1.8s each. That was 87% of the
  query time.

**Fix (no re-indexing needed — the reranker runs at query time only):**
1. Swapped to `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, the standard
   fast reranker) — rerank dropped from ~58s to a few seconds.
2. Trimmed the candidate pool 20+20 -> 15+15 (fewer pairs to score).
3. Capped the reranker's scored length to 256 tokens (`max_length=256`) — the
   full chunk text still goes to the LLM; only the relevance score uses the cap.

**Result:** ~15s/query on Sonnet 5 (~7s retrieve + ~8s LLM), ~10s on Gemini
Flash. The one-time model load (~20s) and Python import (~40s) happen once at
process start, not per query, so they don't affect a warm server.

**Anthropic backend note:** the Claude adapter sends `thinking: {type:
"disabled"}` — grounded extraction/citation doesn't need extended reasoning, and
Sonnet 5 / Opus have adaptive thinking ON by default, which otherwise made each
answer slow and more expensive.

## Data-completeness caveat (from the earlier corpus review)

We cross-checked the corpus against RBI's public notification history and found
one genuine gap (the Responsible Business Conduct first amendment), since added.
That check used secondary sources (taxguru, mondaq, etc.) because rbi.org.in
blocks automated access with a CAPTCHA. Someone with RBI-portal access should do
a final gap check before this is relied upon.

## Citation architecture (why every [n] link is trustworthy)

A citation in this app may only point at a PDF page that verifiably contains
the cited fact. Nothing is guessed at render time.

**Ground truth.** `src/build_pages.py` extracts the exact text of every page of
all 34 PDFs into `data/pages.jsonl` (1-based physical pages — the same numbers
the browser's `#page=N` viewer uses). `src/pagetext.py` provides the shared
normalizer ("₹1,500 crore" matches "1500") and quote-locating helpers.

**Guides (Key points / What changed / Implications).** Citations are baked
OFFLINE by `src/bake_guide_citations.py`: each guide line's distinctive anchors
(numbers, number+unit bigrams, bold terms, document-rare words) are matched
against real page text; a line is cited only if it passes a confidence gate,
and ambiguous or unverifiable lines get NO citation instead of a wrong one.
Section-opening summary lines cite p.1 (the title page states the document's
scope). Human corrections live in `data/citation_overrides.json` and are merged
on every re-bake (re-running never clobbers a review). The verified pages are
stored in each `data/explanations/*.json` under "citations", keyed by a hash of
the line text — the app (`render_cited_markdown`) only renders what is baked.
`data/citation_report.md` shows the evidence snippet behind every decision.

**Chat.** The model must append a verbatim supporting quote per cited source
(`===CITES===` block, stripped before display). `pagetext.locate_page` finds
that quote in the per-page text within the chunk's page span, pinning the [n]
link to the exact page (chunks span multiple pages, so the chunk's first page
alone is wrong ~a third of the time). Verified quotes are shown under
"Sources & citations"; unverifiable quotes fall back to the chunk's first page
and are flagged in the audit log (`cite_verified = false`).

**Regression check.** `python src/eval_citations.py` re-verifies every baked
citation against page text (hard-fails on any unverifiable auto citation).
Run it after re-baking guides, re-chunking, or adding documents. As of
2026-07-07: 513 citations across 34 documents, 0 failures.

## Why the models run on ONNX, not PyTorch (2026-07-14)

**Symptom.** On Streamlit Community Cloud the app hung forever on a query —
spinner, no answer, no error. Not a code bug: the container was being
OOM-killed mid-query and silently restarted.

**Cause.** The free tier caps at ~1 GB. Measured RSS with the torch stack:
`torch` + `sentence-transformers` cost **410 MB before a single model loads**;
after one real query the process sat at **898 MB**, and Streamlit itself adds
~150 MB. The app was surviving on a knife-edge; adding the page-image/figure
features (+45 MB) tipped it over.

**Fix.** Same models, different runtime. `fastembed` runs
BAAI/bge-small-en-v1.5 and ms-marco-MiniLM-L-6-v2 on **onnxruntime** instead of
torch. The weights are identical — the ONNX query vectors match the torch ones
to **cosine 0.999999** (max element diff 3.5e-04), so the *existing Chroma index
stayed valid and did not need rebuilding*.

**The trap.** The naive port made things *worse*: 2,539 MB. onnxruntime sizes
its memory arena from the largest activation tensor (batch x sequence) and never
releases it. Reranking 40 chunks of ~1.7k chars in batches of 8 blew the arena
up. Counter-intuitively, `batch_size=1` uses the **least** memory *and* scores
**best**, because the cross-encoder then sees each full passage instead of a
truncated head. Hence `RERANK_BATCH = 1` in retrieval.py — that line is load
bearing, do not "optimise" it back up.

**Measured on the 28 answerable eval questions (data/eval_set.jsonl):**

| stack | hit@8 | MRR | peak RSS |
|---|---|---|---|
| torch (before) | 26/28 (92.9%) | 0.798 | 898 MB |
| onnx, batch=8, truncated | 26/28 (92.9%) | 0.798 | 610 MB |
| onnx, batch=8, full length | 28/28 (100%) | 0.905 | 2539 MB |
| **onnx, batch=1, full length (shipped)** | **28/28 (100%)** | **0.905** | **444 MB** |

Retrieval quality went *up* (both previous misses now found) while memory
halved. Citations were structurally unaffected — guide citations are baked
offline and chat citations are located by matching the model's verbatim quote
against real page text, so neither path touches the embedding model.
`eval_citations.py` still reports 0 hard failures.
