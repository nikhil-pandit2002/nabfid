# AGENTS.md — RBI Compliance Assistant (NaBFID)

## Who is building this

Nikhil Prakash Pandit, PGDM (Big Data Analytics) student at Goa Institute of Management, interning in the Risk Management Division at NaBFID (National Bank for Financing Infrastructure and Development). This project has VP approval. It is one of two approved projects (the other, an Early Warning System for the loan portfolio, comes later — do not build it here).

NaBFID context that shapes decisions: it is a Development Finance Institution (DFI) — no retail deposits; it raises money via bonds and lends long-term (15–20 year tenors) to large infrastructure projects. It is a regulated government institution, so RBI compliance is existential and data governance matters. Align design choices with RBI's FREE-AI framework: human oversight, explainability, audit trails.

## What we are building

An internal web application — the **RBI Compliance Assistant** — that lets NaBFID staff search, browse, and ask questions about RBI circulars, notifications, master directions, and amendments, with grounded, citation-backed answers.

### Core principles (non-negotiable)

1. **Grounded answers only.** The chatbot answers ONLY from the ingested RBI document corpus (RAG). It must never answer regulatory questions from the model's general memory. If the answer is not in the retrieved context, it must say so plainly ("I could not find this in the circulars I have"). Abstention is a feature, not a failure.
2. **Every claim cited.** Every answer must cite the specific document (circular number, title, date) and show the retrieved source snippets so a human can verify at a glance. Uncited claims are bugs.
3. **Amendments are final.** When a direction/circular and an amendment conflict, the amendment always wins. See "Amendment logic" below — this is the heart of the system.
4. **Human in the loop.** The tool accelerates the compliance expert; it does not replace them. Every answer carries a standing note to verify with the compliance team before acting.
5. **Swappable model layer.** All LLM calls go through a single adapter/interface so the model can be switched (Gemini API now → local LLM on-premise later) as a config change, not a rewrite. Never scatter direct API calls through the codebase.
6. **Portable, standard code.** Plain Python, standard libraries, no platform lock-in. This prototype is web-hosted (Option B) but must migrate cleanly to an on-premise server (Option A) later.

## The data

The RBI documents are already downloaded locally, organised folder-wise by division/department. Within divisions there are **directions** (master directions / master circulars) and **amendments**, plus circulars/notifications.

Critical facts about this data, told by the owner:

- RBI, when it releases an amendment, ALSO updates the master direction file on its website. The master files here were downloaded recently, so a given master file **may or may not already incorporate** a given amendment. Older notifications/files may not reflect amendments at all.
- Therefore: **the amendment file is always the final word**, and every amendment has an **applicability date** ("applicable from") that must be captured and respected.

### First task: inspect and structure the data

Before writing any pipeline code:

1. Walk the existing folder structure and report it back (divisions found, doc counts per folder, file naming patterns, any anomalies — scanned vs text PDFs, duplicates, non-PDF files).
2. Generate a **manifest.csv** — one row per document — as the single source of truth. Columns:
   - `doc_id` (unique key, derive from filename or circular number)
   - `division` (from folder structure)
   - `doc_type` — one of: `master_direction`, `circular`, `notification`, `amendment`
   - `title`
   - `circular_no` (RBI reference number, extract from PDF first page where possible)
   - `issue_date`
   - `applicable_from` (effective date — extract from document text; critical for amendments)
   - `amends` (doc_id of the parent document an amendment modifies — the supersedence chain)
   - `consolidated_as_of` (for master directions: the date up to which RBI has already incorporated amendments into that file — usually stated on the RBI page or the document's first pages)
   - `file_path` (relative path to the PDF)
   - `source_url` (rbi.org.in link, for citation deep-links; may be filled later)
3. Auto-fill as much of the manifest as possible by parsing PDF first pages (dates, circular numbers, titles). Flag rows needing human verification (especially `amends` links and `consolidated_as_of`) rather than guessing silently.

### Amendment logic (encode exactly this)

- An amendment supersedes the document in its `amends` chain. When answering, if retrieved content comes from a document that has amendments, the amendment's text wins for any conflicting provision.
- For a master direction with `consolidated_as_of = D`: amendments with `issue_date <= D` are already baked into the master file — do NOT double-apply them. Amendments with `issue_date > D` are NOT yet reflected — the amendment file is authoritative for those changes.
- `applicable_from` governs when a rule takes effect. If asked "what is the current rule", apply only provisions whose `applicable_from` is on or before today; if a change is announced but not yet applicable, say so explicitly ("amended by X, effective from DATE").
- Answers touching amended areas should state the version story: "Per [master direction], as amended by [amendment, effective DATE]: …"

## Architecture (already designed — follow it)

Three zones plus a cross-cutting layer:

1. **Ingestion pipeline (offline, run on schedule or manually):**
   scrape/refresh RBI documents → parse PDFs (pdfplumber/PyMuPDF; pytesseract OCR fallback for scanned files) → chunk by section/paragraph (not blind fixed-size), attaching doc_id + section reference to every chunk → embed (open-source sentence-transformers, e.g. BGE/E5, run locally/CPU) → index.
2. **Knowledge base (stored with us):**
   - Vector index (Chroma preferred for the prototype; MongoDB Atlas vector search acceptable) — semantic retrieval.
   - Keyword index (BM25) — exact terms, circular numbers, section references.
   - Metadata store (the manifest, in SQLite or as CSV loaded to SQLite) — dates, divisions, amendment chains.
3. **Query pipeline (live):**
   user question → hybrid retrieval (vector + BM25, merged) → rerank top ~20 to best 4–6 (cross-encoder reranker, e.g. BGE reranker) → apply amendment/version filtering from metadata → LLM generates the answer FROM ONLY the retrieved chunks → response with citations + shown source snippets.
4. **Cross-cutting:** simple login/access gate (prototype-level is fine: shared access code or basic auth — it must not be an open public URL) and an **audit log** (every query, answer, sources used, timestamp — SQLite table).

### LLM and prompt rules

- Prototype brain: **free Gemini API (Flash tier)** from Google AI Studio. Design for Flash — do NOT build against Gemini Pro free tier (it is effectively unavailable/too limited). Handle 429 rate-limit errors with retry/backoff.
- System prompt for generation must instruct: answer only from provided context; cite circular number + date for every claim; if not found in context, say so; do not speculate; temperature at or near 0.
- The adapter interface must make swapping to a local model (Ollama — Llama 3.1 / Qwen2.5) trivial later.
- Embeddings should be open-source and local (not tied to the Gemini API) so the knowledge base is portable to on-premise unchanged.

## Frontend (Streamlit for the prototype)

Pages/sections:

1. **Browse by division** — show all divisions; clicking a division lists its documents (directions, circulars, notifications, amendments) with type, date, and status (in-force / amended / superseded). Clicking a document opens a **PDF preview** in-app.
2. **Document explanation view** — for a selected circular/notification/amendment: a detailed plain-language explanation with practical examples. Rules for this content:
   - The **regulatory substance** (what the rule says) must come from the ingested documents, cited.
   - **Explanatory framing and illustrative examples** MAY draw on the model's general knowledge to aid understanding, but must be clearly relevant to NaBFID's context (a DFI doing large infrastructure lending), must never contradict or add to the regulatory substance, and must be visually distinguished from the cited regulatory content (e.g. an "Illustrative example" callout).
   - Where a document has amendments, show the version story prominently (amended by X, effective DATE).
3. **Chatbot** — conversational Q&A over the whole corpus ("personal assistant" style: ask questions, doubts, ask to explain a circular, compare versions). Every answer grounded + cited + source snippets shown, with the abstention behaviour above.
4. **Standing UI elements:** a visible banner — "Prototype hosted on external infrastructure. Ask about regulations in general terms; do not enter confidential deal or client details." Plus the human-in-the-loop verification note on answers.

## Deployment plan

- **Phase 1 (now, this codebase):** web-hosted prototype, free tiers — Streamlit Community Cloud or Hugging Face Spaces (or DigitalOcean via GitHub Student Pack credits). Free Gemini Flash API key. Access gate on. Goal: prove value to the VP and pilot users.
- **Phase 2 (if adopted; do not build now but do not block it):** on-premise NaBFID server, local LLM via Ollama, SSO login, same codebase — only the model adapter config and hosting change.

## Quality bar before anyone relies on it

- Build a small **evaluation set** (start with 30–50 real questions with known answers and their source documents). Measure: does retrieval find the right document; is the answer supported by the cited text; does it abstain when it should. Run it after significant changes.
- Prefer boring, readable code over clever abstractions. The owner must be able to read, understand, and defend every part of this system in front of a VP — explain non-obvious choices in comments or a NOTES.md.

## Working style with the owner

- He knows Python, SQL, ML, Excel/Power BI; he is learning RAG, vector stores, and scraping at production quality. Explain what you're doing and why as you go — this is his project to own and present.
- Prototype-first: get a thin end-to-end slice working (a few documents → manifest → index → one grounded cited answer in the UI), then widen to the full corpus. Do not build all components to completion before anything runs end-to-end.
- Ask before destructive operations on the data folders. The downloaded PDFs are the source of truth — never modify them in place; all derived data (text, chunks, indexes) goes in separate directories.
- Keep secrets (API keys) in a `.env` file, never in code or git. Add `.gitignore` early (data folders, .env, index files).
