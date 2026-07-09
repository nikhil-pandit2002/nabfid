---
title: RBI Compliance Assistant
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
short_description: Grounded, citation-backed Q&A over RBI circulars (NaBFID)
---

# RBI Compliance Assistant

An internal assistant that lets staff search, browse, and ask questions about
RBI circulars, master directions, and amendments — with **grounded,
citation-backed answers**. Every answer is drawn only from the ingested
document corpus; each claim links to the exact page of the source PDF.

Developed by **Nikhil Pandit** (Risk Management, NaBFID).

## Features
- **Chatbot** — grounded Q&A over the whole corpus, with inline `[n]` citations
  that open the source PDF at the exact, quote-verified page. Conversations are
  saved and can be revisited.
- **Browse by division** — the document library with an in-app scrollable PDF
  viewer.
- **Document explanation** — hand-authored Key points / What changed /
  Implications for all 34 documents, each line citing the exact source page
  (verified offline against the real page text).
- **Audit log** — every query recorded, for governance.

## Configuration (Hugging Face → Settings → Variables and secrets)
Set these as **Secrets** (never commit them):

| Secret | Value |
|---|---|
| `GEMINI_API_KEY` | your Google AI Studio key (free Gemini Flash tier) |
| `ACCESS_CODE`    | the shared access code for the login gate |

The LLM layer is swappable via env vars (defaults shown):
`LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-2.5-flash`. To use Claude instead, set
`LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-haiku-4-5`, and add `ANTHROPIC_API_KEY`.

## Notes
- First load takes ~1 minute while the local embedding + reranker models load
  (retrieval runs entirely on-device; only answer generation calls the LLM).
- The free Gemini tier is rate-limited (~15 requests/min, ~1,500/day), shared
  across everyone who uses the app.
- Answers are AI-generated — **verify with the compliance team before acting.**
