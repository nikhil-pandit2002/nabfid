"""
app.py — RBI Compliance Assistant (NaBFID) Streamlit prototype.

Four surfaces (CLAUDE.md), behind a simple access gate, with a standing banner
and an audit log:

  1. Chatbot                — grounded, cited Q&A over the whole corpus.
  2. Browse by division     — the document library with in-app PDF preview.
  3. Document explanation   — key points + what-changed + implications + PDF +
                              a chat scoped to that circular.
  (Admin) Audit log         — recent questions, for governance visibility.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

import streamlit as st

# Make the src/ modules importable when Streamlit runs app.py from the root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import ACCESS_CODE, LLM_MODEL, PROJECT_ROOT  # noqa: E402
import audit  # noqa: E402
import bake_guide_citations as bake_cite  # noqa: E402  (shared line walk/hash)
import conversations  # noqa: E402
import docstore  # noqa: E402
import explain  # noqa: E402
from query import answer, VERIFY_NOTE  # noqa: E402
from retrieval import scope_for_document  # noqa: E402

st.set_page_config(page_title="RBI Compliance Assistant — NaBFID",
                   page_icon="📘", layout="wide")

# --------------------------------------------------------------------------
# Explanation sections are read from the persistent on-disk store (data/
# explanations/). If a doc was pre-generated (baked) the app shows it with NO
# LLM call and it survives restarts. If not, the user can opt in to generate it
# once (which saves it to disk for good).
_SECTION = {
    "key_points": explain.key_points,
    "what_changed": explain.what_changed,
    "implications": explain.implications,
}


def explanation_tab(doc: dict, kind: str, *, note: bool = False) -> None:
    """Show a saved explanation section, or an opt-in one-time generate button.
    Rendering always goes through render_cited_markdown so every section gets
    end-of-line citations to the source PDF (defined further below, but
    Python resolves it at call time so definition order doesn't matter)."""
    doc_id = doc["doc_id"]
    saved = explain.load_saved(doc_id)
    if saved.get(kind):
        render_cited_markdown(saved[kind], doc_id,
                               (saved.get("citations") or {}).get(kind))
        if note:
            st.caption(VERIFY_NOTE)
        return
    st.info("Not pre-generated yet — this section is served from disk with no AI "
            "cost once created.")
    if st.button("⚙️ Generate & save this once (uses AI, then free forever)",
                 key=f"gen_{kind}_{doc_id}"):
        with st.spinner("Generating and saving…"):
            text = _SECTION[kind](doc)
            explain.save_section(doc_id, kind, text)
        st.rerun()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _pdf_page_count(rel_path: str) -> int:
    import fitz
    with fitz.open(PROJECT_ROOT / rel_path) as doc:
        return doc.page_count


def _is_mobile() -> bool:
    """True when the request comes from a phone/tablet browser. Mobile browsers
    can't render PDFs in an iframe (Android Chrome shows a blank box, iOS Safari
    shows only page 1) and ignore #page=N anchors, so several surfaces switch to
    image-based page rendering instead. NaBFID office laptops block the hosted
    URL, so mobile is the PRIMARY access path for pilot users — not an edge case."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:  # very old Streamlit or non-request context
        return False
    return bool(re.search(r"Mobi|Android|iPhone|iPad", ua, re.IGNORECASE))


def _is_android() -> bool:
    """True on Android browsers (Chrome etc.). Android hands PDFs to an external
    viewer (Google Drive/preview) that IGNORES the "#page=N" anchor, so a
    citation link opens page 1 instead of the cited page. iOS Safari and desktop
    both honour the anchor. So on Android we point citations at an image of the
    exact page instead (see _citation_href), which every mobile browser renders
    correctly."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    return "android" in ua.lower()


# Page-image zoom. 1.5x still reads clearly when pinch-zoomed on a phone, but the
# rendered pixmap costs ~2.25x less memory than 2x (cost scales with zoom^2). The
# whole app runs near the ~1 GB ceiling of free hosts, where going over is an
# OOM kill (the app appears to hang), so this margin is worth more than the extra
# sharpness. Same reason the cache is kept small.
_PAGE_ZOOM = 1.5


@st.cache_data(show_spinner=False, max_entries=8)
def _page_png(rel_path: str, page: int) -> bytes:
    """Render one PDF page to a PNG. Cached — repeat views of a page are free."""
    import fitz
    with fitz.open(PROJECT_ROOT / rel_path) as doc:
        pix = doc[page - 1].get_pixmap(
            matrix=fitz.Matrix(_PAGE_ZOOM, _PAGE_ZOOM))
        return pix.tobytes("png")


@st.cache_data(show_spinner=False)
def _relpath_for_doc(doc_id: str) -> str | None:
    d = docstore.get_document(doc_id)
    return d["file_path"] if d else None


# A figure/table caption at the start of a line, e.g. "Figure 1: ...",
# "Table 2 -", "Chart 3". RBI documents label their visuals this way, so a
# match on a cited page means that page carries a diagram/table worth showing
# alongside the answer (see render_figures).
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|chart)\s+\d+\b.*", re.IGNORECASE)


@st.cache_data(show_spinner=False)
def _figure_pages(rel_path: str) -> dict[int, str]:
    """Map {page_no: caption} of every page in the document that carries a
    labelled figure/table/chart. Computed in a single pass per document and
    cached, so figure lookups while rendering answers are instant."""
    import fitz
    out: dict[int, str] = {}
    with fitz.open(PROJECT_ROOT / rel_path) as doc:
        for i in range(doc.page_count):
            for line in doc[i].get_text().splitlines():
                if _FIGURE_CAPTION_RE.match(line):
                    out[i + 1] = " ".join(line.split())[:120]
                    break
    return out


# The source PDFs are served (as copies) from ./static/ so the browser's native,
# scrollable PDF viewer can open them at a specific page via "#page=N". Streamlit
# exposes ./static/ at the URL path "app/static/". Copies are made on demand and
# cached; the originals in "AIFI latest/" are never touched.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@st.cache_data(show_spinner=False)
def _static_pdf_url(rel_path: str) -> str | None:
    """Ensure a served copy of the PDF exists and return its "app/static/..." URL."""
    src = PROJECT_ROOT / rel_path
    if not src.exists():
        return None
    STATIC_DIR.mkdir(exist_ok=True)
    name = hashlib.md5(rel_path.encode("utf-8")).hexdigest() + ".pdf"
    dest = STATIC_DIR / name
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        shutil.copyfile(src, dest)
    return f"app/static/{name}"


@st.cache_data(show_spinner=False)
def _static_url_for_doc(doc_id: str) -> str | None:
    """Served PDF URL for a doc_id (used by inline citation links)."""
    d = docstore.get_document(doc_id)
    return _static_pdf_url(d["file_path"]) if d else None


@st.cache_data(show_spinner=False)
def _static_page_png_url(rel_path: str, page: int) -> str | None:
    """Render one PDF page to a PNG served from ./static/ and return its
    "app/static/...png" URL. Used for Android citations: opening this image in a
    new tab shows the exact cited page on every mobile browser, sidestepping
    Android's PDF viewer that ignores "#page=N"."""
    src = PROJECT_ROOT / rel_path
    if not src.exists():
        return None
    STATIC_DIR.mkdir(exist_ok=True)
    name = hashlib.md5(f"{rel_path}::p{page}".encode("utf-8")).hexdigest() + ".png"
    dest = STATIC_DIR / name
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        import fitz
        with fitz.open(src) as doc:
            page = max(1, min(page, doc.page_count))
            doc[page - 1].get_pixmap(
                matrix=fitz.Matrix(_PAGE_ZOOM, _PAGE_ZOOM)).save(str(dest))
    return f"app/static/{name}"


def _citation_href(doc_id: str, url: str | None, page: int) -> str | None:
    """Where a citation should point. Android: an image of the exact page (its
    PDF viewer ignores "#page=N" and would strand the reader on page 1). iOS
    Safari + desktop: the real PDF opened at "#page=N" (full scrollable viewer,
    better when it works)."""
    if _is_android():
        rel = _relpath_for_doc(doc_id)
        img = _static_page_png_url(rel, page) if rel else None
        if img:
            return img
    return f"{url}#page={page}&view=FitH" if url else None


# --------------------------------------------------------------------------
# Inline citation links.
#
# History: an inline onclick that updated an on-page <iframe> via JS was
# tried, but Streamlit's markdown sanitizer strips onclick/javascript:
# attributes even under unsafe_allow_html (confirmed: href="javascript:..."
# comes back as href="#" and onclick vanishes entirely — a hard security
# boundary). The follow-up (a plain, non-interactive [n] marker + a separate
# "preview" button below the answer) worked, but the marker still looked
# clickable and the real click target wasn't where the citation itself was —
# so at the owner's request, make the number itself the link again.
#
# A REAL <a href="..."> (no onclick, no javascript: URI) is NOT stripped by
# the sanitizer, so this works — the earlier trap was only ever the
# JS-based approaches. The remaining risk from the very first attempt (a
# same-tab navigation reloading the whole Streamlit session, so Back landed
# on the access gate) is avoided with target="_blank" rel="noopener
# noreferrer": a genuine mouse click on a real link opens a separate
# browser tab reliably (unlike a script-simulated click), so the chat tab
# never navigates and there is nothing for Back to undo there.
# --------------------------------------------------------------------------
def _citation_link(idx: int, href: str | None, page: int, title: str) -> str:
    """One inline, clickable citation marker — opens the cited page in a new tab
    (a #page=N PDF link on desktop/iOS, a page image on Android; the caller
    picks via _citation_href). On mobile the marker shows the page number too
    ("1·p12"): touch screens have no hover tooltip, so the label itself tells
    the reader which page it points at."""
    label = f"{idx}·p{page}" if _is_mobile() else str(idx)
    if not href:
        return f"<sup>[{label}]</sup>"
    return (f'<a href="{href}" target="_blank" rel="noopener noreferrer" '
            f'title="{title}" style="text-decoration:none;color:#1f6feb;'
            'font-size:0.72em;font-weight:700;vertical-align:super;'
            'background:#eef3fb;border:1px solid #d6e2f5;border-radius:4px;'
            f'padding:0 4px;margin:0 1px;white-space:nowrap;">{label}</a>')


# --------------------------------------------------------------------------
# Inline, end-of-line citations for the baked Key points / What changed /
# Implications guides — read from the VERIFIED citations baked into each
# data/explanations/<doc_id>.json by src/bake_guide_citations.py.
#
# History: render-time matching (first chapter-level, then fuzzy word/number
# overlap against chunks) proved unreliable — measured ~1 in 3 word-only lines
# cited a wrong page, and chunk page_start is wrong for the 74% of chunks that
# span multiple pages. Citations are now computed OFFLINE against the exact
# per-page PDF text (data/pages.jsonl), pass a confidence gate, are human-
# reviewed (data/citation_overrides.json), and verified by
# src/eval_citations.py. The renderer only ever shows a baked, verified page —
# a line without a baked citation gets NO marker (honest absence beats a
# guessed page). Lines are matched by the same hash + walk the baker uses
# (bake_guide_citations.line_hash / iter_guide_lines), so they cannot drift.
# --------------------------------------------------------------------------
def render_cited_markdown(text: str, doc_id: str,
                           cites: dict[str, list[int]] | None) -> None:
    """Render a guide section, appending a clickable [n] link (exact verified
    page, new tab) to every line that has a baked citation. Citation numbers
    are per distinct page in order of first appearance (footnote style)."""
    if not cites:
        st.markdown(text)
        return
    url = _static_url_for_doc(doc_id)
    page_to_num: dict[int, int] = {}
    out: list[str] = []
    for raw, citable, _heading in bake_cite.iter_guide_lines(text):
        pages = cites.get(bake_cite.line_hash(citable)) if citable else None
        if not pages:
            out.append(raw)
            continue
        chips = ""
        for page in pages:
            num = page_to_num.setdefault(page, len(page_to_num) + 1)
            href = _citation_href(doc_id, url, page)
            chips += _citation_link(num, href, page, f"Source — page {page}")
        base = raw.rstrip()
        if base.endswith("|"):  # markdown table row: keep the chip inside the
            out.append(f"{base[:-1].rstrip()} {chips} |")  # last cell.
        else:
            out.append(f"{base} {chips}")
    st.markdown("\n".join(out), unsafe_allow_html=True)


def pdf_preview(doc: dict) -> None:
    """PDF viewer with two modes. Desktop default: the browser's embedded,
    scrollable + searchable PDF reader (iframe). Mobile default: page-by-page
    images rendered by PyMuPDF — phone browsers cannot render a PDF inside an
    iframe at all (Android Chrome: blank box; iOS Safari: first page only), and
    NaBFID office laptops block the hosted URL, so phones are the primary way
    pilot users will read documents. Both modes are always offered via a toggle;
    only the default follows the device."""
    rel_path = doc["file_path"]
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        st.warning(f"PDF not found: {rel_path}")
        return
    url = _static_pdf_url(rel_path)
    if not url:
        st.warning("PDF could not be prepared for viewing.")
        return

    n_pages = _pdf_page_count(rel_path)
    cite_page = st.session_state.get(f"citepage_{doc['doc_id']}")
    page = min(max(1, cite_page), n_pages) if cite_page else 1
    if cite_page:
        st.success(f"📄 Opened at cited page {page} — scroll freely from here.")

    MODE_PAGES, MODE_EMBED = "🖼 Page-by-page", "📜 Full document"
    mode = st.radio("Viewer mode", [MODE_PAGES, MODE_EMBED],
                    index=0 if _is_mobile() else 1, horizontal=True,
                    key=f"pdfmode_{doc['doc_id']}",
                    label_visibility="collapsed")

    if mode == MODE_EMBED:
        st.caption("Full document — scroll to read; use the viewer toolbar or "
                   "Ctrl+F to search. (Blank on phones? Switch to page-by-page.)")
        # Real URL + "#page=N" so the native viewer lands on the right page.
        src = f"{url}#page={page}&view=FitH&toolbar=1"
        st.markdown(
            f'<iframe src="{src}" width="100%" height="900" '
            f'style="border:1px solid #ddd;"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        pg_key = f"pdfpage_{doc['doc_id']}"
        if cite_page:  # a citation jump overrides whatever page was open before
            st.session_state[pg_key] = page
            st.session_state.pop(f"citepage_{doc['doc_id']}", None)
        st.session_state.setdefault(pg_key, page)
        c_prev, c_num, c_next = st.columns([1, 2, 1])
        # Buttons are handled BEFORE the number_input widget is instantiated,
        # so mutating its session-state value here is allowed.
        if c_prev.button("◀ Prev", key=f"{pg_key}_prev", use_container_width=True,
                         disabled=st.session_state[pg_key] <= 1):
            st.session_state[pg_key] = max(1, st.session_state[pg_key] - 1)
        if c_next.button("Next ▶", key=f"{pg_key}_next", use_container_width=True,
                         disabled=st.session_state[pg_key] >= n_pages):
            st.session_state[pg_key] = min(n_pages, st.session_state[pg_key] + 1)
        cur = c_num.number_input(f"Page (1–{n_pages})", min_value=1,
                                 max_value=n_pages, key=pg_key)
        st.image(_page_png(rel_path, cur), use_container_width=True)
        st.caption(f"Page {cur} of {n_pages} — pinch to zoom on mobile.")

    st.download_button("⬇️ Download full PDF", path.read_bytes(),
                       file_name=path.name, mime="application/pdf")


# Label used for the PDF section in the doc-view selector.
PDF_SECTION = "📄 PDF preview"


# Matches inline citation markers the model emits: [1], [1,3], etc.
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def render_answer(text: str, sources: list[dict]) -> None:
    """Render the answer with Perplexity-style inline citations: each [n]
    marker becomes a small clickable link that opens source n's PDF at the
    exact cited page in a new browser tab. `cite_page` is where the model's
    verbatim supporting quote was located (see query.answer / pagetext);
    page_start is the fallback when no quote could be verified."""
    if not sources:
        st.markdown(text)
        return

    def repl(match: "re.Match") -> str:
        chips = []
        for ns in re.findall(r"\d+", match.group(1)):
            i = int(ns)
            if not (1 <= i <= len(sources)):
                continue
            c = sources[i - 1]
            url = _static_url_for_doc(c["doc_id"])
            page = c.get("cite_page", c["page_start"])
            title = f"{c['circular_no']} ({c['issue_date']}), page {page}"
            href = _citation_href(c["doc_id"], url, page)
            chips.append(_citation_link(i, href, page, title))
        return "".join(chips) if chips else match.group(0)

    st.markdown(_CITE_RE.sub(repl, text), unsafe_allow_html=True)


def _open_pdf_button(doc_id: str, url: str | None, page: int) -> str:
    """A full-width 'Open p.N' link styled as a button that opens the cited page
    in a NEW browser tab — so the chat is never left and the reader can keep
    clicking other citations. Routes through _citation_href, so on Android it
    opens a page image (its PDF viewer ignores #page=N) and on desktop/iOS the
    real PDF at that page."""
    href = _citation_href(doc_id, url, page)
    if not href:
        return f"<div style='text-align:center;color:#888;'>p.{page}</div>"
    return (f'<a href="{href}" target="_blank" rel="noopener noreferrer" '
            'style="display:block;width:100%;box-sizing:border-box;'
            'text-align:center;padding:0.35rem 0.5rem;border:1px solid #d0d7de;'
            'border-radius:8px;text-decoration:none;color:#1f6feb;'
            f'font-weight:600;font-size:0.85rem;">📄 Open p.{page} ↗</a>')


def render_sources(sources: list[dict], key_prefix: str = "src") -> None:
    """Sources under an answer. Each one is a citation (document, date, page,
    section) with a 📄 link that opens the exact cited page in the scrollable
    PDF viewer in a NEW browser tab — the chat stays open behind it, so the
    reader can come back and open other citations. `cite_page` is the page the
    model's verbatim quote was located on (exact); page_start is the fallback
    when no quote could be verified."""
    if not sources:
        return
    mobile = _is_mobile()
    with st.expander(f"📎 Sources & citations ({len(sources)}) — click 📄 to open "
                     "the cited page (opens in a new tab; your chat stays here)",
                     expanded=False):
        for i, c in enumerate(sources, start=1):
            page = c.get("cite_page", c["page_start"])
            col1, col2 = st.columns([4, 1.3])
            with col1:
                st.markdown(
                    f"**[{i}] {c['circular_no']}** ({c['issue_date']}) — "
                    f"{c.get('title') or c['division']}"
                )
                ent = c.get("entity") or ""
                ent_tag = f"{ENTITY_ICON.get(ent, '')} **{ent}** · " if ent else ""
                st.caption(f"{ent_tag}{c['division']} · `{c['doc_type']}` · "
                           f"p.{page} · {c['section_ref'] or '—'}")
                if c.get("cite_quote") and c.get("cite_verified"):
                    st.markdown(f"> *“{c['cite_quote']}”* — verified on p.{page}")
                else:
                    st.caption('"' + " ".join(c["text"].split())[:280] + '…"')
            with col2:
                url = _static_url_for_doc(c["doc_id"])
                st.markdown(_open_pdf_button(c["doc_id"], url, page),
                            unsafe_allow_html=True)
            # Mobile: phone PDF viewers ignore the #page=N anchor, so an
            # "Open p.N" link would strand the reader on page 1 of a long
            # direction. Show the exact cited page inline as an image instead.
            if mobile:
                if st.toggle(f"🖼 Show cited page {page} here",
                             key=f"{key_prefix}_pg_{i}"):
                    rel = _relpath_for_doc(c["doc_id"])
                    if rel:
                        st.image(_page_png(rel, int(page)),
                                 use_container_width=True)


# Common words to ignore when matching a figure caption to the question, so
# overlap reflects the topic (e.g. "investments", "capital", "insurance") not
# filler. Kept small and generic on purpose.
_FIG_STOP = {
    "the", "and", "for", "with", "that", "outside", "scope", "such", "all",
    "under", "from", "into", "over", "per", "cent", "shall", "based", "other",
    "this", "these", "which", "where", "applicable", "including", "any",
}


def _caption_overlap(caption: str, query: str) -> int:
    """How many topic words a figure caption shares with the question — a cheap,
    embedding-free relevance signal. Figure 1's caption is nearly the user's
    query verbatim (high overlap); an unrelated 'Foreign PSE risk weights' table
    scores ~0 and is filtered out."""
    cw = set(re.findall(r"[a-z]{4,}", caption.lower())) - _FIG_STOP
    qw = set(re.findall(r"[a-z]{4,}", query.lower())) - _FIG_STOP
    return len(cw & qw)


def render_example(example: str) -> None:
    """The plain-language 'how this applies at NaBFID' scenario, in its own
    bordered callout.

    CLAUDE.md allows illustrative framing to draw on the model's general
    knowledge so a non-expert can follow the rule — but ONLY if it is visually
    distinguished from the cited regulatory substance and never adds to it. The
    model returns it as a separate ===EXAMPLE=== block (see query._split_blocks)
    precisely so it can never be mistaken for the cited answer above it, and it
    carries no citations of its own."""
    if not example:
        return
    with st.container(border=True):
        st.markdown("💡 **Illustrative example — how this applies at an AIFI "
                    "like NaBFID**")
        st.markdown(example)
        st.caption("Plain-language illustration to aid understanding — general "
                   "framing, not regulatory text. The rule itself, with "
                   "citations, is above.")


def render_figures(sources: list[dict], query: str) -> None:
    """Show figures/tables/charts that the answer draws on, right under it — so a
    diagram (e.g. the capital-instruments deduction flowchart) is visible, not
    just linked. Pages are detected deterministically from their captions in the
    real PDF (never hallucinated). Candidates come from the cited chunks' page
    spans; we then keep only those whose caption is topically relevant to the
    question and show the best few — otherwise a table on a tangentially-
    retrieved page could crowd out the figure the user actually asked about."""
    if not sources or not query:
        return
    seen: set[tuple[str, int]] = set()
    cands: list[tuple[int, int, str, str, str]] = []  # score, pg, rel, cap, circ
    for c in sources:
        rel = _relpath_for_doc(c["doc_id"])
        if not rel:
            continue
        figpages = _figure_pages(rel)
        if not figpages:
            continue
        # Scan the cited chunk's whole page span, plus one page either side (a
        # figure can sit just before/after the paragraph that references it).
        lo = int(c.get("page_start") or c.get("cite_page") or 1)
        hi = int(c.get("page_end") or lo)
        for pg in range(lo - 1, hi + 2):
            if pg < 1 or pg not in figpages or (rel, pg) in seen:
                continue
            seen.add((rel, pg))
            cap = figpages[pg]
            cands.append((_caption_overlap(cap, query), pg, rel, cap,
                          c["circular_no"]))
    # Keep only captions clearly on-topic (>=2 shared words), best first.
    cands = sorted((x for x in cands if x[0] >= 2), key=lambda x: -x[0])
    for rank, (_score, pg, rel, cap, circ) in enumerate(cands[:3]):
        with st.expander(f"📊 {cap}  ·  {circ} — p.{pg}", expanded=(rank == 0)):
            st.image(_page_png(rel, pg), use_container_width=True)
            st.caption(f"Figure/table from {circ}, page {pg}. Shown from the "
                       "source document — verify against the full circular.")


def goto(page: str, doc_id: str | None = None) -> None:
    # `nav` is bound to the sidebar radio widget, so it can't be written after
    # that widget is instantiated. Stash the target and apply it at the top of
    # main() before the radio is created.
    st.session_state._goto = page
    if doc_id:
        st.session_state.selected_doc = doc_id
    st.rerun()


# --------------------------------------------------------------------------
# Access gate
# --------------------------------------------------------------------------
def gate() -> bool:
    if st.session_state.get("authed"):
        return True
    st.title("📘 RBI Compliance Assistant")
    st.caption("NaBFID — Risk Management. Internal prototype.")
    code = st.text_input("Access code", type="password")
    if st.button("Enter"):
        if code == ACCESS_CODE:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.caption("Developed by Nikhil Pandit")
    return False


# --------------------------------------------------------------------------
# Page: Chatbot (whole corpus)
# --------------------------------------------------------------------------
def _new_chat() -> None:
    """Start a fresh, empty conversation (the current one is already saved)."""
    st.session_state["chat"] = []
    st.session_state["active_conv_id"] = None
    goto("Chatbot")


def page_chatbot() -> None:
    c1, c2 = st.columns([5, 1])
    c1.header("💬 Chatbot")
    if c2.button("➕ New chat", use_container_width=True):
        _new_chat()
    active = st.session_state.get("active_conv_id")
    if active:
        meta = conversations.load(active) or {}
        st.caption(f"Conversation: **{meta.get('title', '—')}** · saved — reopen "
                   "or start a new one from the sidebar.")
    else:
        st.caption("Ask anything about the RBI directions in the corpus. Answers "
                   "are grounded and cited. This chat is saved automatically.")

    # Which rulebook to answer from. RBI issues the same topics separately for
    # AIFIs and Commercial Banks, so answering an AIFI question from a
    # Commercial Bank direction would be a correctness failure, not a nuance.
    # Default is NaBFID's own rulebook (AIFI); "Both" is for comparison.
    scope_ids = _entity_scope_picker("chat")

    history = st.session_state.setdefault("chat", [])
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["content"], msg.get("sources") or [])
                render_figures(msg.get("sources") or [],
                               history[i - 1]["content"] if i > 0 else "")
                render_example(msg.get("example") or "")
            else:
                st.markdown(msg["content"])
            if msg.get("sources") is not None:
                render_sources(msg["sources"], key_prefix=f"chat_h{i}")

    q = st.chat_input("e.g. What is the single-counterparty exposure limit?")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        history.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            with st.spinner("Searching the circulars…"):
                res = answer(q, scope_doc_ids=scope_ids)
                audit.log("chatbot", q, res)
            render_answer(res["answer"], res["sources"])
            render_figures(res["sources"], q)
            render_example(res.get("example") or "")
            render_sources(res["sources"], key_prefix=f"chat_h{len(history)}")
            st.caption(VERIFY_NOTE)
        history.append({"role": "assistant", "content": res["answer"],
                        "sources": res["sources"],
                        "example": res.get("example") or ""})
        # Persist to disk so it can be revisited later. Create the conversation
        # on the first exchange; keep updating the same one after that.
        conv_id = st.session_state.get("active_conv_id") or conversations.new_id()
        st.session_state["active_conv_id"] = conv_id
        conversations.save(conv_id, conversations.derive_title(history), history)
        st.rerun()  # refresh the sidebar list + active-conversation caption


# --------------------------------------------------------------------------
# Page: Browse by division
# --------------------------------------------------------------------------
ENTITY_ICON = {"AIFI": "🏛️", "Commercial Bank": "🏦",
               "All Regulated Entities": "🌐"}
BOTH_ENTITIES = "Both (compare)"


def _entity_scope_picker(key: str) -> set[str] | None:
    """Let the reader choose which rulebook the chatbot answers from.

    Returns a set of doc_ids to restrict retrieval to, or None for the whole
    corpus. This matters for correctness, not just convenience: the AIFI and
    Commercial Bank directions cover the same topics with different numbers, so
    an unscoped question can silently mix the two. NaBFID is an AIFI, so that is
    the default; "Both" exists for deliberate comparison.
    """
    # Only the real institution types are offered as scopes. Cross-entity
    # directions ("All Regulated Entities" — those that bind banks AND AIFIs
    # alike) are folded into every scope by docstore.doc_ids_for_entity, so
    # offering them as a separate choice would wrongly narrow the corpus to
    # just those few documents.
    names = [e["entity"] for e in docstore.entities()
             if e["entity"] != docstore.ENTITY_ALL_RES]
    if len(names) < 2:
        return None
    default = names.index("AIFI") if "AIFI" in names else 0
    choice = st.radio(
        "Answer from", names + [BOTH_ENTITIES], index=default, horizontal=True,
        key=f"scope_{key}",
        help="RBI issues separate directions for AIFIs and Commercial Banks. "
             "NaBFID is an AIFI. Pick 'Both' only to compare them.",
    )
    if choice == BOTH_ENTITIES:
        return None
    return docstore.doc_ids_for_entity(choice)


def page_browse() -> None:
    """Library, partitioned by regulated ENTITY first, then division.

    RBI issues the same topics (capital adequacy, credit risk, ...) separately
    for AIFIs and for Commercial Banks, so entity is the top-level split — a
    user must always know which rulebook they are reading."""
    st.header("🗂️ Browse circulars")
    ents = docstore.entities()
    if not ents:                       # pre-entity manifest: fall back to flat
        names = [d["division"] for d in docstore.divisions()]
        entity, sel = None, st.selectbox("Division", names)
    else:
        labels = [f"{ENTITY_ICON.get(e['entity'], '📁')} {e['entity']}  "
                  f"({e['n']})" for e in ents]
        idx = st.radio("Regulated entity", range(len(ents)),
                       format_func=lambda i: labels[i], horizontal=True,
                       key="browse_entity")
        entity = ents[idx]["entity"]
        divs = docstore.divisions(entity)
        counts = {d["division"]: d["n"] for d in divs}
        sel = st.selectbox("Division", [d["division"] for d in divs],
                           format_func=lambda n: f"{n}  ({counts[n]})",
                           key=f"browse_div_{entity}")

    st.divider()
    for doc in docstore.documents_in(sel, entity):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"**{doc['title']}**")
            st.caption(
                f"{doc['circular_no']} · issued {doc['issue_date']} · "
                f"`{doc['doc_type']}` · {doc['status']}"
            )
        with c2:
            if st.button("Open", key=f"open_{doc['doc_id']}"):
                # Fresh open from the library starts at page 1 (drop any page
                # left over from a previous citation jump to this doc).
                st.session_state.pop(f"citepage_{doc['doc_id']}", None)
                goto("Document explanation", doc["doc_id"])


# --------------------------------------------------------------------------
# Page: Document explanation view
# --------------------------------------------------------------------------
def page_document() -> None:
    doc_id = st.session_state.get("selected_doc")
    if not doc_id:
        st.info("Pick a document from **Browse by division** first.")
        return
    doc = docstore.get_document(doc_id)
    if not doc:
        st.error("Document not found.")
        return

    st.header(doc["title"])
    st.caption(f"{doc['circular_no']} · issued {doc['issue_date']} · "
               f"`{doc['doc_type']}` · {doc['status']}")

    # Version story: amendments that touch this master direction.
    amds = docstore.amendments_of(doc_id)
    if amds:
        st.warning("**Amended by:** " + " · ".join(
            f"{a['title'].split(') ')[-1]} (eff {a['applicable_from'] or a['issue_date']})"
            for a in amds))

    # Section selector (replaces st.tabs so a citation click can navigate here).
    sections = ["📌 Key points", "🔁 What changed", "💡 Implications",
                PDF_SECTION, "💬 Ask about this circular"]
    sec_key = f"sec_{doc_id}"
    pending = f"_pending_sec_{doc_id}"
    if pending in st.session_state:              # apply a citation-triggered switch
        st.session_state[sec_key] = st.session_state.pop(pending)
    st.session_state.setdefault(sec_key, sections[0])
    sec = st.radio("Section", sections, key=sec_key, horizontal=True,
                   label_visibility="collapsed")

    if sec == "📌 Key points":
        st.caption("Each **[n]** is clickable — it opens the source PDF at the "
                   "exact page that states that point.")
        explanation_tab(doc, "key_points", note=True)
    elif sec == "🔁 What changed":
        if doc["doc_type"] != "amendment":
            st.info("This is a base direction. Its amendments are listed at the "
                    "top; open an amendment to see exactly what it changed.")
        else:
            st.caption("Each **[n]** is clickable — it opens the source PDF at "
                       "the exact page that states that change.")
            explanation_tab(doc, "what_changed", note=True)
    elif sec == "💡 Implications":
        st.caption("Explanatory framing — general knowledge, not a rule. "
                   "Regulatory substance is in Key points, cited. (The "
                   "*Illustrative example* line is general framing and is "
                   "not itself cited.)")
        explanation_tab(doc, "implications")
    elif sec == PDF_SECTION:
        pdf_preview(doc)                          # no AI — renders the source page
    else:
        _document_chat(doc)


def _document_chat(doc: dict) -> None:
    st.caption(f"Answers scoped to **{doc['circular_no']}** and its amendment "
               "chain.")
    scope = scope_for_document(doc["doc_id"], doc["doc_type"], doc["amends"])
    key = f"chat_{doc['doc_id']}"
    history = st.session_state.setdefault(key, [])
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["content"], msg.get("sources") or [])
                render_figures(msg.get("sources") or [],
                               history[i - 1]["content"] if i > 0 else "")
                render_example(msg.get("example") or "")
            else:
                st.markdown(msg["content"])
            if msg.get("sources") is not None:
                render_sources(msg["sources"],
                               key_prefix=f"dchat_{doc['doc_id']}_h{i}")

    q = st.chat_input("Ask a doubt about this circular…", key=f"in_{doc['doc_id']}")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        history.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            with st.spinner("Searching this circular…"):
                res = answer(q, scope_doc_ids=scope)
                audit.log("document_chat", q, res, scope_doc=doc["doc_id"])
            render_answer(res["answer"], res["sources"])
            render_figures(res["sources"], q)
            render_example(res.get("example") or "")
            render_sources(res["sources"],
                           key_prefix=f"dchat_{doc['doc_id']}_h{len(history)}")
            st.caption(VERIFY_NOTE)
        history.append({"role": "assistant", "content": res["answer"],
                        "sources": res["sources"],
                        "example": res.get("example") or ""})


# --------------------------------------------------------------------------
# Page: Audit log (admin visibility)
# --------------------------------------------------------------------------
def page_audit() -> None:
    st.header("🧾 Audit log")
    st.caption("Every question, answer, and sources used — for governance.")
    rows = audit.recent(100)
    if not rows:
        st.info("No queries logged yet.")
        return
    for r in rows:
        st.markdown(f"**{r['ts_utc']}** · `{r['surface']}`"
                    + (f" · doc `{r['scope_doc']}`" if r["scope_doc"] else ""))
        st.markdown(f"**Q:** {r['question']}")
        st.caption(("Abstained. " if r["abstained"] else "")
                   + (r["answer"][:240] + "…" if len(r["answer"]) > 240 else r["answer"]))
        st.divider()


# --------------------------------------------------------------------------
# Sidebar: saved chat conversations (Claude-style history)
# --------------------------------------------------------------------------
def _sidebar_conversations() -> None:
    sb = st.sidebar
    sb.markdown("### 💬 Chats")
    if sb.button("➕ New chat", use_container_width=True, key="sb_new_chat"):
        _new_chat()

    convs = conversations.list_all()
    if not convs:
        sb.caption("Your saved chats will appear here.")
        return

    active = st.session_state.get("active_conv_id")
    for conv in convs:
        is_active = conv["id"] == active
        open_col, del_col = sb.columns([6, 1])
        label = ("🟢 " if is_active else "") + conv["title"]
        if open_col.button(label, key=f"open_conv_{conv['id']}",
                           use_container_width=True,
                           help="Reopen this conversation"):
            loaded = conversations.load(conv["id"])
            st.session_state["chat"] = loaded["messages"] if loaded else []
            st.session_state["active_conv_id"] = conv["id"]
            goto("Chatbot")
        if del_col.button("🗑", key=f"del_conv_{conv['id']}",
                          help="Delete this conversation"):
            conversations.delete(conv["id"])
            if is_active:
                st.session_state["chat"] = []
                st.session_state["active_conv_id"] = None
            st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    if not gate():
        return

    # Mobile-friendliness (office laptops block the hosted URL, so phones are
    # the primary access path): wide guide tables scroll sideways inside their
    # own box instead of stretching the page, and the main-area padding tightens
    # on narrow screens so content gets the width.
    st.markdown("""<style>
    [data-testid="stMarkdownContainer"] table {
        display: block; overflow-x: auto; max-width: 100%;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.9rem; padding-right: 0.9rem; padding-top: 2.5rem;
        }
    }
    </style>""", unsafe_allow_html=True)

    st.sidebar.title("📘 RBI Compliance Assistant")
    st.sidebar.caption(f"NaBFID · model: `{LLM_MODEL}`")
    st.sidebar.caption("Developed by Nikhil Pandit")
    # Apply any pending programmatic navigation before the radio is instantiated.
    if "_goto" in st.session_state:
        st.session_state.nav = st.session_state.pop("_goto")
    st.session_state.setdefault("nav", "Chatbot")
    st.sidebar.radio(
        "Go to",
        ["Chatbot", "Browse by division", "Document explanation", "Audit log"],
        key="nav",
    )
    st.sidebar.divider()
    _sidebar_conversations()

    page = st.session_state.nav
    if page == "Chatbot":
        page_chatbot()
    elif page == "Browse by division":
        page_browse()
    elif page == "Document explanation":
        page_document()
    elif page == "Audit log":
        page_audit()


main()
