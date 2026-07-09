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
def _citation_link(idx: int, url: str | None, page: int, title: str) -> str:
    """One inline, clickable citation marker — opens the source PDF at the
    right page in a new browser tab."""
    if not url:
        return f"<sup>[{idx}]</sup>"
    href = f"{url}#page={page}&view=FitH"
    return (f'<a href="{href}" target="_blank" rel="noopener noreferrer" '
            f'title="{title}" style="text-decoration:none;color:#1f6feb;'
            'font-size:0.72em;font-weight:700;vertical-align:super;'
            'background:#eef3fb;border:1px solid #d6e2f5;border-radius:4px;'
            f'padding:0 4px;margin:0 1px;">{idx}</a>')


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
            chips += _citation_link(num, url, page, f"Source — page {page}")
        base = raw.rstrip()
        if base.endswith("|"):  # markdown table row: keep the chip inside the
            out.append(f"{base[:-1].rstrip()} {chips} |")  # last cell.
        else:
            out.append(f"{base} {chips}")
    st.markdown("\n".join(out), unsafe_allow_html=True)


def pdf_preview(doc: dict) -> None:
    """Full, scrollable + searchable PDF viewer. If the user arrived via a
    citation, it opens jumped to that page; otherwise page 1. (No single-page
    image view — the whole document is scrollable in the browser's PDF reader.)"""
    rel_path = doc["file_path"]
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        st.warning(f"PDF not found: {rel_path}")
        return
    url = _static_pdf_url(rel_path)
    if not url:
        st.warning("PDF could not be prepared for viewing.")
        return

    cite_page = st.session_state.get(f"citepage_{doc['doc_id']}")
    page = min(max(1, cite_page), _pdf_page_count(rel_path)) if cite_page else 1
    if cite_page:
        st.success(f"📄 Opened at cited page {page} — scroll freely from here.")
    st.caption("Full document — scroll to read; use the viewer toolbar or Ctrl+F "
               "to search.")
    # Real URL + "#page=N" so the native viewer lands on the right page; the
    # cache-buster keeps the iframe from re-jumping to the page on every rerun.
    src = f"{url}#page={page}&view=FitH&toolbar=1"
    st.markdown(
        f'<iframe src="{src}" width="100%" height="900" '
        f'style="border:1px solid #ddd;"></iframe>',
        unsafe_allow_html=True,
    )
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
            chips.append(_citation_link(i, url, page, title))
        return "".join(chips) if chips else match.group(0)

    st.markdown(_CITE_RE.sub(repl, text), unsafe_allow_html=True)


def _open_pdf_button(url: str | None, page: int) -> str:
    """A full-width 'Open p.N' link styled as a button that opens the source
    PDF at that page in a NEW browser tab — so the chat is never left and the
    reader can keep clicking other citations (this is what the inline [n]
    citations do too; the old in-app-navigation button had no way back)."""
    if not url:
        return f"<div style='text-align:center;color:#888;'>p.{page}</div>"
    href = f"{url}#page={page}&view=FitH"
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
                st.caption(f"{c['division']} · `{c['doc_type']}` · "
                           f"p.{page} · {c['section_ref'] or '—'}")
                if c.get("cite_quote") and c.get("cite_verified"):
                    st.markdown(f"> *“{c['cite_quote']}”* — verified on p.{page}")
                else:
                    st.caption('"' + " ".join(c["text"].split())[:280] + '…"')
            with col2:
                url = _static_url_for_doc(c["doc_id"])
                st.markdown(_open_pdf_button(url, page),
                            unsafe_allow_html=True)


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

    history = st.session_state.setdefault("chat", [])
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["content"], msg.get("sources") or [])
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
                res = answer(q)
                audit.log("chatbot", q, res)
            render_answer(res["answer"], res["sources"])
            render_sources(res["sources"], key_prefix=f"chat_h{len(history)}")
            st.caption(VERIFY_NOTE)
        history.append({"role": "assistant", "content": res["answer"],
                        "sources": res["sources"]})
        # Persist to disk so it can be revisited later. Create the conversation
        # on the first exchange; keep updating the same one after that.
        conv_id = st.session_state.get("active_conv_id") or conversations.new_id()
        st.session_state["active_conv_id"] = conv_id
        conversations.save(conv_id, conversations.derive_title(history), history)
        st.rerun()  # refresh the sidebar list + active-conversation caption


# --------------------------------------------------------------------------
# Page: Browse by division
# --------------------------------------------------------------------------
def page_browse() -> None:
    st.header("🗂️ Browse by division")
    divs = docstore.divisions()
    names = [d["division"] for d in divs]
    counts = {d["division"]: d["n"] for d in divs}

    sel = st.selectbox("Division", names,
                       format_func=lambda n: f"{n}  ({counts[n]})")
    st.divider()
    for doc in docstore.documents_in(sel):
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
            render_sources(res["sources"],
                           key_prefix=f"dchat_{doc['doc_id']}_h{len(history)}")
            st.caption(VERIFY_NOTE)
        history.append({"role": "assistant", "content": res["answer"],
                        "sources": res["sources"]})


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
