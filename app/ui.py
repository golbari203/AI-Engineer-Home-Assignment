"""
Streamlit chat interface.

Why Streamlit rather than a web app
-----------------------------------
The brief says the investor is not technical, so a command line fails it. It
also gives two to three days, and a React frontend would consume most of one
without improving a single answer. Streamlit gives a usable chat interface in
pure Python, keeps the stack one language, and leaves the time for the parts
that decide whether the answers are right.

What the screen has to carry
----------------------------
The answer to "how much Costco do I own?" is "2,300 shares, about $2.11m", but
it has to arrive with two custodian names, two as-of dates, two currencies and
a warning that the client's own position file is missing 900 of those shares.
That is unreadable in a terminal and natural here: answer first, sources
underneath, data quality always visible in the sidebar rather than buried in a
caveat the investor may skim past.

Four pages, one job each
------------------------
Ask is the conversation. Portfolio, Companies and Sources show what the
assistant reads, so the investor can check an answer against the record
without having to ask a second question. Every page can hand a question to
Ask, and every cited document opens to the page it was quoted from.

All logic lives in the helpers below and in ui_data so it can be tested; the
Streamlit calls are presentation only.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from pathlib import Path

# `streamlit run app/ui.py` puts app/ on the path, not the project root, so
# `from app import ...` fails wherever PYTHONPATH is not set (Streamlit Cloud).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A relative PORTFOLIO_DIR is read from the project root, whatever the cwd.
PORTFOLIO_DIR = ROOT / os.environ.get("PORTFOLIO_DIR", "/mnt/user-data/uploads")

RECORDS_QUESTION = "Is anything wrong with my records?"

SEVERITY_ICON = {"error": "🔴", "warning": "🟠", "info": "🔵"}
SEVERITY_WORD = {"error": "Discrepancy", "warning": "To review", "info": "Note"}

# Reconciliation check names, in words an investor would use.
CHECK_LABELS = {
    "quantity_mismatch": "Share count differs",
    "missing_in_books": "Missing from your records",
    "missing_in_custodian": "Missing from the custodian statement",
    "line_value": "Line value doesn't add up",
    "statement_total": "Statement total doesn't add up",
    "dividend_implies_different_quantity": "Dividends imply a different share count",
    "cost_basis_fx_rate": "Cost basis uses an unexpected FX rate",
    "cost_basis_vs_trades": "Cost basis differs from trade history",
    "no_purchase_recorded": "No purchase on record",
    "trade_history_does_not_net_to_position": "Trades don't add up to the position",
    "split_adjustment_required": "Stock split adjustment applied",
}


# ---------------------------------------------------------------------------
# Presentation helpers (no Streamlit, so they can be tested)
# ---------------------------------------------------------------------------

def severity_badge(exceptions) -> tuple[str, str]:
    """Returns (label, level) for the sidebar indicator.

    Shown permanently rather than appended to answers: a data-quality problem
    the investor only sees when they happen to ask is one they will miss.
    """
    errors = sum(1 for e in exceptions if e.severity == "error")
    warnings = sum(1 for e in exceptions if e.severity == "warning")
    if errors:
        noun = "discrepancy" if errors == 1 else "discrepancies"
        return (f"{errors} {noun} found in your records", "error")
    if warnings:
        return (f"{warnings} item{'' if warnings == 1 else 's'} to review", "warning")
    return ("Records reconcile", "ok")


def money_short(x) -> str:
    """$20.1m rather than $20,125,684, for places too narrow for the full figure."""
    if not isinstance(x, (int, float)):
        return "-"
    if abs(x) >= 1e9:
        return f"${x / 1e9:.2f}bn"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}m"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}k"
    return f"${x:,.0f}"


def as_of_label(holdings: dict) -> str:
    """'30 Jun 2026', or 'several dates' when the statements disagree."""
    import datetime as dt
    dates = {str(a.get("as_of_date")) for a in holdings.get("account_summary") or []
             if a.get("as_of_date")}
    dates |= {str(p.get("as_of_date")) for p in holdings.get("positions") or []
              if p.get("as_of_date")}
    if len(dates) != 1:
        return "several dates" if dates else "date unknown"
    (only,) = dates
    try:
        return dt.date.fromisoformat(only[:10]).strftime("%-d %b %Y")
    except ValueError:
        return only


def holdings_rows(holdings: dict) -> list[dict]:
    """One row per position per account, largest first.

    The same company can sit in two accounts (Costco does), so rows are not
    merged: merging would hide which custodian holds what.
    """
    total = holdings.get("total_portfolio_value_usd")
    rows = []
    for p in holdings.get("positions") or []:
        value = p.get("market_value_usd")
        rows.append({
            "Ticker": p.get("ticker"),
            "Held at": p.get("custodian"),
            "Shares": p.get("quantity"),
            "Price": p.get("price"),
            "Currency": p.get("currency"),
            "Value (USD)": value,
            "% of portfolio": (100 * value / total
                               if isinstance(value, (int, float)) and total else None),
        })
    return sorted(rows, key=lambda r: r["Value (USD)"] or 0, reverse=True)


def quality_items(quality: dict) -> list[dict]:
    """Record-check findings, worst first, with a plain-language title."""
    items = []
    for bucket, severity in (("errors", "error"), ("warnings", "warning"),
                             ("info", "info")):
        for item in quality.get(bucket) or []:
            check = item.get("check") or ""
            title = CHECK_LABELS.get(check, check.replace("_", " ").capitalize())
            if item.get("ticker"):
                title = f"{item['ticker']}: {title}"
            items.append({"severity": severity, "icon": SEVERITY_ICON[severity],
                          "word": SEVERITY_WORD[severity], "ticker": item.get("ticker"),
                          "title": title, "description": item.get("description", "")})
    return items


def portfolio_summary(holdings: dict) -> list[tuple[str, str]]:
    def money(x) -> str:
        return f"${x:,.0f}" if isinstance(x, (int, float)) else "-"
    return [
        ("Total value", money(holdings.get("total_portfolio_value_usd"))),
        ("Securities", money(holdings.get("total_securities_usd"))),
        ("Cash", money(holdings.get("cash_usd"))),
        ("Positions", str(len(holdings.get("positions") or []))),
        ("Accounts", str(len(holdings.get("accounts") or []))),
    ]


def format_citation(citation: dict) -> str:
    bits = [citation.get("file") or "unknown source"]
    if citation.get("date"):
        bits.append(str(citation["date"]))
    if citation.get("pages"):
        bits.append(f"p.{citation['pages']}")
    if citation.get("detail"):
        bits.append(citation["detail"])
    line = " · ".join(bits)
    if citation.get("from_ocr"):
        line += "  ⚠ text recovered from a scan"
    if citation.get("other_versions"):
        line += f"  ⚠ another version exists: {', '.join(citation['other_versions'])}"
    return line


def dedupe_citations(citations: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in citations:
        key = (c.get("file"), c.get("pages"), c.get("detail"))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def answer_caption(turn) -> str:
    """What was checked and how long it took. Token counts are left out: they
    mean nothing to the investor."""
    return f"{tools_caption(turn)} · {turn.latency_ms / 1000:.1f}s"


def tools_caption(turn) -> str:
    names = {
        "get_holdings": "your positions", "get_transactions": "your trade history",
        "get_facts": "reported company figures", "search_documents": "document text",
        "get_research": "broker research", "get_consensus": "analyst estimates",
        "get_data_quality": "record checks",
    }
    used = [names.get(n, n) for n in dict.fromkeys(turn.tools_used)]
    return "Checked " + ", ".join(used) if used else "Answered without lookups"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# Small, stable additions to the theme in .streamlit/config.toml: a visible
# keyboard focus ring, a quieter eyebrow line, and readable line lengths.
CSS = """
<style>
:root { --accent: #1E4D45; }
*:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px !important; }
.era-eyebrow { font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase;
               color: #5C5A55; margin: 0 0 0.25rem 0; }
.era-lede { font-size: 1.06rem; color: #3D3B37; max-width: 46rem; line-height: 1.55; }
.era-muted { color: #5C5A55; font-size: 0.9rem; }
[data-testid="stChatMessage"] { max-width: 52rem; margin-inline: auto; }
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li { line-height: 1.6; }
[data-testid="stMetricValue"] { font-family: "Source Serif Pro", "Source Serif 4", Georgia, serif; }
</style>
"""

AVATAR = {"user": ":material/person:", "assistant": ":material/account_balance:"}


def main() -> None:
    import streamlit as st

    from app import ui_data as data
    from app.agent import ask
    from app.offline_store import attach_index, build_store
    from app.pipeline import register_corpus
    from app.tools import run_tool
    from app.watcher import FolderWatcher, handle_path

    st.set_page_config(page_title="Equity Research Assistant", page_icon="📊",
                       layout="wide", initial_sidebar_state="expanded",
                       # No "Get help" / "Report a bug" links out to streamlit.io.
                       menu_items={"Get help": None, "Report a bug": None,
                                   "About": "Answers come only from the documents on file."})
    st.markdown(CSS, unsafe_allow_html=True)

    # New documents are classified and their figures extracted by the model,
    # when there is a key to call it with. Without one they are still read,
    # indexed and searchable; only the model-dependent steps are skipped.
    llm_enabled = bool(os.environ.get("AIEHA_API_KEY")) and not os.environ.get("ERA_OFFLINE")

    @st.cache_resource(show_spinner="Preparing your documents. The first visit "
                                    "after a restart takes a minute or two…")
    def load_store():
        store = build_store(PORTFOLIO_DIR, with_index=False)
        # Real embeddings unless ERA_OFFLINE asks for the test stand-in, which
        # matches words rather than meaning.
        attach_index(store, PORTFOLIO_DIR,
                     offline_embedder=bool(os.environ.get("ERA_OFFLINE")))
        register_corpus(store, PORTFOLIO_DIR)
        # Watch the folder for the life of the process, on this same store, so a
        # file dropped into portfolio/ is answerable here without a restart.
        log = logging.getLogger("app.watcher")
        if not log.handlers:                            # one line per file it reads
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("watcher: %(message)s"))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
        try:
            watcher = FolderWatcher(PORTFOLIO_DIR, store, use_llm=llm_enabled)
            watcher.start()
            store._watcher = watcher                     # noqa: SLF001  keep it alive
        except Exception as exc:                        # noqa: BLE001
            logging.getLogger(__name__).warning("folder watcher not started: %s", exc)
        return store

    @st.cache_data(show_spinner="Cataloguing your documents…", max_entries=4)
    def load_catalog(folder_state: tuple, index_size: int):
        # Both arguments are only the cache key. Any file added, removed or
        # rewritten changes folder_state; index_size changes once a new file
        # has been fully read, so the catalog picks up its classification.
        return data.build_catalog(PORTFOLIO_DIR)

    def folder_state() -> tuple:
        return tuple(sorted((p.name, p.stat().st_mtime_ns)
                            for p in PORTFOLIO_DIR.glob("*.pdf")))

    store = load_store()
    catalog = load_catalog(folder_state(),
                           len(getattr(store.retrieval_backend, "records", {}) or {}))
    holdings = run_tool(store, "get_holdings", {})
    quality = run_tool(store, "get_data_quality", {})
    as_of = as_of_label(holdings)
    universe = data.company_universe(holdings, catalog, store.broker_notes,
                                     store.price_targets)
    owned = {c["ticker"] for c in universe if c["owned"]}
    # Only broker notes count as research: the vendor aggregate has no text to
    # quote, so suggesting "what do analysts say" on it would invite a thin answer.
    researched = {c["ticker"] for c in universe if c["broker_notes"]}
    findings = quality_items(quality)
    badge_label, badge_level = severity_badge(store.exceptions)
    documents = [d for d in catalog if not d["duplicate_of"]]
    workbooks = sorted(p.name for p in PORTFOLIO_DIR.glob("*.xlsx")
                       if not p.name.startswith("~$"))
    backend = store.retrieval_backend
    filename_by_document = ({r.document_id: r.filename for r in backend.records.values()}
                            if hasattr(backend, "records") else {})
    entities = [s.get("entity") or "" for s in store.statements]
    household = " ".join(os.path.commonprefix([e.split() for e in entities])) if entities else ""

    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("history", [])
    ss.setdefault("uploads_done", set())

    pages: dict = {}

    def ask_about(question: str) -> None:
        """Hand a question to the Ask page from anywhere."""
        ss.pending = question
        st.switch_page(pages["ask"])

    def badge_alert(text: str | None = None) -> None:
        show = {"error": st.error, "warning": st.warning, "ok": st.success}[badge_level]
        icon = {"error": ":material/error:", "warning": ":material/warning:",
                "ok": ":material/check_circle:"}[badge_level]
        show(text or badge_label, icon=icon)

    # -- Documents --------------------------------------------------------------

    def show_document(filename: str, cited: list[int] | None = None, key: str = "doc") -> None:
        """A document's metadata and its extracted text, opened at the cited page."""
        meta = next((d for d in catalog if d["filename"] == filename), None)
        pages_ = data.document_pages(PORTFOLIO_DIR, filename)
        if not pages_:
            st.warning("This file is no longer in the portfolio folder.", icon=":material/draft:")
            return
        if meta:
            st.markdown(f"**{meta['title']}**")
            facts = [meta["type"], f"{meta['reliability']} source",
                     data.fmt_date(meta["published_date"]) or "date not stated",
                     f"{meta['pages']} pages"]
            st.caption(" · ".join(facts) + f"  \nFile: {filename}")
            if meta["duplicate_of"]:
                st.info(f"Identical to **{meta['duplicate_of']}**, so it is only read once.",
                        icon=":material/content_copy:")
        numbers = [p["page_no"] for p in pages_]
        start = next((n for n in (cited or []) if n in numbers), numbers[0])
        if len(numbers) > 1:
            page_no = st.selectbox(
                "Page", numbers, index=numbers.index(start), key=f"{key}-page",
                format_func=lambda n: f"Page {n}" + ("  · cited" if cited and n in cited else ""))
        else:
            page_no = numbers[0]
        page = next(p for p in pages_ if p["page_no"] == page_no)
        if page.get("method") == "ocr":
            st.caption(":material/document_scanner: This page is a scan. Its text was "
                       "recovered by character recognition and may contain misread characters.")
        with st.container(height=420, border=True):
            st.text(page["text"].strip() or "This page has no readable text.")

    @st.dialog("Source document", width="large")
    def inspect(filename: str, cited: list[int]) -> None:
        show_document(filename, cited, key="dialog")

    def show_sources(cards: list[dict], key: str, tools_note: str) -> None:
        if not cards:
            st.caption(f":material/fact_check: {tools_note}. No document passages were "
                       "needed; the figures come from your statements and record files.")
            return
        with st.expander(f"Sources ({len(cards)})", icon=":material/menu_book:"):
            for i, card in enumerate(cards):
                text, action = st.columns([5, 1], vertical_alignment="center")
                bits = [card["kind"], data.fmt_date(card["date"])]
                if card["pages"]:
                    bits.append(f"pages {card['pages']}")
                lines = [f"**{card['title']}**", " · ".join(b for b in bits if b)]
                if card["positions"]:
                    lines.append(f"Supports: {', '.join(card['positions'])}")
                if card["from_ocr"]:
                    lines.append(":material/document_scanner: Read from a scan; "
                                 "check figures against the original")
                if card["other_versions"]:
                    lines.append(":material/layers: Another version exists ("
                                 + ", ".join(card["other_versions"])
                                 + "); this one was preferred as more reliable")
                text.markdown("  \n".join(lines))
                if action.button("Inspect", key=f"{key}-src-{i}", icon=":material/visibility:",
                                 disabled=not card["file"],
                                 help=None if card["file"] else "The file isn't in the folder"):
                    inspect(card["file"], data.page_numbers(card["pages"]))

    # -- Ask --------------------------------------------------------------------

    def ask_page() -> None:
        with st.sidebar:
            st.markdown("#### Portfolio at a glance")
            st.metric("Total value", portfolio_summary(holdings)[0][1])
            st.caption(f"{money_short(holdings.get('total_securities_usd'))} securities · "
                       f"{money_short(holdings.get('cash_usd'))} cash · as of {as_of}")
            badge_alert()
            st.page_link(pages["portfolio"], label="Review record checks",
                         icon=":material/arrow_forward:")
            st.divider()
            st.markdown("#### This conversation")
            asked = sum(1 for m in ss.messages if m["role"] == "user")
            st.caption(f"{asked} question{'s' if asked != 1 else ''} so far. The assistant "
                       "remembers earlier questions, so follow-ups can be short.")
            if st.button("New conversation", icon=":material/restart_alt:", width="stretch",
                         disabled=not ss.messages):
                ss.messages, ss.history = [], []
                st.rerun()
            with st.expander("How answers are made", icon=":material/info:"):
                st.markdown(
                    f"- The assistant reads only your statements, {len(workbooks)} record "
                    f"files and {len(documents)} documents. It does not use outside knowledge.\n"
                    "- When documents disagree, it prefers filings, then company releases, "
                    "transcripts, presentations, and broker notes last, and says so.\n"
                    "- Each answer lists its sources. **Inspect** opens the page it came from.\n"
                    "- If the documents don't cover a question, it says so rather than guess.")

        if not ss.messages:
            if household:
                st.markdown(f"<p class='era-eyebrow'>{household} · figures as of {as_of}</p>",
                            unsafe_allow_html=True)
            st.title("What would you like to know?")
            st.markdown(
                "<p class='era-lede'>Ask about your holdings, the companies behind them, or "
                f"the research on file. Answers come only from your {len(documents)} documents "
                "and record files, and every figure shows where it came from.</p>",
                unsafe_allow_html=True)

            conc = data.concentration(holdings)
            cols = st.columns(4)
            cols[0].metric("Portfolio value", money_short(holdings.get("total_portfolio_value_usd")),
                           border=True, help=f"Custodian figures as of {as_of}")
            if conc["largest"]:
                cols[1].metric("Largest holding", conc["largest"]["Company"], border=True,
                               help=f"{conc['largest']['% of securities']:.1f}% of securities")
            cols[2].metric("Record checks", badge_label.split(" found")[0].capitalize(),
                           border=True, help="Where your own records and the custodians disagree")
            cols[3].metric("On file", f"{len(documents)} documents", border=True,
                           help=f"Covering {len(universe)} companies")

            st.subheader("Start with a question")
            groups = list(data.suggested_questions(holdings, store.exceptions, universe).items())
            grid = st.columns(2, gap="large")
            for i, (group, questions) in enumerate(groups):
                with grid[i % 2]:
                    st.markdown(f"<p class='era-eyebrow'>{group}</p>", unsafe_allow_html=True)
                    for q in questions:
                        if st.button(q, key=f"suggest-{group}-{q}", width="stretch"):
                            ss.pending = q
                            st.rerun()

        for index, message in enumerate(ss.messages):
            with st.chat_message(message["role"], avatar=AVATAR[message["role"]]):
                if message.get("error"):
                    st.error(message["content"], icon=":material/cloud_off:")
                    st.caption(message.get("hint", ""))
                    retry, _ = st.columns([1, 3])
                    if index == len(ss.messages) - 1 and retry.button(
                            "Try again", key=f"retry-{index}", icon=":material/refresh:"):
                        question = message["question"]
                        ss.messages = ss.messages[:-2]
                        ss.pending = question
                        st.rerun()
                    with st.expander("Technical details"):
                        st.code(message["error"], language=None, wrap_lines=True)
                    continue
                if message["role"] == "assistant" and message.get("unsupported"):
                    st.info("Your documents don't cover this. Nothing below comes from "
                            "outside your files.", icon=":material/search_off:")
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    show_sources(message.get("cards") or [], f"m{index}",
                                 message.get("tools_note", "Checked your records"))
                    if message.get("caption"):
                        st.caption(message["caption"])

        last = ss.messages[-1] if ss.messages else None
        if last and last["role"] == "assistant" and not last.get("error") and last.get("follow_ups"):
            st.markdown("<p class='era-eyebrow' style='max-width:52rem;margin:1rem auto 0.4rem'>"
                        "Keep going</p>", unsafe_allow_html=True)
            row = st.container(horizontal=True)
            for i, q in enumerate(last["follow_ups"]):
                if row.button(q, key=f"follow-{len(ss.messages)}-{i}", icon=":material/subdirectory_arrow_right:"):
                    ss.pending = q
                    st.rerun()

        question = st.chat_input("Ask about a holding, a company, or your records…")
        question = question or ss.pop("pending", None)
        if not question:
            return

        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATAR["user"]):
            st.markdown(question)
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            try:
                with st.spinner("Reading your statements and documents. This usually "
                                "takes 5 to 20 seconds…"):
                    turn, history = ask(question, store, history=ss.history)
            except Exception as exc:                            # noqa: BLE001
                # A failed call must not lose the conversation, and the investor
                # needs to know whether to retry, not read a traceback.
                what, hint = data.friendly_error(exc)
                ss.messages.append({"role": "assistant", "content": what, "hint": hint,
                                    "question": question,
                                    "error": f"{type(exc).__name__}: {exc}"})
                st.rerun()
        ss.history = history
        ss.messages.append({
            "role": "assistant", "content": turn.answer,
            "cards": data.source_cards(dedupe_citations(turn.citations), catalog,
                                       filename_by_document),
            "tools_note": tools_caption(turn),
            "caption": answer_caption(turn),
            "unsupported": data.is_unsupported(turn.answer),
            "follow_ups": data.follow_ups(question, turn.answer, owned, researched),
        })
        st.rerun()

    # -- Portfolio --------------------------------------------------------------

    def portfolio_page() -> None:
        accounts = data.account_allocation(holdings, store.statements)
        conc = data.concentration(holdings)
        st.title("Portfolio")
        notes = [f"Custodian figures as of {as_of}, in US dollars."]
        for a in accounts:
            if a["fx_rate"]:
                notes.append(f"The {a['custodian']} account reports in {a['currency']} and is "
                             f"converted at the statement's rate of {a['fx_rate']:.4f} "
                             f"{a['currency']} per US dollar.")
        st.markdown(f"<p class='era-lede'>{' '.join(notes)}</p>", unsafe_allow_html=True)

        total = holdings.get("total_portfolio_value_usd")
        cols = st.columns(4)
        cols[0].metric("Total value", portfolio_summary(holdings)[0][1], border=True)
        cols[1].metric("Securities", money_short(holdings.get("total_securities_usd")), border=True,
                       help=f"{100 - (conc['cash_pct'] or 0):.1f}% of the portfolio")
        cols[2].metric("Cash", money_short(holdings.get("cash_usd")), border=True,
                       help=f"{conc['cash_pct']:.1f}% of the portfolio" if conc["cash_pct"] else None)
        cols[3].metric("Companies held", f"{conc['companies']}", border=True,
                       help=f"{len(holdings.get('positions') or [])} positions across "
                            f"{len(accounts)} accounts")

        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.subheader("Where it's held")
            for a in accounts:
                with st.container(border=True):
                    st.markdown(f"**{a['custodian']}**")
                    st.caption(f"{a['entity'] or 'Entity not stated'} · account {a['account']} · "
                               f"reports in {a['currency']}")
                    if a["pct"] is not None:
                        st.progress(min(a["pct"] / 100, 1.0),
                                    text=f"{money_short(a['value_usd'])} · {a['pct']:.1f}% of the portfolio")
                    extra = f"Cash {money_short(a['cash_usd'])}"
                    if a["fx_rate"] and a["stated_total"]:
                        extra += (f" · stated as {a['currency']} {a['stated_total']:,.0f} "
                                  "on the statement")
                    st.caption(extra)
        with right:
            st.subheader("Concentration")
            if conc["largest"]:
                st.markdown(
                    f"**{conc['largest']['Company']}** is the largest holding at "
                    f"{conc['largest']['% of securities']:.1f}% of securities. The three largest "
                    f"make up {conc['top3_pct']:.1f}%, and cash is {conc['cash_pct']:.1f}% "
                    "of the whole portfolio.")
            st.dataframe(
                conc["rows"], hide_index=True, width="stretch",
                column_order=["Company", "Value (USD)", "% of securities", "Accounts"],
                column_config={
                    "Value (USD)": st.column_config.NumberColumn(format="dollar"),
                    "% of securities": st.column_config.ProgressColumn(
                        format="%.1f%%", min_value=0, max_value=100),
                    "Accounts": st.column_config.NumberColumn(
                        help="Costco is held in two accounts; its rows are combined here"),
                })

        st.subheader("Holdings")
        search, which = st.columns([2, 3], vertical_alignment="bottom")
        query = search.text_input("Search holdings", placeholder="Company, ticker or custodian",
                                  key="holdings-search")
        custodians = [a["custodian"] for a in accounts]
        chosen = which.segmented_control("Custodian", ["All accounts", *custodians],
                                         default="All accounts", key="holdings-custodian")
        rows = data.filter_rows(data.holdings_table(holdings), query,
                                Custodian=None if chosen in (None, "All accounts") else chosen)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch", column_config={
                "Shares": st.column_config.NumberColumn(format="localized"),
                "Price (USD)": st.column_config.NumberColumn(format="%.2f"),
                "Value (USD)": st.column_config.NumberColumn(format="dollar"),
                "% of portfolio": st.column_config.NumberColumn(format="%.1f%%"),
                "Cost basis": st.column_config.NumberColumn(
                    format="localized", help="In the currency shown in the next column"),
            })
            st.caption("Cost bases are shown in each account's own currency. Gains aren't "
                       "calculated here: the currencies differ, and one conversion rate is "
                       "under review in the record checks below.")
        else:
            st.info("No holdings match that search.", icon=":material/search_off:")

        st.subheader("Record checks")
        st.caption("Your family office's own records compared with the custodians' statements, "
                   "trade history and dividends.")
        badge_alert()
        if not findings:
            st.caption("Nothing to review.")
        for i, item in enumerate(findings):
            with st.container(border=True):
                text, action = st.columns([5, 1.3], vertical_alignment="center")
                text.markdown(f"{item['icon']} **{item['word']}: {item['title']}**  \n"
                              f"{item['description']}")
                if action.button("Ask about this", key=f"finding-{i}", icon=":material/forum:",
                                 width="stretch"):
                    ask_about(f"Explain this record issue and what I should do about it: "
                              f"{item['description']}")

    # -- Companies --------------------------------------------------------------

    def companies_page() -> None:
        st.title("Companies")
        st.markdown("<p class='era-lede'>Profiles built from the filings, releases, transcripts "
                    "and broker research on file. Companies you own are marked; the others are "
                    "ones you have research on but no position in.</p>", unsafe_allow_html=True)

        picker, profile = st.columns([1, 2.8], gap="large")
        with picker:
            show = st.segmented_control("Show", ["All", "Owned", "Watching"], default="All",
                                        key="companies-filter")
            query = st.text_input("Find a company", placeholder="Name or ticker",
                                  key="companies-search")
            pool = [c for c in universe
                    if (show in (None, "All") or c["owned"] == (show == "Owned"))
                    and (not query.strip() or query.strip().lower() in
                         f"{c['name']} {c['ticker']}".lower())]
            if not pool:
                st.info("No companies match.", icon=":material/search_off:")
                return
            tickers = [c["ticker"] for c in pool]
            current = ss.get("company") if ss.get("company") in tickers else tickers[0]
            by_ticker = {c["ticker"]: c for c in universe}
            ticker = st.radio(
                "Company", tickers, index=tickers.index(current), key="company-radio",
                format_func=lambda t: f"{by_ticker[t]['name']} · {t}"
                                      + ("" if by_ticker[t]["owned"] else "  (watching)"))
            ss.company = ticker

        c = by_ticker[ticker]
        name = c["name"]
        with profile:
            st.header(name)
            badges = st.container(horizontal=True)
            with badges:
                if c["owned"]:
                    st.badge("Owned", icon=":material/check_circle:", color="green")
                else:
                    st.badge("Watching · not owned", icon=":material/visibility:", color="gray")
                st.badge(ticker, color="blue")
                st.badge(f"{c['documents']} documents", icon=":material/description:", color="gray")
            if c["owned"]:
                position = data.company_positions(holdings)[ticker]
                total = holdings.get("total_portfolio_value_usd")
                share = f", {100 * position['value_usd'] / total:.1f}% of the portfolio" if total else ""
                st.markdown(f"You hold **{position['shares']:,.0f} shares** worth "
                            f"**{money_short(position['value_usd'])}**{share}, in "
                            f"{len(position['accounts'])} account"
                            f"{'s' if len(position['accounts']) != 1 else ''} (as of {as_of}).")
            else:
                st.markdown(f"Not in your portfolio. On file: {c['documents']} documents and "
                            f"{c['broker_notes']} broker note{'s' if c['broker_notes'] != 1 else ''}.")

            asks = st.container(horizontal=True)
            for i, q in enumerate(data.company_questions(ticker, c["owned"], ticker in researched)):
                if asks.button(q, key=f"company-ask-{ticker}-{i}", icon=":material/forum:"):
                    ask_about(q)

            overview, figures, estimates, research, docs = st.tabs(
                ["Overview", "Figures", "Estimates", "Research", "Documents"])
            with figures:
                rows = data.figure_rows(store.facts, ticker, filename_by_document)
                if rows:
                    query = st.text_input("Search figures", placeholder="Revenue, EPS, Q2 2026…",
                                          key=f"figures-search-{ticker}")
                    shown = data.filter_rows(rows, query)
                    st.dataframe(shown, hide_index=True, width="stretch", column_config={
                        "Page": st.column_config.NumberColumn(format="%d")})
                    disputed = sum(1 for r in rows if r["Disputed"])
                    st.caption(
                        f"{len(rows)} figures read from {name}'s documents, exactly as "
                        "printed and with their accounting basis. "
                        + (f"{disputed} are stated differently by another document; "
                           "ask about one to see which source wins and why."
                           if disputed else "No two documents disagree on any of them."))
                else:
                    st.info(f"No figures have been extracted for {name} yet. Its documents "
                            "are still searchable, so questions about it are answered from "
                            "their text.", icon=":material/table_rows:")
            street = data.street_view(store.price_targets, ticker)
            with overview:
                if street:
                    cols = st.columns(3)
                    cols[0].metric("Last price", f"${street['last_price']:,.2f}" if street["last_price"] else "—",
                                   border=True, help=f"As of {data.fmt_date(street['as_of_date'])}")
                    cols[1].metric("Street target", f"${street['consensus_pt']:,.0f}" if street["consensus_pt"] else "—",
                                   border=True,
                                   help=(f"{street['upside_pct']:+.1f}% from the last price"
                                         if street["upside_pct"] is not None else None))
                    cols[2].metric("Ratings", f"{street['buy']:.0f} · {street['hold']:.0f} · {street['sell']:.0f}",
                                   border=True, help="Buy · Hold · Sell")
                    st.caption(f"Street figures are a data vendor's aggregate across all analysts, "
                               f"as of {data.fmt_date(street['as_of_date'])}. They are not research "
                               "you can inspect; see the Research tab for notes on file.")
                else:
                    st.caption("No street price target or rating counts on file.")
                if c["owned"]:
                    st.markdown("**Your positions**")
                    st.dataframe(
                        [{"Custodian": p["custodian"], "Account": p["account"],
                          "Shares": p["quantity"], "Value (USD)": p["market_value_usd"],
                          "Cost basis": p["cost_basis"], "Cost currency": p["currency"]}
                         for p in data.company_positions(holdings)[ticker]["accounts"]],
                        hide_index=True, width="stretch", column_config={
                            "Shares": st.column_config.NumberColumn(format="localized"),
                            "Value (USD)": st.column_config.NumberColumn(format="dollar"),
                            "Cost basis": st.column_config.NumberColumn(format="localized")})
                issues = [f for f in findings if f["ticker"] == ticker]
                for item in issues:
                    (st.error if item["severity"] == "error" else st.warning
                     if item["severity"] == "warning" else st.info)(
                        f"**{item['word']}: {item['title']}.** {item['description']}")
            with estimates:
                rows = data.estimate_rows(store.consensus, ticker)
                if rows:
                    st.dataframe(rows, hide_index=True, width="stretch", column_config={
                        "Period end": st.column_config.DateColumn(format="D MMM YYYY"),
                        "Reported": st.column_config.DateColumn(format="D MMM YYYY"),
                        "Revenue est. ($m)": st.column_config.NumberColumn(format="localized"),
                        "Revenue actual ($m)": st.column_config.NumberColumn(format="localized"),
                        "EPS est.": st.column_config.NumberColumn(format="%.2f"),
                        "EPS actual": st.column_config.NumberColumn(format="%.2f"),
                        "Estimates": st.column_config.NumberColumn(format="%d"),
                    })
                    st.caption("From the consensus file dated 30 Jun 2026; results reported after "
                               "that date are blank. EPS basis differs by company, so compare "
                               "like with like.")
                else:
                    st.info(f"No consensus estimates on file for {name}.", icon=":material/search_off:")
            with research:
                rows = data.broker_rows(store.broker_notes, ticker)
                if rows:
                    st.dataframe(rows, hide_index=True, width="stretch", column_config={
                        "Date": st.column_config.DateColumn(format="D MMM YYYY"),
                        "Price target (USD)": st.column_config.NumberColumn(format="dollar")})
                    st.caption("Each house rates on its own scale, so ratings and targets are "
                               "shown side by side and never averaged.")
                else:
                    st.info(f"No broker notes on file for {name}.", icon=":material/search_off:")
            with docs:
                company_docs = sorted((d for d in catalog if d["ticker"] == ticker),
                                      key=lambda d: d["published_date"] or dt.date.min, reverse=True)
                if not company_docs:
                    st.info(f"No documents on file for {name}.", icon=":material/search_off:")
                for i, d in enumerate(company_docs):
                    text, action = st.columns([5, 1], vertical_alignment="center")
                    flags = []
                    if d["scanned_pages"]:
                        flags.append(f"{d['scanned_pages']} scanned page{'s' if d['scanned_pages'] != 1 else ''}")
                    if d["duplicate_of"]:
                        flags.append("duplicate copy")
                    text.markdown(f"**{d['title']}**  \n"
                                  + " · ".join(b for b in [d["reliability"] + " source",
                                                           data.fmt_date(d["published_date"]),
                                                           f"{d['pages']} pages", *flags] if b))
                    if action.button("Inspect", key=f"cdoc-{ticker}-{i}", icon=":material/visibility:"):
                        inspect(d["filename"], [])

    # -- Sources ----------------------------------------------------------------

    def sources_page() -> None:
        st.title("Sources")
        st.markdown(f"<p class='era-lede'>{len(documents)} documents and {len(workbooks)} record "
                    "files. The assistant answers only from these; select a document to read "
                    "the text it works from.</p>", unsafe_allow_html=True)

        q, who, kind = st.columns([2, 1.5, 1.5], vertical_alignment="bottom")
        query = q.text_input("Search documents", placeholder="Title, company, type or file name",
                             key="sources-search")
        companies = who.multiselect("Company", sorted({d["company"] for d in catalog}),
                                    key="sources-company", placeholder="All companies")
        types = kind.multiselect("Type", sorted({d["type"] for d in catalog}),
                                 key="sources-type", placeholder="All types")
        rows = [{"Title": d["title"], "Company": d["company"], "Type": d["type"],
                 "Reliability": d["reliability"], "Date": d["published_date"],
                 "Pages": d["pages"], "Scanned": d["scanned_pages"],
                 "File": d["filename"]} for d in catalog]
        rows = data.filter_rows(rows, query, Company=companies or None, Type=types or None)
        if not rows:
            st.info("No documents match. Clear a filter or try a different word.",
                    icon=":material/search_off:")
        else:
            picked = st.dataframe(
                rows, hide_index=True, width="stretch", on_select="rerun",
                selection_mode="single-row", key="sources-table", column_config={
                    "Date": st.column_config.DateColumn(format="D MMM YYYY"),
                    "Scanned": st.column_config.NumberColumn(
                        help="Pages whose text was recovered from a scan"),
                })
            st.caption(f"Showing {len(rows)} of {len(catalog)}. Select a row to read it.")
            selected = picked.selection.rows if picked and picked.selection else []
            if selected:
                st.subheader("Document")
                show_document(rows[selected[0]]["File"], key="sources")

        with st.expander("How reliability is ranked", icon=":material/info:"):
            st.markdown(
                "When documents disagree, the assistant prefers the more reliable source "
                "and tells you it did:\n\n"
                "1. **Filing** — 10-K and 10-Q reports\n2. **Company release** — earnings "
                "releases and announcements\n3. **Corrected transcript**\n4. **Automatic "
                "transcript**\n5. **Presentation**\n6. **Broker note** — one firm's view\n"
                "7. **Internal record** — your own family office files")
        with st.expander(f"Record files ({len(workbooks)})", icon=":material/table_chart:"):
            for name in workbooks:
                st.markdown(f"**{name}**  \n{data.workbook_role(name)}")

        st.subheader("Add a document")
        uploaded = st.file_uploader(
            "Add a PDF to the portfolio folder", type=["pdf"],
            help="Earnings releases, filings, statements or broker notes, for any "
                 "company. It becomes answerable straight away, and its figures "
                 "appear on the company's profile."
                 + ("" if llm_enabled else " (Figures need the AI service, "
                                           "which isn't configured.)"))
        # The uploader keeps its file across reruns, so remember what has been
        # read; otherwise every click would re-ingest it.
        if uploaded is not None and uploaded.file_id not in ss.uploads_done:
            with st.spinner(f"Reading {uploaded.name}. Identifying the company and "
                            "extracting its figures can take up to a minute…"):
                target = PORTFOLIO_DIR / uploaded.name
                target.write_bytes(uploaded.getbuffer())
                # The same path the watcher uses, so a new custodian statement
                # also re-runs the record checks.
                event = handle_path(target, store, PORTFOLIO_DIR, wait=False,
                                    use_llm=llm_enabled)
            ss.uploads_done.add(uploaded.file_id)
            if event.result and event.result.status == "ingested":
                st.success(event.summary(), icon=":material/check_circle:")
            else:
                st.info(event.summary(), icon=":material/info:")

    pages["ask"] = st.Page(ask_page, title="Ask", icon=":material/forum:",
                           url_path="ask", default=True)
    pages["portfolio"] = st.Page(portfolio_page, title="Portfolio",
                                 icon=":material/account_balance_wallet:", url_path="portfolio")
    pages["companies"] = st.Page(companies_page, title="Companies",
                                 icon=":material/domain:", url_path="companies")
    pages["sources"] = st.Page(sources_page, title="Sources",
                               icon=":material/menu_book:", url_path="sources")
    st.navigation(list(pages.values()), position="top").run()


if __name__ == "__main__":
    main()
