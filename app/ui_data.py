"""
Data shaping for the interface.

Everything the screens show is computed here from the store, the tool payloads
and the portfolio folder, so it can be tested without Streamlit. Nothing here
invents a figure: where the files are silent the helpers return None and the
screen says so.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from app.retrieval import TIER_NAMES

# The ten companies the classifier recognises (classify.KNOWN_TICKERS), named
# the way their own documents name them.
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "JPM": "JPMorgan Chase", "XOM": "ExxonMobil", "COST": "Costco",
    "UNH": "UnitedHealth", "TSLA": "Tesla", "LLY": "Eli Lilly",
}

# Patterns that find a company in free text, mirroring classify.NAME_TO_TICKER.
_MENTIONS = {
    "AAPL": r"\bApple\b|\bAAPL\b", "MSFT": r"\bMicrosoft\b|\bMSFT\b",
    "NVDA": r"\bNVIDIA\b|\bNvidia\b|\bNVDA\b", "AMZN": r"\bAmazon\b|\bAWS\b|\bAMZN\b",
    "JPM": r"\bJPMorgan\b|\bJ\.P\. Morgan\b|\bJPM\b",
    "XOM": r"\bExxon\s?Mobil\b|\bExxon\b|\bXOM\b", "COST": r"\bCostco\b|\bCOST\b",
    "UNH": r"\bUnitedHealth\b|\bUNH\b", "TSLA": r"\bTesla\b|\bTSLA\b",
    "LLY": r"\bEli Lilly\b|\bLilly\b|\bLLY\b",
}

DOC_TYPE_LABELS = {
    "filing_10k": "Annual report (10-K)", "filing_10q": "Quarterly report (10-Q)",
    "press_release": "Company release", "transcript": "Call transcript",
    "presentation": "Presentation", "broker_note": "Broker note",
    "custodian_statement": "Custodian statement",
    "corporate_action": "Corporate action", "internal_workbook": "Record file",
}

# What each record workbook contributes, for the Sources page.
WORKBOOK_ROLES = {
    "holdings": "Your family office's own positions and trade blotter",
    "prices": "Month-end closing prices, dividends and the NVIDIA split",
    "fx": "Exchange rates used to check currency conversions",
    "consensus": "Analyst estimates, street price targets and broker ratings, as of 30 Jun 2026",
}


# Companies outside the original ten, named by the classifier when their first
# document arrived. Filled by build_catalog from cached classifications, so a
# new company keeps its name across restarts.
_DISCOVERED: dict[str, str] = {}


def remember_company(ticker: str | None, name: str | None) -> None:
    if ticker and name and ticker not in COMPANY_NAMES:
        _DISCOVERED[ticker] = name.strip()


def company_name(ticker: str | None) -> str:
    if not ticker:
        return "General"
    return COMPANY_NAMES.get(ticker) or _DISCOVERED.get(ticker) or ticker


def _mention_patterns() -> dict[str, str]:
    patterns = dict(_MENTIONS)
    for ticker, name in _DISCOVERED.items():
        patterns[ticker] = rf"\b{re.escape(name)}\b|\b{re.escape(ticker)}\b"
    return patterns


def tier_label(tier: int | None) -> str:
    return TIER_NAMES.get(tier or 0, "document").capitalize()


def doc_type_label(doc_type: str | None) -> str:
    return DOC_TYPE_LABELS.get(doc_type or "", (doc_type or "Document").replace("_", " ").capitalize())


def fmt_date(value) -> str:
    """'30 Jun 2026'. Accepts dates and ISO strings; '' when unknown."""
    if not value:
        return ""
    if isinstance(value, str):
        try:
            value = dt.date.fromisoformat(value[:10])
        except ValueError:
            return value
    return f"{value.day} {value:%b %Y}"


def _fold(s: str) -> str:
    """Lowercase without accents, so 'Léman' matches 'Leman'."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def company_positions(holdings: dict) -> dict[str, dict]:
    """Positions summed per company across accounts. Costco is held in two."""
    out: dict[str, dict] = {}
    for p in holdings.get("positions") or []:
        row = out.setdefault(p["ticker"], {"ticker": p["ticker"], "shares": 0.0,
                                           "value_usd": 0.0, "accounts": []})
        row["shares"] += p.get("quantity") or 0
        row["value_usd"] += p.get("market_value_usd") or 0
        row["accounts"].append(p)
    return out


def concentration(holdings: dict) -> dict:
    """Weights by company, as a share of securities and of the whole portfolio."""
    securities = holdings.get("total_securities_usd") or 0
    total = holdings.get("total_portfolio_value_usd") or 0
    rows = sorted(company_positions(holdings).values(),
                  key=lambda r: r["value_usd"], reverse=True)
    table = [{
        "Company": company_name(r["ticker"]), "Ticker": r["ticker"],
        "Value (USD)": r["value_usd"],
        "% of securities": 100 * r["value_usd"] / securities if securities else None,
        "% of portfolio": 100 * r["value_usd"] / total if total else None,
        "Accounts": len(r["accounts"]),
    } for r in rows]
    weights = [r["% of securities"] or 0 for r in table]
    return {
        "rows": table,
        "companies": len(table),
        "largest": table[0] if table else None,
        "top3_pct": sum(weights[:3]) if table else None,
        "cash_pct": 100 * (holdings.get("cash_usd") or 0) / total if total else None,
    }


def account_allocation(holdings: dict, statements: list[dict]) -> list[dict]:
    """One row per custodian account, with the rate used to convert it."""
    total = holdings.get("total_portfolio_value_usd") or 0
    by_account = {s.get("account_no"): s for s in statements}
    rows = []
    for a in holdings.get("account_summary") or []:
        statement = by_account.get(a.get("account"), {})
        value = a.get("stated_total_usd")
        rows.append({
            "custodian": a.get("custodian"), "account": a.get("account"),
            "entity": statement.get("entity"), "currency": a.get("currency"),
            "value_usd": value, "cash_usd": a.get("cash_usd"),
            "stated_total": a.get("stated_total"),
            "pct": 100 * value / total if total and value else None,
            "fx_rate": statement.get("fx_rate") if a.get("currency") != "USD" else None,
            "as_of": a.get("as_of_date"),
        })
    return rows


def holdings_table(holdings: dict) -> list[dict]:
    """Every position in every account, with its cost basis in its own currency."""
    total = holdings.get("total_portfolio_value_usd")
    rows = []
    for p in holdings.get("positions") or []:
        value = p.get("market_value_usd")
        rows.append({
            "Company": company_name(p.get("ticker")), "Ticker": p.get("ticker"),
            "Custodian": p.get("custodian"), "Account": p.get("account"),
            "Shares": p.get("quantity"), "Price (USD)": p.get("price"),
            "Value (USD)": value,
            "% of portfolio": 100 * value / total if total and value is not None else None,
            "Cost basis": p.get("cost_basis"), "Cost currency": p.get("currency"),
        })
    return sorted(rows, key=lambda r: r["Value (USD)"] or 0, reverse=True)


def filter_rows(rows: list[dict], query: str = "", **equals) -> list[dict]:
    """Case-insensitive text search across all columns, plus exact filters."""
    q = _fold(query.strip())
    out = []
    for r in rows:
        if any(v and r.get(k) not in (v if isinstance(v, (list, set, tuple)) else [v])
               for k, v in equals.items()):
            continue
        if q and q not in _fold(" ".join(str(v) for v in r.values() if v is not None)):
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

def company_universe(holdings: dict, catalog: list[dict], broker_notes: list[dict],
                     price_targets: list[dict]) -> list[dict]:
    """Owned companies first, then those with research on file but no position."""
    owned = company_positions(holdings)
    docs = defaultdict(int)
    for d in catalog:
        if d.get("ticker"):
            docs[d["ticker"]] += 1
    researched = {n["ticker"] for n in broker_notes} | set(docs)
    tickers = set(owned) | researched
    rows = [{
        "ticker": t, "name": company_name(t), "owned": t in owned,
        "value_usd": owned[t]["value_usd"] if t in owned else None,
        "shares": owned[t]["shares"] if t in owned else None,
        "documents": docs.get(t, 0),
        "broker_notes": sum(1 for n in broker_notes if n["ticker"] == t),
        "has_street_view": any(p["ticker"] == t for p in price_targets),
    } for t in tickers]
    return sorted(rows, key=lambda r: (not r["owned"], -(r["value_usd"] or 0), r["name"]))


def street_view(price_targets: list[dict], ticker: str) -> dict | None:
    row = next((p for p in price_targets if p["ticker"] == ticker), None)
    if not row:
        return None
    last, target = row.get("last_price"), row.get("consensus_pt")
    return {**row, "upside_pct": 100 * (target / last - 1) if last and target else None}


def estimate_rows(consensus: list[dict], ticker: str) -> list[dict]:
    rows = sorted((c for c in consensus if c["ticker"] == ticker),
                  key=lambda c: c.get("period_end") or dt.date.min, reverse=True)
    return [{
        "Period": c.get("fiscal_period"), "Period end": c.get("period_end"),
        "Reported": c.get("report_date"),
        "Revenue est. ($m)": c.get("revenue_consensus"),
        "Revenue actual ($m)": c.get("revenue_actual"),
        "EPS est.": c.get("eps_consensus"), "EPS actual": c.get("eps_actual"),
        "EPS basis": c.get("eps_basis"), "Estimates": c.get("n_estimates"),
    } for c in rows]


def broker_rows(broker_notes: list[dict], ticker: str) -> list[dict]:
    rows = sorted((n for n in broker_notes if n["ticker"] == ticker),
                  key=lambda n: n.get("note_date") or dt.date.min, reverse=True)
    return [{"Date": n.get("note_date"), "House": n.get("house"),
             "Rating (house scale)": n.get("rating"), "Price target (USD)": n.get("price_target"),
             "Analyst": n.get("analyst")} for n in rows]


METRIC_LABELS = {
    "eps_basic": "EPS (basic)", "eps_diluted": "EPS (diluted)",
    "gross_margin_pct": "Gross margin %", "operating_margin_pct": "Operating margin %",
    "cet1_ratio": "CET1 ratio", "rotce": "ROTCE", "cfo_name": "Chief financial officer",
    "ceo_name": "Chief executive", "guidance_revenue_low": "Revenue guidance (low)",
    "guidance_revenue_high": "Revenue guidance (high)",
    "guidance_operating_income_low": "Operating income guidance (low)",
    "guidance_operating_income_high": "Operating income guidance (high)",
}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())


def fact_value(fact) -> str:
    """A figure as printed: '$24,733m', '$4.68', '72.6%'."""
    if fact.value_text:
        return fact.value_text
    v = fact.value_num
    if v is None:
        return "—"
    unit = (fact.unit or "").lower()
    if unit == "usd_millions":
        return f"${v:,.0f}m" if abs(v) >= 100 else f"${v:,.1f}m"
    if unit == "usd":
        return f"${v:,.2f}"
    if unit == "percent":
        return f"{v:,.1f}%"
    if unit == "gwh":
        return f"{v:,.1f} GWh"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def figure_rows(facts: list, ticker: str, filename_by_document: dict[int, str]) -> list[dict]:
    """Every extracted figure for a company, newest period first.

    Figures that other documents state differently are marked, because the
    disagreement is often the most important thing about a number.
    """
    from app.ingest.facts import find_conflicts
    mine = [f for f in facts if f.ticker == ticker]
    disputed = {c.versions[0].key for c in find_conflicts(mine)} if mine else set()
    rows = []
    for f in sorted(mine, key=lambda f: (f.period_end or dt.date.min, f.metric), reverse=True):
        kind = "Guidance" if f.is_guidance else ("Restated" if f.is_restated else "Reported")
        rows.append({
            "Period": f.fiscal_period, "Figure": metric_label(f.metric)
                                                 + (f" · {f.segment}" if f.segment else ""),
            "Value": fact_value(f), "Basis": f.basis or "—", "Kind": kind,
            "Disputed": "Yes" if f.key in disputed else "",
            "Source": filename_by_document.get(f.document_id, "—"), "Page": f.page_no,
        })
    return rows


def company_questions(ticker: str, owned: bool, has_research: bool) -> list[str]:
    name = company_name(ticker)
    questions = [f"What did {name} report in its latest quarter?",
                 f"What did {name} guide to for next quarter?"]
    if has_research:
        questions.append(f"What do analysts say about {name}?")
    if owned:
        questions.append(f"How much {name} do I own, and what did I pay?")
    return questions


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def suggested_questions(holdings: dict, exceptions: list, universe: list[dict]
                        ) -> dict[str, list[str]]:
    """Starter questions built from what is actually on file."""
    owned = [c for c in universe if c["owned"]]
    watching = [c for c in universe if not c["owned"]]
    groups: dict[str, list[str]] = {}

    portfolio = ["What is my portfolio worth?"]
    if owned:
        portfolio.append(f"How much {owned[0]['name']} do I own?")
    portfolio.append("How concentrated is my portfolio?")
    groups["Your portfolio"] = portfolio

    flagged = []
    for e in sorted(exceptions, key=lambda e: {"error": 0, "warning": 1}.get(e.severity, 2)):
        if e.ticker and e.ticker not in flagged and e.severity == "error":
            flagged.append(e.ticker)
    if flagged:
        groups["Your records"] = (
            ["Is anything wrong with my records?"]
            + [f"Why don't my {company_name(t)} records match the custodian?"
               for t in flagged[:2]])

    if owned:
        companies = [f"What did {owned[0]['name']} report in its latest quarter?"]
        researched = next((c for c in owned if c["broker_notes"]), None)
        if researched:
            companies.append(f"What do analysts say about {researched['name']}?")
        if len(owned) > 1:
            companies.append(f"Compare {owned[0]['name']} and {owned[1]['name']}'s latest results")
        groups["Companies you own"] = companies

    if watching:
        groups["Companies you're watching"] = [
            f"What do analysts say about {c['name']}?" for c in watching[:2]
            if c["broker_notes"]] or [f"What is on file about {watching[0]['name']}?"]
    return groups


def tickers_in(text: str) -> list[str]:
    """Companies mentioned in text, in order of first mention."""
    found = []
    for ticker, pattern in _mention_patterns().items():
        m = re.search(pattern, text or "")
        if m:
            found.append((m.start(), ticker))
    return [t for _, t in sorted(found)]


def follow_ups(question: str, answer: str, owned: set[str], researched: set[str],
               limit: int = 3) -> list[str]:
    """Next questions that build on the last answer, never repeating it."""
    in_question = tickers_in(question)
    in_answer = tickers_in(answer)
    # A long list of companies in an answer is a portfolio answer, not a
    # company one; following up on the first name in the list would be noise.
    focus = in_question or (in_answer if len(in_answer) <= 2 else [])
    asked = question.strip().lower()
    out: list[str] = []
    if focus:
        t = focus[0]
        name = company_name(t)
        candidates = []
        if t in owned and "own" not in asked:
            candidates.append(f"How much {name} do I own, and what did I pay?")
        if t in researched and "analyst" not in asked:
            candidates.append(f"What do analysts say about {name}?")
        if "guid" not in asked:
            candidates.append(f"What did {name} guide to for next quarter?")
        if "expectation" not in asked and "beat" not in asked:
            candidates.append(f"Did {name} beat expectations last quarter?")
        if len(focus) > 1:
            candidates.insert(0, f"Compare {name} and {company_name(focus[1])}'s latest results")
        out = candidates
    else:
        out = ["What is my portfolio worth?",
               "Is anything wrong with my records?",
               "How concentrated is my portfolio?"]
    return [q for q in out if q.strip().lower() != asked][:limit]


_UNSUPPORTED = re.compile(
    r"\b(nothing (is )?on file|no [\w\s,-]{0,60}? on file|not (in|covered by) your "
    r"(documents|files)|isn'?t in your (documents|files)|(there is|there's) no "
    r"[\w\s-]{0,40}?(in your documents|on file)|can'?t (answer|give you)[^.]{0,60}"
    r"without inventing|your (documents|files) (have|has|contain) no)\b",
    re.IGNORECASE)


def is_unsupported(answer: str) -> bool:
    """True when the answer opens by saying the documents don't cover it.

    Only the opening is checked: a caveat deep in a full answer ("no Q4 release
    on file") qualifies one figure, it does not mean the question went unanswered.
    """
    opening = re.split(r"\n\s*\n", (answer or "").strip(), maxsplit=1)[0][:400]
    return bool(_UNSUPPORTED.search(opening))


def statement_file(custodian: str, catalog: list[dict]) -> str | None:
    """The statement PDF for a custodian, matched on the firm's first word."""
    first = _fold((custodian or "").split()[0]) if custodian else ""
    for d in catalog:
        if d.get("doc_type") == "custodian_statement" and first and first in _fold(d["filename"]):
            return d["filename"]
    return None


def source_cards(citations: list[dict], catalog: list[dict],
                 filename_by_document: dict[int, str] | None = None) -> list[dict]:
    """Citations as the investor should read them.

    The holdings tool cites one statement line per position, so a portfolio
    answer arrives with nine copies of the same statement. They collapse into
    one card per statement that lists the positions it supports.
    """
    by_file = {d["filename"]: d for d in catalog}
    cards: list[dict] = []
    statements: dict[str, dict] = {}
    seen = set()
    for c in citations:
        label = c.get("file") or "unknown source"
        if label.endswith(" statement") and label not in by_file:
            custodian = label[: -len(" statement")]
            card = statements.get(label)
            if card is None:
                card = statements[label] = {
                    "title": f"{custodian} — custodian statement",
                    "file": statement_file(custodian, catalog), "date": c.get("date"),
                    "kind": "Custodian statement", "pages": None, "positions": [],
                    "from_ocr": False, "other_versions": None}
                cards.append(card)
            detail = (c.get("detail") or "").replace(" position", "")
            if detail and detail not in card["positions"]:
                card["positions"].append(detail)
            continue
        m = re.fullmatch(r"document (\d+)", label)
        if m and filename_by_document:
            label = filename_by_document.get(int(m.group(1)), label)
        key = (label, c.get("pages"))
        if key in seen:
            continue
        seen.add(key)
        meta = by_file.get(label, {})
        cards.append({
            "title": meta.get("title") or label, "file": label if meta else None,
            "date": c.get("date") or meta.get("published_date"),
            "kind": tier_label(c.get("tier") or meta.get("tier")) if (c.get("tier") or meta) else "Record",
            "pages": c.get("pages"), "positions": [],
            "from_ocr": bool(c.get("from_ocr")),
            "other_versions": c.get("other_versions"),
        })
    return cards


def page_numbers(pages: str | None) -> list[int]:
    """'3-5' -> [3, 4, 5]; None -> []."""
    if not pages:
        return []
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", str(pages))
    if not m:
        return []
    start, end = int(m.group(1)), int(m.group(2) or m.group(1))
    return list(range(start, max(start, end) + 1))


def friendly_error(exc: BaseException) -> tuple[str, str]:
    """(what happened, what to do), in words for the investor."""
    name = type(exc).__name__
    text = str(exc)
    if "credit balance" in text.lower():
        return ("The AI service account has run out of credit.",
                "Whoever manages the Anthropic account needs to add credit under "
                "Plans & Billing. Your documents and portfolio pages still work.")
    if name == "KeyError" and "AIEHA_API_KEY" in text or name == "AuthenticationError":
        return ("The assistant isn't connected to its AI service.",
                "The API key is missing or was rejected. Whoever set up the app "
                "should check AIEHA_API_KEY, then restart it.")
    if name in ("RateLimitError", "OverloadedError") or "overloaded" in text.lower():
        return ("The AI service is busy right now.",
                "Wait a minute and try again. Nothing was lost.")
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "TimeoutError"):
        return ("The assistant couldn't reach its AI service.",
                "Check the internet connection, then try again.")
    return ("Something went wrong while answering.",
            "Try again. If it keeps happening, rephrase the question or ask it in smaller parts.")


# ---------------------------------------------------------------------------
# Document catalog
# ---------------------------------------------------------------------------

def workbook_role(filename: str) -> str:
    f = _fold(filename)
    for key, role in (("holdings", WORKBOOK_ROLES["holdings"]),
                      ("price", WORKBOOK_ROLES["prices"]),
                      ("fx", WORKBOOK_ROLES["fx"]),
                      ("consensus", WORKBOOK_ROLES["consensus"])):
        if key in f:
            return role
    return "Record file"


def build_catalog(folder: str | Path) -> list[dict]:
    """Every PDF on file with what the classifier made of it.

    Uses the same extraction cache and classifier as the search index, so the
    catalog and the citations always agree about a document.
    """
    from app.ingest.cache import extract_cached
    from app.ingest.classify import classify_cached

    catalog, first_by_hash = [], {}
    for path in sorted(Path(folder).glob("*.pdf")):
        extracted = extract_cached(path)
        pages = extracted["pages"]
        head = "\n\n".join(p["text"] for p in pages[:2])
        digest = extracted["content_hash"]
        meta = classify_cached(head, path.name, digest, use_llm=False)
        remember_company(meta.ticker, meta.company)
        # The classifier's title is the page's first line, which for a filing
        # is "UNITED STATES"; a title built from what it found reads better.
        if meta.ticker:
            title = f"{company_name(meta.ticker)} — {doc_type_label(meta.doc_type)}"
            if meta.fiscal_period:
                title += f", {meta.fiscal_period}"
            if meta.doc_type == "broker_note" and meta.publisher:
                title += f" ({meta.publisher})"
        else:
            title = path.stem
        catalog.append({
            "filename": path.name, "ticker": meta.ticker,
            "company": company_name(meta.ticker) if meta.ticker else "—",
            "doc_type": meta.doc_type, "type": doc_type_label(meta.doc_type),
            "tier": meta.source_tier, "reliability": tier_label(meta.source_tier),
            "published_date": meta.published_date or meta.as_of_date,
            "fiscal_period": meta.fiscal_period, "publisher": meta.publisher,
            "title": title,
            "pages": len(pages),
            "scanned_pages": sum(1 for p in pages if p.get("method") == "ocr"),
            "duplicate_of": first_by_hash.get(digest),
        })
        first_by_hash.setdefault(digest, path.name)
    return catalog


def document_pages(folder: str | Path, filename: str) -> list[dict]:
    """Extracted page text, from the same cache the index was built from."""
    from app.ingest.cache import extract_cached
    path = Path(folder) / filename
    if not path.exists():
        return []
    return extract_cached(path)["pages"]
