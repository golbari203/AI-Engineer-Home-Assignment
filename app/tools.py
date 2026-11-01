"""
The seven tools the agent can call.

Why fixed functions and not text-to-SQL
---------------------------------------
Text-to-SQL demos beautifully and fails quietly. On this schema a model would
have to know that holdings carry two rows per position from disagreeing
sources, that NVIDIA quantities need split adjustment before any arithmetic,
that prices are stored unadjusted on purpose, and that facts must be grouped by
basis or a GAAP figure gets compared with a non-GAAP one. Each of those is a
join or a filter the model would have to get right every time, with no way for
the system to tell when it did not.

Hand-written queries behind named functions cannot produce a wrong join. The
cost is expressiveness: a question none of the seven covers cannot be answered,
and the agent should say so rather than improvise. That trade is the right one
when the output is a number someone may act on.

What the tools guarantee
------------------------
* Portfolio figures are complete, because the query returns every matching row
  rather than the top k.
* Disagreements travel with the answer. get_holdings does not pick between the
  custodian and the family office book; it returns the custodian figure and
  attaches the disagreement.
* Every number carries its provenance: which document, which date, and for
  company figures which basis.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class Store:
    """Everything the tools read. Populated from MySQL in production, and
    from the loaders directly in tests and the offline demo."""
    statements: list[dict] = field(default_factory=list)
    positions: list[Any] = field(default_factory=list)      # internal book
    trades: list[Any] = field(default_factory=list)
    prices: list[dict] = field(default_factory=list)
    fx: list[dict] = field(default_factory=list)
    corporate_actions: list[dict] = field(default_factory=list)
    consensus: list[dict] = field(default_factory=list)
    price_targets: list[dict] = field(default_factory=list)
    broker_notes: list[dict] = field(default_factory=list)
    facts: list[Any] = field(default_factory=list)
    exceptions: list[Any] = field(default_factory=list)
    tier_by_document: dict[int, int] = field(default_factory=dict)
    retrieval_backend: Any = None
    embedder: Any = None

    def split_factor(self, ticker: str, when: dt.date | None) -> float:
        """Multiplier converting a quantity traded on `when` to today's basis."""
        factor = 1.0
        for a in self.corporate_actions:
            if a["ticker"] != ticker or not when:
                continue
            if when < a["effective_date"]:
                factor *= a["ratio_num"] / a["ratio_den"]
        return factor

    def rate(self, base: str, quote: str, when: dt.date | None) -> float | None:
        if base == quote:
            return 1.0
        rows = [r for r in self.fx
                if r["base_ccy"] == base and r["quote_ccy"] == quote
                and (when is None or r["rate_date"] <= when)]
        return max(rows, key=lambda r: r["rate_date"])["rate"] if rows else None


# ---------------------------------------------------------------------------
# 1. Holdings
# ---------------------------------------------------------------------------

def get_holdings(store: Store, ticker: str | None = None,
                 account: str | None = None) -> dict:
    """Positions from the custodian statements, with book disagreements attached.

    Precedence is applied here and nowhere else: the custodian is authoritative
    for what the client owns. The family office book is reported alongside when
    it differs, never substituted and never silently dropped.
    """
    ticker = ticker.upper() if ticker else None
    book = {(p.account_no, p.ticker): p for p in store.positions}
    rows, warnings = [], []

    for s in store.statements:
        if account and account not in (s["account_no"], s["custodian"]):
            continue
        as_of = s["as_of_date"]
        as_of = dt.date.fromisoformat(str(as_of)[:10]) if as_of else None
        fx = s.get("fx_rate") or 1.0
        for p in s["positions"]:
            if ticker and p["ticker"] != ticker:
                continue
            usd = (p["market_value"] / fx) if s["currency"] != "USD" else p["market_value"]
            row = {
                "ticker": p["ticker"], "custodian": s["custodian"],
                "account": s["account_no"], "entity": s.get("entity"),
                "quantity": p["quantity"], "price": p["price"],
                "market_value": p["market_value"], "currency": s["currency"],
                "market_value_usd": round(usd, 2),
                "cost_basis": p.get("cost_basis"),
                "as_of_date": str(as_of) if as_of else None,
                "source": "custodian statement",
            }
            b = book.get((s["account_no"], p["ticker"]))
            if b is None:
                row["book_disagreement"] = (
                    "this position does not appear in the family office "
                    "position file")
                warnings.append(f"{p['ticker']}: missing from internal records")
            elif abs(b.quantity - p["quantity"]) > 1e-6:
                row["book_disagreement"] = (
                    f"the family office position file records "
                    f"{b.quantity:,.0f} shares, not {p['quantity']:,.0f}")
                warnings.append(
                    f"{p['ticker']}: custodian {p['quantity']:,.0f} vs "
                    f"book {b.quantity:,.0f}")
            rows.append(row)

    securities = round(sum(r["market_value_usd"] for r in rows), 2)
    out: dict = {"positions": rows, "total_securities_usd": securities,
                 "accounts": sorted({r["account"] for r in rows})}

    # Cash is a quarter of this portfolio. A holdings tool that returned only
    # securities would answer "what is my portfolio worth?" with 75% of the
    # number. Reported separately so the two are never conflated.
    if not ticker:
        summaries, cash_usd = [], 0.0
        for st in store.statements:
            if account and account not in (st["account_no"], st["custodian"]):
                continue
            fx = st.get("fx_rate") or 1.0
            divisor = fx if st["currency"] != "USD" else 1.0
            cash = (st.get("cash") or 0.0) / divisor
            cash_usd += cash
            summaries.append({
                "account": st["account_no"], "custodian": st["custodian"],
                "currency": st["currency"],
                "cash_usd": round(cash, 2),
                "stated_total": st.get("total_value"),
                "stated_total_usd": round((st.get("total_value") or 0) / divisor, 2),
                "as_of_date": str(st["as_of_date"])[:10],
            })
        out["cash_usd"] = round(cash_usd, 2)
        out["total_portfolio_value_usd"] = round(securities + cash_usd, 2)
        out["account_summary"] = summaries

    if ticker:
        out["total_quantity"] = sum(r["quantity"] for r in rows)
        out["held_in_accounts"] = len(rows)
    if warnings:
        out["data_quality_warnings"] = warnings
    if not rows:
        out["note"] = ("no position found"
                       + (f" for {ticker}" if ticker else "")
                       + " on any custodian statement")
    return out


# ---------------------------------------------------------------------------
# 2. Transactions
# ---------------------------------------------------------------------------

def get_transactions(store: Store, ticker: str | None = None,
                     type: str | None = None, since: str | None = None,
                     split_adjusted: bool = True) -> dict:
    """Trade and income history, split-adjusted by default.

    Unadjusted, NVIDIA's 1,800-share purchase and 4,000-share sale net to a
    short position of 2,200 shares the client does not have.
    """
    ticker = ticker.upper() if ticker else None
    cutoff = dt.date.fromisoformat(since) if since else None
    rows = []
    for t in store.trades:
        if ticker and t.ticker != ticker:
            continue
        if type and t.type != type.upper():
            continue
        if cutoff and (t.trade_date or dt.date.min) < cutoff:
            continue
        # Only position-affecting lines are restated. Adjusting a dividend
        # line turns "1,800 shares at $0.01" into "18,000 at $0.001", a
        # per-share rate that never existed and that the investor would not
        # recognise from the statement in front of them.
        adjusts = split_adjusted and t.ticker and t.type in ("BUY", "SELL")
        factor = store.split_factor(t.ticker, t.trade_date) if adjusts else 1.0
        rows.append({
            "trade_date": str(t.trade_date) if t.trade_date else None,
            "ticker": t.ticker, "type": t.type,
            "quantity": (t.quantity * factor) if t.quantity else None,
            "price": (t.price / factor) if (t.price and factor != 1) else t.price,
            "amount": t.amount, "account": t.account_no,
            "split_adjusted": bool(factor != 1),
            "memo": t.memo,
        })
    rows.sort(key=lambda r: r["trade_date"] or "")

    out: dict = {"transactions": rows, "count": len(rows)}
    if ticker:
        net = sum((r["quantity"] or 0) * (1 if r["type"] == "BUY" else -1)
                  for r in rows if r["type"] in ("BUY", "SELL"))
        out["net_shares"] = net
        if any(r["split_adjusted"] for r in rows):
            raw = sum((t.quantity or 0) * (1 if t.type == "BUY" else -1)
                      for t in store.trades
                      if t.ticker == ticker and t.type in ("BUY", "SELL"))
            out["note"] = (
                f"quantities are adjusted for a corporate action; unadjusted "
                f"the same trades net to {raw:,.0f} shares, which is not the "
                f"position the client holds")
    return out


# ---------------------------------------------------------------------------
# 3. Facts
# ---------------------------------------------------------------------------

def get_facts(store: Store, ticker: str, metric: str | None = None,
              fiscal_period: str | None = None, segment: str | None = None
              ) -> dict:
    """Every recorded version of a figure, with conflicts made explicit.

    Returning only the winning version would hide that a figure is disputed,
    which on this corpus is often the most important thing about it.
    """
    from app.ingest.facts import find_conflicts, resolve

    ticker = ticker.upper()
    matches = [f for f in store.facts
               if f.ticker == ticker
               and (metric is None or f.metric == metric)
               and (fiscal_period is None or f.fiscal_period == fiscal_period)
               and (segment is None or f.segment == segment)]

    conflicts = find_conflicts(matches)
    disputed = {c.versions[0].key for c in conflicts}

    def render(f) -> dict:
        return {
            "metric": f.metric, "fiscal_period": f.fiscal_period,
            "value": f.value_text if f.value_text is not None else f.value_num,
            "unit": f.unit, "basis": f.basis, "segment": f.segment,
            "is_guidance": f.is_guidance, "is_restated": f.is_restated,
            "document_id": f.document_id, "page": f.page_no,
            "stated_on": str(f.stated_on) if f.stated_on else None,
            "confidence": f.confidence,
            "notes": f.notes or None,
        }

    out = {
        "ticker": ticker,
        "facts": [render(f) for f in matches if f.key not in disputed],
        "conflicts": [{
            "metric": c.metric, "fiscal_period": c.fiscal_period,
            "basis": c.basis, "segment": c.segment,
            "versions": [render(v) for v in c.versions],
            "authoritative": render(resolve(c, store.tier_by_document)),
            "summary": c.describe(),
        } for c in conflicts],
    }
    if not matches:
        out["note"] = (f"no recorded figures for {ticker}"
                       + (f" metric '{metric}'" if metric else "")
                       + (f" in {fiscal_period}" if fiscal_period else "")
                       + ". Try search_documents for the underlying text.")
    return out


# ---------------------------------------------------------------------------
# 4. Documents
# ---------------------------------------------------------------------------

def search_documents(store: Store, query: str, ticker: str | None = None,
                     k: int = 8) -> dict:
    """Hybrid retrieval over document text, tier-labelled and deduplicated."""
    from app.retrieval import format_for_prompt, search

    if store.retrieval_backend is None:
        return {"passages": [], "note": "no document index is loaded"}

    hits = search(query, store.retrieval_backend, store.embedder,
                  k=k, ticker=ticker.upper() if ticker else None)
    if not hits:
        return {"passages": [],
                "note": f"nothing on file matching '{query}'"
                        + (f" for {ticker}" if ticker else "")}
    return {
        "passages": format_for_prompt(hits),
        "sources": [{"file": h.record.filename,
                     "tier": h.record.source_tier,
                     "date": h.record.published_date,
                     "pages": f"{h.record.page_from}-{h.record.page_to}",
                     "from_ocr": h.record.is_ocr,
                     "other_versions": h.suppressed or None} for h in hits],
    }


# ---------------------------------------------------------------------------
# 5. Research
# ---------------------------------------------------------------------------

def get_research(store: Store, ticker: str) -> dict:
    """Ratings and targets, never averaged across houses.

    Each house uses its own three-point scale and says in its own disclosures
    that its ratings are not comparable with other firms'. Lilly carries a Buy
    at $890, an Overweight at $820 and a Market Perform at $705 within 33 days,
    on three different scales. A mean of those is a number with no meaning.
    """
    ticker = ticker.upper()
    notes = sorted((n for n in store.broker_notes if n["ticker"] == ticker),
                   key=lambda n: n["note_date"] or dt.date.min, reverse=True)
    target = next((t for t in store.price_targets if t["ticker"] == ticker), None)

    out = {
        "ticker": ticker,
        "notes_on_file": [{
            "date": str(n["note_date"]), "house": n["house"],
            "analyst": n["analyst"], "rating": n["rating"],
            "price_target": n["price_target"],
            "scale_note": f"{n['house']}'s own scale; the house states its "
                          f"ratings are not comparable with other firms'",
        } for n in notes],
        "street_aggregate": ({
            "consensus_price_target": target["consensus_pt"],
            "buy": target["buy"], "hold": target["hold"], "sell": target["sell"],
            "last_price": target["last_price"],
            "as_of": str(target["as_of_date"]),
            "note": "a vendor aggregate across the full analyst universe, not "
                    "a summary of the notes on file",
        } if target else None),
    }
    if notes:
        spread = [n["price_target"] for n in notes if n["price_target"]]
        if len(spread) > 1:
            out["target_range"] = {"low": min(spread), "high": max(spread),
                                   "note": "targets are not averaged because "
                                           "the scales differ"}
    else:
        out["note"] = f"no broker research on file for {ticker}"
    return out


# ---------------------------------------------------------------------------
# 6. Consensus
# ---------------------------------------------------------------------------

def get_consensus(store: Store, ticker: str,
                  fiscal_period: str | None = None) -> dict:
    """Estimates against actuals, always on the company's own EPS basis.

    The basis differs per company: GAAP for Apple and Amazon, non-GAAP for
    NVIDIA and Tesla, Adjusted for UnitedHealth, ex identified items for Exxon.
    Comparing an estimate with an actual on the wrong basis is the classic way
    to invent a beat or a miss.
    """
    ticker = ticker.upper()
    rows = [c for c in store.consensus
            if c["ticker"] == ticker
            and (fiscal_period is None or c["fiscal_period"] == fiscal_period)]
    rows.sort(key=lambda c: c["period_end"] or dt.date.min)

    def render(c: dict) -> dict:
        out = {
            "fiscal_period": c["fiscal_period"],
            "period_end": str(c["period_end"]) if c["period_end"] else None,
            "eps_basis": c["eps_basis"],
            "revenue_consensus_musd": c["revenue_consensus"],
            "eps_consensus": c["eps_consensus"],
            "estimates": c["n_estimates"],
            "revenue_actual_musd": c["revenue_actual"],
            "eps_actual": c["eps_actual"],
        }
        if c["eps_actual"] and c["eps_consensus"]:
            out["eps_surprise_pct"] = round(
                (c["eps_actual"] / c["eps_consensus"] - 1) * 100, 2)
        elif c["revenue_consensus"] and not c["revenue_actual"]:
            out["status"] = "not yet reported as of the file date"
        return out

    as_of = rows[0]["as_of_date"] if rows else None
    return {
        "ticker": ticker,
        "eps_basis": rows[0]["eps_basis"] if rows else None,
        "as_of_date": str(as_of) if as_of else None,
        "periods": [render(c) for c in rows],
        "note": (f"the consensus file is dated {as_of}; results reported after "
                 f"that date appear as actuals in the company documents but "
                 f"not here" if rows else f"no consensus data for {ticker}"),
    }


# ---------------------------------------------------------------------------
# 7. Data quality
# ---------------------------------------------------------------------------

def get_data_quality(store: Store, ticker: str | None = None) -> dict:
    """Reconciliation exceptions, so caveats are grounded rather than vague."""
    ticker = ticker.upper() if ticker else None
    rows = [e for e in store.exceptions
            if ticker is None or getattr(e, "ticker", None) == ticker]
    by_severity: dict[str, list] = defaultdict(list)
    for e in rows:
        by_severity[e.severity].append({
            "check": e.check_name, "ticker": e.ticker,
            "account": e.account_no, "description": e.description,
            "source_a": e.source_a, "source_b": e.source_b,
        })
    return {
        "exception_count": len(rows),
        "errors": by_severity.get("error", []),
        "warnings": by_severity.get("warning", []),
        "info": by_severity.get("info", []),
        "note": ("no reconciliation exceptions"
                 + (f" for {ticker}" if ticker else "")) if not rows else None,
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "get_holdings",
        "description": (
            "Positions the client actually owns, from the custodian statements. "
            "Use for any question about what is held, how much, what it is "
            "worth, or cost basis. Returns every matching position across all "
            "accounts, so totals are complete. Where the family office's own "
            "records disagree with the custodian, the disagreement is attached "
            "to the position. Never answer a holdings question from document "
            "text; use this."),
        "input_schema": {"type": "object", "properties": {
            "ticker": {"type": "string",
                       "description": "Symbol, e.g. COST. Omit for the whole portfolio."},
            "account": {"type": "string",
                        "description": "Account number or custodian name."}}},
    },
    {
        "name": "get_transactions",
        "description": (
            "Trade and income history: purchases, sales, dividends, fees and "
            "withholding tax. Quantities are adjusted for corporate actions by "
            "default. Use for cost basis questions, dividend income, or when a "
            "position was bought."),
        "input_schema": {"type": "object", "properties": {
            "ticker": {"type": "string"},
            "type": {"type": "string", "enum": ["BUY", "SELL", "DIV", "FEE", "TAX"]},
            "since": {"type": "string", "description": "YYYY-MM-DD"}}},
    },
    {
        "name": "get_facts",
        "description": (
            "Reported company figures extracted from filings, releases and "
            "transcripts: revenue, EPS, margins, guidance, segment results, "
            "officer names. Returns EVERY recorded version, and lists any that "
            "disagree along with which source is authoritative. Use this for "
            "'what was X in period Y', not search_documents."),
        "input_schema": {"type": "object", "properties": {
            "ticker": {"type": "string"},
            "metric": {"type": "string",
                       "description": "e.g. revenue, eps_diluted, cfo_name"},
            "fiscal_period": {"type": "string", "description": "e.g. 'Q4 2025'"},
            "segment": {"type": "string", "description": "e.g. 'AWS'"}},
            "required": ["ticker"]},
    },
    {
        "name": "search_documents",
        "description": (
            "Search the text of all documents on file. Use for explanations, "
            "reasons, risks, management commentary and analyst arguments: the "
            "questions that have no single correct value. Each passage is "
            "labelled with its source and reliability tier."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"},
            "ticker": {"type": "string", "description": "Restrict to one company."},
            "k": {"type": "integer", "description": "Passages to return, default 8."}},
            "required": ["query"]},
    },
    {
        "name": "get_research",
        "description": (
            "Broker ratings and price targets on file, plus the street "
            "aggregate. Ratings are reported per house with its own scale and "
            "are never averaged across houses."),
        "input_schema": {"type": "object", "properties": {
            "ticker": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_consensus",
        "description": (
            "Analyst estimates against reported actuals, on the company's own "
            "EPS basis. Use for beat/miss questions."),
        "input_schema": {"type": "object", "properties": {
            "ticker": {"type": "string"},
            "fiscal_period": {"type": "string"}}, "required": ["ticker"]},
    },
    {
        "name": "get_data_quality",
        "description": (
            "Known problems in the client's records: disagreements between the "
            "custodian statements and the family office books, missing "
            "positions, unexplained cost bases. Use when asked whether the "
            "records are correct, and to ground any caveat about a figure."),
        "input_schema": {"type": "object", "properties": {
            "ticker": {"type": "string"}}},
    },
]

DISPATCH = {
    "get_holdings": get_holdings,
    "get_transactions": get_transactions,
    "get_facts": get_facts,
    "search_documents": search_documents,
    "get_research": get_research,
    "get_consensus": get_consensus,
    "get_data_quality": get_data_quality,
}


def run_tool(store: Store, name: str, arguments: dict) -> dict:
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return fn(store, **arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
