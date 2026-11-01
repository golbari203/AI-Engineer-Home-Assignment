"""
Custodian statement extraction.

Why an LLM instead of a parser per bank
---------------------------------------
There are two custodians today and they share nothing. Northgate prints a US
brokerage statement in USD with a holdings table, an activity table and a trade
history. Banque Leman prints a bilingual French/English portfolio statement in
CHF, with prices in USD, an exchange rate column, apostrophe thousands
separators, dd.mm.yyyy dates and a negative CHF cash balance. A third custodian
would share nothing with either. A family office will not commission a parser
each time it opens an account, so extraction is driven by a schema the model
fills in, and correctness is enforced afterwards by arithmetic.

The division of labour
----------------------
This module answers one question: **does the document agree with itself?**
Line values must equal quantity x price x fx. Positions plus cash must equal
the stated total. The account roll-forward must close. If those fail, the
extraction is wrong and must not be stored.

Whether the document agrees with *other* documents is reconcile.py's job.

That distinction matters for a specific case. The Banque Leman statement records
a Costco purchase of CHF 681,822 plus CHF 1,705 brokerage, and states a cost
basis of CHF 679,590 for the same position -- roughly CHF 3,900 unexplained,
inside one document. The extractor must reproduce both numbers exactly. It is
not the extractor's business to decide which is right, and an extractor that
"fixed" the inconsistency would destroy the finding.

So: structural failures are rejected and retried; data discrepancies are
preserved and flagged.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field

from app.ingest.workbooks import to_date, to_number

log = logging.getLogger(__name__)

# Statements are stated to the cent, so tolerances here are absolute, not
# relative. A 5bp relative tolerance sounds reasonable and is not: on a
# $17.2m account total it would accept an error of $8,597.
CENT = 0.05
FLOAT_TOL = 1e-6          # only to absorb binary floating-point drift


def _tol(expected: float, lines: int = 1) -> float:
    """Cent-level tolerance, widened by one cent per line for rounding."""
    return max(CENT + 0.01 * (lines - 1), abs(expected) * FLOAT_TOL)

# Normalising transaction types across languages and houses. "Achat" and
# "Bought" are the same event; "Courtage", "Brokerage commission" and
# "Frais de garde" are all costs that must not be mistaken for trades.
TYPE_SYNONYMS = {
    "BUY":  r"\bbought\b|\bbuy\b|\bachat\b|\bpurchase\b",
    "SELL": r"\bsold\b|\bsell\b|\bvente\b",
    "DIV":  r"\bdividend\b|\bdividende\b",
    "TAX":  r"withholding|imp[oô]t\s+[aà]\s+la\s+source|\btax\b",
    "FEE":  r"brokerage|commission|courtage|frais\s+de\s+garde|custody\s+fee",
}


@dataclass
class StatementIssue:
    code: str
    severity: str            # 'structural' rejects the extraction; 'data' is flagged
    description: str
    value_a: float | None = None
    value_b: float | None = None
    delta: float | None = None


@dataclass
class ExtractedStatement:
    data: dict = field(default_factory=dict)
    issues: list[StatementIssue] = field(default_factory=list)

    @property
    def is_storable(self) -> bool:
        return not any(i.severity == "structural" for i in self.issues)

    @property
    def data_flags(self) -> list[StatementIssue]:
        return [i for i in self.issues if i.severity == "data"]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You transcribe custodian account statements into JSON.

You are transcribing, not interpreting. Copy every figure exactly as printed,
including any that look inconsistent with each other. Do not correct, round,
reconcile or recompute anything. If a statement contradicts itself, reproduce
the contradiction; it will be checked separately.

Return ONLY a JSON object with these keys:

  custodian          institution name as printed
  account_no         account or relation number
  entity             the account holder
  as_of_date         valuation date, YYYY-MM-DD
  period_start       YYYY-MM-DD or null
  currency           ISO code the statement totals are denominated in
  price_currency     ISO code the position prices are quoted in
  fx_rate            price_currency -> currency rate printed on the statement,
                     or null when they are the same
  cash               total cash in `currency`, negative balances included
  total_value        the stated total, in `currency`
  beginning_value    opening value for the period, or null
  income             dividends and interest for the period, or null
  market_change      change in market value for the period, or null
  positions          list of:
      ticker         listing symbol
      description    as printed
      quantity       number of units
      price          price per unit in `price_currency`
      market_value   value in `currency`
      cost_basis     in `currency`, or null
  transactions       list of, for every movement line:
      trade_date     YYYY-MM-DD or null
      settle_date    YYYY-MM-DD or null
      type           one of BUY, SELL, DIV, TAX, FEE, OTHER
      ticker         symbol, or null for account-level charges
      quantity       or null
      price          or null
      amount         signed: negative is cash out
      currency       ISO code of `amount`
      fx_rate        rate printed on that line, or null
      description    as printed

Rules:
- Numbers may be printed as 1'160'069.55, 1,160,069.55 or (1 234.56) for a
  negative. Return plain numbers: 1160069.55 and -1234.56.
- Dates may be printed as 30.06.2026 or June 30, 2026. Return YYYY-MM-DD.
- Include every transaction line, including fees, custody charges and
  withholding tax, each as its own entry.
- If a value is not printed, return null. Never infer one."""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_statement(text: str, client=None,
                      model: str = "claude-sonnet-5-5") -> dict:
    if client is None:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["AIEHA_API_KEY"])

    resp = client.messages.create(
        model=model, max_tokens=8000, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text[:30000]}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    return json.loads(raw)


def normalise_type(raw: str | None) -> str:
    if not raw:
        return "OTHER"
    s = str(raw).lower()
    if s.upper() in TYPE_SYNONYMS:
        return s.upper()
    # Order matters: a dividend line and its withholding line both mention the
    # dividend, so TAX and FEE are tested before DIV.
    for key in ("TAX", "FEE", "BUY", "SELL", "DIV"):
        if re.search(TYPE_SYNONYMS[key], s):
            return key
    return "OTHER"


def normalise(raw: dict) -> dict:
    """Coerce types and defend against the formats the model may echo back."""
    out = dict(raw)
    for key in ("as_of_date", "period_start"):
        out[key] = to_date(raw.get(key))
    for key in ("cash", "total_value", "fx_rate", "beginning_value",
                "income", "market_change"):
        out[key] = to_number(raw.get(key))

    out["positions"] = [{
        "ticker": (p.get("ticker") or "").strip().upper() or None,
        "description": (p.get("description") or "").strip(),
        "quantity": to_number(p.get("quantity")),
        "price": to_number(p.get("price")),
        "market_value": to_number(p.get("market_value")),
        "cost_basis": to_number(p.get("cost_basis")),
    } for p in raw.get("positions") or []]

    out["transactions"] = [{
        "trade_date": to_date(t.get("trade_date")),
        "settle_date": to_date(t.get("settle_date")),
        "type": normalise_type(t.get("type") or t.get("description")),
        "ticker": (t.get("ticker") or "").strip().upper() or None,
        "quantity": to_number(t.get("quantity")),
        "price": to_number(t.get("price")),
        "amount": to_number(t.get("amount")),
        "currency": (t.get("currency") or out.get("currency") or "").upper() or None,
        "fx_rate": to_number(t.get("fx_rate")),
        "description": (t.get("description") or "").strip(),
    } for t in raw.get("transactions") or []]
    return out


# ---------------------------------------------------------------------------
# Document-internal validation
# ---------------------------------------------------------------------------

REQUIRED = ("custodian", "account_no", "as_of_date", "currency",
            "total_value", "positions")


def validate(s: dict) -> list[StatementIssue]:
    issues: list[StatementIssue] = []

    for key in REQUIRED:
        if s.get(key) in (None, "", []):
            issues.append(StatementIssue(
                "missing_field", "structural", f"required field '{key}' is absent"))
    if issues:
        return issues

    same_ccy = (s.get("price_currency") or s["currency"]) == s["currency"]
    fx = 1.0 if same_ccy else (s.get("fx_rate") or 0)
    if not same_ccy and not fx:
        issues.append(StatementIssue(
            "missing_fx", "structural",
            f"prices are in {s.get('price_currency')} but totals in "
            f"{s['currency']} with no exchange rate"))
        return issues

    # 1. Line values. Catches both a transcription slip and OCR damage.
    for p in s["positions"]:
        if None in (p["quantity"], p["price"], p["market_value"]):
            continue
        expected = p["quantity"] * p["price"] * fx
        delta = p["market_value"] - expected
        if abs(delta) > _tol(expected):
            issues.append(StatementIssue(
                "line_value", "structural",
                f"{p['ticker']}: {p['quantity']:,.0f} x {p['price']:,.2f} "
                f"{'x ' + format(fx, '.4f') + ' ' if not same_ccy else ''}"
                f"= {expected:,.2f} but the line shows {p['market_value']:,.2f}",
                expected, p["market_value"], delta))

    # 2. Positions plus cash against the stated total.
    subtotal = sum(p["market_value"] for p in s["positions"]
                   if p["market_value"] is not None) + (s.get("cash") or 0.0)
    delta = s["total_value"] - subtotal
    if abs(delta) > _tol(s["total_value"], len(s["positions"]) + 1):
        issues.append(StatementIssue(
            "statement_total", "structural",
            f"positions plus cash come to {subtotal:,.2f} but the stated total "
            f"is {s['total_value']:,.2f}", subtotal, s["total_value"], delta))

    # 3. Account roll-forward, when the statement prints the components.
    if None not in (s.get("beginning_value"), s.get("income"),
                    s.get("market_change")):
        rolled = (s["beginning_value"] + s["income"] + s["market_change"]
                  + (s.get("net_flows") or 0.0))
        delta = s["total_value"] - rolled
        if abs(delta) > _tol(s["total_value"], 4):
            issues.append(StatementIssue(
                "roll_forward", "structural",
                f"opening {s['beginning_value']:,.2f} plus income and market "
                f"change gives {rolled:,.2f}, but the closing value is "
                f"{s['total_value']:,.2f}", rolled, s["total_value"], delta))

    issues += check_cost_basis_internal(s)
    return issues


def check_cost_basis_internal(s: dict) -> list[StatementIssue]:
    """Does the statement's own transaction history support its own cost basis?

    Severity is 'data', not 'structural'. The Banque Leman Costco line is a real
    discrepancy printed on the page: purchase CHF 681,822 plus CHF 1,705
    brokerage against a stated basis of CHF 679,590. Rejecting the extraction
    would hide it; the right behaviour is to store it faithfully and flag it.
    """
    issues = []
    ccy = s["currency"]

    purchases: dict[str, float] = {}
    fees: dict[str, float] = {}
    for t in s["transactions"]:
        if not t["ticker"] or t["amount"] is None:
            continue
        rate = t["fx_rate"] if (t["currency"] != ccy and t["fx_rate"]) else 1.0
        value = abs(t["amount"]) * rate
        if t["type"] == "BUY":
            purchases[t["ticker"]] = purchases.get(t["ticker"], 0.0) + value
        elif t["type"] == "FEE":
            fees[t["ticker"]] = fees.get(t["ticker"], 0.0) + value

    for p in s["positions"]:
        booked = purchases.get(p["ticker"])
        if not booked or p["cost_basis"] is None:
            continue
        fee = fees.get(p["ticker"], 0.0)
        # Accept either convention: cost basis with or without commission.
        for total, label in ((booked, "excluding commission"),
                             (booked + fee, "including commission")):
            if abs(p["cost_basis"] - total) <= _tol(total, 2):
                break
        else:
            issues.append(StatementIssue(
                "cost_basis_vs_movements", "data",
                f"{p['ticker']}: the purchase recorded on this statement is "
                f"{booked:,.2f} {ccy}"
                + (f" plus {fee:,.2f} brokerage" if fee else "")
                + f", but the position line states a cost basis of "
                  f"{p['cost_basis']:,.2f}",
                booked + fee, p["cost_basis"], p["cost_basis"] - booked - fee))
    return issues


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_statement(text: str, client=None, retries: int = 1) -> ExtractedStatement:
    """Extract, normalise and validate. Structural failures trigger a retry.

    A structural failure means the transcription disagrees with the page's own
    arithmetic, so re-asking is reasonable. A data flag means the page
    disagrees with itself, so re-asking would change nothing and the result is
    kept as-is.
    """
    last = ExtractedStatement()
    for attempt in range(retries + 1):
        try:
            raw = extract_statement(text, client=client)
        except (json.JSONDecodeError, ValueError) as exc:
            last = ExtractedStatement(issues=[StatementIssue(
                "unparseable_json", "structural", str(exc))])
            continue

        data = normalise(raw)
        last = ExtractedStatement(data=data, issues=validate(data))
        if last.is_storable:
            return last
        log.warning("statement extraction attempt %d failed validation: %s",
                    attempt + 1, [i.code for i in last.issues])
    return last
