"""
Reconciliation checks.

The portfolio has three sources that disagree: two custodian statements and the
family office's own blotter. Rather than picking one at load time and hoping,
every source is stored and these checks run after ingestion. They produce rows
in reconciliation_exceptions, which does two jobs:

  * answers "is anything wrong with my records?" directly, and
  * gives the agent an honest basis for caveats when a number is disputed.

Precedence rule used throughout: the custodian statement is authoritative for
quantities and valuations; the blotter is authoritative for trade history. Where
they disagree, the discrepancy is reported, never silently resolved.

Each check returns Exception records. Nothing here needs an LLM: these are
arithmetic facts about the data, and arithmetic is cheaper and more reliable
than a model for this job.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable

# Tolerances. Statements round to the cent; FX-converted lines drift a little more.
CENT = 0.02
VALUE_TOL = 1.0          # absolute currency units
RATIO_TOL = 0.0005       # 5bp relative


@dataclass
class Exception_:
    check_name: str
    severity: str                 # info | warning | error
    description: str
    ticker: str | None = None
    account_no: str | None = None
    source_a: str | None = None
    source_b: str | None = None
    value_a: float | None = None
    value_b: float | None = None
    delta: float | None = None

    def as_row(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Custodian statement internal arithmetic
# ---------------------------------------------------------------------------

def check_line_values(statement: dict) -> list[Exception_]:
    """quantity x price (x fx) should equal the stated market value.

    Catches transcription and OCR damage before anything downstream trusts it.
    """
    out = []
    fx = statement.get("fx_rate") or 1.0
    for p in statement["positions"]:
        if p.get("price") is None or p.get("market_value") is None:
            continue
        expected = p["quantity"] * p["price"] * (fx if statement.get("price_currency")
                                                 != statement["currency"] else 1.0)
        delta = p["market_value"] - expected
        if abs(delta) > max(VALUE_TOL, abs(expected) * RATIO_TOL):
            out.append(Exception_(
                "line_value", "error",
                f"{p['ticker']}: {p['quantity']:,.0f} x {p['price']:,.2f} "
                f"= {expected:,.2f} but the statement shows {p['market_value']:,.2f}",
                ticker=p["ticker"], account_no=statement["account_no"],
                value_a=expected, value_b=p["market_value"], delta=delta))
    return out


def check_statement_total(statement: dict) -> list[Exception_]:
    """Positions plus cash should equal the stated total."""
    total = sum(p["market_value"] for p in statement["positions"]
                if p.get("market_value") is not None)
    total += statement.get("cash", 0.0)
    stated = statement.get("total_value")
    if stated is None:
        return []
    delta = stated - total
    if abs(delta) > max(CENT, abs(stated) * RATIO_TOL):
        return [Exception_(
            "statement_total", "error",
            f"Positions plus cash come to {total:,.2f} but the statement total is "
            f"{stated:,.2f}", account_no=statement["account_no"],
            value_a=total, value_b=stated, delta=delta)]
    return []


def _average_cost_per_share(trades, account_no: str, ticker: str,
                            split_ratio: float, split_date) -> tuple[float, float] | None:
    """Walk the blotter chronologically and return (qty, average cost per share).

    Three things make the naive version wrong:
      * disposals reduce the cost pool, so NVIDIA's 4,000-share sale has to be
        applied or the remaining 14,000 shares look far too expensive;
      * pre-split purchases must be expanded before averaging;
      * commissions are booked on separate FEE lines, and both custodians state
        that cost basis is the average cost of open lots excluding commission,
        so FEE lines are deliberately left out.
    Amounts stay in the trade currency (USD); conversion happens in the caller.
    """
    rows = sorted((t for t in trades
                   if t.account_no == account_no and t.ticker == ticker
                   and t.type in ("BUY", "SELL") and t.quantity and t.amount),
                  key=lambda t: t.trade_date or dt.date.min)
    qty = 0.0
    pool = 0.0
    for t in rows:
        factor = split_ratio if (split_date and t.trade_date
                                 and t.trade_date < split_date) else 1.0
        q = t.quantity * factor
        if t.type == "BUY":
            qty += q
            pool += abs(t.amount)
        else:
            if qty <= 0:
                continue
            pool -= (pool / qty) * q      # average-cost disposal
            qty -= q
    return (qty, pool / qty) if qty > 0 else None


def check_cost_basis_against_trades(statement: dict, trades: Iterable,
                                    fx: list[dict] | None = None,
                                    actions: list[dict] | None = None) -> list[Exception_]:
    """Compare average cost per share, not total cost.

    Comparing totals conflates two different problems: a wrong share count and a
    wrong cost. Per-share isolates the cost question, so a quantity break is
    reported once by check_books_against_custodian instead of twice.

    For a foreign-currency statement the comparison also recovers the exchange
    rate implied by the stated cost basis and tests it against the rate on the
    trade date, because the statement claims cost is converted at the historical
    rate.
    """
    out: list[Exception_] = []
    ratios = {a["ticker"]: (a["ratio_num"] / a["ratio_den"], a["effective_date"])
              for a in (actions or [])}
    trades = list(trades)

    for p in statement["positions"]:
        ticker = p["ticker"]
        stated_total = p.get("cost_basis")
        if stated_total is None:
            continue
        ratio, split_date = ratios.get(ticker, (1.0, None))
        booked = _average_cost_per_share(trades, statement["account_no"], ticker,
                                         ratio, split_date)
        if booked is None:
            out.append(Exception_(
                "no_purchase_recorded", "warning",
                f"{ticker}: the statement reports a cost basis of {stated_total:,.2f} "
                f"{statement['currency']} but the blotter records no purchase for "
                f"this account",
                ticker=ticker, account_no=statement["account_no"],
                source_a="custodian statement", source_b="trade blotter"))
            continue

        _, booked_avg = booked                       # trade currency, USD
        stated_avg = stated_total / p["quantity"]    # statement currency

        if statement["currency"] == "USD":
            delta = stated_avg - booked_avg
            if abs(delta) > max(0.01, abs(booked_avg) * RATIO_TOL):
                out.append(Exception_(
                    "cost_basis_vs_trades", "warning",
                    f"{ticker}: the blotter implies an average cost of "
                    f"{booked_avg:,.4f} per share, the statement implies "
                    f"{stated_avg:,.4f}",
                    ticker=ticker, account_no=statement["account_no"],
                    source_a="trade blotter", source_b="custodian statement",
                    value_a=booked_avg, value_b=stated_avg, delta=delta))
            continue

        # Foreign-currency statement: recover the implied conversion rate.
        implied = stated_avg / booked_avg
        trade_dates = [t.trade_date for t in trades
                       if t.account_no == statement["account_no"]
                       and t.ticker == ticker and t.type == "BUY" and t.trade_date]
        expected = _rate_near(fx or [], min(trade_dates)) if trade_dates else None
        if expected and abs(implied - expected) > 0.01:
            out.append(Exception_(
                "cost_basis_fx_rate", "warning",
                f"{ticker}: the stated cost basis implies a USD/{statement['currency']} "
                f"rate of {implied:.4f}, but the rate around the purchase date "
                f"({min(trade_dates)}) was {expected:.4f}. The statement says cost is "
                f"converted at the historical rate on the purchase date",
                ticker=ticker, account_no=statement["account_no"],
                source_a="implied by statement", source_b="FX reference file",
                value_a=implied, value_b=expected, delta=implied - expected))
    return out


def _rate_near(fx: list[dict], when: dt.date) -> float | None:
    """Most recent USD->CHF rate on or before `when`."""
    candidates = [r for r in fx if r["rate_date"] <= when and r["quote_ccy"] == "CHF"]
    return max(candidates, key=lambda r: r["rate_date"])["rate"] if candidates else None


# ---------------------------------------------------------------------------
# 2. Custodian statements against the family office books
# ---------------------------------------------------------------------------

def check_books_against_custodian(statements: list[dict], positions) -> list[Exception_]:
    """Compare quantities, and flag positions missing from either side.

    Finds both known breaks: JPM 7,250 vs 7,520, and the Trust's 900 Costco
    shares absent from the Positions sheet entirely.
    """
    out: list[Exception_] = []
    book = {(p.account_no, p.ticker): p for p in positions}
    custodian: dict[tuple[str, str], dict] = {}
    for s in statements:
        for p in s["positions"]:
            custodian[(s["account_no"], p["ticker"])] = {**p, "account_no": s["account_no"]}

    for key, c in custodian.items():
        b = book.get(key)
        if b is None:
            out.append(Exception_(
                "missing_in_books", "error",
                f"{c['ticker']}: {c['quantity']:,.0f} shares appear on the "
                f"{key[0]} statement but not in the family office position file",
                ticker=c["ticker"], account_no=key[0],
                source_a="custodian statement", source_b="internal book",
                value_a=c["quantity"], value_b=0.0, delta=c["quantity"]))
        elif abs(b.quantity - c["quantity"]) > 1e-6:
            out.append(Exception_(
                "quantity_mismatch", "error",
                f"{c['ticker']}: the custodian reports {c['quantity']:,.0f} shares, "
                f"the internal book records {b.quantity:,.0f}",
                ticker=c["ticker"], account_no=key[0],
                source_a="custodian statement", source_b="internal book",
                value_a=c["quantity"], value_b=b.quantity,
                delta=b.quantity - c["quantity"]))

    for key, b in book.items():
        if key not in custodian:
            out.append(Exception_(
                "missing_in_custodian", "warning",
                f"{b.ticker}: the internal book records {b.quantity:,.0f} shares in "
                f"account {key[0]}, with no matching line on any statement",
                ticker=b.ticker, account_no=key[0],
                value_a=0.0, value_b=b.quantity, delta=-b.quantity))
    return out


# ---------------------------------------------------------------------------
# 3. Internal consistency of the blotter itself
# ---------------------------------------------------------------------------

def check_dividends_imply_quantity(trades, positions) -> list[Exception_]:
    """A cash dividend divided by its per-share rate implies a share count.

    The blotter's own dividend lines disagree with its own position file for
    JPMorgan, which is how we know 7,520 is the typo and not 7,250.
    """
    out = []
    latest: dict[tuple[str, str], tuple[dt.date, float]] = {}
    for t in trades:
        if t.type != "DIV" or not (t.price and t.amount and t.trade_date):
            continue
        implied = t.amount / t.price
        key = (t.account_no, t.ticker)
        if key not in latest or t.trade_date > latest[key][0]:
            latest[key] = (t.trade_date, implied)

    book = {(p.account_no, p.ticker): p for p in positions}
    for (acct, ticker), (date, implied) in latest.items():
        b = book.get((acct, ticker))
        if b is None or abs(implied - b.quantity) <= 1e-6:
            continue
        out.append(Exception_(
            "dividend_implies_different_quantity", "error",
            f"{ticker}: the dividend paid on {date} implies {implied:,.0f} shares, "
            f"but the position file records {b.quantity:,.0f}",
            ticker=ticker, account_no=acct,
            source_a="blotter dividend line", source_b="blotter position file",
            value_a=implied, value_b=b.quantity, delta=b.quantity - implied))
    return out


def check_split_adjusted_quantities(trades, statements, actions) -> list[Exception_]:
    """Net the trade history, expanding pre-split quantities, and compare.

    Without the expansion NVIDIA nets to 1,800 - 4,000 = -2,200 shares, i.e. a
    short position the family office does not have.
    """
    out = []
    ratios = {a["ticker"]: (a["ratio_num"] / a["ratio_den"], a["effective_date"])
              for a in actions}

    net: dict[tuple[str, str], float] = defaultdict(float)
    net_raw: dict[tuple[str, str], float] = defaultdict(float)
    for t in trades:
        if t.type not in ("BUY", "SELL") or t.quantity is None or not t.ticker:
            continue
        sign = 1 if t.type == "BUY" else -1
        factor = 1.0
        if t.ticker in ratios:
            ratio, eff = ratios[t.ticker]
            if t.trade_date and t.trade_date < eff:
                factor = ratio
        net[(t.account_no, t.ticker)] += sign * t.quantity * factor
        net_raw[(t.account_no, t.ticker)] += sign * t.quantity

    custodian = {(s["account_no"], p["ticker"]): p["quantity"]
                 for s in statements for p in s["positions"]}

    for key, qty in net.items():
        actual = custodian.get(key)
        if actual is None:
            continue
        if abs(qty - actual) > 1e-6:
            out.append(Exception_(
                "trade_history_does_not_net_to_position", "warning",
                f"{key[1]}: split-adjusted trades net to {qty:,.0f} shares but the "
                f"custodian reports {actual:,.0f}",
                ticker=key[1], account_no=key[0],
                value_a=qty, value_b=actual, delta=qty - actual))
        elif abs(net_raw[key] - actual) > 1e-6:
            out.append(Exception_(
                "split_adjustment_required", "info",
                f"{key[1]}: trade quantities only net to the reported position "
                f"after applying the {ratios[key[1]][0]:.0f}-for-1 split of "
                f"{ratios[key[1]][1]}; unadjusted they net to {net_raw[key]:,.0f}",
                ticker=key[1], account_no=key[0],
                value_a=net_raw[key], value_b=actual))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(statements: list[dict], workbooks) -> list[Exception_]:
    exceptions: list[Exception_] = []
    for s in statements:
        exceptions += check_line_values(s)
        exceptions += check_statement_total(s)
        exceptions += check_cost_basis_against_trades(
            s, workbooks.trades, workbooks.fx, workbooks.corporate_actions)
    exceptions += check_books_against_custodian(statements, workbooks.positions)
    exceptions += check_dividends_imply_quantity(workbooks.trades, workbooks.positions)
    exceptions += check_split_adjusted_quantities(
        workbooks.trades, statements, workbooks.corporate_actions)
    return exceptions
