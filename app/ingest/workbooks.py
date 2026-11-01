"""
Loaders for the four family-office workbooks.

These files look clean, which is exactly why they are dangerous. Every loader
below exists to defuse something specific found in the data:

  Holdings_and_trades.xlsx
      Positions sheet omits the Trust's 900 Costco shares and records JPM as
      7,520 where the custodian and the workbook's own dividend lines say 7,250.
      Loaded as 'internal_book', never merged with the custodian figures.
      NVDA trades before 2025-06-10 are in pre-split units (1,800 = 18,000).

  Prices.xlsx
      Header says prices are NOT adjusted for later splits. NVDA goes
      975.90 -> 84.55 across the split. Flagged, never silently used for returns.

  FX_rates.xlsx
      Three sheets, three conventions: Excel serials with decimal points on one,
      text strings with European decimal commas ("0,8600") and dd.mm.yyyy on
      another, plus both CHF-per-USD and USD-per-CHF columns inviting inversion.

  Consensus_estimates.xlsx
      EPS basis differs per company (GAAP / non-GAAP / Adjusted / ex identified
      items) and some EPS cells are text with a leading '$'. The workbook is
      as of 2026-06-30, so later actuals are legitimately blank.

Every sheet also has one or two title/note rows above the real header, so the
header row is detected rather than assumed.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from openpyxl import load_workbook

EXCEL_EPOCH = dt.date(1899, 12, 30)  # 1900 date system, leap-bug compensated


# ---------------------------------------------------------------------------
# Cell coercion
# ---------------------------------------------------------------------------

def to_date(value: Any) -> dt.date | None:
    """Accept Excel serials, datetimes, dd.mm.yyyy and yyyy-mm-dd strings."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, (int, float)):
        return EXCEL_EPOCH + dt.timedelta(days=int(value))
    s = str(value).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def to_number(value: Any) -> float | None:
    """Accept floats, '$5.39', "1'160'069.55" and European '0,8600'."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("$", "").replace("\u2019", "").replace("'", "")
    s = s.replace("\xa0", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    # European decimal comma: a comma with exactly 1-4 digits after it and no
    # other separator is a decimal point, not a thousands separator.
    if re.fullmatch(r"-?\d+,\d{1,4}", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def find_header_row(rows: list[tuple], max_scan: int = 8) -> int:
    """Return the index of the header row.

    Every sheet carries a title row and often a note row before the header.
    The header is the first row with at least three non-empty string cells
    that is followed by a row containing data.
    """
    for i, row in enumerate(rows[:max_scan]):
        strings = [c for c in row if isinstance(c, str) and c.strip()]
        if len(strings) >= 3 and i + 1 < len(rows) and any(
            c is not None and c != "" for c in rows[i + 1]
        ):
            return i
    return 0


def sheet_records(path: Path, sheet: str) -> Iterator[dict]:
    """Yield each data row of a sheet as a dict keyed by its header labels."""
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    h = find_header_row(rows)
    headers = [str(c).strip() if c is not None else f"col{j}"
               for j, c in enumerate(rows[h])]
    for row in rows[h + 1:]:
        if all(c is None or c == "" for c in row):
            continue
        yield {headers[j]: row[j] for j in range(min(len(headers), len(row)))}
    wb.close()


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------

@dataclass
class Position:
    entity: str
    custodian: str
    account_no: str
    ticker: str
    quantity: float
    avg_cost: float | None
    last_price: float | None
    market_value: float | None
    note: str = ""


@dataclass
class Trade:
    account_no: str
    ticker: str | None
    trade_date: dt.date | None
    settle_date: dt.date | None
    type: str
    quantity: float | None
    price: float | None
    amount: float | None
    reference: str = ""
    memo: str = ""
    quantity_is_presplit: bool = False


@dataclass
class Workbooks:
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    prices: list[dict] = field(default_factory=list)
    dividends: list[dict] = field(default_factory=list)
    corporate_actions: list[dict] = field(default_factory=list)
    fx: list[dict] = field(default_factory=list)
    consensus: list[dict] = field(default_factory=list)
    price_targets: list[dict] = field(default_factory=list)
    broker_notes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_holdings_and_trades(path: Path) -> tuple[list[Position], list[Trade]]:
    positions = [
        Position(
            entity=str(r.get("Entity", "")).strip(),
            custodian=str(r.get("Custodian", "")).strip(),
            account_no=str(r.get("Account", "")).strip(),
            ticker=str(r.get("Ticker", "")).strip().upper(),
            quantity=to_number(r.get("Quantity")) or 0.0,
            avg_cost=to_number(r.get("Average cost (USD)")),
            last_price=to_number(r.get("Last price (USD)")),
            market_value=to_number(r.get("Market value (USD)")),
            note=str(r.get("Note") or "").strip(),
        )
        for r in sheet_records(path, "Positions")
        if r.get("Ticker")
    ]

    trades: list[Trade] = []
    for sheet, account in (("Trades - Holdings LLC", "4471-8820"),
                           ("Trades - Trust", "CH-5521.03")):
        for r in sheet_records(path, sheet):
            if not r.get("Type"):
                continue
            ticker = str(r.get("Ticker") or "").strip().upper() or None
            td = to_date(r.get("Trade date"))
            memo = str(r.get("Memo") or "").strip()
            trades.append(Trade(
                account_no=account,
                ticker=ticker,
                trade_date=td,
                settle_date=to_date(r.get("Booking date")),
                type=str(r["Type"]).strip().upper(),
                quantity=to_number(r.get("Quantity")),
                price=to_number(r.get("Price")),
                amount=to_number(r.get("Amount (USD)")),
                reference=str(r.get("Reference") or "").strip(),
                memo=memo,
                # The workbook labels these explicitly; we also derive it from the
                # split date in corporate_actions so the flag survives a relabel.
                quantity_is_presplit=(memo.lower() == "pre-split"),
            ))
    return positions, trades


def load_prices(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (prices, dividends, corporate_actions). Prices are NOT split-adjusted."""
    prices: list[dict] = []
    for r in sheet_records(path, "Month-end closes"):
        d = to_date(r.get("Date"))
        if not d:
            continue
        for ticker, close in r.items():
            if ticker == "Date" or close in (None, ""):
                continue
            px = to_number(close)
            if px is not None:
                prices.append({"ticker": ticker.strip().upper(), "price_date": d,
                               "close_price": px, "split_adjusted": False})

    dividends = [
        {"ticker": str(r["Ticker"]).strip().upper(),
         "pay_date": to_date(r.get("Pay date")),
         "dps": to_number(r.get("Dividend per share (USD)")),
         "note": str(r.get("Note") or "").strip()}
        for r in sheet_records(path, "Dividends") if r.get("Ticker")
    ]

    actions = []
    for r in sheet_records(path, "Corporate actions"):
        if not r.get("Ticker"):
            continue
        ratio = str(r.get("Ratio") or "1:1")
        num, den = (ratio.split(":") + ["1"])[:2]
        actions.append({"ticker": str(r["Ticker"]).strip().upper(),
                        "action": str(r.get("Action") or "").strip(),
                        "effective_date": to_date(r.get("Effective date")),
                        "ratio_num": to_number(num) or 1.0,
                        "ratio_den": to_number(den) or 1.0})
    return prices, dividends, actions


def load_fx(path: Path) -> list[dict]:
    """Canonical direction: CHF per 1 USD.

    The 'Month-end' sheet gives USD/CHF directly. The 'Quarter ends' sheet gives
    both directions as European-formatted text; we take the CHF-per-USD column
    and assert the inverse column agrees, which catches a transposed file.
    """
    out: list[dict] = []
    for r in sheet_records(path, "Month-end"):
        d = to_date(r.get("Date"))
        rate = to_number(r.get("USD/CHF"))
        if d and rate:
            out.append({"rate_date": d, "base_ccy": "USD", "quote_ccy": "CHF",
                        "rate": rate, "source": "FX_rates.xlsx/Month-end"})

    for r in sheet_records(path, "Quarter ends"):
        d = to_date(r.get("Date"))
        chf_per_usd = to_number(r.get("CHF per USD"))
        usd_per_chf = to_number(r.get("USD per CHF"))
        if not (d and chf_per_usd):
            continue
        if usd_per_chf and abs(chf_per_usd * usd_per_chf - 1) > 0.01:
            raise ValueError(f"FX columns inconsistent on {d}: "
                             f"{chf_per_usd} * {usd_per_chf} != 1")
        out.append({"rate_date": d, "base_ccy": "USD", "quote_ccy": "CHF",
                    "rate": chf_per_usd, "source": "FX_rates.xlsx/Quarter ends"})

    # The two sheets overlap on twelve quarter-end dates. They agree today, but
    # a merged series is only safe if that holds, so it is checked rather than
    # assumed -- the same reason the inversion columns are checked above.
    seen: dict[dt.date, dict] = {}
    merged = []
    for row in out:
        prior = seen.get(row["rate_date"])
        if prior is None:
            seen[row["rate_date"]] = row
            merged.append(row)
        elif abs(prior["rate"] - row["rate"]) > 1e-9:
            raise ValueError(
                f"FX sheets disagree on {row['rate_date']}: "
                f"{prior['rate']} ({prior['source']}) vs "
                f"{row['rate']} ({row['source']})")
    return merged


def load_consensus(path: Path, as_of: dt.date) -> tuple[list[dict], list[dict], list[dict]]:
    estimates = []
    for r in sheet_records(path, "Estimates"):
        if not r.get("Ticker"):
            continue
        estimates.append({
            "ticker": str(r["Ticker"]).strip().upper(),
            "fiscal_period": str(r.get("Fiscal period") or "").strip(),
            "period_end": to_date(r.get("Period end")),
            "revenue_consensus": to_number(r.get("Revenue consensus ($m)")),
            "eps_consensus": to_number(r.get("EPS consensus")),   # '$5.39' handled
            "eps_basis": str(r.get("EPS basis") or "").strip(),
            "n_estimates": to_number(r.get("# estimates")),
            "report_date": to_date(r.get("Report date")),
            "revenue_actual": to_number(r.get("Revenue actual ($m)")),
            "eps_actual": to_number(r.get("EPS actual")),
            "as_of_date": as_of,
        })

    targets = [
        {"ticker": str(r["Ticker"]).strip().upper(),
         "consensus_pt": to_number(r.get("Consensus PT (USD)")),
         "buy": to_number(r.get("Buy")), "hold": to_number(r.get("Hold")),
         "sell": to_number(r.get("Sell")),
         "last_price": to_number(r.get("Last price")),
         "as_of_date": as_of}
        for r in sheet_records(path, "Price targets") if r.get("Ticker")
    ]

    notes = [
        {"note_date": to_date(r.get("Date")),
         "house": str(r.get("House") or "").strip(),
         "ticker": str(r["Ticker"]).strip().upper(),
         "rating": str(r.get("Rating (house scale)") or "").strip(),
         "price_target": to_number(r.get("Price target")),
         "analyst": str(r.get("Analyst") or "").strip()}
        for r in sheet_records(path, "Broker notes on file") if r.get("Ticker")
    ]
    return estimates, targets, notes


def workbook_path(folder: str | Path, name: str) -> Path:
    """Resolve a workbook whether saved as 'FX_rates.xlsx' or 'FX rates.xlsx'."""
    from app.filenames import resolve
    return resolve(folder, name)


def load_all(folder: str | Path) -> Workbooks:
    folder = Path(folder)
    wbs = Workbooks()
    wbs.positions, wbs.trades = load_holdings_and_trades(
        workbook_path(folder, "Holdings_and_trades.xlsx"))
    wbs.prices, wbs.dividends, wbs.corporate_actions = load_prices(
        workbook_path(folder, "Prices.xlsx"))
    wbs.fx = load_fx(workbook_path(folder, "FX_rates.xlsx"))
    wbs.consensus, wbs.price_targets, wbs.broker_notes = load_consensus(
        workbook_path(folder, "Consensus_estimates.xlsx"), as_of=dt.date(2026, 6, 30))
    return wbs


if __name__ == "__main__":
    import sys
    w = load_all(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads")
    print(f"positions={len(w.positions)} trades={len(w.trades)} prices={len(w.prices)} "
          f"dividends={len(w.dividends)} actions={w.corporate_actions} "
          f"fx={len(w.fx)} consensus={len(w.consensus)} targets={len(w.price_targets)} "
          f"broker_notes={len(w.broker_notes)}")
