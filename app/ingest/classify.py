"""
Document classification: type, ticker, dates, fiscal period, tier, event key.

Design: deterministic signals first, LLM second.

The facts that matter most for adjudication are stated literally in the
documents. "CORRECTED TRANSCRIPT" and "Transcript auto-generated... may contain
errors" are printed on the two NVIDIA transcripts; "FORM 10-Q" is printed on the
filings; "(Nasdaq: AMZN)" names the issuer. Extracting those with regex costs
nothing, is reproducible across runs, and cannot hallucinate. An LLM asked to
decide which transcript is authoritative might get it right; a string match on
"CORRECTED TRANSCRIPT" always does.

So the pipeline is:

    1. extract_signals()   deterministic, high precision, free
    2. classify_with_llm() fills whatever is still missing
    3. merge()             deterministic wins where it was confident

Two outputs drive later stages:

  source_tier   adjudication order when documents disagree about a company fact
  event_key     clusters different renderings of ONE event, so the two
                transcripts of the NVIDIA Q4 FY2026 call are recognised as one
                source rather than two independent witnesses

Note on scope: source_tier ranks documents as evidence about *companies*.
Precedence for *portfolio* data is a different question and is handled by
holdings.source_kind, because a custodian statement outranks everything about
what the client owns while saying nothing about Amazon's revenue.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

log = logging.getLogger(__name__)

KNOWN_TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM",
                 "COST", "UNH", "TSLA", "LLY"}

# Presentations and some scans never print a ticker, so the company name is the
# only signal available.
NAME_TO_TICKER = {
    r"Apple\b": "AAPL", r"Microsoft": "MSFT", r"NVIDIA": "NVDA",
    r"Amazon": "AMZN", r"JPMorgan|J\.P\. Morgan": "JPM",
    r"Exxon\s?Mobil|ExxonMobil": "XOM", r"Costco": "COST",
    r"UnitedHealth": "UNH", r"Tesla": "TSLA", r"Eli Lilly|\bLilly\b": "LLY",
}

# --------------------------------------------------------------------------
# Document types and tiers
# --------------------------------------------------------------------------

# Ordered: the first pattern that matches wins, so specific beats general.
DOC_TYPE_PATTERNS: list[tuple[str, str]] = [
    # NOT a bare "FORM 10-K": every press release cites it in the
    # forward-looking-statements boilerplate. Anchor on the SEC cover page.
    ("filing_10k",          r"ANNUAL REPORT PURSUANT TO SECTION 13"),
    ("filing_10q",          r"QUARTERLY REPORT PURSUANT TO SECTION 13"),
    ("custodian_statement", r"Brokerage Account Statement|Relev. de portefeuille"
                            r"|Portfolio statement"),
    # NOT a bare "earnings call": the Microsoft CFO release mentions the
    # upcoming call in prose. Require structural transcript markers.
    ("transcript",          r"CORRECTED TRANSCRIPT|Earnings Call Transcript"
                            r"|MANAGEMENT DISCUSSION SECTION|CORPORATE PARTICIPANTS"
                            r"|QUESTION AND ANSWER SECTION|Call duration"),
    ("broker_note",         r"\|\s*Equity Research\s*\||Analyst certification"
                            r"|Rating definitions"),
    ("presentation",        r"Investor Day|R&D;?\s*Day|Investor Overview"
                            r"|confidential draft for attendees"),
    ("corporate_action",    r"Stock Split|Chief Financial Officer transition"
                            r"|announces? .{0,40}(?:Split|transition)"),
    ("press_release",       r"today announced financial results"
                            r"|Reports (?:First|Second|Third|Fourth) Quarter"
                            r"|press release"),
]

# Tier 1 is the most authoritative. Lower numbers win when facts conflict.
TIER_BY_TYPE = {
    "filing_10k": 1, "filing_10q": 1,
    "press_release": 2, "corporate_action": 2, "custodian_statement": 2,
    "transcript": 4,            # refined below by TIER_MARKERS
    "presentation": 5,
    "broker_note": 6,
    "internal_workbook": 7,
}

# A transcript's trustworthiness is printed on it. This is the rule that makes
# "$41.4 billion" beat "$41.4 million".
TIER_MARKERS: list[tuple[int, str]] = [
    (3, r"CORRECTED TRANSCRIPT|Verbatim Transcripts"),
    (4, r"auto-generated|may contain (?:material )?errors"),
]

EVENT_FAMILY = {
    "filing_10k": "annual_report", "filing_10q": "quarterly_report",
    "press_release": "earnings_release", "transcript": "earnings_call",
    "presentation": "investor_event", "broker_note": "research_note",
    "corporate_action": "corporate_action",
    "custodian_statement": "account_statement",
}

# --------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------

MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
         r"October|November|December)")
LONG_DATE = rf"{MONTH}\s+\d{{1,2}},\s+\d{{4}}"

TICKER_PARENS = re.compile(r"\((?:Nasdaq|NYSE|NASDAQ|NYSE American)\s*:\s*([A-Z.]{1,6})\)")
TICKER_BARE = re.compile(r"\(([A-Z]{2,5})\)")
TICKER_SYMBOL_ROW = re.compile(r"Trading symbol[\s\S]{0,120}?\b([A-Z]{2,5})\b")

PERIOD_END_PATTERNS = [
    re.compile(rf"fiscal year ended\s+({LONG_DATE})", re.I),
    re.compile(rf"quarterly period ended\s+({LONG_DATE})", re.I),
    re.compile(rf"quarter (?:of|ended)[^.]{{0,30}}?ended\s+({LONG_DATE})", re.I),
    re.compile(rf"ended\s+({LONG_DATE})", re.I),
]

PUBLISHED_PATTERNS = [
    re.compile(rf"—\s*({LONG_DATE})\s*—"),                 # press release dateline
    re.compile(rf"Event date:\s*({LONG_DATE})", re.I),     # Verbatim transcripts
    re.compile(rf"\|\s*Equity Research[^|]*\|\s*({LONG_DATE})"),  # broker notes
    re.compile(rf"^({LONG_DATE})\s*·", re.M),              # CallScript transcripts
    re.compile(rf"Filed\s+({LONG_DATE})", re.I),
    re.compile(rf"\b({LONG_DATE})\b"),                     # last resort
]

QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
FISCAL_PATTERNS = [
    # "its fiscal 2026 second quarter"  /  "Second Quarter Fiscal 2026"
    re.compile(r"fiscal\s+(\d{4})\s+(first|second|third|fourth)\s+quarter", re.I),
    re.compile(r"(first|second|third|fourth)\s+quarter\s+fiscal\s+(?:year\s+)?(\d{4})", re.I),
    # "its fourth quarter of 2025"
    re.compile(r"(first|second|third|fourth)\s+quarter\s+of\s+(\d{4})", re.I),
    # literal "Q4 FY2026" / "Q1 2026"
    re.compile(r"\bQ([1-4])\s*(?:FY)?\s*(\d{4})\b"),
]
FY_PATTERN = re.compile(r"fiscal year ended\s+" + MONTH + r"\s+\d{1,2},\s+(\d{4})", re.I)

# Companies whose fiscal-year label runs ahead of the calendar year of the
# period end. Needed so NVIDIA's quarter ending Jan 2026 is Q4 FY2026, not FY2025.
FY_LEADS = {"NVDA", "AAPL", "MSFT", "COST"}


@dataclass
class Classification:
    doc_type: str | None = None
    ticker: str | None = None
    company: str | None = None      # only needed for tickers outside KNOWN_TICKERS
    publisher: str | None = None
    title: str | None = None
    published_date: dt.date | None = None
    as_of_date: dt.date | None = None
    fiscal_period: str | None = None
    source_tier: int = 5
    event_key: str | None = None
    signals: dict = field(default_factory=dict)   # what was found deterministically

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("signals", None)
        return d


# --------------------------------------------------------------------------
# Deterministic layer
# --------------------------------------------------------------------------

def _parse_long_date(s: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(s.strip(), "%B %d, %Y").date()
    except ValueError:
        return None


def detect_doc_type(text: str) -> str | None:
    for doc_type, pattern in DOC_TYPE_PATTERNS:
        if re.search(pattern, text, re.I):
            return doc_type
    return None


def detect_tier(doc_type: str | None, text: str) -> int:
    for tier, pattern in TIER_MARKERS:
        if re.search(pattern, text, re.I):
            # Markers only override within transcripts; a 10-K that happens to
            # contain the word "errors" is still tier 1.
            if doc_type == "transcript" or doc_type is None:
                return tier
    return TIER_BY_TYPE.get(doc_type or "", 5)


def detect_ticker(text: str, filename: str = "") -> str | None:
    for rx in (TICKER_PARENS, TICKER_SYMBOL_ROW, TICKER_BARE):
        for m in rx.finditer(text[:6000]):
            cand = m.group(1).replace(".", "").upper()
            if cand in KNOWN_TICKERS:
                return cand
    # A company outside the original ten, named by its exchange listing in the
    # dateline ("(NASDAQ: ZPHR)"). Checked before the name patterns so a new
    # company's release that mentions Apple in passing is not filed as Apple.
    if m := TICKER_PARENS.search(text[:800]):
        return m.group(1).replace(".", "").upper()
    # Company name in the opening lines.
    for pattern, tick in NAME_TO_TICKER.items():
        if re.search(pattern, text[:1500], re.I):
            return tick
    # Filename is the weakest signal, used only when the text yields nothing.
    upper = filename.upper()
    for t in KNOWN_TICKERS:
        if re.search(rf"(?:^|[^A-Z]){t}(?:[^A-Z]|$)", upper):
            return t
    return None


def detect_dates(text: str) -> tuple[dt.date | None, dt.date | None]:
    """Returns (published_date, as_of_date)."""
    head = text[:6000]
    published = None
    for rx in PUBLISHED_PATTERNS:
        m = rx.search(head)
        if m and (published := _parse_long_date(m.group(1))):
            break

    if published is None:
        m = re.search(rf"\b({MONTH})\s+(\d{{4}})\b", head)
        if m:
            published = dt.date(int(m.group(2)),
                                dt.datetime.strptime(m.group(1), "%B").month, 1)

    as_of = None
    for rx in PERIOD_END_PATTERNS:
        m = rx.search(head)
        if m and (as_of := _parse_long_date(m.group(1))):
            break

    # Custodian statements: "statement June 30, 2026", "Holdings as of June 30, 2026"
    if as_of is None:
        m = re.search(rf"(?:statement|Holdings as of|as of)\s+({LONG_DATE})", head, re.I)
        if m:
            as_of = _parse_long_date(m.group(1))
    # Swiss statement: "Valuation date: 30.06.2026"
    if as_of is None:
        m = re.search(r"(?:Valuation date|date)[:\s]+(\d{2})\.(\d{2})\.(\d{4})", head, re.I)
        if m:
            as_of = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return published, as_of


def detect_fiscal_period(text: str, ticker: str | None,
                         as_of: dt.date | None) -> str | None:
    head = text[:6000]

    for rx in FISCAL_PATTERNS[:2]:                    # explicit "fiscal YYYY Qn"
        if m := rx.search(head):
            groups = m.groups()
            word, year = (groups if groups[0].lower() in QUARTER_WORDS
                          else (groups[1], groups[0]))
            return f"Q{QUARTER_WORDS[word.lower()]} FY{year}"

    if m := FISCAL_PATTERNS[2].search(head):          # "fourth quarter of 2025"
        return f"Q{QUARTER_WORDS[m.group(1).lower()]} {m.group(2)}"

    if m := FISCAL_PATTERNS[3].search(head):          # literal "Q4 FY2026"
        q, year = m.groups()
        label = "FY" if (ticker in FY_LEADS or "FY" in m.group(0)) else ""
        return f"Q{q} {label}{year}"

    if m := FY_PATTERN.search(head):
        return f"FY{m.group(1)}"

    if as_of:                                         # fall back to the period end
        q = (as_of.month - 1) // 3 + 1
        return f"Q{q} {as_of.year}"
    return None


def build_event_key(doc_type: str | None, ticker: str | None,
                    fiscal_period: str | None, published: dt.date | None,
                    as_of: dt.date | None) -> str | None:
    """Identifies ONE event, so different renderings of it collapse together.

    Deliberately keeps distinct events distinct: Amazon's Q4 2025 press release
    and the Q4 2025 call transcript are genuinely different sources published on
    the same day, so they get different keys. Only the two vendors' transcripts
    of the same call collide, which is the intent.
    """
    family = EVENT_FAMILY.get(doc_type or "", "document")
    if family == "account_statement":
        anchor = (as_of or published)
        return f"STATEMENT|{anchor.isoformat() if anchor else 'unknown'}"
    if not ticker:
        return None
    if family == "research_note":
        anchor = published.isoformat() if published else "unknown"
    elif family in ("corporate_action", "account_statement"):
        anchor = (as_of or published).isoformat() if (as_of or published) else "unknown"
    else:
        anchor = fiscal_period or (published.isoformat() if published else "unknown")
    return f"{ticker}|{family}|{anchor}"


def detect_publisher(text: str, doc_type: str | None) -> str | None:
    head = text[:2000]
    if doc_type == "broker_note":
        if m := re.match(r"\s*([A-Z][\w&.\' ]{2,40}?)\s*\|\s*Equity Research", head):
            return m.group(1).strip()
        if m := re.match(r"\s*([A-Z][\w&.\' ]{2,40}?)\s*—", head):
            return m.group(1).strip()
    if doc_type == "transcript":
        for name in ("CallScript.io", "Verbatim Transcripts"):
            if name.lower() in head.lower():
                return name
    if doc_type == "custodian_statement":
        if m := re.match(r"\s*([A-Z][\w&.\'é ]{2,60}?)(?:\s+(?:LLC|SA|Ltd))", head):
            return m.group(0).strip()
    return None


def extract_signals(text: str, filename: str = "") -> Classification:
    doc_type = detect_doc_type(text)
    ticker = detect_ticker(text, filename)
    published, as_of = detect_dates(text)
    fiscal = detect_fiscal_period(text, ticker, as_of)
    tier = detect_tier(doc_type, text)
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")

    if doc_type == "custodian_statement":
        ticker = None           # multi-ticker by nature
        fiscal = None           # a statement period is not a fiscal quarter
        published = published or as_of

    c = Classification(
        doc_type=doc_type, ticker=ticker,
        publisher=detect_publisher(text, doc_type),
        title=first_line[:200] or None,
        published_date=published, as_of_date=as_of,
        fiscal_period=fiscal, source_tier=tier,
    )
    c.event_key = build_event_key(doc_type, ticker, fiscal, published, as_of)
    c.signals = {k: v for k, v in (
        ("doc_type", doc_type), ("ticker", ticker), ("published_date", published),
        ("as_of_date", as_of), ("fiscal_period", fiscal)) if v}
    return c


# --------------------------------------------------------------------------
# LLM layer — fills whatever the deterministic pass left empty
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You classify financial documents for a family office.

Return ONLY a JSON object, no prose and no code fences, with these keys:
  doc_type       one of: filing_10k, filing_10q, press_release, transcript,
                 presentation, broker_note, custodian_statement, corporate_action
  ticker         the US listing symbol of the company the document is ABOUT,
                 or null. For a broker note this is the covered company, not
                 the publishing house.
  company        that company's name as the document prints it, without legal
                 suffixes such as "Inc." or "Corporation", or null.
  publisher      the research house or transcript vendor, or null for
                 company-issued documents.
  title          a short descriptive title, max 100 characters.
  published_date the date the document was issued, YYYY-MM-DD or null.
  as_of_date     the end of the period the content describes, YYYY-MM-DD or null.
  fiscal_period  the company's own label, e.g. "Q4 FY2026" or "Q2 2025" or
                 "FY2025", or null.

Rules:
- Use only what the document says. Do not use outside knowledge about these
  companies; the figures and people in these documents may differ from reality.
- Fiscal labels follow the company's convention, so a quarter ending in
  January 2026 may belong to fiscal 2026.
- If a field is not stated, return null rather than guessing."""


def classify_with_llm(text: str, client=None, model: str = "claude-sonnet-5-5") -> dict:
    """Ask the model for the fields the regex pass could not determine."""
    if client is None:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["AIEHA_API_KEY"])

    resp = client.messages.create(
        model=model, max_tokens=400, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text[:4000]}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("classifier returned non-JSON: %.120s", raw)
        return {}


def merge(signals: Classification, llm: dict) -> Classification:
    """Deterministic findings win; the LLM only fills blanks.

    The exception that matters: source_tier and event_key are never taken from
    the model. They decide which document wins a conflict, so they stay with
    the printed markers.
    """
    def pick(current, key, parse=lambda v: v):
        if current is not None:
            return current
        value = llm.get(key)
        return parse(value) if value not in (None, "", "null") else None

    def as_date(v):
        try:
            return dt.date.fromisoformat(str(v)[:10])
        except (ValueError, TypeError):
            return None

    def as_ticker(v):
        cand = re.sub(r"^[A-Za-z ]+:\s*", "", str(v)).replace(".", "").strip().upper()
        return cand if re.fullmatch(r"[A-Z]{1,6}", cand) else None

    out = Classification(
        doc_type=pick(signals.doc_type, "doc_type"),
        ticker=pick(signals.ticker, "ticker", as_ticker),
        company=pick(signals.company, "company", clean_company_name),
        publisher=pick(signals.publisher, "publisher"),
        title=pick(signals.title, "title"),
        published_date=pick(signals.published_date, "published_date", as_date),
        as_of_date=pick(signals.as_of_date, "as_of_date", as_date),
        fiscal_period=pick(signals.fiscal_period, "fiscal_period"),
        signals=signals.signals,
    )
    out.source_tier = (signals.source_tier if signals.doc_type
                       else TIER_BY_TYPE.get(out.doc_type or "", 5))
    out.event_key = build_event_key(out.doc_type, out.ticker, out.fiscal_period,
                                    out.published_date, out.as_of_date)
    return out


NEEDS_TICKER = {"filing_10k", "filing_10q", "press_release", "transcript",
                "presentation", "broker_note", "corporate_action"}
NEEDS_PERIOD = {"filing_10k", "filing_10q", "press_release", "transcript"}


def _is_complete(c: Classification) -> bool:
    """What counts as complete depends on the type.

    A custodian statement covers eight tickers and no fiscal period, so
    demanding both would send every statement to the LLM for nothing.
    """
    if not c.doc_type or not c.published_date:
        return False
    if c.doc_type in NEEDS_TICKER and not c.ticker:
        return False
    if c.doc_type in NEEDS_PERIOD and not c.fiscal_period:
        return False
    if c.doc_type == "custodian_statement" and not c.as_of_date:
        return False
    return True


def classify_document(text: str, filename: str = "", client=None,
                      use_llm: bool = True) -> Classification:
    signals = extract_signals(text, filename)
    complete = _is_complete(signals)
    if complete or not use_llm:
        return signals
    return merge(signals, classify_with_llm(text, client=client))


def clean_company_name(name) -> str | None:
    """'Zephyr Robotics, Inc.' -> 'Zephyr Robotics'."""
    if not name or not str(name).strip():
        return None
    out = re.sub(r",?\s+(Inc\.?|Incorporated|Corp\.?|Corporation|Co\.?|Ltd\.?|"
                 r"Limited|plc|PLC|N\.V\.|S\.A\.|AG|SE|Holdings)$", "", str(name).strip())
    return out.strip(" ,") or None


def needs_llm(c: Classification) -> bool:
    """Incomplete, or a company outside the original ten with no name yet."""
    unknown = c.ticker is not None and c.ticker not in KNOWN_TICKERS and not c.company
    return not _is_complete(c) or unknown


def classify_cached(text: str, filename: str, content_hash: str, use_llm: bool = True,
                    client=None) -> Classification:
    """classify_document, with the model's answer cached by content hash.

    The model is asked at most once per document, when a key is available.
    Every later reader -- the search index, the Sources catalog, bootstrap
    after a restart -- reuses that answer for free, so a new company's name
    and ticker survive restarts instead of falling back to the regex pass.
    """
    from app.ingest import cache

    signals = extract_signals(text, filename)
    if not needs_llm(signals):
        return signals
    llm = cache.get(content_hash, "classify")
    if llm is None and use_llm:
        try:
            llm = classify_with_llm(text, client=client)
        except Exception as exc:                        # noqa: BLE001
            log.warning("classification call failed for %s: %s", filename, exc)
            llm = None
        if llm:
            cache.put(content_hash, "classify", llm)
    return merge(signals, llm) if llm else signals


# --------------------------------------------------------------------------
# CLI: run the deterministic pass over a folder and report coverage
# --------------------------------------------------------------------------

def _document_text(path: Path) -> str:
    """First two pages are enough to classify.

    Goes through the content-hash cache, so a scanned document is OCR'd once
    and every later stage reads the same text. Without this a full-folder
    sweep re-runs OCR on every pass and does not finish.
    """
    if path.suffix.lower() != ".pdf":
        return ""
    from app.ingest.cache import head_text
    return head_text(path, pages=2)


if __name__ == "__main__":
    import sys
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads")
    rows = []
    for p in sorted(folder.glob("*.pdf")):
        c = classify_document(_document_text(p), p.name, use_llm=False)
        rows.append((p.name, c))
        print(f"{p.name[:52]:<52} {str(c.doc_type):<20} {str(c.ticker):<5} "
              f"t{c.source_tier}  {str(c.published_date):<11} "
              f"{str(c.fiscal_period):<10} {c.event_key or ''}")

    missing = [n for n, c in rows if not _is_complete(c)]
    print(f"\n{len(rows) - len(missing)}/{len(rows)} fully resolved without an LLM call")
    if missing:
        print("needs the LLM pass:", ", ".join(missing))
