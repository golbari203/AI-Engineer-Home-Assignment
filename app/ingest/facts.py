"""
Fact extraction: figures from prose documents into queryable rows.

Three jobs, in order of how much they matter.

1. Basis and period are mandatory
   A bare number is meaningless in this corpus. Lilly's Q1 2026 EPS is $2.87
   GAAP and $4.68 non-GAAP. NVIDIA's Q1 FY2026 EPS is $7.33 pre-split and the
   same quarter reads $0.73 in the restated 10-K. Storing 4.68 without
   "non-GAAP" is storing a future error, so a fact with no basis is dropped.

2. Cross-footing, because OCR corrupts digits silently
   The OCR'd JPMorgan income statement returned net interest income as 18,268
   against a true 18,208, at 90% mean confidence. Confidence does not catch
   that; arithmetic does, because the page states its own total:
       15,025 + 18,268 = 33,293  != 33,233   reject
       15,025 + 18,208 = 33,233  == 33,233   accept

3. Conflict detection becomes a query
   Four rows for AWS Q4 2025 revenue -- two values, one marked restated --
   make disagreement a property of the data rather than an accident of which
   chunks retrieval happened to return.

A warning about what NOT to check
---------------------------------
Segment revenues sum to total revenue; segment operating income does not.
Amazon's Q2 2026 release shows segment operating income of 4,743 + 1,383 +
9,194 = 15,320, a segment-table total of 13,933, and a consolidated operating
income of 13,523 -- three different numbers in one document. Validating that
identity would raise a false error on every Amazon filing. Only identities
that hold across the whole corpus are enforced; the rest are informational.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, asdict, field

from app.ingest.workbooks import to_date, to_number

log = logging.getLogger(__name__)

VALID_BASES = {"GAAP", "non-GAAP", "adjusted", "ex_identified_items",
               "managed", "constant_currency", None}

# Canonical metric names. The model is given this vocabulary so that the same
# concept from ten different houses lands in one row group.
METRICS = [
    "revenue", "net_interest_income", "noninterest_revenue", "cost_of_sales",
    "gross_margin", "operating_expenses", "operating_income", "income_before_taxes",
    "income_tax_expense", "net_income", "eps_basic", "eps_diluted",
    "operating_cash_flow", "free_cash_flow", "capital_expenditures",
    "dividends_per_share", "shareholder_returns", "total_assets",
    "total_liabilities", "shareholders_equity", "cash_and_investments",
    "total_debt", "gross_margin_pct", "operating_margin_pct", "medical_care_ratio",
    "cet1_ratio", "rotce", "comparable_sales_growth", "membership_renewal_rate",
    "paid_members", "warehouses", "vehicle_deliveries", "energy_storage_deployed",
    "guidance_revenue_low", "guidance_revenue_high", "guidance_operating_income_low",
    "guidance_operating_income_high", "price_target", "rating",
    "cfo_name", "ceo_name",
]

TEXT_METRICS = {"cfo_name", "ceo_name", "rating"}

# Fiscal calendars. Needed so "Q1 FY2026" resolves to real dates and
# "last quarter" means something different for Apple than for Amazon.
FISCAL_CALENDAR = {
    "AAPL": {"fy_end_month": 9,  "label_leads": True},
    "MSFT": {"fy_end_month": 6,  "label_leads": True},
    "NVDA": {"fy_end_month": 1,  "label_leads": True},
    "COST": {"fy_end_month": 8,  "label_leads": True},
    "AMZN": {"fy_end_month": 12, "label_leads": False},
    "JPM":  {"fy_end_month": 12, "label_leads": False},
    "XOM":  {"fy_end_month": 12, "label_leads": False},
    "UNH":  {"fy_end_month": 12, "label_leads": False},
    "TSLA": {"fy_end_month": 12, "label_leads": False},
    "LLY":  {"fy_end_month": 12, "label_leads": False},
}


@dataclass
class Fact:
    ticker: str | None
    metric: str
    fiscal_period: str | None = None
    period_start: dt.date | None = None
    period_end: dt.date | None = None
    value_num: float | None = None
    value_text: str | None = None
    unit: str | None = None
    basis: str | None = None
    segment: str | None = None
    is_guidance: bool = False
    is_restated: bool = False
    document_id: int | None = None
    page_no: int | None = None
    stated_on: dt.date | None = None
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        """What makes two facts comparable, and therefore conflicting."""
        return (self.ticker, self.metric, self.fiscal_period,
                self.basis, self.segment, self.is_guidance)

    def as_row(self) -> dict:
        d = asdict(self)
        d["notes"] = json.dumps(d["notes"]) if d["notes"] else None
        return d


# ---------------------------------------------------------------------------
# Fiscal periods
# ---------------------------------------------------------------------------

def quarter_end(ticker: str, label: str) -> dt.date | None:
    """Approximate the calendar quarter-end a fiscal label refers to.

    Documents state their own period end, which is always preferred. This is
    the fallback for when OCR has damaged the date, and the basis for
    answering "the most recent quarter" per company.
    """
    m = re.fullmatch(r"Q([1-4])\s+(?:FY)?(\d{4})", (label or "").strip())
    if not m:
        return None
    q, year = int(m.group(1)), int(m.group(2))
    cal = FISCAL_CALENDAR.get(ticker, {"fy_end_month": 12, "label_leads": False})
    fy_end = cal["fy_end_month"]

    # Month in which the fiscal quarter closes, counting forward from FY start.
    month = ((fy_end + 3 * q - 1) % 12) + 1
    # When the label leads the calendar, FY2026 Q1 lands in calendar 2025.
    if cal["label_leads"] and month > fy_end:
        year -= 1
    elif not cal["label_leads"]:
        year = year
    last = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    if month == 2 and year % 4 == 0:
        last = 29
    return dt.date(year, month, last)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You extract financial facts from documents into JSON rows.

Return ONLY a JSON array. Each element:

  metric         one of: {", ".join(METRICS)}
  value          a number, or a string for: {", ".join(sorted(TEXT_METRICS))}
  unit           "USD_millions", "USD", "percent", "count", "shares", "GWh"
  basis          "GAAP", "non-GAAP", "adjusted", "ex_identified_items",
                 "managed", "constant_currency", or null when the document
                 states no basis
  fiscal_period  the company's own label, e.g. "Q4 FY2026", "Q2 2025", "FY2025"
  period_end     YYYY-MM-DD as printed, or null
  segment        the business segment, or null for consolidated figures
  is_guidance    true for forward-looking figures, false for reported results
  is_restated    true when the document labels the figure as recast, restated
                 or reclassified from a previously reported number
  page           the page number the figure appears on, if known

Rules:
- Use only what the document says. Do not use outside knowledge about these
  companies; figures and people here may differ from reality.
- basis is not optional guesswork: if a company reports both GAAP and non-GAAP
  EPS, emit two rows with different basis values, never one.
- When a document gives both a previously reported and a recast figure for the
  same period, emit BOTH, marking the recast one is_restated true.
- Report figures exactly as printed. Do not convert units, annualise, or
  compute anything that is not stated.
- Omit a fact entirely rather than guessing a field."""


def extract_facts(text: str, meta: dict, client=None,
                  model: str = "claude-sonnet-5-5") -> list[dict]:
    if client is None:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["AIEHA_API_KEY"])

    context = (f"Document: {meta.get('title') or ''}\n"
               f"Company: {meta.get('ticker') or 'unknown'}\n"
               f"Type: {meta.get('doc_type') or 'unknown'}\n"
               f"Published: {meta.get('published_date') or 'unknown'}\n\n")
    resp = client.messages.create(
        model=model, max_tokens=8000, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context + text[:30000]}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    return parse_rows(raw)


def parse_rows(raw: str) -> list[dict]:
    """The model's JSON array, keeping every complete row if it was cut off.

    A long filing can exhaust max_tokens mid-row. Dropping the whole document
    for one unfinished row would lose dozens of good figures.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        end = raw.rfind("}")
        if end < 0:
            raise
        parsed = json.loads(raw[:end + 1].rstrip().rstrip(",") + "]")
        log.warning("fact extraction output was truncated; kept %d complete rows",
                    len(parsed))
    return parsed if isinstance(parsed, list) else []


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_facts(rows: list[dict], meta: dict) -> tuple[list[Fact], list[str]]:
    """Returns (facts, rejections). A fact missing basis or period is dropped."""
    facts: list[Fact] = []
    rejected: list[str] = []
    ticker = meta.get("ticker")

    for r in rows:
        metric = (r.get("metric") or "").strip()
        if metric not in METRICS:
            rejected.append(f"unknown metric '{metric}'")
            continue

        basis = r.get("basis")
        basis = basis.strip() if isinstance(basis, str) and basis.strip() else None
        if basis not in VALID_BASES:
            rejected.append(f"{metric}: invalid basis '{basis}'")
            continue

        period = (r.get("fiscal_period") or meta.get("fiscal_period") or "").strip()
        if not period and metric not in TEXT_METRICS:
            rejected.append(f"{metric}: no fiscal period")
            continue

        is_text = metric in TEXT_METRICS
        value = r.get("value")
        if is_text:
            value_text, value_num = (str(value).strip() if value else None), None
            if not value_text:
                rejected.append(f"{metric}: empty value")
                continue
        else:
            value_num, value_text = to_number(value), None
            if value_num is None:
                rejected.append(f"{metric}: unparseable value {value!r}")
                continue

        period_end = to_date(r.get("period_end")) or quarter_end(ticker, period)
        facts.append(Fact(
            ticker=ticker, metric=metric, fiscal_period=period or None,
            period_end=period_end, value_num=value_num, value_text=value_text,
            unit=(r.get("unit") or None), basis=basis,
            segment=(r.get("segment") or None),
            is_guidance=bool(r.get("is_guidance")),
            is_restated=bool(r.get("is_restated")),
            document_id=meta.get("document_id"),
            page_no=r.get("page"), stated_on=to_date(meta.get("published_date")),
        ))
    return facts, rejected


# ---------------------------------------------------------------------------
# Cross-footing
# ---------------------------------------------------------------------------

# Identities verified to hold across every document in the corpus.
# Each entry: (total_metric, [(sign, component_metric), ...])
IDENTITIES: list[tuple[str, list[tuple[int, str]]]] = [
    ("revenue", [(1, "net_interest_income"), (1, "noninterest_revenue")]),
    ("gross_margin", [(1, "revenue"), (-1, "cost_of_sales")]),
    ("operating_income", [(1, "gross_margin"), (-1, "operating_expenses")]),
    ("net_income", [(1, "income_before_taxes"), (-1, "income_tax_expense")]),
]

# Deliberately absent: segment operating income summing to the consolidated
# figure. It does not hold for Amazon in any quarter. Segment revenue summing
# to total revenue does hold and is checked separately below.

CROSSFOOT_TOL = 1.0     # the statements are in whole millions


def crossfoot(facts: list[Fact], ocr_pages: set[int] | None = None
              ) -> list[str]:
    """Check stated totals against their stated components.

    Facts on pages that came from OCR have their confidence reduced when an
    identity fails, because that is the signal that a digit was misread.
    """
    ocr_pages = ocr_pages or set()
    problems: list[str] = []

    by_key: dict[tuple, dict[str, Fact]] = defaultdict(dict)
    for f in facts:
        if f.value_num is None or f.segment or f.is_guidance:
            continue
        by_key[(f.ticker, f.fiscal_period, f.basis)][f.metric] = f

    for group in by_key.values():
        for total_metric, components in IDENTITIES:
            total = group.get(total_metric)
            parts = [(sign, group.get(m)) for sign, m in components]
            if total is None or any(p is None for _, p in parts):
                continue
            expected = sum(sign * p.value_num for sign, p in parts)
            delta = total.value_num - expected
            if abs(delta) <= CROSSFOOT_TOL:
                continue
            involved = [total] + [p for _, p in parts]
            from_ocr = [f for f in involved if f.page_no in ocr_pages]
            for f in involved:
                f.confidence = min(f.confidence, 0.4 if from_ocr else 0.7)
                f.notes.append(f"failed identity {total_metric}: "
                               f"components give {expected:,.0f}, "
                               f"stated {total.value_num:,.0f}")
            problems.append(
                f"{total.ticker} {total.fiscal_period}: {total_metric} stated "
                f"{total.value_num:,.0f} but components give {expected:,.0f}"
                + (" (figures from OCR'd pages)" if from_ocr else ""))
    return problems


def check_segment_revenue_sums(facts: list[Fact]) -> list[str]:
    """Segment revenues do sum to total revenue. Segment income does not."""
    problems = []
    totals = {(f.ticker, f.fiscal_period): f for f in facts
              if f.metric == "revenue" and not f.segment and not f.is_guidance}
    segs: dict[tuple, list[Fact]] = defaultdict(list)
    for f in facts:
        if f.metric == "revenue" and f.segment and not f.is_guidance:
            segs[(f.ticker, f.fiscal_period)].append(f)

    for key, parts in segs.items():
        total = totals.get(key)
        if total is None or len(parts) < 2:
            continue
        summed = sum(p.value_num for p in parts if p.value_num is not None)
        if abs(summed - total.value_num) > CROSSFOOT_TOL:
            for f in parts + [total]:
                f.confidence = min(f.confidence, 0.7)
            problems.append(
                f"{key[0]} {key[1]}: segment revenue sums to {summed:,.0f} "
                f"against a stated total of {total.value_num:,.0f}")
    return problems


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    ticker: str | None
    metric: str
    fiscal_period: str | None
    basis: str | None
    segment: str | None
    versions: list[Fact]

    def describe(self) -> str:
        parts = []
        for f in sorted(self.versions, key=lambda x: (x.stated_on or dt.date.min)):
            value = f.value_text if f.value_text is not None else f"{f.value_num:,.0f}"
            parts.append(f"{value} (doc {f.document_id}"
                         + (", restated" if f.is_restated else "")
                         + (f", {f.stated_on}" if f.stated_on else "") + ")")
        label = f"{self.ticker} {self.metric}"
        if self.segment:
            label += f" [{self.segment}]"
        if self.basis:
            label += f" {self.basis}"
        return f"{label} {self.fiscal_period}: " + " vs ".join(parts)


def find_conflicts(facts: list[Fact]) -> list[Conflict]:
    """Same ticker, metric, period, basis and segment but different values.

    Deliberately groups on basis and segment too: Lilly reporting $2.87 GAAP
    and $4.68 non-GAAP for one quarter is not a conflict, it is two facts.
    """
    groups: dict[tuple, list[Fact]] = defaultdict(list)
    for f in facts:
        groups[(f.ticker, f.metric, f.fiscal_period, f.basis, f.segment)].append(f)

    conflicts = []
    for key, versions in groups.items():
        distinct = {(v.value_num, v.value_text) for v in versions}
        if len(distinct) > 1:
            conflicts.append(Conflict(*key, versions=versions))
    return conflicts


def resolve(conflict: Conflict, tier_by_document: dict[int, int]) -> Fact:
    """Pick the authoritative version: best tier, then most recent.

    A restated figure supersedes the figure it restates regardless of tier,
    because a restatement is the issuer saying the earlier number is wrong.
    """
    restated = [f for f in conflict.versions if f.is_restated]
    if restated:
        return max(restated, key=lambda f: f.stated_on or dt.date.min)
    return min(conflict.versions,
               key=lambda f: (tier_by_document.get(f.document_id or -1, 9),
                              -(f.stated_on or dt.date.min).toordinal()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_facts(text: str, meta: dict, client=None,
               ocr_pages: set[int] | None = None) -> tuple[list[Fact], list[str]]:
    rows = extract_facts(text, meta, client=client)
    facts, rejected = normalise_facts(rows, meta)
    problems = crossfoot(facts, ocr_pages) + check_segment_revenue_sums(facts)
    return facts, rejected + problems


# Documents that state company figures. Custodian statements have their own
# extractor, and a corporate action carries names (cfo_name), so it stays in.
FACT_DOC_TYPES = {"filing_10k", "filing_10q", "press_release", "transcript",
                  "presentation", "broker_note", "corporate_action"}


def facts_cached(extracted: dict, meta: dict, use_llm: bool = True, client=None
                 ) -> tuple[list[Fact], list[str]]:
    """Figures for one document, extracted at most once.

    The model's raw rows are cached by content hash, not the normalised facts,
    so the document's current id and metadata are applied on every load and
    the validation rules can change without paying for extraction again.
    Without a cached result and without use_llm, returns nothing rather than
    failing: the document is still searchable as text.
    """
    from app.ingest import cache

    if meta.get("doc_type") not in FACT_DOC_TYPES:
        return [], []
    content_hash = extracted["content_hash"]
    hit = cache.get(content_hash, "facts")
    if hit is not None:
        rows = hit.get("rows") or []
    elif use_llm:
        text = "\n\n".join(f"[page {p['page_no']}]\n{p['text']}"
                           for p in extracted["pages"])
        try:
            rows = extract_facts(text, meta, client=client)
        except Exception as exc:                        # noqa: BLE001
            log.warning("fact extraction failed for %s: %s",
                        meta.get("filename"), exc)
            return [], [f"extraction failed: {exc}"]
        cache.put(content_hash, "facts", {"rows": rows})
    else:
        return [], []

    ocr_pages = {p["page_no"] for p in extracted["pages"] if p.get("method") == "ocr"}
    facts, rejected = normalise_facts(rows, meta)
    problems = crossfoot(facts, ocr_pages) + check_segment_revenue_sums(facts)
    return facts, rejected + problems
