"""
Tests for the classifier.

The LLM path is exercised with a stub client, so these run with no API key and
no network. What is being tested is not whether the model is clever, but whether
the merge rules hold: the model must never be able to overturn a printed signal,
and must never set source_tier or event_key, because those decide which document
wins a conflict.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.classify import (                     # noqa: E402
    Classification, build_event_key, classify_document, detect_doc_type,
    detect_tier, detect_ticker, extract_signals, merge,
)

UPLOADS = Path("/mnt/user-data/uploads")


class StubClient:
    """Returns whatever JSON the test wants, and records that it was called."""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        block = type("B", (), {"type": "text", "text": self.payload})
        return type("R", (), {"content": [block]})


# ---------------------------------------------------------------------------
# Deterministic signals
# ---------------------------------------------------------------------------

def test_press_release_is_not_a_filing():
    """Every press release cites Form 10-K in its boilerplate.

    A bare match on "FORM 10-K" classified all 26 of them as filings.
    """
    text = ("Amazon Reports Fourth Quarter 2025 Results\n"
            "SEATTLE, WASHINGTON — January 29, 2026 — Amazon.com, Inc. "
            "(Nasdaq: AMZN) today announced financial results...\n"
            "Factors that could cause results to differ include those described "
            "under 'Risk Factors' in the Company's most recent Annual Report on "
            "Form 10-K and subsequent Quarterly Reports on Form 10-Q.")
    assert detect_doc_type(text) == "press_release"


def test_mentioning_an_earnings_call_is_not_a_transcript():
    """The Microsoft CFO release says the transition will be discussed on the call."""
    text = ("Microsoft announces Chief Financial Officer transition\n"
            "The transition does not change the company's financial outlook, "
            "which will be discussed on the fiscal third-quarter earnings call "
            "in April.")
    assert detect_doc_type(text) == "corporate_action"


def test_transcript_tier_comes_from_the_printed_marker():
    """This is the rule that makes $41.4 billion beat $41.4 million."""
    corrected = "CORRECTED TRANSCRIPT\nNVIDIA Corporation (Nasdaq: NVDA)"
    auto = ("CallScript.io · Earnings Call Transcript\nNVIDIA (NVDA) Q4 FY2026\n"
            "Transcript auto-generated with human review; may contain errors.")
    assert detect_tier("transcript", corrected) == 3
    assert detect_tier("transcript", auto) == 4
    assert detect_tier("transcript", corrected) < detect_tier("transcript", auto)


def test_company_name_resolves_the_ticker_when_no_symbol_is_printed():
    """Presentations never print a ticker."""
    assert detect_ticker("Tesla Investor Day 2026\nAustin, Texas") == "TSLA"
    assert detect_ticker("Lilly R&D Day\nIndianapolis — March 5, 2026") == "LLY"


def test_statements_get_no_ticker():
    """A statement covering eight positions has no single ticker.

    Before this rule Northgate was labelled AAPL and Banque Leman COST,
    purely because those rows appear first in the holdings table.
    """
    text = ("Northgate Securities LLC\nBrokerage Account Statement\n"
            "Holdings as of June 30, 2026\nAAPL Apple Inc. 12,000")
    c = extract_signals(text, "Northgate_statement_2026-06-30.pdf")
    assert c.doc_type == "custodian_statement"
    assert c.ticker is None
    assert c.fiscal_period is None
    assert c.event_key == "STATEMENT|2026-06-30"


# ---------------------------------------------------------------------------
# Event clustering
# ---------------------------------------------------------------------------

def test_two_vendors_of_one_call_share_an_event_key():
    a = build_event_key("transcript", "NVDA", "Q4 FY2026", dt.date(2026, 2, 18), None)
    b = build_event_key("transcript", "NVDA", "Q4 FY2026", dt.date(2026, 2, 18), None)
    assert a == b == "NVDA|earnings_call|Q4 FY2026"


def test_release_and_call_stay_separate():
    """Same company, same quarter, same day — but genuinely different sources."""
    release = build_event_key("press_release", "AMZN", "Q4 2025",
                              dt.date(2026, 1, 29), None)
    call = build_event_key("transcript", "AMZN", "Q4 2025",
                           dt.date(2026, 1, 29), None)
    assert release != call


# ---------------------------------------------------------------------------
# Merge rules
# ---------------------------------------------------------------------------

def test_llm_fills_blanks_only():
    signals = Classification(doc_type="press_release", ticker="UNH",
                             published_date=dt.date(2026, 7, 29),
                             fiscal_period=None, source_tier=2)
    out = merge(signals, {"doc_type": "transcript", "ticker": "AAPL",
                          "published_date": "2020-01-01", "fiscal_period": "Q2 2026"})
    assert out.doc_type == "press_release"               # not overturned
    assert out.ticker == "UNH"                           # not overturned
    assert out.published_date == dt.date(2026, 7, 29)    # not overturned
    assert out.fiscal_period == "Q2 2026"                # blank, so filled


def test_llm_cannot_set_tier_or_event_key():
    """Tier decides conflicts, so it never comes from the model."""
    signals = Classification(doc_type="transcript", ticker="NVDA",
                             fiscal_period="Q4 FY2026", source_tier=4)
    out = merge(signals, {"source_tier": 1, "event_key": "anything"})
    assert out.source_tier == 4
    assert out.event_key == "NVDA|earnings_call|Q4 FY2026"


def test_llm_is_not_called_when_signals_are_complete():
    text = ("Amazon Reports Fourth Quarter 2025 Results\n"
            "SEATTLE, WASHINGTON — January 29, 2026 — Amazon.com, Inc. "
            "(Nasdaq: AMZN) today announced financial results for its "
            "fourth quarter of 2025 ended December 31, 2025.")
    stub = StubClient("{}")
    c = classify_document(text, "AMZN-Q4-2025-Earnings-Release.pdf", client=stub)
    assert stub.calls == 0
    assert (c.doc_type, c.ticker, c.fiscal_period) == ("press_release", "AMZN", "Q4 2025")


def test_llm_is_called_for_ocr_damaged_text():
    """scan_20260730_1142.pdf: OCR turned "2026" into "aoey" and "30" into "$0"."""
    text = ("UnitedHealth Group incorporaied\n"
            "UnitedHealth Reports Second Quarter 2026 Results\n"
            "MINNETONKA, MINNESOTA — July 29, 2026 — UnitedHealth Group "
            "Incorporated (NYSE: UNH) today 'announced financial results for "
            "its second quarter of aoey ended June $0, 2026.")
    stub = StubClient('{"fiscal_period": "Q2 2026", "as_of_date": "2026-06-30"}')
    c = classify_document(text, "scan_20260730_1142.pdf", client=stub)
    assert stub.calls == 1
    assert c.fiscal_period == "Q2 2026"
    assert c.as_of_date == dt.date(2026, 6, 30)
    assert c.ticker == "UNH"            # still from the printed symbol
    assert c.source_tier == 2


def test_code_fenced_json_is_parsed():
    from app.ingest.classify import classify_with_llm
    stub = StubClient('```json\n{"ticker": "LLY"}\n```')
    assert classify_with_llm("text", client=stub) == {"ticker": "LLY"}


# ---------------------------------------------------------------------------
# Corpus-level checks (skipped if the portfolio folder is absent)
# ---------------------------------------------------------------------------

def test_corpus_coverage():
    if not UPLOADS.exists():
        return
    from app.ingest.cache import head_text
    from app.ingest.classify import _is_complete
    resolved = 0
    pdfs = sorted(UPLOADS.glob("*.pdf"))
    for p in pdfs:
        if _is_complete(classify_document(head_text(p, 2), p.name, use_llm=False)):
            resolved += 1
    assert resolved >= len(pdfs) - 2, f"only {resolved}/{len(pdfs)} resolved"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n{'all tests passed' if not failed else f'{failed} failed'}")
