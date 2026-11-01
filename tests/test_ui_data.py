"""
Tests for the interface's data shaping.

The screens may only show what the files say. These tests pin the arithmetic
(weights, allocation, conversion), the grounding of every suggested question,
and the trust signals: which citations collapse, which answers are flagged as
unsupported, and what an error tells the investor.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ui_data as data                                 # noqa: E402
from app.ingest.classify import KNOWN_TICKERS                   # noqa: E402
from app.reconcile import Exception_                            # noqa: E402

UPLOADS = Path("/mnt/user-data/uploads")

HOLDINGS = {
    "total_portfolio_value_usd": 1000.0, "total_securities_usd": 800.0, "cash_usd": 200.0,
    "positions": [
        {"ticker": "COST", "custodian": "Northgate Securities LLC", "account": "A",
         "quantity": 1400, "price": 1.0, "market_value_usd": 300.0,
         "cost_basis": 10.0, "currency": "USD"},
        {"ticker": "COST", "custodian": "Banque Léman & Cie SA", "account": "B",
         "quantity": 900, "price": 1.0, "market_value_usd": 100.0,
         "cost_basis": 9.0, "currency": "CHF"},
        {"ticker": "MSFT", "custodian": "Northgate Securities LLC", "account": "A",
         "quantity": 10, "price": 1.0, "market_value_usd": 400.0,
         "cost_basis": 5.0, "currency": "USD"},
    ],
    "account_summary": [
        {"account": "A", "custodian": "Northgate Securities LLC", "currency": "USD",
         "cash_usd": 150.0, "stated_total": 850.0, "stated_total_usd": 850.0,
         "as_of_date": "2026-06-30"},
        {"account": "B", "custodian": "Banque Léman & Cie SA", "currency": "CHF",
         "cash_usd": 50.0, "stated_total": 117.0, "stated_total_usd": 150.0,
         "as_of_date": "2026-06-30"},
    ],
}
STATEMENTS = [{"account_no": "A", "entity": "Harbor Oak Holdings LLC", "fx_rate": None},
              {"account_no": "B", "entity": "Harbor Oak 2019 Family Trust", "fx_rate": 0.78}]
CATALOG = [
    {"filename": "Northgate statement 2026-06-30.pdf", "doc_type": "custodian_statement",
     "ticker": None, "title": "Northgate statement", "tier": 7, "published_date": None},
    {"filename": "Banque Leman - releve 30.06.2026.pdf", "doc_type": "custodian_statement",
     "ticker": None, "title": "Banque Leman", "tier": 7, "published_date": None},
    {"filename": "msft.pdf", "doc_type": "press_release", "ticker": "MSFT",
     "title": "Microsoft — Company release", "tier": 2, "published_date": dt.date(2026, 7, 27)},
    {"filename": "lly.pdf", "doc_type": "broker_note", "ticker": "LLY",
     "title": "Eli Lilly — Broker note", "tier": 6, "published_date": dt.date(2026, 5, 5)},
]
NOTES = [{"ticker": "LLY", "house": "Halvard", "rating": "Market Perform",
          "price_target": 705.0, "note_date": dt.date(2026, 5, 5), "analyst": "B"}]


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def test_every_company_the_classifier_knows_has_a_name():
    assert set(data.COMPANY_NAMES) == KNOWN_TICKERS


def test_companies_are_found_in_text_in_order_of_mention():
    assert data.tickers_in("Lilly beat; Microsoft and AWS lagged") == ["LLY", "MSFT", "AMZN"]
    assert data.tickers_in("cost of capital") == []        # 'cost' is not Costco


# ---------------------------------------------------------------------------
# Portfolio arithmetic
# ---------------------------------------------------------------------------

def test_concentration_combines_one_company_held_in_two_accounts():
    conc = data.concentration(HOLDINGS)
    by = {r["Ticker"]: r for r in conc["rows"]}
    assert by["COST"]["Value (USD)"] == 400.0 and by["COST"]["Accounts"] == 2
    assert by["COST"]["% of securities"] == 50.0
    assert conc["companies"] == 2
    assert conc["cash_pct"] == 20.0


def test_allocation_names_the_conversion_rate_only_for_foreign_accounts():
    rows = {r["account"]: r for r in data.account_allocation(HOLDINGS, STATEMENTS)}
    assert rows["A"]["fx_rate"] is None
    assert rows["B"]["fx_rate"] == 0.78
    assert rows["B"]["entity"] == "Harbor Oak 2019 Family Trust"
    assert rows["A"]["pct"] == 85.0


def test_holdings_keep_cost_basis_in_its_own_currency():
    rows = data.holdings_table(HOLDINGS)
    swiss = next(r for r in rows if r["Account"] == "B")
    assert swiss["Cost currency"] == "CHF" and swiss["Cost basis"] == 9.0
    assert [r["Ticker"] for r in rows] == ["MSFT", "COST", "COST"]


def test_search_ignores_case_and_accents():
    rows = data.holdings_table(HOLDINGS)
    assert len(data.filter_rows(rows, "leman")) == 1
    assert len(data.filter_rows(rows, "", Custodian="Northgate Securities LLC")) == 2
    assert data.filter_rows(rows, "tesla") == []


# ---------------------------------------------------------------------------
# Companies and suggestions
# ---------------------------------------------------------------------------

def test_watched_companies_are_kept_apart_from_owned_ones():
    universe = data.company_universe(HOLDINGS, CATALOG, NOTES, [])
    # Owned first, largest first; equal values (400 each here) fall back to name.
    assert [(c["ticker"], c["owned"]) for c in universe] == [
        ("COST", True), ("MSFT", True), ("LLY", False)]


def test_suggestions_name_only_companies_on_file():
    universe = data.company_universe(HOLDINGS, CATALOG, NOTES, [])
    exceptions = [Exception_("missing_in_books", "error", "x", ticker="COST")]
    groups = data.suggested_questions(HOLDINGS, exceptions, universe)
    text = " ".join(q for qs in groups.values() for q in qs)
    mentioned = set(data.tickers_in(text))
    assert mentioned <= {"MSFT", "COST", "LLY"}
    assert "Why don't my Costco records match the custodian?" in groups["Your records"]
    assert groups["Companies you're watching"] == ["What do analysts say about Eli Lilly?"]


def test_follow_ups_build_on_the_company_and_never_repeat_the_question():
    q = "What do analysts say about Eli Lilly?"
    out = data.follow_ups(q, "Three notes on Lilly…", owned={"MSFT"}, researched={"LLY"})
    assert q not in out and out
    assert all("Eli Lilly" in f for f in out)
    assert not any("own" in f for f in out)                    # not a holding


def test_a_portfolio_answer_gets_portfolio_follow_ups():
    answer = "Apple, Microsoft, NVIDIA, Amazon and Costco make up most of it."
    out = data.follow_ups("What is my portfolio worth?", answer, set(), set())
    assert "Is anything wrong with my records?" in out


# ---------------------------------------------------------------------------
# Trust signals
# ---------------------------------------------------------------------------

def test_an_answer_that_opens_with_nothing_on_file_is_flagged():
    assert data.is_unsupported("There is nothing on file about Meta's outlook.\n\nMore.")
    assert data.is_unsupported("**Your documents have no broker research on Exxon.**")


def test_a_caveat_deep_in_a_full_answer_is_not_flagged():
    answer = ("Lilly's 2025 revenue was **$49.3 billion**.\n\n- No Q4 release on file, "
              "so I couldn't cross-check it.")
    assert not data.is_unsupported(answer)


def test_statement_citations_collapse_into_one_card_per_statement():
    citations = [{"file": "Northgate Securities LLC statement", "date": "2026-06-30",
                  "detail": f"{t} position"} for t in ("AAPL", "MSFT", "COST")]
    citations.append({"file": "Banque Léman & Cie SA statement", "date": "2026-06-30",
                      "detail": "COST position"})
    cards = data.source_cards(citations, CATALOG)
    assert len(cards) == 2
    assert cards[0]["positions"] == ["AAPL", "MSFT", "COST"]
    assert cards[0]["file"] == "Northgate statement 2026-06-30.pdf"
    assert cards[1]["file"] == "Banque Leman - releve 30.06.2026.pdf"   # accent-insensitive


def test_document_citations_carry_reliability_and_flags():
    cards = data.source_cards([
        {"file": "msft.pdf", "tier": 2, "date": "2026-07-27", "pages": "1-2",
         "from_ocr": True, "other_versions": ["copy.pdf"]},
        {"file": "msft.pdf", "tier": 2, "date": "2026-07-27", "pages": "1-2"},
        {"file": "document 7", "date": None},
    ], CATALOG, filename_by_document={7: "lly.pdf"})
    assert len(cards) == 2
    assert cards[0]["kind"] == "Company release" and cards[0]["from_ocr"]
    assert cards[0]["other_versions"] == ["copy.pdf"]
    assert cards[1]["file"] == "lly.pdf" and cards[1]["kind"] == "Broker note"


def test_cited_pages_expand_to_a_list():
    assert data.page_numbers("3-5") == [3, 4, 5]
    assert data.page_numbers("2") == [2]
    assert data.page_numbers(None) == []


def test_errors_tell_the_investor_what_to_do():
    class AuthenticationError(Exception):
        pass
    what, hint = data.friendly_error(AuthenticationError("401"))
    assert "isn't connected" in what and "AIEHA_API_KEY" in hint
    what, _ = data.friendly_error(KeyError("AIEHA_API_KEY"))
    assert "isn't connected" in what
    what, hint = data.friendly_error(ValueError("boom"))
    assert "Try again" in hint and "boom" not in what

    class BadRequestError(Exception):
        pass
    what, hint = data.friendly_error(BadRequestError(
        "Error code: 400 - Your credit balance is too low to access the Anthropic API."))
    assert "run out of credit" in what and "Plans & Billing" in hint


def test_dates_read_as_day_month_year():
    assert data.fmt_date(dt.date(2026, 6, 30)) == "30 Jun 2026"
    assert data.fmt_date("2026-01-05") == "5 Jan 2026"
    assert data.fmt_date(None) == ""


# ---------------------------------------------------------------------------
# The real portfolio
# ---------------------------------------------------------------------------

def test_the_real_catalog_reads_every_pdf_and_marks_the_duplicate():
    if not UPLOADS.exists():
        return
    catalog = data.build_catalog(UPLOADS)
    assert len(catalog) == len(list(UPLOADS.glob("*.pdf")))
    duplicates = [d for d in catalog if d["duplicate_of"]]
    assert len(duplicates) == 1
    assert all(d["title"] != "UNITED STATES" for d in catalog)


def test_the_real_universe_owns_eight_and_watches_two():
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store
    from app.tools import run_tool
    store = build_store(UPLOADS, with_index=False)
    holdings = run_tool(store, "get_holdings", {})
    universe = data.company_universe(holdings, data.build_catalog(UPLOADS),
                                     store.broker_notes, store.price_targets)
    assert sum(c["owned"] for c in universe) == 8
    assert {c["ticker"] for c in universe if not c["owned"]} == {"LLY", "TSLA"}


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
            except Exception as e:                       # noqa: BLE001
                failed += 1
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'all tests passed' if not failed else f'{failed} failed'}")
