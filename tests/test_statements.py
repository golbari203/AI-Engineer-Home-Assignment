"""
Tests for custodian statement extraction.

Run with a stub client, so no API key and no network. What matters is not
whether the model transcribes well, but whether the validation gate behaves:

  * a transcription that disagrees with the page's own arithmetic is rejected
    and retried, because the extraction is wrong;
  * a page that disagrees with itself is kept and flagged, because the
    discrepancy is the finding.

The Banque Leman payload below carries the real Costco cost-basis gap. A test
that passed only on clean data would prove nothing.
"""

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.statements import (                   # noqa: E402
    check_cost_basis_internal, load_statement, normalise, normalise_type, validate,
)


class StubClient:
    """Returns a scripted payload per call, so retries can be tested."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        block = type("B", (), {"type": "text", "text": payload})
        return type("R", (), {"content": [block]})


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

LEMAN = {
    "custodian": "Banque Léman & Cie SA", "account_no": "CH-5521.03",
    "entity": "Harbor Oak 2019 Family Trust",
    "as_of_date": "2026-06-30", "period_start": "2026-01-01",
    "currency": "CHF", "price_currency": "USD", "fx_rate": 0.7801,
    "cash": 1126737.44, "total_value": 2286806.99,
    "beginning_value": None, "income": None, "market_change": None,
    "positions": [
        {"ticker": "UNH", "description": "UnitedHealth Group Incorporated",
         "quantity": 2600, "price": 253.63,
         "market_value": 514427.58, "cost_basis": 736029.24},
        {"ticker": "COST", "description": "Costco Wholesale Corporation",
         "quantity": 900, "price": 919.60,
         "market_value": 645641.96, "cost_basis": 679589.74},
    ],
    "transactions": [
        {"trade_date": "2026-03-11", "settle_date": "2026-03-12", "type": "BUY",
         "ticker": "COST", "quantity": 900, "price": 953.77,
         "amount": -858393.00, "currency": "USD", "fx_rate": 0.7943,
         "description": "Achat / Purchase"},
        {"trade_date": "2026-03-11", "settle_date": "2026-03-12", "type": "FEE",
         "ticker": "COST", "quantity": None, "price": None,
         "amount": -2145.98, "currency": "USD", "fx_rate": 0.7943,
         "description": "Courtage / Brokerage"},
        {"trade_date": None, "settle_date": "2026-05-23", "type": "DIV",
         "ticker": "UNH", "quantity": 2600, "price": None, "amount": 5668.00,
         "currency": "USD", "fx_rate": 0.7897,
         "description": "Dividende brut / Gross dividend"},
        {"trade_date": None, "settle_date": "2026-05-23", "type": "TAX",
         "ticker": "UNH", "quantity": None, "price": None, "amount": -850.20,
         "currency": "USD", "fx_rate": 0.7897,
         "description": "Impôt à la source US / US withholding"},
        {"trade_date": None, "settle_date": "2026-06-28", "type": "FEE",
         "ticker": None, "quantity": None, "price": None, "amount": -480.00,
         "currency": "CHF", "fx_rate": None,
         "description": "Frais de garde / Custody fee"},
    ],
}

NORTHGATE = {
    "custodian": "Northgate Securities LLC", "account_no": "4471-8820",
    "entity": "Harbor Oak Holdings LLC",
    "as_of_date": "2026-06-30", "period_start": "2026-04-01",
    "currency": "USD", "price_currency": "USD", "fx_rate": None,
    "cash": 4022308.25, "total_value": 17194255.75,
    "beginning_value": 17889004.25, "income": 30884.00,
    "market_change": -725632.50,
    "positions": [
        {"ticker": "AAPL", "description": "Apple Inc.", "quantity": 12000,
         "price": 209.07, "market_value": 2508840.00, "cost_basis": 1958160.00},
        {"ticker": "MSFT", "description": "Microsoft Corporation", "quantity": 6500,
         "price": 400.51, "market_value": 2603315.00, "cost_basis": 1941095.00},
        {"ticker": "NVDA", "description": "NVIDIA Corporation", "quantity": 14000,
         "price": 141.15, "market_value": 1976100.00, "cost_basis": 1091580.00},
        {"ticker": "AMZN", "description": "Amazon.com, Inc.", "quantity": 12000,
         "price": 170.90, "market_value": 2050800.00, "cost_basis": 1664970.00},
        {"ticker": "JPM", "description": "JPMorgan Chase & Co.", "quantity": 7250,
         "price": 231.89, "market_value": 1681202.50, "cost_basis": 1442460.00},
        {"ticker": "XOM", "description": "Exxon Mobil Corporation", "quantity": 11000,
         "price": 96.75, "market_value": 1064250.00, "cost_basis": 1151480.00},
        {"ticker": "COST", "description": "Costco Wholesale Corporation",
         "quantity": 1400, "price": 919.60,
         "market_value": 1287440.00, "cost_basis": 1114946.00},
    ],
    "transactions": [
        {"trade_date": None, "settle_date": "2026-05-31", "type": "DIV",
         "ticker": "JPM", "quantity": None, "price": None, "amount": 9425.00,
         "currency": "USD", "fx_rate": None, "description": "Qualified dividend"},
    ],
}


def _json(d: dict) -> str:
    return json.dumps(d)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_swiss_number_and_date_formats():
    raw = dict(LEMAN, cash="1'126'737.44", total_value="2'286'806.99",
               as_of_date="30.06.2026")
    s = normalise(raw)
    assert s["cash"] == 1126737.44
    assert s["total_value"] == 2286806.99
    assert s["as_of_date"] == dt.date(2026, 6, 30)


def test_negative_cash_survives():
    """The Swiss CHF current account runs negative at -4,800."""
    s = normalise(dict(LEMAN, cash="(4 800.00)"))
    assert s["cash"] == -4800.00


def test_transaction_types_normalise_across_languages():
    assert normalise_type("Achat / Purchase") == "BUY"
    assert normalise_type("Courtage / Brokerage") == "FEE"
    assert normalise_type("Frais de garde / Custody fee") == "FEE"
    assert normalise_type("Dividende brut / Gross dividend") == "DIV"
    assert normalise_type("Impôt à la source US / US withholding") == "TAX"
    assert normalise_type("Bought") == "BUY"


def test_withholding_is_not_read_as_a_dividend():
    """Both lines mention the dividend; only one is income."""
    assert normalise_type("US withholding tax on dividend") == "TAX"


# ---------------------------------------------------------------------------
# Validation: structural
# ---------------------------------------------------------------------------

def test_clean_statement_validates():
    assert validate(normalise(NORTHGATE)) == []


def test_foreign_currency_line_values_validate():
    """2,600 x 253.63 x 0.7801 = 514,427.58 exactly as printed."""
    issues = [i for i in validate(normalise(LEMAN)) if i.code == "line_value"]
    assert issues == []


def test_corrupted_market_value_is_rejected():
    bad = json.loads(_json(LEMAN))
    bad["positions"][0]["market_value"] = 541427.58      # digits transposed
    bad["total_value"] = 2313806.99
    issues = validate(normalise(bad))
    assert any(i.code == "line_value" and i.severity == "structural" for i in issues)


def test_missing_fx_rate_is_structural():
    bad = json.loads(_json(LEMAN))
    bad["fx_rate"] = None
    issues = validate(normalise(bad))
    assert any(i.code == "missing_fx" for i in issues)


def test_total_mismatch_is_rejected():
    bad = json.loads(_json(NORTHGATE))
    bad["cash"] = 4022308.25 + 1000
    issues = validate(normalise(bad))
    assert any(i.code == "statement_total" for i in issues)


def test_roll_forward_is_checked_when_components_are_printed():
    """17,889,004.25 + 30,884.00 - 725,632.50 = 17,194,255.75."""
    assert not any(i.code == "roll_forward" for i in validate(normalise(NORTHGATE)))
    bad = json.loads(_json(NORTHGATE))
    bad["income"] = 130884.00
    assert any(i.code == "roll_forward" for i in validate(normalise(bad)))


# ---------------------------------------------------------------------------
# Validation: data discrepancies are preserved, not rejected
# ---------------------------------------------------------------------------

def test_leman_costco_cost_basis_gap_is_found():
    """Purchase CHF 681,821.56 + CHF 1,704.55 brokerage vs stated CHF 679,589.74."""
    issues = check_cost_basis_internal(normalise(LEMAN))
    gap = [i for i in issues if i.code == "cost_basis_vs_movements"]
    assert len(gap) == 1
    assert gap[0].severity == "data"
    assert abs(gap[0].delta + 3936.37) < 1.0


def test_a_data_flag_does_not_block_storage():
    """The gap is printed on the page. Rejecting it would hide the finding."""
    result = load_statement("...", client=StubClient(_json(LEMAN)))
    assert result.is_storable
    assert [i.code for i in result.data_flags] == ["cost_basis_vs_movements"]
    assert result.data["positions"][1]["cost_basis"] == 679589.74   # unaltered


def test_either_cost_basis_convention_is_accepted():
    """Northgate excludes commission; some houses include it."""
    with_fee = json.loads(_json(LEMAN))
    with_fee["positions"][1]["cost_basis"] = 683526.11
    assert check_cost_basis_internal(normalise(with_fee)) == []


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

def test_structural_failure_triggers_a_retry():
    bad = json.loads(_json(NORTHGATE))
    bad["positions"][0]["market_value"] = 2508840.00 + 500
    result = load_statement("...", client=StubClient(_json(bad), _json(NORTHGATE)))
    assert result.is_storable
    assert result.data["positions"][0]["market_value"] == 2508840.00


def test_unparseable_output_is_structural():
    result = load_statement("...", client=StubClient("not json at all"), retries=0)
    assert not result.is_storable
    assert result.issues[0].code == "unparseable_json"


def test_a_data_flag_does_not_trigger_a_retry():
    """Re-asking cannot fix a contradiction printed on the page."""
    stub = StubClient(_json(LEMAN))
    load_statement("...", client=stub, retries=2)
    assert stub.calls == 1


# ---------------------------------------------------------------------------
# Round trip into the reconciliation layer
# ---------------------------------------------------------------------------

def test_extracted_statements_feed_reconcile():
    from app.ingest.workbooks import load_all
    from app.reconcile import run_all

    uploads = Path("/mnt/user-data/uploads")
    if not uploads.exists():
        return

    statements = [load_statement("...", client=StubClient(_json(p))).data
                  for p in (NORTHGATE, LEMAN)]
    for s in statements:                       # reconcile expects ISO strings
        s["as_of_date"] = str(s["as_of_date"])

    codes = {e.check_name for e in run_all(statements, load_all(uploads))}
    assert "quantity_mismatch" in codes        # JPM 7,250 vs 7,520
    assert "missing_in_books" in codes         # the Trust's 900 Costco shares
    assert "split_adjustment_required" in codes


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
