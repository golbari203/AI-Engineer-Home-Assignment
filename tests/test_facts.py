"""
Tests for fact extraction.

The cases are taken from the corpus, not invented, so a passing suite means the
real documents are handled. Three things are being proved:

  * a fact without a basis is dropped, because a bare 4.68 is a future error;
  * cross-footing catches the OCR digit error that 90% confidence missed;
  * conflicts are found by grouping, not by hoping retrieval returns both sides.
"""

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.facts import (                        # noqa: E402
    Fact, check_segment_revenue_sums, crossfoot, find_conflicts, load_facts,
    normalise_facts, quarter_end, resolve,
)


class StubClient:
    def __init__(self, payload):
        self.payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        block = type("B", (), {"type": "text", "text": self.payload})
        return type("R", (), {"content": [block]})


AMZN_META = {"ticker": "AMZN", "doc_type": "press_release",
             "published_date": "2026-01-29", "document_id": 11,
             "fiscal_period": "Q4 2025"}


# ---------------------------------------------------------------------------
# Basis and period are mandatory
# ---------------------------------------------------------------------------

def test_both_eps_bases_are_kept_as_separate_facts():
    """Lilly Q1 2026: $2.87 GAAP and $4.68 non-GAAP. Not a conflict."""
    rows = [
        {"metric": "eps_diluted", "value": 2.87, "unit": "USD",
         "basis": "GAAP", "fiscal_period": "Q1 2026"},
        {"metric": "eps_diluted", "value": 4.68, "unit": "USD",
         "basis": "non-GAAP", "fiscal_period": "Q1 2026"},
    ]
    facts, rejected = normalise_facts(rows, {"ticker": "LLY", "document_id": 1})
    assert len(facts) == 2 and not rejected
    assert find_conflicts(facts) == []


def test_invalid_basis_is_rejected():
    rows = [{"metric": "eps_diluted", "value": 4.68, "basis": "headline",
             "fiscal_period": "Q1 2026"}]
    facts, rejected = normalise_facts(rows, {"ticker": "LLY"})
    assert facts == [] and "invalid basis" in rejected[0]


def test_a_fact_without_a_period_is_dropped():
    rows = [{"metric": "revenue", "value": 13100, "basis": "GAAP"}]
    facts, rejected = normalise_facts(rows, {"ticker": "LLY"})
    assert facts == [] and "no fiscal period" in rejected[0]


def test_officer_names_need_no_period():
    rows = [{"metric": "cfo_name", "value": "Grace Lindqvist"}]
    facts, rejected = normalise_facts(rows, {"ticker": "MSFT"})
    assert len(facts) == 1 and facts[0].value_text == "Grace Lindqvist"


def test_unknown_metrics_are_rejected_not_invented():
    rows = [{"metric": "vibes", "value": 10, "fiscal_period": "Q1 2026"}]
    facts, rejected = normalise_facts(rows, {"ticker": "LLY"})
    assert facts == [] and "unknown metric" in rejected[0]


# ---------------------------------------------------------------------------
# Fiscal calendars
# ---------------------------------------------------------------------------

def test_fiscal_labels_resolve_per_company():
    """Apple's Q1 FY2026 closed in December 2025; Amazon's Q1 2026 in March."""
    assert quarter_end("AAPL", "Q1 FY2026").year == 2025
    assert quarter_end("AAPL", "Q1 FY2026").month == 12
    assert quarter_end("AMZN", "Q1 2026") == dt.date(2026, 3, 31)
    assert quarter_end("NVDA", "Q4 FY2026").month == 1
    assert quarter_end("MSFT", "Q2 FY2026").month == 12


def test_a_printed_period_end_beats_the_computed_one():
    rows = [{"metric": "revenue", "value": 104201, "basis": "GAAP",
             "fiscal_period": "Q1 FY2026", "period_end": "2025-12-27"}]
    facts, _ = normalise_facts(rows, {"ticker": "AAPL"})
    assert facts[0].period_end == dt.date(2025, 12, 27)


# ---------------------------------------------------------------------------
# Cross-footing
# ---------------------------------------------------------------------------

def _jpm(nii: float) -> list[Fact]:
    """JPMorgan Q4 2025, from the OCR'd page 3 of the earnings release."""
    return [
        Fact("JPM", "noninterest_revenue", "Q4 2025", value_num=15025,
             unit="USD_millions", basis="GAAP", page_no=3, document_id=7),
        Fact("JPM", "net_interest_income", "Q4 2025", value_num=nii,
             unit="USD_millions", basis="GAAP", page_no=3, document_id=7),
        Fact("JPM", "revenue", "Q4 2025", value_num=33233,
             unit="USD_millions", basis="GAAP", page_no=3, document_id=7),
    ]


def test_crossfoot_accepts_the_true_figures():
    assert crossfoot(_jpm(18208), ocr_pages={3}) == []


def test_crossfoot_catches_the_ocr_digit_error():
    """18,268 for 18,208: one digit, 90% confidence, undetectable by confidence."""
    facts = _jpm(18268)
    problems = crossfoot(facts, ocr_pages={3})
    assert len(problems) == 1
    assert "33,293" in problems[0] and "OCR" in problems[0]
    assert all(f.confidence <= 0.4 for f in facts)


def test_confidence_drops_less_on_a_clean_text_page():
    """A failed identity on a text page is a discrepancy, not a misread digit."""
    facts = _jpm(18268)
    crossfoot(facts, ocr_pages=set())
    assert all(abs(f.confidence - 0.7) < 1e-9 for f in facts)


def test_segment_revenue_sums_are_checked():
    """Amazon Q2 2026: 76,497 + 32,427 + 25,539 = 134,463 exactly."""
    facts = [
        Fact("AMZN", "revenue", "Q2 2026", value_num=76497, segment="North America"),
        Fact("AMZN", "revenue", "Q2 2026", value_num=32427, segment="International"),
        Fact("AMZN", "revenue", "Q2 2026", value_num=25539, segment="AWS"),
        Fact("AMZN", "revenue", "Q2 2026", value_num=134463),
    ]
    assert check_segment_revenue_sums(facts) == []


def test_segment_operating_income_is_not_cross_footed():
    """Amazon Q2 2026 shows 4,743 + 1,383 + 9,194 = 15,320 against a
    consolidated 13,523 and a segment-table total of 13,933. Three numbers,
    one document. Enforcing this identity would fail every Amazon filing."""
    facts = [
        Fact("AMZN", "operating_income", "Q2 2026", value_num=4743, segment="North America"),
        Fact("AMZN", "operating_income", "Q2 2026", value_num=1383, segment="International"),
        Fact("AMZN", "operating_income", "Q2 2026", value_num=9194, segment="AWS"),
        Fact("AMZN", "operating_income", "Q2 2026", value_num=13523),
    ]
    assert crossfoot(facts) == []
    assert check_segment_revenue_sums(facts) == []
    assert all(f.confidence == 1.0 for f in facts)


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------

def _aws_q4() -> list[Fact]:
    """The real four rows: press release, transcript, and both 10-Q versions."""
    return [
        Fact("AMZN", "revenue", "Q4 2025", value_num=23863, segment="AWS",
             document_id=11, stated_on=dt.date(2026, 1, 29)),
        Fact("AMZN", "revenue", "Q4 2025", value_num=24700, segment="AWS",
             document_id=19, stated_on=dt.date(2026, 1, 29)),
        Fact("AMZN", "revenue", "Q4 2025", value_num=23863, segment="AWS",
             document_id=20, stated_on=dt.date(2026, 5, 4)),
        Fact("AMZN", "revenue", "Q4 2025", value_num=24733, segment="AWS",
             document_id=20, stated_on=dt.date(2026, 5, 4), is_restated=True),
    ]


def test_aws_conflict_is_found_without_being_looked_for():
    conflicts = find_conflicts(_aws_q4())
    assert len(conflicts) == 1
    assert {23863, 24700, 24733} == {v.value_num for v in conflicts[0].versions}


def test_restated_figure_wins():
    """A restatement is the issuer saying the earlier number was wrong."""
    tiers = {11: 2, 19: 4, 20: 1}
    assert resolve(find_conflicts(_aws_q4())[0], tiers).value_num == 24733


def test_tier_decides_when_nothing_is_restated():
    """$41.4 billion beats $41.4 million because the corrected transcript
    outranks the auto-generated one."""
    facts = [
        Fact("NVDA", "guidance_revenue_low", "Q1 FY2027", value_num=41.4,
             unit="USD_millions", is_guidance=True, document_id=30,
             stated_on=dt.date(2026, 2, 18)),
        Fact("NVDA", "guidance_revenue_low", "Q1 FY2027", value_num=41400,
             unit="USD_millions", is_guidance=True, document_id=31,
             stated_on=dt.date(2026, 2, 18)),
    ]
    conflict = find_conflicts(facts)[0]
    assert resolve(conflict, {30: 4, 31: 3}).value_num == 41400


def test_officer_conflict_is_detected_like_any_other():
    """Three documents say Lindqvist; three later ones still say Keller."""
    facts = [
        Fact("MSFT", "cfo_name", None, value_text="Grace Lindqvist",
             document_id=40, stated_on=dt.date(2026, 2, 17)),
        Fact("MSFT", "cfo_name", None, value_text="Thomas Keller",
             document_id=41, stated_on=dt.date(2026, 7, 27)),
    ]
    assert len(find_conflicts(facts)) == 1


def test_same_value_from_many_sources_is_not_a_conflict():
    facts = [Fact("AMZN", "revenue", "Q4 2025", value_num=157254, document_id=i)
             for i in (11, 19, 20)]
    assert find_conflicts(facts) == []


def test_conflict_description_is_readable():
    text = find_conflicts(_aws_q4())[0].describe()
    assert "AWS" in text and "restated" in text and "vs" in text


# ---------------------------------------------------------------------------
# End to end with a stub
# ---------------------------------------------------------------------------

def test_load_facts_round_trip():
    payload = [
        {"metric": "revenue", "value": 157254, "unit": "USD_millions",
         "basis": "GAAP", "fiscal_period": "Q4 2025", "period_end": "2025-12-31"},
        {"metric": "revenue", "value": 23863, "unit": "USD_millions",
         "basis": "GAAP", "fiscal_period": "Q4 2025", "segment": "AWS"},
        {"metric": "eps_diluted", "value": 1.15, "unit": "USD",
         "basis": "GAAP", "fiscal_period": "Q4 2025"},
        {"metric": "revenue", "value": "not a number", "basis": "GAAP",
         "fiscal_period": "Q4 2025"},
    ]
    facts, problems = load_facts("...", AMZN_META, client=StubClient(payload))
    assert len(facts) == 3
    assert any("unparseable" in p for p in problems)
    assert facts[0].stated_on == dt.date(2026, 1, 29)
    assert facts[0].document_id == 11


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
