"""
Tests for the tools and the agent loop.

The store is built from the real portfolio folder when it is present, so these
assert on actual figures rather than invented ones. The agent is driven by a
stub client: what is being tested is the loop, the dispatch and the accounting,
not the model's judgement.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import SYSTEM_PROMPT, ask                        # noqa: E402
from app.ingest.facts import Fact                               # noqa: E402
from app.tools import (                                         # noqa: E402
    TOOL_SCHEMAS, Store, run_tool,
)

UPLOADS = Path("/mnt/user-data/uploads")
_STORE = None


def store(with_index: bool = False) -> Store | None:
    global _STORE
    if not UPLOADS.exists():
        return None
    if _STORE is None:
        from app.offline_store import build_store
        _STORE = build_store(with_index=with_index)
    return _STORE


# ---------------------------------------------------------------------------
# get_holdings
# ---------------------------------------------------------------------------

def test_costco_is_complete_across_both_accounts():
    """1,400 at Northgate plus 900 at Banque Leman. The headline failure of a
    retrieval-only system is answering 1,400."""
    s = store()
    if not s:
        return
    out = run_tool(s, "get_holdings", {"ticker": "COST"})
    assert out["total_quantity"] == 2300
    assert out["held_in_accounts"] == 2
    assert len(out["accounts"]) == 2


def test_foreign_currency_is_converted_at_the_statement_rate():
    """CHF 645,641.96 at 0.7801 is USD 827,640."""
    s = store()
    if not s:
        return
    out = run_tool(s, "get_holdings", {"ticker": "COST"})
    chf = [p for p in out["positions"] if p["currency"] == "CHF"][0]
    assert abs(chf["market_value_usd"] - 827639.99) < 1.0
    assert abs(out["total_securities_usd"] - 2115079.99) < 1.0


def test_a_missing_position_is_reported_not_hidden():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_holdings", {"ticker": "COST"})
    chf = [p for p in out["positions"] if p["currency"] == "CHF"][0]
    assert "does not appear in the family office position file" in \
        chf["book_disagreement"]
    assert out["data_quality_warnings"]


def test_jpm_disagreement_travels_with_the_position():
    """The custodian says 7,250; the book says 7,520. Both are reported."""
    s = store()
    if not s:
        return
    out = run_tool(s, "get_holdings", {"ticker": "JPM"})
    position = out["positions"][0]
    assert position["quantity"] == 7250                  # custodian wins
    assert "7,520" in position["book_disagreement"]      # book is not hidden


def test_whole_portfolio_separates_securities_from_cash():
    """Cash is a quarter of this portfolio. Reporting securities alone would
    answer "what is it worth?" with 75% of the number."""
    s = store()
    if not s:
        return
    out = run_tool(s, "get_holdings", {})
    assert len(out["positions"]) == 9                    # 7 + 2
    assert abs(out["total_securities_usd"] - 14_659_025.49) < 1
    assert abs(out["cash_usd"] - 5_466_911.14) < 2_000
    assert 19_500_000 < out["total_portfolio_value_usd"] < 20_500_000


def test_account_summary_carries_the_statement_totals():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_holdings", {})
    assert {a["account"] for a in out["account_summary"]} == \
        {"4471-8820", "CH-5521.03"}
    northgate = [a for a in out["account_summary"] if a["currency"] == "USD"][0]
    assert northgate["stated_total"] == 17_194_255.75


def test_a_ticker_query_omits_cash():
    s = store()
    if not s:
        return
    assert "cash_usd" not in run_tool(s, "get_holdings", {"ticker": "COST"})


def test_an_unheld_ticker_says_so():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_holdings", {"ticker": "TSLA"})
    assert out["positions"] == []
    assert "no position found" in out["note"]


# ---------------------------------------------------------------------------
# get_transactions
# ---------------------------------------------------------------------------

def test_nvidia_nets_to_the_held_position_after_the_split():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_transactions", {"ticker": "NVDA"})
    assert out["net_shares"] == 14000
    assert "-2,200" in out["note"]


def test_only_position_affecting_lines_are_restated():
    """Adjusting a dividend turns 1,800 at $0.01 into 18,000 at $0.001, a
    per-share rate that never existed."""
    s = store()
    if not s:
        return
    out = run_tool(s, "get_transactions", {"ticker": "NVDA", "type": "DIV"})
    early = [t for t in out["transactions"] if t["trade_date"] < "2025-06-10"]
    assert early and all(t["quantity"] == 1800 for t in early)
    assert all(not t["split_adjusted"] for t in early)


def test_buys_are_restated():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_transactions", {"ticker": "NVDA", "type": "BUY"})
    assert out["transactions"][0]["quantity"] == 18000
    assert out["transactions"][0]["price"] == 77.97


# ---------------------------------------------------------------------------
# get_facts
# ---------------------------------------------------------------------------

def _aws_facts() -> list[Fact]:
    return [
        Fact("AMZN", "revenue", "Q4 2025", value_num=23863, segment="AWS",
             document_id=11, stated_on=dt.date(2026, 1, 29), unit="USD_millions"),
        Fact("AMZN", "revenue", "Q4 2025", value_num=24700, segment="AWS",
             document_id=19, stated_on=dt.date(2026, 1, 29), unit="USD_millions"),
        Fact("AMZN", "revenue", "Q4 2025", value_num=24733, segment="AWS",
             document_id=20, stated_on=dt.date(2026, 5, 4), is_restated=True,
             unit="USD_millions"),
        Fact("AMZN", "eps_diluted", "Q4 2025", value_num=1.15, basis="GAAP",
             document_id=11, stated_on=dt.date(2026, 1, 29), unit="USD"),
    ]


def test_conflicts_are_returned_not_resolved_away():
    s = Store(facts=_aws_facts(), tier_by_document={11: 2, 19: 4, 20: 1})
    out = run_tool(s, "get_facts", {"ticker": "AMZN", "metric": "revenue"})
    assert len(out["conflicts"]) == 1
    assert len(out["conflicts"][0]["versions"]) == 3
    assert out["conflicts"][0]["authoritative"]["value"] == 24733


def test_undisputed_facts_are_not_listed_as_conflicts():
    s = Store(facts=_aws_facts(), tier_by_document={11: 2})
    out = run_tool(s, "get_facts", {"ticker": "AMZN", "metric": "eps_diluted"})
    assert out["conflicts"] == []
    assert out["facts"][0]["basis"] == "GAAP"


def test_missing_facts_point_at_the_other_tool():
    out = run_tool(Store(), "get_facts", {"ticker": "XOM"})
    assert "search_documents" in out["note"]


# ---------------------------------------------------------------------------
# get_research
# ---------------------------------------------------------------------------

def test_lilly_targets_are_reported_not_averaged():
    """Buy at 890, Overweight at 820, Market Perform at 705, in 33 days, on
    three incomparable scales. A mean of those means nothing."""
    s = store()
    if not s:
        return
    out = run_tool(s, "get_research", {"ticker": "LLY"})
    assert len(out["notes_on_file"]) == 3
    assert out["target_range"] == {"low": 705.0, "high": 890.0,
                                   "note": "targets are not averaged because "
                                           "the scales differ"}
    assert all("not comparable" in n["scale_note"] for n in out["notes_on_file"])


def test_street_aggregate_is_kept_separate_from_the_notes():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_research", {"ticker": "LLY"})
    assert out["street_aggregate"]["consensus_price_target"] == 902.0
    assert "not a summary of the notes on file" in out["street_aggregate"]["note"]


def test_a_company_with_no_research_says_so():
    s = store()
    if not s:
        return
    assert "no broker research" in run_tool(s, "get_research", {"ticker": "XOM"})["note"]


# ---------------------------------------------------------------------------
# get_consensus
# ---------------------------------------------------------------------------

def test_consensus_carries_the_company_eps_basis():
    s = store()
    if not s:
        return
    assert run_tool(s, "get_consensus", {"ticker": "UNH"})["eps_basis"] == "Adjusted"
    assert run_tool(s, "get_consensus", {"ticker": "NVDA"})["eps_basis"] == "Non-GAAP"
    assert run_tool(s, "get_consensus", {"ticker": "XOM"})["eps_basis"] == \
        "Ex identified items"


def test_a_period_after_the_file_date_is_marked_unreported():
    """UNH Q2 2026 was reported on 29 July; the consensus file is 30 June."""
    s = store()
    if not s:
        return
    out = run_tool(s, "get_consensus", {"ticker": "UNH", "fiscal_period": "Q2 2026"})
    assert out["periods"][0]["status"] == "not yet reported as of the file date"
    assert "2026-06-30" in out["note"]


def test_a_reported_period_shows_the_surprise():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_consensus", {"ticker": "UNH", "fiscal_period": "Q4 2025"})
    assert out["periods"][0]["eps_surprise_pct"] is not None


# ---------------------------------------------------------------------------
# get_data_quality
# ---------------------------------------------------------------------------

def test_exceptions_are_grouped_by_severity():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_data_quality", {})
    assert out["exception_count"] == 7
    assert len(out["errors"]) == 3


def test_exceptions_can_be_filtered_to_one_holding():
    s = store()
    if not s:
        return
    out = run_tool(s, "get_data_quality", {"ticker": "JPM"})
    assert out["exception_count"] >= 2
    assert all(e["ticker"] == "JPM" for e in out["errors"])


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_every_schema_has_an_implementation():
    from app.tools import DISPATCH
    assert {t["name"] for t in TOOL_SCHEMAS} == set(DISPATCH)


def test_unknown_tool_is_an_error_not_a_crash():
    assert "unknown tool" in run_tool(Store(), "get_weather", {})["error"]


def test_bad_arguments_are_an_error_not_a_crash():
    assert "bad arguments" in run_tool(Store(), "get_facts", {"nope": 1})["error"]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

class Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ScriptedClient:
    """Replays a sequence of responses, recording usage like the real client."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _tool_use(name, inp, tool_id="t1"):
    return Block(content=[Block(type="tool_use", id=tool_id, name=name, input=inp)],
                 stop_reason="tool_use",
                 usage=Block(input_tokens=1200, output_tokens=80))


def _final(text):
    return Block(content=[Block(type="text", text=text)],
                 stop_reason="end_turn",
                 usage=Block(input_tokens=1800, output_tokens=150))


def test_the_loop_calls_a_tool_then_answers():
    s = store()
    if not s:
        return
    client = ScriptedClient(_tool_use("get_holdings", {"ticker": "COST"}),
                            _final("You hold 2,300 Costco shares."))
    turn, messages = ask("How much Costco do I own?", s, client=client)
    assert turn.tools_used == ["get_holdings"]
    assert "2,300" in turn.answer
    assert turn.iterations == 2
    assert len(messages) == 4         # user, assistant, tool_result, assistant


def test_token_usage_is_accumulated_across_iterations():
    """Cost in DESIGN.md is measured, not estimated."""
    s = store()
    if not s:
        return
    client = ScriptedClient(_tool_use("get_holdings", {}), _final("done"))
    turn, _ = ask("q", s, client=client)
    assert turn.input_tokens == 3000
    assert turn.output_tokens == 230
    assert turn.latency_ms >= 0


def test_the_loop_stops_rather_than_looping_forever():
    s = store()
    if not s:
        return
    client = ScriptedClient(_tool_use("get_holdings", {}))
    turn, _ = ask("q", s, client=client, max_iterations=3)
    assert turn.iterations == 3
    assert "smaller parts" in turn.answer


def test_a_failing_tool_is_reported_back_not_raised():
    s = store()
    if not s:
        return
    client = ScriptedClient(_tool_use("get_nonsense", {}),
                            _final("I cannot answer that from your documents."))
    turn, _ = ask("q", s, client=client)
    assert turn.answer.startswith("I cannot")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def test_the_prompt_states_the_rules_that_the_data_requires():
    for phrase in ("override anything you think you know",
                   "never from your own arithmetic",
                   "basis and the period",
                   "show the disagreement",
                   "most specific dated statement wins",
                   "say so"):
        assert phrase in SYSTEM_PROMPT, phrase


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
