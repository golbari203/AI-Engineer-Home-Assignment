"""
Tests for the UI helpers and incremental ingest.

Streamlit itself is not tested; everything with logic lives in plain functions
so that the untested part is presentation only.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.filenames import resolve                               # noqa: E402
from app.reconcile import Exception_                            # noqa: E402
from app.ui import (                                            # noqa: E402
    answer_caption, as_of_label, dedupe_citations, format_citation,
    holdings_rows, money_short, portfolio_summary, quality_items,
    severity_badge, tools_caption,
)

UPLOADS = Path("/mnt/user-data/uploads")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def test_errors_outrank_warnings_in_the_badge():
    """A discrepancy the investor only sees if they happen to ask is one they
    will miss, so the badge is permanent and leads with the worst item."""
    exceptions = [Exception_("a", "warning", "x"), Exception_("b", "error", "y")]
    assert severity_badge(exceptions) == ("1 discrepancy found in your "
                                          "records", "error")


def test_badge_counts_read_as_plain_english():
    errors = [Exception_("a", "error", "x"), Exception_("b", "error", "y")]
    assert severity_badge(errors)[0] == "2 discrepancies found in your records"
    assert severity_badge([Exception_("a", "warning", "x")])[0] == "1 item to review"


def test_a_clean_portfolio_reads_as_clean():
    assert severity_badge([]) == ("Records reconcile", "ok")


def test_the_real_portfolio_shows_errors():
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store
    label, level = severity_badge(build_store(with_index=False).exceptions)
    assert level == "error"
    assert "3 discrepancies" in label


def test_summary_separates_securities_from_cash():
    rows = dict(portfolio_summary({
        "total_portfolio_value_usd": 20_125_684, "total_securities_usd": 14_659_025,
        "cash_usd": 5_466_658, "positions": [1] * 9, "accounts": ["a", "b"]}))
    assert rows["Total value"] == "$20,125,684"
    assert rows["Securities"] == "$14,659,025"
    assert rows["Cash"] == "$5,466,658"
    assert rows["Positions"] == "9"


def test_summary_survives_missing_values():
    assert dict(portfolio_summary({}))["Total value"] == "-"


def test_short_money_fits_a_narrow_column():
    assert money_short(14_659_025) == "$14.7m"
    assert money_short(2_286_806_990) == "$2.29bn"
    assert money_short(None) == "-"


def test_as_of_reads_as_a_date_and_admits_disagreement():
    same = {"account_summary": [{"as_of_date": "2026-06-30"}],
            "positions": [{"as_of_date": "2026-06-30"}]}
    assert as_of_label(same) == "30 Jun 2026"
    mixed = {"account_summary": [{"as_of_date": "2026-06-30"},
                                 {"as_of_date": "2026-05-31"}]}
    assert as_of_label(mixed) == "several dates"
    assert as_of_label({}) == "date unknown"


def test_holdings_keep_each_account_separate_and_largest_first():
    """Costco sits in two accounts; merging the rows would hide which
    custodian holds what."""
    rows = holdings_rows({
        "total_portfolio_value_usd": 1000,
        "positions": [
            {"ticker": "COST", "custodian": "A", "market_value_usd": 100},
            {"ticker": "AAPL", "custodian": "A", "market_value_usd": 500},
            {"ticker": "COST", "custodian": "B", "market_value_usd": 200},
        ]})
    assert [(r["Ticker"], r["Held at"]) for r in rows] == [
        ("AAPL", "A"), ("COST", "B"), ("COST", "A")]
    assert rows[0]["% of portfolio"] == 50.0


def test_record_findings_are_worst_first_and_in_plain_words():
    items = quality_items({
        "info": [{"check": "split_adjustment_required", "ticker": "NVDA",
                  "description": "split"}],
        "errors": [{"check": "quantity_mismatch", "ticker": "JPM",
                    "description": "7,250 vs 7,520"}],
        "warnings": [{"check": "some_new_check", "description": "w"}],
    })
    assert [i["severity"] for i in items] == ["error", "warning", "info"]
    assert items[0]["title"] == "JPM: Share count differs"
    assert items[1]["title"] == "Some new check"
    assert all("_" not in i["title"] for i in items)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def test_a_citation_names_the_document_and_date():
    line = format_citation({"file": "Northgate_statement_2026-06-30.pdf",
                            "date": "2026-06-30", "pages": "1-1"})
    assert "Northgate_statement_2026-06-30.pdf" in line and "2026-06-30" in line


def test_ocr_text_is_marked_in_the_source_list():
    line = format_citation({"file": "JPMorgan_Chase_4Q25_earnings.pdf",
                            "from_ocr": True})
    assert "recovered from a scan" in line


def test_a_suppressed_duplicate_is_surfaced_to_the_reader():
    """The investor should be able to see that a second transcript exists."""
    line = format_citation({"file": "NVIDIA_Q4_FY2026_earnings_call_-_Verbatim.pdf",
                            "other_versions": ["NVDA_Q4_FY26_call_transcript.pdf"]})
    assert "another version exists" in line
    assert "NVDA_Q4_FY26_call_transcript.pdf" in line


def test_repeated_citations_collapse():
    citations = [{"file": "a.pdf", "pages": "1-1"}, {"file": "a.pdf", "pages": "1-1"},
                 {"file": "a.pdf", "pages": "2-3"}]
    assert len(dedupe_citations(citations)) == 2


def test_tools_caption_is_written_for_a_non_technical_reader():
    class T:
        tools_used = ["get_holdings", "get_data_quality", "get_holdings"]
    caption = tools_caption(T())
    assert caption == "Checked your positions, record checks"
    assert "get_" not in caption


def test_answer_caption_leaves_out_token_counts():
    class T:
        tools_used = ["get_holdings"]
        latency_ms = 3200
        input_tokens, output_tokens = 4000, 300
    assert answer_caption(T()) == "Checked your positions · 3.2s"


# ---------------------------------------------------------------------------
# Incremental ingest
# ---------------------------------------------------------------------------

def test_the_same_file_under_a_new_name_is_skipped():
    """Re-sending a document with a different filename is routine at a family
    office. The content hash, not the name, decides."""
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store
    from app.pipeline import ingest_file, register_corpus

    store = build_store(with_index=True)
    register_corpus(store, UPLOADS)
    before = len(store.retrieval_backend.records)

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "renamed.pdf"
        shutil.copy(resolve(UPLOADS, "Microsoft_CFO_transition.pdf"), copy)
        result = ingest_file(copy, store)

    assert result.status == "duplicate"
    assert len(store.retrieval_backend.records) == before


def test_ingest_splices_into_the_live_index_without_a_restart():
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store
    from app.pipeline import ingest_file, register_corpus

    store = build_store(with_index=True)
    register_corpus(store, UPLOADS)
    before = len(store.retrieval_backend.records)

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "renamed.pdf"
        shutil.copy(resolve(UPLOADS, "Microsoft_CFO_transition.pdf"), copy)
        store._content_hashes = set()                    # pretend it is unseen
        result = ingest_file(copy, store)

    assert result.status == "ingested"
    assert result.ticker == "MSFT"
    assert result.doc_type == "corporate_action"
    assert len(store.retrieval_backend.records) == before + result.chunks


def test_a_duplicate_event_is_still_collapsed_at_retrieval():
    """Belt and braces: if a copy does reach the index, event-key dedup stops
    the investor seeing the same announcement twice."""
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store
    from app.pipeline import ingest_file, register_corpus
    from app.tools import run_tool

    store = build_store(with_index=True)
    register_corpus(store, UPLOADS)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "renamed.pdf"
        shutil.copy(resolve(UPLOADS, "Microsoft_CFO_transition.pdf"), copy)
        store._content_hashes = set()
        ingest_file(copy, store)

    hits = run_tool(store, "search_documents",
                    {"query": "chief financial officer steps down",
                     "ticker": "MSFT"})
    names = [s["file"] for s in hits["sources"]]
    original = resolve(UPLOADS, "Microsoft_CFO_transition.pdf").name
    assert not ("renamed.pdf" in names and original in names)


def test_a_non_pdf_is_refused_cleanly():
    from app.pipeline import ingest_file
    from app.tools import Store
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notes.txt"
        path.write_text("hello")
        result = ingest_file(path, Store())
        assert result.status == "unsupported"
        assert "unsupported" in result.summary()


def test_ingest_summary_mentions_ocr_pages():
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store
    from app.pipeline import ingest_file

    store = build_store(with_index=True)
    result = ingest_file(resolve(UPLOADS, "ExxonMobil_2Q25_earnings.pdf"), store)
    assert result.ocr_pages == 5
    assert "OCR" in result.summary()


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
