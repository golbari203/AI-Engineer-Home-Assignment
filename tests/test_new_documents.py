"""
Tests for documents that arrive after setup.

A new file must get the same treatment as the original corpus: identified,
indexed, and its figures extracted, even when it is about a company outside
the original ten. The model is replaced by a stub that records its calls, so
these run without a key and prove each document is sent to it at most once.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest import cache                                    # noqa: E402
from app.ingest.classify import (                               # noqa: E402
    KNOWN_TICKERS, classify_cached, clean_company_name, detect_ticker, merge,
    extract_signals,
)
from app.ingest.facts import facts_cached, parse_rows           # noqa: E402

RELEASE = """Zephyr Robotics Reports Second Quarter 2026 Results
SAN JOSE, Calif., July 30, 2026 (NASDAQ: ZPHR) -- Zephyr Robotics, Inc. today
announced results for the second quarter of 2026, ended June 30, 2026.
Revenue was $412 million, up 18% from a year ago. GAAP diluted EPS was $0.42
and non-GAAP diluted EPS was $0.55. Unlike Apple, Zephyr sells to factories."""

CLASSIFY_REPLY = {"doc_type": "press_release", "ticker": "ZPHR",
                  "company": "Zephyr Robotics, Inc.", "publisher": None,
                  "title": "Zephyr Robotics Q2 2026 results",
                  "published_date": "2026-07-30", "as_of_date": "2026-06-30",
                  "fiscal_period": "Q2 2026"}
FACT_ROWS = [
    {"metric": "revenue", "value": 412, "unit": "USD_millions", "basis": "GAAP",
     "fiscal_period": "Q2 2026", "period_end": "2026-06-30", "page": 1},
    {"metric": "eps_diluted", "value": 0.42, "unit": "USD", "basis": "GAAP",
     "fiscal_period": "Q2 2026", "page": 1},
    {"metric": "eps_diluted", "value": 0.55, "unit": "USD", "basis": "non-GAAP",
     "fiscal_period": "Q2 2026", "page": 1},
    {"metric": "eps_diluted", "value": 0.50, "unit": "USD",          # invented basis
     "basis": "pro forma-ish", "fiscal_period": "Q2 2026"},
]


class StubClient:
    """Answers like the model would, and counts how often it was asked."""

    def __init__(self):
        self.calls = {"classify": 0, "facts": 0}
        self.messages = self

    def create(self, model, max_tokens, system, messages):
        kind = "classify" if system.startswith("You classify") else "facts"
        self.calls[kind] += 1
        reply = CLASSIFY_REPLY if kind == "classify" else FACT_ROWS
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(reply))])


class TempCache:
    """Point the content-hash cache at a scratch folder for one test."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved, cache.CACHE_DIR = cache.CACHE_DIR, Path(self.tmp.name)
        return self

    def __exit__(self, *exc):
        cache.CACHE_DIR = self.saved
        self.tmp.cleanup()


def write_pdf(path: Path, text: str) -> None:
    """A minimal one-page PDF with a real text layer."""
    lines = [l.replace("(", r"\(").replace(")", r"\)") for l in text.splitlines()]
    content = "BT /F1 10 Tf 13 TL 50 760 Td " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    stream = content.encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offsets:
        out += b"%010d 00000 n \n" % o
    out += b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# Identifying a company outside the original ten
# ---------------------------------------------------------------------------

def test_an_exchange_listing_names_a_new_company():
    assert "ZPHR" not in KNOWN_TICKERS
    assert detect_ticker(RELEASE) == "ZPHR"


def test_a_passing_mention_of_a_known_company_does_not_win():
    """The release mentions Apple; it is still Zephyr's release."""
    assert extract_signals(RELEASE, "zephyr.pdf").ticker == "ZPHR"


def test_known_companies_are_unchanged():
    assert detect_ticker("Apple Inc. (Nasdaq: AAPL) today announced") == "AAPL"


def test_legal_suffixes_are_dropped_from_names():
    assert clean_company_name("Zephyr Robotics, Inc.") == "Zephyr Robotics"
    assert clean_company_name("Acme Corporation") == "Acme"
    assert clean_company_name("  ") is None


def test_a_model_ticker_is_normalised_or_refused():
    signals = extract_signals("nothing useful here", "x.pdf")
    assert merge(signals, {"ticker": "NYSE: brk.b"}).ticker == "BRKB"
    assert merge(signals, {"ticker": "not a ticker"}).ticker is None


def test_the_model_is_asked_once_and_remembered():
    with TempCache():
        client = StubClient()
        first = classify_cached(RELEASE, "zephyr.pdf", "hash-1", client=client)
        again = classify_cached(RELEASE, "zephyr.pdf", "hash-1", use_llm=False)
        assert first.ticker == again.ticker == "ZPHR"
        assert first.company == again.company == "Zephyr Robotics"
        assert client.calls["classify"] == 1


def test_a_complete_known_document_never_calls_the_model():
    text = ("Apple Inc. (Nasdaq: AAPL) Reports Fourth Quarter Results\n"
            "CUPERTINO, Calif., October 24, 2025 -- Apple today announced financial "
            "results for its fiscal 2025 fourth quarter ended September 27, 2025.")
    with TempCache():
        client = StubClient()
        c = classify_cached(text, "apple.pdf", "hash-2", client=client)
        if c.doc_type and c.published_date and c.fiscal_period:
            assert client.calls["classify"] == 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _extracted(text: str, h: str = "hash-3") -> dict:
    return {"content_hash": h, "pages": [{"page_no": 1, "text": text, "method": "text_layer"}]}


def test_figures_keep_their_basis_and_drop_rows_with_an_invented_one():
    meta = {"ticker": "ZPHR", "doc_type": "press_release", "document_id": 7,
            "published_date": "2026-07-30"}
    with TempCache():
        client = StubClient()
        facts, notes = facts_cached(_extracted(RELEASE), meta, client=client)
        assert len(facts) == 3
        eps = {f.basis: f.value_num for f in facts if f.metric == "eps_diluted"}
        assert eps == {"GAAP": 0.42, "non-GAAP": 0.55}
        assert any("invalid basis" in n for n in notes)
        assert all(f.document_id == 7 and f.ticker == "ZPHR" for f in facts)


def test_figures_are_extracted_once_and_reloaded_free():
    meta = {"ticker": "ZPHR", "doc_type": "press_release", "document_id": 1}
    with TempCache():
        client = StubClient()
        facts_cached(_extracted(RELEASE), meta, client=client)
        reloaded, _ = facts_cached(_extracted(RELEASE), {**meta, "document_id": 9},
                                   use_llm=False)
        assert client.calls["facts"] == 1
        assert {f.document_id for f in reloaded} == {9}          # current id applied


def test_without_a_key_or_cache_a_document_simply_has_no_figures():
    with TempCache():
        facts, notes = facts_cached(_extracted(RELEASE, "hash-4"),
                                    {"ticker": "ZPHR", "doc_type": "press_release"},
                                    use_llm=False)
        assert facts == [] and notes == []


def test_statements_are_not_sent_for_figure_extraction():
    with TempCache():
        client = StubClient()
        facts_cached(_extracted("statement"), {"doc_type": "custodian_statement"},
                     client=client)
        assert client.calls["facts"] == 0


def test_a_cut_off_reply_keeps_its_complete_rows():
    raw = '[{"metric": "revenue", "value": 1}, {"metric": "net_income", "value": 2}, {"metr'
    assert [r["metric"] for r in parse_rows(raw)] == ["revenue", "net_income"]


# ---------------------------------------------------------------------------
# End to end: a new company's PDF arriving in the folder
# ---------------------------------------------------------------------------

def test_a_new_companys_pdf_is_identified_indexed_and_its_figures_extracted():
    from app.ingest.embed import get_embedder
    from app.pipeline import ingest_file
    from app.tools import Store, run_tool

    with TempCache(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Zephyr Q2 2026.pdf"
        write_pdf(path, RELEASE + "\n" + "\n".join(
            f"Segment note {i}: factory automation demand remained strong in the quarter."
            for i in range(8)))
        store = Store(embedder=get_embedder(offline=True))
        client = StubClient()
        result = ingest_file(path, store, use_llm=True, client=client)

        assert result.status == "ingested"
        assert result.ticker == "ZPHR" and result.company == "Zephyr Robotics"
        assert result.facts == 3 and "3 figures" in result.summary()
        found = run_tool(store, "get_facts", {"ticker": "ZPHR", "metric": "revenue"})
        assert [f["value"] for f in found["facts"]] == [412]
        assert found["facts"][0]["basis"] == "GAAP"
        hits = run_tool(store, "search_documents",
                        {"query": "Zephyr factory automation", "ticker": "ZPHR"})
        assert hits["sources"] and hits["sources"][0]["file"] == path.name


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

def test_a_discovered_company_gets_a_name_and_is_found_in_text():
    from app import ui_data as data
    data.remember_company("ZPHR", "Zephyr Robotics")
    try:
        assert data.company_name("ZPHR") == "Zephyr Robotics"
        assert data.tickers_in("How did Zephyr Robotics do?") == ["ZPHR"]
        data.remember_company("AAPL", "Something Else")            # never overrides
        assert data.company_name("AAPL") == "Apple"
    finally:
        data._DISCOVERED.pop("ZPHR", None)


def test_figures_table_marks_disagreements_and_formats_values():
    from app import ui_data as data
    from app.ingest.facts import Fact
    facts = [Fact("ZPHR", "revenue", "Q2 2026", value_num=412, unit="USD_millions",
                  basis="GAAP", document_id=1),
             Fact("ZPHR", "revenue", "Q2 2026", value_num=415, unit="USD_millions",
                  basis="GAAP", document_id=2),
             Fact("ZPHR", "eps_diluted", "Q2 2026", value_num=0.55, unit="USD",
                  basis="non-GAAP", document_id=1)]
    rows = data.figure_rows(facts, "ZPHR", {1: "a.pdf", 2: "b.pdf"})
    by = {(r["Figure"], r["Value"]): r for r in rows}
    assert by[("Revenue", "$412m")]["Disputed"] == "Yes"
    assert by[("EPS (diluted)", "$0.55")]["Disputed"] == ""
    assert by[("Revenue", "$415m")]["Source"] == "b.pdf"


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
