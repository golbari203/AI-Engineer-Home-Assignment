"""
Tests for the folder watcher.

watchdog itself is not tested: the Observer wiring is three one-line methods
that forward to handle_path, and handle_path is where everything that can go
wrong lives. These tests cover filtering, the partial-write guard, and the
reconciliation refresh that a new workbook or statement must trigger.
"""

import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.filenames import resolve                               # noqa: E402
from app.tools import Store                                     # noqa: E402
from app.watcher import (                                       # noqa: E402
    FolderWatcher, handle_path, looks_complete, refresh_portfolio,
    should_ignore, wait_until_stable,
)

UPLOADS = Path("/mnt/user-data/uploads")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_documents_and_workbooks_are_accepted():
    assert not should_ignore("Northgate_statement_2026-06-30.pdf")
    assert not should_ignore("Holdings_and_trades.xlsx")


def test_partial_writes_are_ignored():
    """A browser, a sync client and Excel each leave their own debris."""
    for name in ("report.pdf.part", "report.pdf.crdownload", "report.pdf.tmp",
                 "~$Holdings_and_trades.xlsx", ".DS_Store", ".hidden.pdf"):
        assert should_ignore(name), name


def test_unsupported_types_are_ignored():
    for name in ("notes.txt", "scan.png", "archive.zip", "noextension"):
        assert should_ignore(name), name


# ---------------------------------------------------------------------------
# Partial-write guard
# ---------------------------------------------------------------------------

def test_a_settled_file_is_stable():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.pdf"
        path.write_bytes(b"x" * 1000)
        assert wait_until_stable(path, checks=2, interval=0.01)


def test_an_empty_file_is_never_stable():
    """A creation event often fires before any bytes are written."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.pdf"
        path.touch()
        assert not wait_until_stable(path, checks=2, interval=0.01, timeout=0.2)


def test_a_growing_file_waits_for_the_last_byte():
    """Reading mid-copy yields a truncated document cached under a hash that
    never recurs, so the damage is permanent rather than transient."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.pdf"
        path.write_bytes(b"x" * 100)

        def grow():
            for _ in range(5):
                time.sleep(0.05)
                with open(path, "ab") as fh:
                    fh.write(b"x" * 100)

        thread = threading.Thread(target=grow)
        thread.start()
        assert wait_until_stable(path, checks=3, interval=0.05, timeout=5)
        thread.join()
        assert path.stat().st_size == 600          # every byte arrived


def test_a_pdf_without_its_end_marker_is_not_read():
    """A steady size can lie through Docker Desktop's file sharing; the PDF's
    own %%EOF marker cannot."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "half.pdf"
        path.write_bytes(b"%PDF-1.4\n" + b"x" * 500)
        assert not looks_complete(path)
        assert handle_path(path, Store(), tmp, wait=False).kind == "unstable"
        path.write_bytes(path.read_bytes() + b"\n%%EOF\n")
        assert looks_complete(path)


def test_a_half_copied_workbook_is_not_read():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Prices.xlsx"
        path.write_bytes(b"PK\x03\x04" + b"x" * 200)
        assert not looks_complete(path)


def test_every_real_document_looks_complete():
    if not UPLOADS.exists():
        return
    for path in UPLOADS.iterdir():
        if not should_ignore(path):
            assert looks_complete(path), path.name


def test_a_vanished_file_does_not_raise():
    assert not wait_until_stable(Path(tempfile.gettempdir()) / "gone.pdf",
                                 checks=1, interval=0.01, timeout=0.1)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_an_ignored_file_is_reported_not_processed():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notes.txt"
        path.write_text("hello")
        event = handle_path(path, Store(), tmp)
        assert event.kind == "ignored"
        assert "ignored" in event.summary()


def test_a_file_still_being_written_is_skipped_this_time():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.pdf"
        path.touch()
        event = handle_path(path, Store(), tmp)
        assert event.kind == "unstable"
        assert "still being written" in event.summary()


def test_a_known_document_is_a_no_op():
    """The startup scan and the live watcher both see the same file."""
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store
    from app.pipeline import register_corpus

    store = build_store(with_index=True)
    register_corpus(store, UPLOADS)
    before = len(store.retrieval_backend.records)
    event = handle_path(resolve(UPLOADS, "Microsoft_CFO_transition.pdf"), store,
                        UPLOADS, wait=False)
    assert event.result.status == "duplicate"
    assert len(store.retrieval_backend.records) == before


# ---------------------------------------------------------------------------
# Portfolio refresh
# ---------------------------------------------------------------------------

def test_a_new_workbook_re_runs_reconciliation():
    """Dropping in an updated blotter changes the portfolio, not the index."""
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store

    store = build_store(with_index=False)
    before = len(store.exceptions)
    event = handle_path(resolve(UPLOADS, "Holdings_and_trades.xlsx"), store, UPLOADS,
                        wait=False)
    assert event.kind == "workbook"
    assert event.exceptions_after == before == 7
    assert "records unchanged" in event.summary()


def test_a_corrected_blotter_clears_the_exception():
    """Fixing the JPM share count in the client's own file should make the
    discrepancy disappear without anyone restarting anything."""
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store

    store = build_store(with_index=False)
    assert any(e.check_name == "quantity_mismatch" for e in store.exceptions)

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("Holdings_and_trades.xlsx", "Prices.xlsx",
                     "FX_rates.xlsx", "Consensus_estimates.xlsx"):
            shutil.copy(resolve(UPLOADS, name), Path(tmp) / name)
        _rewrite_jpm_quantity(Path(tmp) / "Holdings_and_trades.xlsx", 7250)
        refresh_portfolio(store, tmp)

    assert not any(e.check_name == "quantity_mismatch" for e in store.exceptions)
    assert any(e.check_name == "missing_in_books" for e in store.exceptions)


def _rewrite_jpm_quantity(path: Path, quantity: int) -> None:
    from openpyxl import load_workbook
    workbook = load_workbook(path)
    sheet = workbook["Positions"]
    for row in sheet.iter_rows():
        values = [c.value for c in row]
        if "JPM" in values:
            row[values.index("JPM") + 2].value = quantity    # Quantity column
    workbook.save(path)


# ---------------------------------------------------------------------------
# Startup scan
# ---------------------------------------------------------------------------

def test_the_scan_covers_the_folder_and_skips_debris():
    if not UPLOADS.exists():
        return
    from app.offline_store import build_store

    store = build_store(with_index=True)
    events = FolderWatcher(UPLOADS, store).scan()
    kinds = {e.kind for e in events}
    assert kinds <= {"document", "workbook"}
    assert sum(1 for e in events if e.kind == "workbook") == 4
    # Everything is already indexed, so a scan of a known folder changes nothing.
    assert all(e.result.status == "duplicate"
               for e in events if e.kind == "document")


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
