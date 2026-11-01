"""
Folder watcher.

The brief requires that new documents can arrive while the system is running.
Most of the work is already in pipeline.ingest_file; this module is about the
three things that go wrong when a real folder is watched rather than a fixture.

1. Files arrive in pieces
   A PDF copied over a network share, synced by Dropbox or saved by a scanner
   fires a creation event long before the last byte lands. Reading it then
   yields a truncated file, which extracts as a short or empty document and is
   then cached under a hash that will never be seen again -- the damage is
   permanent, not transient. So a file is read only once its size has stopped
   changing.

2. Editors and sync clients litter the folder
   .part, .crdownload, ~$ lock files and dot-files all appear in a watched
   directory and none of them are documents.

3. A spreadsheet is not a document
   Dropping in an updated blotter or a new statement changes the portfolio,
   not the document index. Those go back through the loaders and trigger a
   fresh reconciliation, so "is anything wrong with my records?" reflects what
   arrived a minute ago.

Idempotence comes from the content hash, so the startup scan and the live
watcher can both see the same file and the second one does nothing.
"""

from __future__ import annotations

import logging
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.pipeline import IngestResult, ingest_file, register_corpus
from app.tools import Store

log = logging.getLogger(__name__)

DOCUMENT_SUFFIXES = {".pdf"}
WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}
SUPPORTED = DOCUMENT_SUFFIXES | WORKBOOK_SUFFIXES

# Partial-write markers used by browsers, sync clients and Office.
TEMP_MARKERS = (".part", ".crdownload", ".tmp", ".download")
TEMP_PREFIXES = (".", "~$")

STABLE_CHECKS = 3          # consecutive identical sizes required
STABLE_INTERVAL = 0.4      # seconds between checks
STABLE_TIMEOUT = 60.0


def should_ignore(path: str | Path) -> bool:
    path = Path(path)
    name = path.name
    if name.startswith(TEMP_PREFIXES):
        return True
    if name.lower().endswith(TEMP_MARKERS):
        return True
    return path.suffix.lower() not in SUPPORTED


def wait_until_stable(path: str | Path, checks: int = STABLE_CHECKS,
                      interval: float = STABLE_INTERVAL,
                      timeout: float = STABLE_TIMEOUT) -> bool:
    """Return True once the file size has held steady, False on timeout.

    Cheaper and more portable than trying to take an exclusive lock, and
    sufficient: a truncated read here would be cached under a hash that never
    recurs, so the cost of being wrong is a permanently broken document.
    """
    path = Path(path)
    deadline = time.time() + timeout
    last, stable = -1, 0
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last and size > 0:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable, last = 0, size
        time.sleep(interval)
    return False


def looks_complete(path: str | Path) -> bool:
    """True when the file's own structure says it has finished arriving.

    A steady size is not enough on its own: through Docker Desktop's file
    sharing, a file still being copied from the Mac can report the same size
    for over a second. A PDF is complete once its %%EOF marker is written, and
    an .xlsx once its zip directory is, so check for those instead.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in DOCUMENT_SUFFIXES:
            size = path.stat().st_size
            with open(path, "rb") as fh:
                fh.seek(max(0, size - 4096))
                return b"%%EOF" in fh.read()
        if suffix in WORKBOOK_SUFFIXES:
            return zipfile.is_zipfile(path)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Handling one path
# ---------------------------------------------------------------------------

@dataclass
class WatchEvent:
    path: Path
    kind: str                  # document | workbook | ignored | unstable
    result: IngestResult | None = None
    exceptions_before: int = 0
    exceptions_after: int = 0

    def summary(self) -> str:
        if self.kind == "ignored":
            return f"{self.path.name}: ignored"
        if self.kind == "unstable":
            return f"{self.path.name}: still being written, skipped"
        if self.kind == "workbook":
            delta = self.exceptions_after - self.exceptions_before
            moved = (f", {abs(delta)} "
                     f"{'more' if delta > 0 else 'fewer'} exception(s)"
                     if delta else ", records unchanged")
            return f"{self.path.name}: portfolio records reloaded{moved}"
        return self.result.summary() if self.result else f"{self.path.name}: no result"


def refresh_portfolio(store: Store, folder: str | Path) -> int:
    """Reload the workbooks and re-run reconciliation. Returns the count."""
    from app.ingest.workbooks import load_all
    from app.reconcile import run_all

    workbooks = load_all(Path(folder))
    store.positions = workbooks.positions
    store.trades = workbooks.trades
    store.prices = workbooks.prices
    store.fx = workbooks.fx
    store.corporate_actions = workbooks.corporate_actions
    store.consensus = workbooks.consensus
    store.price_targets = workbooks.price_targets
    store.broker_notes = workbooks.broker_notes
    store.exceptions = run_all(store.statements, workbooks)
    return len(store.exceptions)


# The watcher thread and the UI upload can see the same file at once. One lock
# makes the second arrival a duplicate rather than a second copy in the index.
_INGEST_LOCK = threading.Lock()


def handle_path(path: str | Path, store: Store, folder: str | Path,
                wait: bool = True, use_llm: bool = False) -> WatchEvent:
    """The single entry point for the scan, the watcher and the UI upload."""
    path = Path(path)
    if should_ignore(path):
        return WatchEvent(path, "ignored")
    if wait and not wait_until_stable(path):
        return WatchEvent(path, "unstable")
    if not looks_complete(path):
        return WatchEvent(path, "unstable")
    with _INGEST_LOCK:
        return _handle(path, store, folder, use_llm)


def _handle(path: Path, store: Store, folder: str | Path, use_llm: bool) -> WatchEvent:
    if path.suffix.lower() in WORKBOOK_SUFFIXES:
        before = len(store.exceptions)
        after = refresh_portfolio(store, folder)
        return WatchEvent(path, "workbook", exceptions_before=before,
                          exceptions_after=after)

    result = ingest_file(path, store, use_llm=use_llm)
    event = WatchEvent(path, "document", result=result)
    # A new statement changes what the client owns, so the checks must re-run.
    if result.status == "ingested" and result.doc_type == "custodian_statement":
        event.exceptions_before = len(store.exceptions)
        event.exceptions_after = refresh_portfolio(store, folder)
    return event


# ---------------------------------------------------------------------------
# Watchdog wiring
# ---------------------------------------------------------------------------

class FolderWatcher:
    """Startup scan plus live watching, both through handle_path."""

    def __init__(self, folder: str | Path, store: Store,
                 on_event: Callable[[WatchEvent], None] | None = None,
                 use_llm: bool = False):
        self.folder = Path(folder)
        self.store = store
        self.on_event = on_event or (lambda e: log.info(e.summary()))
        self.use_llm = use_llm
        self._observer = None

    def scan(self) -> list[WatchEvent]:
        """Catch anything added while the process was down."""
        register_corpus(self.store, self.folder)
        events = []
        for path in sorted(self.folder.iterdir()):
            if should_ignore(path):
                continue
            event = handle_path(path, self.store, self.folder, wait=False,
                                use_llm=self.use_llm)
            events.append(event)
            self.on_event(event)
        return events

    def start(self) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher = self

        class Handler(FileSystemEventHandler):
            def _handle(self, path: str) -> None:
                event = handle_path(path, watcher.store, watcher.folder,
                                    use_llm=watcher.use_llm)
                if event.kind != "ignored":
                    watcher.on_event(event)

            def on_created(self, event):
                if not event.is_directory:
                    self._handle(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    self._handle(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    self._handle(event.dest_path)

        self._observer = Observer()
        self._observer.schedule(Handler(), str(self.folder), recursive=False)
        self._observer.start()
        log.info("watching %s", self.folder)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def run_forever(self) -> None:
        self.scan()
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


if __name__ == "__main__":
    import sys

    from app.offline_store import build_store

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads")
    FolderWatcher(target, build_store(target, with_index=True),
                  on_event=lambda e: print(e.summary())).run_forever()
