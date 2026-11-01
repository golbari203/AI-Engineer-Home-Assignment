"""
Incremental ingest.

The brief says new documents can arrive while the system is running, so adding
a file must not mean rebuilding the index or restarting the process. One
function does the whole job and is called from three places: the startup scan,
the folder watcher, and the upload control in the UI. Routing all three through
the same code means the upload path cannot drift from the batch path.

Idempotence comes from the content hash. The startup scan and the watcher can
both see the same file and the second one is a no-op, which is also how the
byte-identical Microsoft Q2 FY2026 duplicate is skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.ingest.cache import extract_cached, file_hash
from app.ingest.chunk import chunk_document, find_boilerplate
from app.ingest.classify import classify_cached
from app.ingest.facts import facts_cached
from app.retrieval import ChunkRecord, InMemoryBackend
from app.tools import Store

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    path: Path
    status: str                 # ingested | duplicate | unsupported | failed
    doc_type: str | None = None
    ticker: str | None = None
    company: str | None = None
    chunks: int = 0
    facts: int = 0
    pages: int = 0
    ocr_pages: int = 0
    warnings: list[str] | None = None
    detail: str = ""

    def summary(self) -> str:
        if self.status == "duplicate":
            return f"{self.path.name}: already on file, skipped"
        if self.status != "ingested":
            return f"{self.path.name}: {self.status} ({self.detail})"
        bits = [f"{self.doc_type or 'document'}"]
        if self.ticker:
            bits.append(f"{self.company} ({self.ticker})" if self.company else self.ticker)
        bits.append(f"{self.chunks} passages")
        if self.facts:
            bits.append(f"{self.facts} figures")
        if self.ocr_pages:
            bits.append(f"{self.ocr_pages} page(s) read by OCR")
        return f"{self.path.name}: " + ", ".join(bits)


def ingest_file(path: str | Path, store: Store, use_llm: bool = False,
                client=None) -> IngestResult:
    """Extract, classify, chunk, embed and splice into the live index.

    With use_llm, a document the rules cannot place is classified by the model
    (which also names a company outside the original ten), and its figures are
    extracted into store.facts. Both answers are cached by content hash.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return IngestResult(path, "unsupported", detail=path.suffix or "no extension")

    content_hash = file_hash(path)
    if content_hash in _hashes(store):
        return IngestResult(path, "duplicate")

    try:
        extracted = extract_cached(path)
    except Exception as exc:                             # noqa: BLE001
        return IngestResult(path, "failed", detail=str(exc))

    head = "\n\n".join(p["text"] for p in extracted["pages"][:2])
    meta = classify_cached(head, path.name, content_hash, use_llm=use_llm,
                           client=client).as_row()
    document_id = _next_document_id(store)
    meta["document_id"] = document_id
    meta["filename"] = path.name
    meta["content_hash"] = content_hash

    chunks = chunk_document(extracted["pages"], meta, drop_hashes=_boilerplate(store))
    if chunks and store.embedder is not None:
        _append(store, meta, chunks)
    # Seen even when it yielded no passages; otherwise every later event for
    # the same file would read it again.
    store._content_hashes = _hashes(store) | {content_hash}     # noqa: SLF001
    store.tier_by_document[document_id] = meta.get("source_tier") or 5

    facts, notes = facts_cached(extracted, meta, use_llm=use_llm, client=client)
    store.facts = list(store.facts) + facts
    if notes:
        log.info("%s: %d figure(s) set aside: %s", path.name, len(notes), "; ".join(notes[:5]))

    return IngestResult(
        path, "ingested", doc_type=meta.get("doc_type"), ticker=meta.get("ticker"),
        company=meta.get("company"), chunks=len(chunks), facts=len(facts),
        pages=extracted["page_count"], ocr_pages=extracted["ocr_page_count"],
        warnings=extracted["warnings"])


def scan_folder(folder: str | Path, store: Store, use_llm: bool = False
                ) -> list[IngestResult]:
    return [ingest_file(p, store, use_llm)
            for p in sorted(Path(folder).glob("*.pdf"))]


# ---------------------------------------------------------------------------
# Index surgery
# ---------------------------------------------------------------------------

def _hashes(store: Store) -> set[str]:
    return getattr(store, "_content_hashes", set())


def _next_document_id(store: Store) -> int:
    return max(store.tier_by_document, default=-1) + 1


def _boilerplate(store: Store) -> set[str]:
    """Boilerplate is identified over the corpus, so a single new document
    inherits the set rather than recomputing it from one file."""
    return getattr(store, "_boilerplate", set())


def _append(store: Store, meta: dict, chunks) -> None:
    import numpy as np

    backend: InMemoryBackend | None = store.retrieval_backend
    existing = list(backend.records.values()) if backend else []
    matrix = backend.matrix if backend else np.zeros((0, store.embedder.dim),
                                                     dtype=np.float32)
    start = max((r.chunk_id for r in existing), default=-1) + 1

    records = [ChunkRecord(
        chunk_id=start + i, document_id=meta["document_id"], text=c.text,
        page_from=c.page_from, page_to=c.page_to, is_ocr=c.is_ocr,
        ticker=meta.get("ticker"), doc_type=meta.get("doc_type"),
        source_tier=meta.get("source_tier") or 5, event_key=meta.get("event_key"),
        published_date=str(meta.get("published_date") or "") or None,
        filename=meta["filename"]) for i, c in enumerate(chunks)]

    vectors = store.embedder.encode([c.text for c in chunks])
    store.retrieval_backend = InMemoryBackend(
        existing + records, np.vstack([matrix, vectors]))


def register_corpus(store: Store, folder: str | Path) -> None:
    """Record the hashes and boilerplate of an index built in bulk.

    Without this a file already in the index would be ingested a second time
    when the watcher starts.
    """
    folder = Path(folder)
    documents = []
    hashes = set()
    for path in sorted(folder.glob("*.pdf")):
        hashes.add(file_hash(path))
        documents.append(({}, extract_cached(path)["pages"]))
    store._content_hashes = hashes                       # noqa: SLF001
    store._boilerplate = find_boilerplate(documents)     # noqa: SLF001
