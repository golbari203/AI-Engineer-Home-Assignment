"""
Builds a populated Store directly from the portfolio folder.

Exists so the tools and the agent can be exercised end to end before MySQL is
wired up, and so the test suite has a realistic store without a database. The
MySQL-backed loader will populate the same Store from tables; nothing above
this layer changes.

Facts are left empty unless supplied: fact extraction needs an API key, so a
store built offline has no facts and get_facts honestly reports that it has
nothing rather than pretending.
"""

from __future__ import annotations

from pathlib import Path

from app.ingest.workbooks import load_all
from app.reconcile import run_all
from app.tools import Store

UPLOADS = Path("/mnt/user-data/uploads")


def build_store(folder: str | Path = UPLOADS, with_index: bool = True,
                facts: list | None = None) -> Store:
    folder = Path(folder)
    wbs = load_all(folder)

    from tests.fixture_statements import ALL as STATEMENTS
    statements = [dict(s) for s in STATEMENTS]

    store = Store(
        statements=statements,
        positions=wbs.positions,
        trades=wbs.trades,
        prices=wbs.prices,
        fx=wbs.fx,
        corporate_actions=wbs.corporate_actions,
        consensus=wbs.consensus,
        price_targets=wbs.price_targets,
        broker_notes=wbs.broker_notes,
        facts=facts or [],
        exceptions=run_all(statements, wbs),
    )

    if with_index:
        attach_index(store, folder)
    return store


def attach_index(store: Store, folder: str | Path = UPLOADS,
                 offline_embedder: bool = True) -> Store:
    """Chunk, embed and index the corpus for search_documents."""
    from app.ingest.cache import extract_cached
    from app.ingest.chunk import chunk_corpus
    from app.ingest.classify import classify_cached
    from app.ingest.embed import get_embedder
    from app.ingest.facts import facts_cached
    from app.retrieval import ChunkRecord, InMemoryBackend

    folder = Path(folder)
    documents, meta_by_id, facts = [], {}, []
    for i, path in enumerate(sorted(folder.glob("*.pdf"))):
        extracted = extract_cached(path)
        head = "\n\n".join(p["text"] for p in extracted["pages"][:2])
        # Cached answers only: no model calls while building the index, but a
        # document the model already classified or read keeps what it found.
        meta = classify_cached(head, path.name, extracted["content_hash"],
                               use_llm=False).as_row()
        meta["document_id"] = i
        meta["filename"] = path.name
        meta_by_id[i] = meta
        documents.append((meta, extracted["pages"]))
        facts += facts_cached(extracted, meta, use_llm=False)[0]
    if facts and not store.facts:
        store.facts = facts

    chunks = chunk_corpus(documents)
    embedder = get_embedder(offline=offline_embedder)
    matrix = embedder.encode([c.text for c in chunks])

    records = []
    for j, chunk in enumerate(chunks):
        meta = meta_by_id[chunk.document_id]
        records.append(ChunkRecord(
            chunk_id=j, document_id=chunk.document_id, text=chunk.text,
            page_from=chunk.page_from, page_to=chunk.page_to,
            is_ocr=chunk.is_ocr, ticker=meta.get("ticker"),
            doc_type=meta.get("doc_type"),
            source_tier=meta.get("source_tier") or 5,
            event_key=meta.get("event_key"),
            published_date=str(meta.get("published_date") or "") or None,
            filename=meta["filename"]))

    store.retrieval_backend = InMemoryBackend(records, matrix)
    store.embedder = embedder
    store.tier_by_document = {i: m.get("source_tier") or 5
                              for i, m in meta_by_id.items()}
    return store
