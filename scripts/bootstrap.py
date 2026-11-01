#!/usr/bin/env python3
"""One-shot setup: apply the schema and ingest the portfolio folder.

Safe to re-run. Documents are keyed by content hash, so a second run
re-reads nothing and reference data is replaced wholesale.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db                                             # noqa: E402
from app.ingest.cache import extract_cached, file_hash          # noqa: E402
from app.ingest.chunk import chunk_document, find_boilerplate   # noqa: E402
from app.ingest.classify import classify_cached                 # noqa: E402
from app.ingest.embed import get_embedder                       # noqa: E402
from app.ingest.facts import FACT_DOC_TYPES, facts_cached       # noqa: E402
from app.ingest.workbooks import load_all, workbook_path        # noqa: E402
from app.reconcile import run_all                               # noqa: E402

log = logging.getLogger("bootstrap")


def wait_for_mysql(attempts: int = 30) -> None:
    for i in range(attempts):
        try:
            db.connect(with_database=False).close()
            return
        except Exception as exc:                                # noqa: BLE001
            log.info("waiting for MySQL (%d/%d): %s", i + 1, attempts, exc)
            time.sleep(2)
    raise SystemExit("MySQL did not become available")


def save_facts(conn, on_file: dict[int, tuple[dict, dict]], use_llm: bool,
               workers: int = 6) -> None:
    """Fill the facts table for every document that has no rows yet.

    Extraction is cached by content hash, so the model reads each document
    once, ever: the first run after a fresh cache takes a few minutes and costs
    a few dollars, and every later start only reads the cache. Calls run in
    parallel because one long filing takes the better part of a minute.
    """
    from concurrent.futures import ThreadPoolExecutor

    with db.cursor(conn) as cur:
        cur.execute("SELECT DISTINCT document_id FROM facts")
        done = {r[0] for r in cur.fetchall()}
    todo = {i: v for i, v in on_file.items()
            if i not in done and v[1].get("doc_type") in FACT_DOC_TYPES}
    if not todo:
        return
    log.info("extracting figures from %d document(s)%s", len(todo),
             "" if use_llm else " (cache only)")

    def one(item):
        document_id, (extracted, meta) = item
        return document_id, meta, facts_cached(extracted, meta, use_llm=use_llm)

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for document_id, meta, (facts, notes) in pool.map(one, todo.items()):
            if facts:
                db.save_facts(conn, facts)
                total += len(facts)
            log.info("  %s: %d figures%s", meta["filename"], len(facts),
                     f", {len(notes)} set aside" if notes else "")
    log.info("figures saved: %d", total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=os.environ.get(
        "PORTFOLIO_DIR", "/data/portfolio"))
    parser.add_argument("--offline-embeddings", action="store_true")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip classification calls; regex only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"portfolio folder not found: {folder}")

    wait_for_mysql()
    db.apply_schema()
    conn = db.connect()

    pdfs = sorted(folder.glob("*.pdf"))
    documents = [({}, extract_cached(p)["pages"]) for p in pdfs]
    boilerplate = find_boilerplate(documents)
    embedder = get_embedder(offline=args.offline_embeddings)

    ingested = skipped = 0
    on_file: dict[int, tuple[dict, dict]] = {}       # document id -> (extracted, meta)
    for path in pdfs:
        extracted = extract_cached(path)
        head = "\n\n".join(p["text"] for p in extracted["pages"][:2])
        meta = classify_cached(head, path.name, extracted["content_hash"],
                               use_llm=not args.no_llm).as_row()
        meta.update(path=str(path), filename=path.name,
                    content_hash=file_hash(path))

        document_id = db.save_document(conn, meta, extracted["pages"],
                                       extracted["warnings"])
        if document_id is None:
            skipped += 1
            existing = db.document_id_for(conn, meta["content_hash"])
            if existing is not None:
                on_file.setdefault(existing, (extracted, {**meta, "document_id": existing}))
            continue
        on_file[document_id] = (extracted, {**meta, "document_id": document_id})

        chunks = chunk_document(extracted["pages"],
                                {**meta, "document_id": document_id},
                                drop_hashes=boilerplate)
        if chunks:
            db.save_chunks(conn, document_id, chunks,
                           embedder.encode([c.text for c in chunks]))
        ingested += 1
        log.info("%s -> %s, %d passages", path.name,
                 meta.get("doc_type"), len(chunks))

    save_facts(conn, on_file, use_llm=not args.no_llm)

    # The spreadsheets are source documents too. Registering them gives the
    # internal book real provenance, so a disputed position can be traced to
    # the file it came from rather than appearing from nowhere.
    workbook_ids = {}
    for path in sorted(folder.glob("*.xlsx")):
        meta = {"filename": path.name, "path": str(path),
                "content_hash": file_hash(path), "doc_type": "internal_workbook",
                "source_tier": 7, "title": path.stem.replace("_", " ")}
        workbook_ids[path.name] = (db.save_document(conn, meta, [], [])
                                   or db.document_id_for(conn, meta["content_hash"]))

    workbooks = load_all(folder)
    db.save_workbooks(conn, workbooks,
                      document_id=workbook_ids.get(
                          workbook_path(folder, "Holdings_and_trades.xlsx").name))

    # Custodian statements. Until the LLM extractor runs these come from the
    # fixture, which matches the two statements line for line; the shape is
    # identical either way, so swapping in live output changes nothing here.
    from tests.fixture_statements import ALL as STATEMENTS
    statements = [dict(s) for s in STATEMENTS]
    with db.cursor(conn, dictionary=True) as cur:
        cur.execute("SELECT id, filename FROM documents "
                    "WHERE doc_type = 'custodian_statement'")
        by_custodian = {r["filename"]: r["id"] for r in cur.fetchall()}
    for statement in statements:
        document_id = next(
            (i for name, i in by_custodian.items()
             if statement["account_no"].split("-")[0] in name
             or statement["custodian"].split()[0] in name),
            next(iter(by_custodian.values()), None))
        if document_id is not None:
            db.save_statement(conn, statement, document_id)

    db.save_exceptions(conn, run_all(statements, workbooks))

    conn.close()
    log.info("done: %d ingested, %d already on file", ingested, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
