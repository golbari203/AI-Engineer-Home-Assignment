"""
MySQL persistence.

Everything above this layer talks to a Store, so this file is the only place
that knows SQL. The offline loader in offline_store.py builds the same Store
from files, which is what lets the whole test suite run without a database.

Embeddings live here too, as JSON beside the FULLTEXT index on the same table.
A separate vector store would be a second thing to keep in step with document
re-ingestion, and at a few hundred chunks there is nothing to gain from one.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def config() -> dict:
    return {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", 3306)),
        "user": os.environ.get("MYSQL_USER", "era"),
        "password": os.environ.get("MYSQL_PASSWORD", "era"),
        "database": os.environ.get("MYSQL_DATABASE", "era"),
    }


def connect(with_database: bool = True):
    import mysql.connector
    settings = config()
    if not with_database:
        settings.pop("database")
    return mysql.connector.connect(**settings)


def _f(value):
    """MySQL returns DECIMAL as Decimal, which will not mix with float.

    Every numeric column in this schema is DECIMAL for exactness in storage,
    so the conversion happens once here rather than failing later inside a
    tool with an unhelpful type error.
    """
    return float(value) if value is not None else None


@contextmanager
def cursor(conn, dictionary: bool = False):
    cur = conn.cursor(dictionary=dictionary)
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def split_statements(sql: str) -> list[str]:
    """Split a schema file into statements.

    Comments are stripped before splitting, not after. The schema explains why
    each table exists and several of those comments contain semicolons, so a
    naive split on ";" cuts a comment in half and feeds the remainder to the
    server as SQL.
    """
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        if "--" in line:
            line = line[:line.index("--")]
        lines.append(line)
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def apply_schema() -> None:
    """Idempotent: every statement is CREATE ... IF NOT EXISTS."""
    conn = connect(with_database=False)
    statements = split_statements(SCHEMA.read_text())
    with cursor(conn) as cur:
        for statement in statements:
            cur.execute(statement)
    conn.close()
    log.info("schema applied (%d statements)", len(statements))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_document(conn, meta: dict, pages: list[dict], warnings: list[str]
                  ) -> int | None:
    """Insert a document and its pages. Returns None if already present.

    The unique constraint on content_hash is what makes re-ingestion free and
    what drops the byte-identical Microsoft duplicate.
    """
    with cursor(conn) as cur:
        cur.execute("SELECT id FROM documents WHERE content_hash = %s",
                    (meta["content_hash"],))
        if cur.fetchone():
            return None

        cur.execute(
            """INSERT INTO documents (path, filename, content_hash, doc_type,
                 source_tier, publisher, ticker, title, published_date,
                 as_of_date, fiscal_period, event_key, page_count,
                 ocr_page_count, language, warnings)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (meta.get("path"), meta["filename"], meta["content_hash"],
             meta.get("doc_type"), meta.get("source_tier") or 5,
             meta.get("publisher"), meta.get("ticker"), meta.get("title"),
             meta.get("published_date"), meta.get("as_of_date"),
             meta.get("fiscal_period"), meta.get("event_key"),
             len(pages), sum(1 for p in pages if p.get("method") == "ocr"),
             meta.get("language", "en"), json.dumps(warnings or [])))
        document_id = cur.lastrowid

        cur.executemany(
            """INSERT INTO document_pages (document_id, page_no, text, method,
                 rotation, ocr_confidence, char_count)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            [(document_id, p["page_no"], p.get("text") or "", p["method"],
              p.get("rotation") or 0, p.get("ocr_confidence"),
              len(p.get("text") or "")) for p in pages])
    return document_id


def document_id_for(conn, content_hash: str) -> int | None:
    """Id of an already-ingested document. save_document returns None for one
    it skipped, which is the signal not to re-chunk; callers that still need
    the id to attach rows to use this."""
    with cursor(conn) as cur:
        cur.execute("SELECT id FROM documents WHERE content_hash = %s",
                    (content_hash,))
        row = cur.fetchone()
    return row[0] if row else None


def save_chunks(conn, document_id: int, chunks, vectors) -> None:
    with cursor(conn) as cur:
        cur.executemany(
            """INSERT INTO chunks (document_id, chunk_index, page_from,
                 page_to, text, embedding) VALUES (%s,%s,%s,%s,%s,%s)""",
            [(document_id, c.chunk_index, c.page_from, c.page_to, c.text,
              json.dumps([round(float(x), 6) for x in v]))
             for c, v in zip(chunks, vectors)])


def save_facts(conn, facts) -> None:
    with cursor(conn) as cur:
        cur.executemany(
            """INSERT INTO facts (ticker, metric, fiscal_period, period_start,
                 period_end, value_num, value_text, unit, basis, segment,
                 is_guidance, is_restated, document_id, page_no, stated_on,
                 confidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(f.ticker, f.metric, f.fiscal_period, f.period_start, f.period_end,
              f.value_num, f.value_text, f.unit, f.basis, f.segment,
              f.is_guidance, f.is_restated, f.document_id, f.page_no,
              f.stated_on, f.confidence) for f in facts])


def save_statement(conn, statement: dict, document_id: int) -> int:
    with cursor(conn) as cur:
        cur.execute(
            """INSERT INTO accounts (custodian, account_no, entity,
                 reporting_currency, cash, total_value, fx_rate, as_of_date)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE entity = VALUES(entity),
                 reporting_currency = VALUES(reporting_currency),
                 cash = VALUES(cash), total_value = VALUES(total_value),
                 fx_rate = VALUES(fx_rate), as_of_date = VALUES(as_of_date),
                 id = LAST_INSERT_ID(id)""",
            (statement["custodian"], statement["account_no"],
             statement.get("entity"), statement["currency"],
             statement.get("cash"), statement.get("total_value"),
             statement.get("fx_rate"), statement["as_of_date"]))
        account_id = cur.lastrowid

        # Re-reading a statement replaces its rows rather than adding to them,
        # so bootstrap stays idempotent and a corrected statement supersedes.
        cur.execute("DELETE FROM holdings WHERE account_id = %s AND "
                    "source_kind = 'custodian_statement'", (account_id,))
        cur.execute("DELETE FROM transactions WHERE account_id = %s AND "
                    "source_kind = 'custodian_statement'", (account_id,))

        cur.executemany(
            """INSERT INTO holdings (account_id, ticker, quantity, price,
                 price_currency, market_value, cost_basis, value_currency,
                 fx_rate_used, as_of_date, source_kind, source_document_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'custodian_statement',%s)""",
            [(account_id, p["ticker"], p["quantity"], p.get("price"),
              statement.get("price_currency"), p.get("market_value"),
              p.get("cost_basis"), statement["currency"],
              statement.get("fx_rate"), statement["as_of_date"], document_id)
             for p in statement["positions"]])

        cur.executemany(
            """INSERT INTO transactions (account_id, ticker, trade_date,
                 settle_date, type, quantity, price, amount, currency, fx_rate,
                 memo, source_kind, source_document_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'custodian_statement',%s)""",
            [(account_id, t["ticker"], t["trade_date"], t["settle_date"],
              t["type"], t["quantity"], t["price"], t["amount"], t["currency"],
              t["fx_rate"], t["description"], document_id)
             for t in statement.get("transactions") or []])
    return account_id


def save_workbooks(conn, workbooks, document_id: int | None = None) -> None:
    """Reference data is replaced wholesale: a corrected file supersedes.

    The family office's own positions and trades are stored alongside the
    custodian's, tagged source_kind='internal_book'. They are never merged:
    the two disagree, and the disagreement is the finding.
    """
    with cursor(conn) as cur:
        accounts: dict[str, int] = {}
        for position in workbooks.positions:
            # 'USD' is only a placeholder until the custodian statement for the
            # account is saved; save_statement overwrites it with the real one.
            cur.execute(
                """INSERT INTO accounts (custodian, account_no, entity,
                     reporting_currency) VALUES (%s,%s,%s,'USD')
                   ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id)""",
                (position.custodian, position.account_no, position.entity))
            accounts[position.account_no] = cur.lastrowid

        cur.execute("DELETE FROM holdings WHERE source_kind = 'internal_book'")
        cur.executemany(
            """INSERT INTO holdings (account_id, ticker, quantity, price,
                 market_value, cost_basis, value_currency, as_of_date,
                 source_kind, source_document_id)
               VALUES (%s,%s,%s,%s,%s,%s,'USD',%s,'internal_book',%s)""",
            [(accounts[p.account_no], p.ticker, p.quantity, p.last_price,
              p.market_value, (p.avg_cost or 0) * p.quantity, "2026-06-30",
              document_id) for p in workbooks.positions])

        cur.execute("DELETE FROM transactions WHERE source_kind = 'internal_book'")
        cur.executemany(
            """INSERT INTO transactions (account_id, ticker, trade_date,
                 settle_date, type, quantity, quantity_is_presplit, price,
                 amount, currency, reference, memo, source_kind,
                 source_document_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'USD',%s,%s,'internal_book',%s)""",
            [(accounts.get(t.account_no) or next(iter(accounts.values())),
              t.ticker, t.trade_date, t.settle_date, t.type, t.quantity,
              t.quantity_is_presplit, t.price, t.amount, t.reference, t.memo,
              document_id) for t in workbooks.trades if accounts])
        cur.execute("DELETE FROM prices")
        cur.executemany(
            "INSERT INTO prices (ticker, price_date, close_price, split_adjusted)"
            " VALUES (%s,%s,%s,%s)"
            " ON DUPLICATE KEY UPDATE close_price = VALUES(close_price)",
            [(p["ticker"], p["price_date"], p["close_price"], False)
             for p in workbooks.prices])

        cur.execute("DELETE FROM fx_rates")
        cur.executemany(
            "INSERT INTO fx_rates (rate_date, base_ccy, quote_ccy, rate, source)"
            " VALUES (%s,%s,%s,%s,%s)"
            " ON DUPLICATE KEY UPDATE rate = VALUES(rate), source = VALUES(source)",
            [(r["rate_date"], r["base_ccy"], r["quote_ccy"], r["rate"],
              r["source"]) for r in workbooks.fx])

        cur.executemany(
            """INSERT IGNORE INTO corporate_actions (ticker, action,
                 effective_date, ratio_num, ratio_den) VALUES (%s,%s,%s,%s,%s)""",
            [(a["ticker"], a["action"], a["effective_date"], a["ratio_num"],
              a["ratio_den"]) for a in workbooks.corporate_actions])

        cur.execute("DELETE FROM consensus")
        cur.executemany(
            """INSERT INTO consensus (ticker, fiscal_period, period_end,
                 revenue_consensus, eps_consensus, eps_basis, n_estimates,
                 report_date, revenue_actual, eps_actual, as_of_date)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(c["ticker"], c["fiscal_period"], c["period_end"],
              c["revenue_consensus"], c["eps_consensus"], c["eps_basis"],
              c["n_estimates"], c["report_date"], c["revenue_actual"],
              c["eps_actual"], c["as_of_date"]) for c in workbooks.consensus])

        cur.execute("DELETE FROM research_ratings")  # noqa: E501
        cur.executemany(
            """INSERT INTO research_ratings (ticker, house, analyst, rating,
                 price_target, note_date) VALUES (%s,%s,%s,%s,%s,%s)""",
            [(n["ticker"], n["house"], n["analyst"], n["rating"],
              n["price_target"], n["note_date"]) for n in workbooks.broker_notes])

        cur.execute("DELETE FROM street_targets")
        cur.executemany(
            """INSERT INTO street_targets (ticker, consensus_pt, buy, hold, sell,
                 last_price, as_of_date) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            [(p["ticker"], p["consensus_pt"], p["buy"], p["hold"], p["sell"],
              p["last_price"], p["as_of_date"]) for p in workbooks.price_targets])


def save_exceptions(conn, exceptions) -> None:
    with cursor(conn) as cur:
        cur.execute("DELETE FROM reconciliation_exceptions")
        cur.executemany(
            """INSERT INTO reconciliation_exceptions (check_name, severity,
                 ticker, description, source_a, source_b, value_a, value_b,
                 delta) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(e.check_name, e.severity, e.ticker, e.description, e.source_a,
              e.source_b, e.value_a, e.value_b, e.delta) for e in exceptions])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_store(conn):
    """Build the same Store the offline loader builds, from tables."""
    from decimal import Decimal

    from app.ingest.embed import get_embedder
    from app.ingest.workbooks import Position, Trade
    from app.ingest.facts import Fact
    from app.reconcile import Exception_
    from app.retrieval import MySQLBackend
    from app.tools import Store

    store = Store()
    with cursor(conn, dictionary=True) as cur:
        cur.execute("""SELECT a.custodian, a.account_no, a.entity,
                              a.reporting_currency, a.cash, a.total_value,
                              a.fx_rate, h.ticker, h.quantity,
                              h.price, h.market_value, h.cost_basis,
                              h.fx_rate_used, h.as_of_date
                       FROM holdings h JOIN accounts a ON a.id = h.account_id
                       WHERE h.source_kind = 'custodian_statement'
                       ORDER BY a.id, h.id""")
        grouped: dict[tuple, dict] = {}
        for row in cur.fetchall():
            key = (row["custodian"], row["account_no"])
            statement = grouped.setdefault(key, {
                "custodian": row["custodian"], "account_no": row["account_no"],
                "entity": row["entity"], "currency": row["reporting_currency"],
                "price_currency": "USD",
                "fx_rate": _f(row["fx_rate"] or row["fx_rate_used"]),
                "as_of_date": row["as_of_date"], "cash": _f(row["cash"]) or 0.0,
                "total_value": _f(row["total_value"]),
                "positions": [], "transactions": []})
            statement["positions"].append({
                "ticker": row["ticker"], "quantity": _f(row["quantity"]),
                "price": _f(row["price"]), "market_value": _f(row["market_value"]),
                "cost_basis": _f(row["cost_basis"])})
        store.statements = list(grouped.values())

        cur.execute("""SELECT ticker, metric, fiscal_period, period_end,
                              value_num, value_text, unit, basis, segment,
                              is_guidance, is_restated, document_id, page_no,
                              stated_on, confidence FROM facts""")
        store.facts = [Fact(
            ticker=r["ticker"], metric=r["metric"],
            fiscal_period=r["fiscal_period"], period_end=r["period_end"],
            value_num=float(r["value_num"]) if r["value_num"] is not None else None,
            value_text=r["value_text"], unit=r["unit"], basis=r["basis"],
            segment=r["segment"], is_guidance=bool(r["is_guidance"]),
            is_restated=bool(r["is_restated"]), document_id=r["document_id"],
            page_no=r["page_no"], stated_on=r["stated_on"],
            confidence=float(r["confidence"] or 1.0)) for r in cur.fetchall()]

        cur.execute("""SELECT check_name, severity, ticker, description,
                              source_a, source_b FROM reconciliation_exceptions""")
        store.exceptions = [Exception_(
            check_name=r["check_name"], severity=r["severity"],
            description=r["description"], ticker=r["ticker"],
            source_a=r["source_a"], source_b=r["source_b"])
            for r in cur.fetchall()]

        cur.execute("SELECT ticker, action, effective_date, ratio_num, ratio_den"
                    " FROM corporate_actions")
        store.corporate_actions = [{
            "ticker": r["ticker"], "action": r["action"],
            "effective_date": r["effective_date"],
            "ratio_num": float(r["ratio_num"]),
            "ratio_den": float(r["ratio_den"])} for r in cur.fetchall()]

        cur.execute("SELECT rate_date, base_ccy, quote_ccy, rate FROM fx_rates")
        store.fx = [{"rate_date": r["rate_date"], "base_ccy": r["base_ccy"],
                     "quote_ccy": r["quote_ccy"], "rate": float(r["rate"])}
                    for r in cur.fetchall()]

        cur.execute("""SELECT a.custodian, a.account_no, a.entity, h.ticker,
                              h.quantity, h.price, h.market_value, h.cost_basis
                       FROM holdings h JOIN accounts a ON a.id = h.account_id
                       WHERE h.source_kind = 'internal_book'""")
        store.positions = [Position(
            entity=r["entity"] or "", custodian=r["custodian"],
            account_no=r["account_no"], ticker=r["ticker"],
            quantity=float(r["quantity"]),
            avg_cost=(float(r["cost_basis"]) / float(r["quantity"]))
            if r["cost_basis"] and r["quantity"] else None,
            last_price=float(r["price"]) if r["price"] else None,
            market_value=float(r["market_value"]) if r["market_value"] else None)
            for r in cur.fetchall()]

        cur.execute("""SELECT a.account_no, t.ticker, t.trade_date,
                              t.settle_date, t.type, t.quantity, t.price,
                              t.amount, t.reference, t.memo,
                              t.quantity_is_presplit
                       FROM transactions t JOIN accounts a ON a.id = t.account_id
                       WHERE t.source_kind = 'internal_book'""")
        store.trades = [Trade(
            account_no=r["account_no"], ticker=r["ticker"],
            trade_date=r["trade_date"], settle_date=r["settle_date"],
            type=r["type"],
            quantity=float(r["quantity"]) if r["quantity"] is not None else None,
            price=float(r["price"]) if r["price"] is not None else None,
            amount=float(r["amount"]) if r["amount"] is not None else None,
            reference=r["reference"] or "", memo=r["memo"] or "",
            quantity_is_presplit=bool(r["quantity_is_presplit"]))
            for r in cur.fetchall()]

        cur.execute("SELECT ticker, price_date, close_price, split_adjusted"
                    " FROM prices")
        store.prices = [{"ticker": r["ticker"], "price_date": r["price_date"],
                         "close_price": float(r["close_price"]),
                         "split_adjusted": bool(r["split_adjusted"])}
                        for r in cur.fetchall()]

        cur.execute("""SELECT ticker, fiscal_period, period_end,
                              revenue_consensus, eps_consensus, eps_basis,
                              n_estimates, report_date, revenue_actual,
                              eps_actual, as_of_date FROM consensus""")
        store.consensus = [{
            k: (float(v) if isinstance(v, Decimal) else v)
            for k, v in r.items()} for r in cur.fetchall()]

        cur.execute("""SELECT ticker, house, analyst, rating, price_target,
                              note_date FROM research_ratings""")
        store.broker_notes = [{
            "ticker": r["ticker"], "house": r["house"], "analyst": r["analyst"],
            "rating": r["rating"], "note_date": r["note_date"],
            "price_target": float(r["price_target"]) if r["price_target"] else None}
            for r in cur.fetchall()]

        cur.execute("""SELECT ticker, consensus_pt, buy, hold, sell, last_price,
                              as_of_date FROM street_targets""")
        store.price_targets = [{
            k: (float(v) if isinstance(v, Decimal) else v)
            for k, v in r.items()} for r in cur.fetchall()]

        cur.execute("SELECT id, source_tier FROM documents")
        store.tier_by_document = {r["id"]: r["source_tier"] for r in cur.fetchall()}

    store.retrieval_backend = MySQLBackend(conn)
    store.embedder = get_embedder(offline=bool(os.environ.get("ERA_OFFLINE")))
    return store
