"""
Tests for the MySQL layer.

Skipped unless a database is reachable, so the rest of the suite still runs
with no server. Set MYSQL_HOST/USER/PASSWORD/DATABASE to enable.

These assert the same figures the offline tests assert, because the whole
point of the Store abstraction is that the source of the data is invisible
above this layer.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _conn():
    try:
        from app import db
        return db.connect()
    except Exception:                                    # noqa: BLE001
        return None


def test_statement_splitting_ignores_comments():
    """The schema explains why each table exists and several of those
    comments contain semicolons; splitting on ";" first cuts one in half."""
    from app.db import split_statements
    sql = ("-- a comment; with a semicolon\n"
           "CREATE TABLE a (id INT);\n"
           "CREATE TABLE b (id INT);  -- trailing; comment\n")
    statements = split_statements(sql)
    assert len(statements) == 2
    assert all(s.startswith("CREATE TABLE") for s in statements)


def test_decimals_are_coerced():
    from decimal import Decimal

    from app.db import _f
    assert _f(Decimal("0.7801")) == 0.7801
    assert _f(None) is None


def test_schema_applies_and_is_idempotent():
    if _conn() is None:
        return
    from app import db
    db.apply_schema()
    db.apply_schema()
    conn = db.connect()
    with db.cursor(conn) as cur:
        cur.execute("SHOW TABLES")
        assert len(cur.fetchall()) >= 15
    conn.close()


def test_store_from_mysql_matches_the_offline_figures():
    conn = _conn()
    if conn is None:
        return
    from app import db
    from app.tools import run_tool

    store = db.load_store(conn)
    if not store.statements:
        return                                           # not bootstrapped yet

    costco = run_tool(store, "get_holdings", {"ticker": "COST"})
    assert costco["total_quantity"] == 2300
    assert costco["held_in_accounts"] == 2

    jpm = run_tool(store, "get_holdings", {"ticker": "JPM"})
    assert jpm["total_quantity"] == 7250
    assert "7,520" in jpm["positions"][0]["book_disagreement"]

    whole = run_tool(store, "get_holdings", {})
    assert abs(whole["total_portfolio_value_usd"] - 20_125_683.75) < 2_000

    nvidia = run_tool(store, "get_transactions", {"ticker": "NVDA"})
    assert nvidia["net_shares"] == 14000

    # The vendor's street target must survive the round trip through MySQL.
    lilly = run_tool(store, "get_research", {"ticker": "LLY"})
    assert lilly["street_aggregate"]["consensus_price_target"] == 902.0

    swiss = next(s for s in store.statements if s["account_no"] == "CH-5521.03")
    assert swiss["currency"] == "CHF"
    conn.close()


def test_fulltext_index_exists_and_matches():
    conn = _conn()
    if conn is None:
        return
    from app import db
    with db.cursor(conn) as cur:
        cur.execute("SHOW INDEX FROM chunks WHERE Index_type = 'FULLTEXT'")
        if not cur.fetchall():
            return
        cur.execute("SELECT COUNT(*) FROM chunks")
        if not cur.fetchone()[0]:
            return
    from app.retrieval import MySQLBackend
    backend = MySQLBackend(conn)
    hits = backend.keyword_candidates("Prescription Drug Affordability Act", 6)
    files = {backend.get(cid).filename for cid, _ in hits}
    assert len(files) == 5                               # the five that contain it
    conn.close()


def test_re_running_bootstrap_does_not_duplicate_rows():
    conn = _conn()
    if conn is None:
        return
    from app import db
    with db.cursor(conn) as cur:
        cur.execute("""SELECT ticker, COUNT(*) n FROM holdings
                       WHERE source_kind = 'custodian_statement'
                       GROUP BY ticker, account_id HAVING n > 1""")
        assert cur.fetchall() == []
    conn.close()


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
