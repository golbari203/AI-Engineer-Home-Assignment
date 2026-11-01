"""
Tests for chunking and embedding.

The corpus tests run against the real portfolio folder when it is present and
are skipped otherwise, so the suite is still useful in isolation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.chunk import (                         # noqa: E402
    MIN_CHARS, Chunk, block_hash, build_prefix, chunk_corpus, chunk_document,
    find_boilerplate, looks_like_table, normalise_block, split_blocks,
)
from app.ingest.embed import (                         # noqa: E402
    HashingEmbedder, embed_chunks, from_json, to_json, top_k,
)

UPLOADS = Path("/mnt/user-data/uploads")

META = {"ticker": "AMZN", "doc_type": "press_release",
        "published_date": "2026-01-29", "title": "Q4 2025 results",
        "document_id": 1}

TABLE = """Segment                Q4 2025    Q4 2024   Change
North America           94,355     85,513    10.3%
International           39,035     36,809     6.0%
AWS                     23,863     21,503    11.0%
Total                  157,254    143,824     9.3%"""

PROSE = ("AWS revenue was $23.9 billion, up 11.0% compared with $21.5 billion "
         "in the prior-year period, reflecting customer migrations to the cloud "
         "and demand for generative AI services, with new capacity coming "
         "online. " * 4)

BOILERPLATE = (
    "This release contains forward-looking statements within the meaning of "
    "the Private Securities Litigation Reform Act of 1995. These statements "
    "involve risks and uncertainties, and actual results may differ materially.")


def _page(no: int, text: str, method: str = "text_layer") -> dict:
    return {"page_no": no, "text": text, "method": method}


# ---------------------------------------------------------------------------
# Context prefix
# ---------------------------------------------------------------------------

def test_prefix_makes_a_chunk_self_describing():
    """Three Amazon releases each say "AWS revenue was $X billion, up Y%"."""
    prefix = build_prefix(META)
    assert prefix == "[AMZN | press_release | 2026-01-29 | Q4 2025 results]"
    chunks = chunk_document([_page(1, PROSE)], META)
    assert all(c.text.startswith(prefix) for c in chunks)


def test_prefix_survives_missing_metadata():
    assert build_prefix({"ticker": "LLY"}) == "[LLY | document]"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_a_segment_table_is_recognised():
    assert looks_like_table(TABLE)
    assert not looks_like_table(PROSE)


def test_a_table_is_never_split():
    """Three rows of an income statement with no header look authoritative
    and have lost both their column labels and the rest of the data."""
    chunks = chunk_document([_page(1, TABLE)], META)
    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1
    for row in ("North America", "International", "AWS", "Total"):
        assert row in tables[0].text


def test_a_table_is_separated_from_surrounding_prose():
    chunks = chunk_document([_page(1, PROSE + "\n\n" + TABLE)], META)
    assert any(c.is_table for c in chunks)
    assert any(not c.is_table for c in chunks)


# ---------------------------------------------------------------------------
# Boilerplate
# ---------------------------------------------------------------------------

def test_digits_are_not_masked_when_detecting_duplicates():
    """Masking digits collapsed nine different EPS tables into one block and
    twenty dividend declarations with different amounts into another."""
    a = "The Board declared a cash dividend of $0.25 per share."
    b = "The Board declared a cash dividend of $1.25 per share."
    assert normalise_block(a) != normalise_block(b)
    assert block_hash(a) != block_hash(b)


def test_identical_legal_text_collapses():
    assert block_hash(BOILERPLATE) == block_hash(BOILERPLATE.replace("\n", " "))


def test_boilerplate_is_dropped_from_chunks():
    docs = [({"ticker": t, "doc_type": "press_release", "document_id": i},
             [_page(1, f"{t} reported results for the quarter. " * 12
                       + "\n\n" + BOILERPLATE)])
            for i, t in enumerate(["AMZN", "AAPL", "MSFT", "JPM", "XOM"])]
    assert len(find_boilerplate(docs)) == 1
    text = " ".join(c.text for c in chunk_corpus(docs))
    assert "Private Securities Litigation Reform Act" not in text
    assert "reported results for the quarter" in text


def test_a_block_in_three_documents_is_not_boilerplate():
    """A company repeating its own segment description across quarters is
    still informative, so the threshold is four documents, not two."""
    docs = [({"ticker": "AMZN", "document_id": i}, [_page(1, BOILERPLATE)])
            for i in range(3)]
    assert find_boilerplate(docs) == set()


def test_tables_are_never_suppressed():
    """A recurring table layout still holds different numbers each time."""
    docs = [({"ticker": t, "document_id": i}, [_page(1, TABLE)])
            for i, t in enumerate(["AMZN", "AAPL", "MSFT", "JPM", "XOM"])]
    assert find_boilerplate(docs) == set()
    assert all(c.is_table for c in chunk_corpus(docs))


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_page_ranges_are_tracked():
    chunks = chunk_document([_page(1, PROSE), _page(2, PROSE)], META)
    assert all(c.page_from <= c.page_to for c in chunks)
    assert max(c.page_to for c in chunks) == 2


def test_ocr_pages_are_flagged_through_to_the_chunk():
    """Needed so retrieval can caveat figures that came from a scan."""
    chunks = chunk_document([_page(1, PROSE, method="ocr")], META)
    assert all(c.is_ocr for c in chunks)


def test_tiny_fragments_are_dropped():
    chunks = chunk_document([_page(1, "Contacts")], META)
    assert chunks == []


def test_chunks_are_indexed_in_order():
    chunks = chunk_document([_page(1, PROSE * 4)], META)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def test_vectors_are_normalised():
    emb = HashingEmbedder()
    v = emb.encode(["AWS revenue was $23.9 billion"])
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-5


def test_json_round_trip_preserves_similarity():
    emb = HashingEmbedder()
    v = emb.encode(["Costco Wholesale Corporation"])[0]
    assert abs(float(from_json(to_json(v)) @ v) - 1.0) < 1e-4


def test_top_k_orders_by_similarity():
    emb = HashingEmbedder()
    corpus = ["Costco warehouse membership fees",
              "NVIDIA data center revenue",
              "Costco membership renewal rate"]
    matrix = emb.encode(corpus)
    query = emb.encode(["Costco membership"], is_query=True)[0]
    assert set(top_k(query, matrix, k=2)) == {0, 2}


def test_embed_chunks_covers_every_chunk():
    chunks = chunk_document([_page(1, PROSE * 3)], META)
    matrix = embed_chunks(chunks, HashingEmbedder())
    assert matrix.shape == (len(chunks), HashingEmbedder().dim)


# ---------------------------------------------------------------------------
# Real corpus
# ---------------------------------------------------------------------------

def _real_documents():
    from app.ingest.cache import extract_cached
    from app.ingest.classify import classify_document
    docs = []
    for i, p in enumerate(sorted(UPLOADS.glob("*.pdf"))):
        extracted = extract_cached(p)
        meta = classify_document(
            "\n\n".join(pg["text"] for pg in extracted["pages"][:2]),
            p.name, use_llm=False).as_row()
        meta["document_id"] = i
        docs.append((meta, extracted["pages"]))
    return docs


def test_corpus_chunks_are_sane():
    if not UPLOADS.exists():
        return
    docs = _real_documents()
    chunks = chunk_corpus(docs)
    assert 300 <= len(chunks) <= 700
    assert all(len(c.body) >= MIN_CHARS for c in chunks)
    assert all(c.text.startswith("[") for c in chunks)
    # Every document contributes something, or the pipeline has lost a file.
    assert len({c.document_id for c in chunks}) == len(docs)


def test_corpus_boilerplate_suppression_is_material_but_bounded():
    if not UPLOADS.exists():
        return
    docs = _real_documents()
    kept = sum(c.char_count for c in chunk_corpus(docs))
    raw = sum(c.char_count for c in chunk_corpus(docs, min_docs=10 ** 6))
    removed = (raw - kept) / raw
    assert 0.01 < removed < 0.20, f"suppressed {removed:.1%} of the corpus"


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
