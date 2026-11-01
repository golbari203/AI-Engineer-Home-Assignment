"""
Chunking: turning pages into retrievable units.

Three decisions, each forced by something in the corpus.

1. Every chunk carries a context prefix
   Amazon's Q3 2025, Q4 2025 and Q2 2026 releases are structurally identical,
   each containing a sentence of the form "AWS revenue was $X billion, up Y%".
   Retrieved bare, three such passages are indistinguishable and the model
   cannot tell which quarter it is quoting. The prefix
   "[AMZN | press_release | 2026-01-29 | Q4 2025 results]" makes the chunk
   self-describing and lets retrieval filter by ticker -- which matters
   because COST is also the ordinary word "cost", appearing in "cost of sales"
   across most of the corpus.

2. Tables are not split mid-table
   A financial statement is a single logical unit. A chunk holding three rows
   of an income statement with no header is worse than useless: it looks
   authoritative and has lost both its column labels and the rest of its data.
   Table-like blocks are kept whole even when that overshoots the target size.

3. Boilerplate is detected and suppressed
   The forward-looking-statements paragraph appears in nearly every press
   release, and "Restated Articles of Incorporation" in every 10-K exhibit
   list. Without suppression a question about risk returns twenty-six copies
   of the same legal paragraph, crowding out the one document that answers it.
   Detection is corpus-level, not a hand-written blocklist: any normalised
   block appearing in several distinct documents is boilerplate by definition.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

# Roughly 500-800 tokens at about four characters per token. Estimating rather
# than tokenising avoids a tokeniser dependency at ingest; the figure only
# needs to be approximately right.
TARGET_CHARS = 2600
MAX_CHARS = 4000
OVERLAP_CHARS = 250
MIN_CHARS = 120

# A block is boilerplate when the same normalised text appears in at least this
# many distinct documents. Two is too aggressive (a company repeating its own
# segment description across quarters is still informative); four is the point
# at which a block is clearly a legal or structural template.
BOILERPLATE_MIN_DOCS = 4


@dataclass
class Chunk:
    document_id: int | None
    chunk_index: int
    text: str                       # includes the prefix
    body: str                       # without the prefix, used for dedup
    page_from: int
    page_to: int
    is_table: bool = False
    is_ocr: bool = False
    is_boilerplate: bool = False
    block_hash: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

NUMERIC = re.compile(r"-?[\d,.'()]+%?")


def looks_like_table(block: str) -> bool:
    """A block is tabular when most of its lines carry several numbers.

    Catches the condensed statements, segment tables and broker estimate
    tables without needing a PDF table extractor.
    """
    lines = [l for l in block.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    numeric_lines = sum(
        1 for l in lines
        if len([t for t in NUMERIC.findall(l) if any(c.isdigit() for c in t)]) >= 3)
    return numeric_lines >= max(2, len(lines) * 0.5)


def split_blocks(text: str) -> list[str]:
    """Paragraph boundaries, with consecutive table lines held together."""
    raw = re.split(r"\n\s*\n", text)
    blocks: list[str] = []
    for part in raw:
        part = part.strip()
        if part:
            blocks.append(part)
    return blocks


def normalise_block(block: str) -> str:
    """For duplicate detection: ignore case, whitespace and punctuation only.

    Digits are deliberately NOT masked. Masking them seemed reasonable and was
    badly wrong on this corpus: nine different earnings-per-share tables hashed
    to one block, and twenty dividend declarations with different amounts
    collapsed together. Suppressing those would break "what dividend did
    JPMorgan declare?". In a financial corpus the digits are the payload, so
    boilerplate means literally the same text, not the same template.
    """
    s = re.sub(r"[^\w]+", " ", block.lower())
    return " ".join(s.split())


def block_hash(block: str) -> str:
    return hashlib.sha1(normalise_block(block).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Context prefix
# ---------------------------------------------------------------------------

def build_prefix(meta: dict) -> str:
    parts = [
        meta.get("ticker") or "UNKNOWN",
        meta.get("doc_type") or "document",
        str(meta.get("published_date") or meta.get("as_of_date") or ""),
        (meta.get("title") or "")[:70],
    ]
    return "[" + " | ".join(p for p in parts if p) + "]"


# ---------------------------------------------------------------------------
# Chunking one document
# ---------------------------------------------------------------------------

def chunk_document(pages: list[dict], meta: dict,
                   drop_hashes: set[str] | None = None) -> list[Chunk]:
    """pages: [{page_no, text, method}] from the extraction cache.

    drop_hashes holds blocks identified corpus-wide as boilerplate. They are
    removed before chunks are assembled, because boilerplate is a paragraph
    sitting inside an otherwise useful chunk, not a chunk of its own. Nothing
    is lost: document_pages still stores the full text of every page.
    """
    drop_hashes = drop_hashes or set()
    prefix = build_prefix(meta)
    chunks: list[Chunk] = []

    buf: list[str] = []
    buf_len = 0
    first_page = last_page = None
    buf_ocr = False

    def flush(is_table: bool = False) -> None:
        nonlocal buf, buf_len, first_page, last_page, buf_ocr
        body = "\n\n".join(buf).strip()
        if len(body) >= MIN_CHARS:
            chunks.append(Chunk(
                document_id=meta.get("document_id"), chunk_index=len(chunks),
                text=f"{prefix}\n{body}", body=body,
                page_from=first_page or 1, page_to=last_page or 1,
                is_table=is_table, is_ocr=buf_ocr, block_hash=block_hash(body)))
        # Carry a tail of the previous chunk so a sentence split across the
        # boundary is still retrievable from one side.
        tail = body[-OVERLAP_CHARS:] if body and not is_table else ""
        buf = [tail] if tail else []
        buf_len = len(tail)
        first_page = last_page
        buf_ocr = False

    for page in pages:
        page_no = page["page_no"]
        is_ocr = page.get("method") == "ocr"
        for block in split_blocks(page.get("text") or ""):
            table = looks_like_table(block)
            if not table and block_hash(block) in drop_hashes:
                continue

            # A table is its own chunk: splitting it strands rows from headers.
            if table:
                if buf:
                    flush()
                first_page = last_page = page_no
                buf, buf_len, buf_ocr = [block], len(block), is_ocr
                flush(is_table=True)
                buf, buf_len = [], 0
                continue

            if buf_len + len(block) > MAX_CHARS and buf:
                flush()
            if first_page is None:
                first_page = page_no
            last_page = page_no
            buf.append(block)
            buf_len += len(block) + 2
            buf_ocr = buf_ocr or is_ocr
            if buf_len >= TARGET_CHARS:
                flush()

    if buf:
        flush()
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


# ---------------------------------------------------------------------------
# Corpus-level boilerplate
# ---------------------------------------------------------------------------

def find_boilerplate(documents: list[tuple[dict, list[dict]]],
                     min_docs: int = BOILERPLATE_MIN_DOCS) -> set[str]:
    """Blocks appearing in at least `min_docs` distinct documents.

    Corpus-level rather than a hand-written blocklist, so it adapts when new
    document types arrive: a second custodian's legal footer is caught by the
    same rule with no code change. Tables are never suppressed, because a
    recurring table layout still holds different numbers each time.
    """
    doc_count: Counter[str] = Counter()
    for _, pages in documents:
        seen: set[str] = set()
        for page in pages:
            for block in split_blocks(page.get("text") or ""):
                if len(block) < MIN_CHARS or looks_like_table(block):
                    continue
                h = block_hash(block)
                if h not in seen:
                    seen.add(h)
                    doc_count[h] += 1
    return {h for h, n in doc_count.items() if n >= min_docs}


def chunk_corpus(documents: list[tuple[dict, list[dict]]],
                 min_docs: int = BOILERPLATE_MIN_DOCS) -> list[Chunk]:
    """Two passes: identify boilerplate across the corpus, then chunk without it."""
    drop = find_boilerplate(documents, min_docs)
    out: list[Chunk] = []
    for i, (meta, pages) in enumerate(documents):
        meta = {**meta, "document_id": meta.get("document_id", i)}
        out.extend(chunk_document(pages, meta, drop_hashes=drop))
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from app.ingest.cache import extract_cached
    from app.ingest.classify import classify_document

    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads")
    docs = []
    for i, p in enumerate(sorted(folder.glob("*.pdf"))):
        extracted = extract_cached(p)
        meta = classify_document(
            "\n\n".join(pg["text"] for pg in extracted["pages"][:2]),
            p.name, use_llm=False).as_row()
        meta["document_id"] = i
        meta["filename"] = p.name
        docs.append((meta, extracted["pages"]))

    from app.ingest.chunk import find_boilerplate
    before = chunk_corpus(docs, min_docs=10 ** 6)        # suppression disabled
    chunks = chunk_corpus(docs)
    drop = find_boilerplate(docs)
    sizes = sorted(c.char_count for c in chunks)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print(f"  boilerplate blocks found : {len(drop)}")
    print(f"  chunks without suppression: {len(before)}")
    print(f"  chars removed             : "
          f"{sum(c.char_count for c in before) - sum(c.char_count for c in chunks):,}")
    print(f"  tables kept whole         : {sum(c.is_table for c in chunks)}")
    print(f"  from OCR'd pages          : {sum(c.is_ocr for c in chunks)}")
    print(f"  chars median/max          : {sizes[len(sizes) // 2]} / {sizes[-1]}")
