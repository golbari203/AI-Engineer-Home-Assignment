"""
Content-addressed cache for extraction and classification.

OCR is the expensive step in this pipeline. A scanned five-page document costs
roughly thirty seconds, and the rotation check in pdf_text.py doubles that on
any page where orientation detection proposes a rotation. Re-reading the same
file during classification, chunking and fact extraction would pay that cost
three times over, and a full-folder sweep would not finish.

Keyed by SHA-256 of the file bytes, so:
  * re-ingesting an unchanged file is free;
  * an edited file gets a new hash and is re-read automatically;
  * the byte-identical Microsoft Q2 FY2026 duplicate hits the same cache entry
    as the original, which is also how the pipeline knows to skip it.

Stored as JSON on disk rather than in MySQL so extraction can run before the
database exists, which keeps the bootstrap order simple.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

CACHE_DIR = Path.home() / ".era_cache"


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _slot(content_hash: str, kind: str) -> Path:
    d = CACHE_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{content_hash}.json"


def get(content_hash: str, kind: str) -> dict | None:
    slot = _slot(content_hash, kind)
    if not slot.exists():
        return None
    try:
        return json.loads(slot.read_text())
    except json.JSONDecodeError:
        slot.unlink(missing_ok=True)
        return None


def put(content_hash: str, kind: str, payload: dict) -> None:
    _slot(content_hash, kind).write_text(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# Cached extraction
# ---------------------------------------------------------------------------

def extract_cached(path: str | Path) -> dict:
    """Full page-aware extraction, cached by content hash.

    Returns {'content_hash', 'pages': [{page_no, text, method, rotation,
    ocr_confidence}], 'warnings', 'page_count', 'ocr_page_count'}.
    """
    path = Path(path)
    h = file_hash(path)
    if (hit := get(h, "extract")) is not None:
        return hit

    from app.ingest.pdf_text import extract_pdf
    doc = extract_pdf(path)
    payload = {
        "content_hash": h,
        "filename": path.name,
        "page_count": doc.page_count,
        "ocr_page_count": doc.ocr_page_count,
        "warnings": doc.warnings,
        "pages": [asdict(p) for p in doc.pages],
    }
    put(h, "extract", payload)
    return payload


def head_text(path: str | Path, pages: int = 2) -> str:
    """Text of the first N pages, from cache. Enough to classify."""
    doc = extract_cached(path)
    return "\n\n".join(p["text"] for p in doc["pages"][:pages])


def full_text(path: str | Path) -> str:
    doc = extract_cached(path)
    return "\n\n".join(p["text"] for p in doc["pages"])


def clear(kind: str | None = None) -> None:
    import shutil
    target = CACHE_DIR / kind if kind else CACHE_DIR
    shutil.rmtree(target, ignore_errors=True)
