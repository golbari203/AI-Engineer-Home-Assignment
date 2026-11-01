"""
Page-aware PDF text extraction.

Why this module is not three lines of pdftotext
-----------------------------------------------
Three documents in the corpus carry a text layer on some pages and bare images
on others:

    JPMorgan_Chase_4Q25_earnings.pdf   pages 2-3 of 4   (income statement)
    Apple_10-K_FY2024.pdf              pages 3-8 of 13  (risk factors, MD&A)
    XOM_10-K_2025.pdf                  pages 7-12 of 13 (financial statements)

A document-level `pdftotext` returns text for these files, so nothing looks
broken, but the most valuable pages are silently missing. The agent would then
tell the investor "that isn't in your documents" about figures that are.

Two further wrinkles, both present in the corpus:
  * ExxonMobil_2Q25_earnings.pdf is scanned 180 degrees rotated. OCR without
    orientation detection returns fluent-looking gibberish.
  * Every PDF repeats a running header on each page ("Amazon.com, Inc. - Q2 2026
    results - July 27, 2026 3"). Poppler glues it onto the body text, which
    pollutes chunks and wastes embedding space.

So extraction is per page, with a decision recorded for each page.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# A page with fewer characters than this is treated as having no usable text
# layer. Real text pages in this corpus run 600-3,300 chars; image-only pages
# come back with exactly 1 (a form feed). 120 leaves wide margin either way.
MIN_CHARS_FOR_TEXT_LAYER = 120

OCR_DPI = 200
OSD_MIN_CONFIDENCE = 1.0  # tesseract OSD confidence on these scans is ~2-7


@dataclass
class Page:
    page_no: int
    text: str
    method: str                 # 'text_layer' | 'ocr'
    rotation: int = 0
    ocr_confidence: float | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ExtractedDoc:
    path: Path
    pages: list[Page] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.method == "ocr")

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


# ---------------------------------------------------------------------------
# poppler helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout


def page_count(path: Path) -> int:
    out = _run(["pdfinfo", str(path)])
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0


def _text_layer(path: Path, page_no: int) -> str:
    # -layout preserves table column alignment, which matters for the financial
    # statements. Without it, columns interleave and numbers lose their headers.
    return _run(["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no),
                 str(path), "-"]).strip()


# ---------------------------------------------------------------------------
# OCR path
# ---------------------------------------------------------------------------

def _ocr_page(path: Path, page_no: int) -> tuple[str, int, float]:
    """Rasterise one page, correct its orientation, OCR it.

    Returns (text, rotation_applied, mean_word_confidence).
    """
    from PIL import Image
    import pytesseract

    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "pg"
        _run(["pdftoppm", "-r", str(OCR_DPI), "-png",
              "-f", str(page_no), "-l", str(page_no), str(path), str(prefix)])
        pngs = sorted(Path(tmp).glob("pg*.png"))
        if not pngs:
            return "", 0, 0.0
        img = Image.open(pngs[0])

        proposed = 0
        try:
            osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
            if osd.get("orientation_conf", 0) >= OSD_MIN_CONFIDENCE:
                proposed = int(osd.get("rotate", 0)) % 360
        except Exception as exc:                       # OSD fails on sparse pages
            log.debug("OSD failed on %s p%d: %s", path.name, page_no, exc)

        # OSD is unreliable on noisy scans. On the Amazon Q4 2025 transcript it
        # claims 180 degrees at confidence 7.3 for pages that are already
        # upright, and rotating them drops OCR confidence from ~80 to ~32.
        # So treat OSD as a proposal and let OCR confidence decide: read the
        # page both ways and keep the better result. Mean word confidence is
        # reliable for detecting gross failure like a wrong orientation, even
        # though it does not catch local errors such as a single wrong digit.
        candidates = [(0, img)]
        if proposed:
            candidates.append((proposed, img.rotate(-proposed, expand=True)))

        best_text, best_rot, best_conf = "", 0, -1.0
        for rot, candidate in candidates:
            data = pytesseract.image_to_data(candidate,
                                             output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data["conf"] if c not in ("-1", -1)]
            conf = sum(confs) / len(confs) if confs else 0.0
            if conf > best_conf:
                best_text = pytesseract.image_to_string(candidate).strip()
                best_rot, best_conf = rot, conf

        text, rotation, confidence = best_text, best_rot, best_conf

    return text, rotation, confidence


# ---------------------------------------------------------------------------
# Running headers and footers
# ---------------------------------------------------------------------------

def strip_running_headers(pages: list[Page], min_share: float = 0.5) -> None:
    """Remove lines that repeat on at least `min_share` of pages.

    Page numbers vary, so lines are normalised by removing digits before
    counting. Mutates the pages in place.
    """
    if len(pages) < 3:
        return

    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter[str] = Counter()
    for p in pages:
        # Only the first three and last three lines of a page can be furniture.
        lines = [l for l in p.text.splitlines() if l.strip()]
        for line in lines[:3] + lines[-3:]:
            if 8 <= len(line.strip()) <= 160:
                counts[norm(line)] += 1

    threshold = max(2, int(len(pages) * min_share))
    furniture = {k for k, v in counts.items() if v >= threshold}
    if not furniture:
        return

    for p in pages:
        kept = [l for l in p.text.splitlines() if norm(l) not in furniture]
        p.text = "\n".join(kept).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_pdf(path: str | Path, ocr: bool = True) -> ExtractedDoc:
    path = Path(path)
    doc = ExtractedDoc(path=path)
    n = page_count(path)
    if n == 0:
        doc.warnings.append("unreadable_pdf")
        return doc

    for i in range(1, n + 1):
        text = _text_layer(path, i)
        if len(text) >= MIN_CHARS_FOR_TEXT_LAYER:
            doc.pages.append(Page(i, text, "text_layer"))
            continue

        if not ocr:
            doc.pages.append(Page(i, text, "text_layer"))
            doc.warnings.append(f"page_{i}_no_text_layer_ocr_disabled")
            continue

        ocr_text, rotation, conf = _ocr_page(path, i)
        doc.pages.append(Page(i, ocr_text, "ocr", rotation, conf))
        if rotation:
            doc.warnings.append(f"page_{i}_rotated_{rotation}")
        if conf and conf < 70:
            doc.warnings.append(f"page_{i}_low_ocr_confidence_{conf:.0f}")

    strip_running_headers(doc.pages)

    if 0 < doc.ocr_page_count < doc.page_count:
        doc.warnings.append("hybrid_text_and_scanned_pages")
    elif doc.ocr_page_count == doc.page_count:
        doc.warnings.append("fully_scanned_document")

    return doc


if __name__ == "__main__":  # smoke test
    import sys
    d = extract_pdf(sys.argv[1])
    print(f"{d.path.name}: {d.page_count} pages, {d.ocr_page_count} OCR'd, "
          f"warnings={d.warnings}")
    for p in d.pages:
        print(f"  p{p.page_no}: {p.method:<10} rot={p.rotation:<3} "
              f"conf={p.ocr_confidence or 0:.0f} chars={p.char_count}")
