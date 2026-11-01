"""
Hybrid retrieval: vector search and keyword search, fused.

Why both halves are necessary on this corpus
--------------------------------------------
Embeddings handle meaning and miss exact tokens. "Prescription Drug
Affordability Act of 2026" is a fictional law appearing in three broker notes;
no embedding model has seen it, and a near-miss is useless when the user asked
about that specific act.

Keywords handle exact tokens and drown in boilerplate. Searching "restated"
returns six hits for "Restated Articles of Incorporation" in the exhibit lists
of six different 10-Ks, against one real recast note in Amazon's Q1 2026 10-Q.

Neither is sufficient. Both are cheap. Reciprocal rank fusion combines them
without needing the two score scales to be comparable, which is its main
attraction: BM25 scores and cosine similarities live on unrelated ranges, and
any attempt to normalise them is a tuning exercise that has to be redone
whenever either side changes.

Three things happen after fusion
--------------------------------
1. Ticker filtering, because COST is both a ticker and the ordinary word
   appearing in "cost of sales" across most of the corpus.
2. Event-key deduplication, because the two vendors' transcripts of the NVIDIA
   Q4 FY2026 call are near-identical and would otherwise occupy half the
   budget with one event. The higher-tier rendering wins.
3. Tier labelling, so the model can see that one passage is a filing and
   another an auto-generated transcript. This is what lets it prefer
   "$41.4 billion" over "$41.4 million" rather than picking whichever scored
   higher.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

# Reciprocal rank fusion. k=60 is the value from the original paper and is
# deliberately not tuned: the point of RRF is that it works without tuning.
RRF_K = 60
CANDIDATES_PER_ARM = 30

# An arm only votes with candidates scoring at least this fraction of its own
# best score. Without it, an arm that found nothing relevant still contributes
# thirty ranked candidates and its noise displaces the other arm's good hits.
# Relative rather than absolute, because BM25 and cosine scores are not
# comparable and the whole point of RRF is not having to make them so.
ARM_FLOOR = 0.2

TIER_NAMES = {1: "filing", 2: "company release", 3: "corrected transcript",
              4: "auto transcript", 5: "presentation", 6: "broker note",
              7: "internal record"}


@dataclass
class ChunkRecord:
    chunk_id: int
    document_id: int
    text: str
    page_from: int
    page_to: int
    is_ocr: bool
    ticker: str | None
    doc_type: str | None
    source_tier: int
    event_key: str | None
    published_date: str | None
    filename: str


@dataclass
class Hit:
    record: ChunkRecord
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    suppressed: list[str] = field(default_factory=list)   # duplicates dropped

    @property
    def provenance(self) -> str:
        tier = TIER_NAMES.get(self.record.source_tier, "document")
        bits = [self.record.filename, tier]
        if self.record.published_date:
            bits.append(str(self.record.published_date))
        if self.record.is_ocr:
            bits.append("OCR")
        return " | ".join(bits)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend(Protocol):
    def vector_candidates(self, query_vec: np.ndarray, k: int,
                          ticker: str | None) -> list[tuple[int, float]]: ...

    def keyword_candidates(self, query: str, k: int,
                           ticker: str | None) -> list[tuple[int, float]]: ...

    def get(self, chunk_id: int) -> ChunkRecord: ...


TOKEN = re.compile(r"[a-z0-9][a-z0-9.'-]*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class InMemoryBackend:
    """BM25 plus exact cosine. Used by the tests and runnable without MySQL.

    At 421 chunks a full scan is about a millisecond, so this is also a
    perfectly serviceable production path for a corpus this size. MySQLBackend
    exists because the brief specifies MySQL, and because the corpus grows.
    """

    K1, B = 1.5, 0.75

    def __init__(self, records: Sequence[ChunkRecord], matrix: np.ndarray):
        self.records = {r.chunk_id: r for r in records}
        self.order = [r.chunk_id for r in records]
        self.matrix = matrix
        self.docs = {r.chunk_id: tokenize(r.text) for r in records}
        self.len_avg = sum(len(t) for t in self.docs.values()) / max(1, len(self.docs))
        self.df: Counter[str] = Counter()
        for tokens in self.docs.values():
            self.df.update(set(tokens))
        self.n = len(self.docs)

    def _allowed(self, chunk_id: int, ticker: str | None) -> bool:
        return ticker is None or self.records[chunk_id].ticker == ticker

    def vector_candidates(self, query_vec, k, ticker=None):
        scores = self.matrix @ query_vec
        pairs = [(cid, float(scores[i])) for i, cid in enumerate(self.order)
                 if self._allowed(cid, ticker)]
        pairs.sort(key=lambda p: -p[1])
        return pairs[:k]

    def keyword_candidates(self, query, k, ticker=None):
        terms = tokenize(query)
        out = []
        for chunk_id, tokens in self.docs.items():
            if not self._allowed(chunk_id, ticker):
                continue
            counts = Counter(tokens)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5)
                               / (self.df[term] + 0.5))
                norm = 1 - self.B + self.B * len(tokens) / self.len_avg
                score += idf * tf * (self.K1 + 1) / (tf + self.K1 * norm)
            if score > 0:
                out.append((chunk_id, score))
        out.sort(key=lambda p: -p[1])
        return out[:k]

    def get(self, chunk_id):
        return self.records[chunk_id]


class MySQLBackend:
    """FULLTEXT for the keyword arm, JSON vectors scored in NumPy.

    Vectors are held in MySQL rather than a separate vector store so the two
    halves cannot drift apart when a document is re-ingested. They are loaded
    into memory once; at this corpus size that is a few megabytes.
    """

    def __init__(self, conn):
        self.conn = conn
        self._matrix = None
        self._order: list[int] = []
        self._records: dict[int, ChunkRecord] = {}

    def _load(self):
        if self._matrix is not None:
            return
        cur = self.conn.cursor(dictionary=True)
        cur.execute("""
            SELECT c.id, c.document_id, c.text, c.page_from, c.page_to,
                   c.embedding, d.ticker, d.doc_type, d.source_tier,
                   d.event_key, d.published_date, d.filename,
                   EXISTS(SELECT 1 FROM document_pages p
                          WHERE p.document_id = c.document_id
                            AND p.page_no BETWEEN c.page_from AND c.page_to
                            AND p.method = 'ocr') AS is_ocr
            FROM chunks c JOIN documents d ON d.id = c.document_id""")
        rows = cur.fetchall()
        vectors = []
        for r in rows:
            self._order.append(r["id"])
            vectors.append(json.loads(r["embedding"]))
            self._records[r["id"]] = ChunkRecord(
                chunk_id=r["id"], document_id=r["document_id"], text=r["text"],
                page_from=r["page_from"], page_to=r["page_to"],
                is_ocr=bool(r["is_ocr"]), ticker=r["ticker"],
                doc_type=r["doc_type"], source_tier=r["source_tier"],
                event_key=r["event_key"],
                published_date=str(r["published_date"]) if r["published_date"] else None,
                filename=r["filename"])
        self._matrix = np.asarray(vectors, dtype=np.float32)

    def vector_candidates(self, query_vec, k, ticker=None):
        self._load()
        scores = self._matrix @ query_vec
        pairs = [(cid, float(scores[i])) for i, cid in enumerate(self._order)
                 if ticker is None or self._records[cid].ticker == ticker]
        pairs.sort(key=lambda p: -p[1])
        return pairs[:k]

    def keyword_candidates(self, query, k, ticker=None):
        self._load()
        sql = ("SELECT c.id, MATCH(c.text) AGAINST (%s IN NATURAL LANGUAGE MODE) "
               "AS score FROM chunks c JOIN documents d ON d.id = c.document_id "
               "WHERE MATCH(c.text) AGAINST (%s IN NATURAL LANGUAGE MODE)")
        params: list = [query, query]
        if ticker:
            sql += " AND d.ticker = %s"
            params.append(ticker)
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(k)
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return [(int(cid), float(score)) for cid, score in cur.fetchall()]

    def get(self, chunk_id):
        self._load()
        return self._records[chunk_id]


# ---------------------------------------------------------------------------
# Fusion and post-processing
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(*rankings: Sequence[int], k: int = RRF_K,
                           weights: Sequence[float] | None = None
                           ) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    weights = weights or [1.0] * len(rankings)
    for ranking, weight in zip(rankings, weights):
        for position, chunk_id in enumerate(ranking):
            scores[chunk_id] += weight / (k + position + 1)
    return scores


def apply_floor(candidates: list[tuple[int, float]],
                floor: float = ARM_FLOOR) -> list[tuple[int, float]]:
    """Keep only candidates scoring at least `floor` of this arm's best."""
    if not candidates:
        return candidates
    best = max(score for _, score in candidates)
    if best <= 0:
        return []
    return [(cid, score) for cid, score in candidates if score >= best * floor]


def deduplicate_events(hits: list[Hit]) -> list[Hit]:
    """Collapse chunks from different renderings of the same event.

    The two NVIDIA transcripts share an event_key, so without this one call
    occupies two of the eight slots with near-identical text. The better tier
    wins; ties go to the higher fused score. What was dropped is recorded on
    the surviving hit, so the agent can still say a second version exists --
    which matters when they disagree.
    """
    best: dict[str, Hit] = {}
    out: list[Hit] = []
    for hit in hits:
        key = hit.record.event_key
        if not key:
            out.append(hit)
            continue
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = hit
            out.append(hit)
            continue
        # Several chunks of the same document collapse onto one hit, so the
        # record of what was dropped is a set of filenames, not a list of
        # chunk events.
        if hit.record.source_tier < incumbent.record.source_tier:
            dropped = set(incumbent.suppressed) | {incumbent.record.filename}
            hit.suppressed = sorted(dropped - {hit.record.filename})
            out[out.index(incumbent)] = hit
            best[key] = hit
        elif hit.record.filename != incumbent.record.filename:
            incumbent.suppressed = sorted(
                set(incumbent.suppressed) | {hit.record.filename})
    return out


def search(query: str, backend: Backend, embedder, k: int = 8,
           ticker: str | None = None, candidates: int = CANDIDATES_PER_ARM,
           dedupe: bool = True, weights: Sequence[float] | None = None,
           floor: float = ARM_FLOOR) -> list[Hit]:
    query_vec = embedder.encode([query], is_query=True)[0]
    vector = apply_floor(backend.vector_candidates(query_vec, candidates, ticker), floor)
    keyword = apply_floor(backend.keyword_candidates(query, candidates, ticker), floor)

    vector_rank = {cid: i for i, (cid, _) in enumerate(vector)}
    keyword_rank = {cid: i for i, (cid, _) in enumerate(keyword)}
    fused = reciprocal_rank_fusion([cid for cid, _ in vector],
                                   [cid for cid, _ in keyword],
                                   weights=weights)

    hits = [Hit(record=backend.get(cid), score=score,
                vector_rank=vector_rank.get(cid),
                keyword_rank=keyword_rank.get(cid))
            for cid, score in sorted(fused.items(), key=lambda p: -p[1])]

    if dedupe:
        hits = deduplicate_events(hits)
    return hits[:k]


# ---------------------------------------------------------------------------
# Rendering for the agent
# ---------------------------------------------------------------------------

def format_for_prompt(hits: Sequence[Hit]) -> str:
    """Every passage arrives labelled with its source and tier.

    Without the tier the model has no reason to prefer a corrected transcript
    over an auto-generated one, and no way to caveat a figure read by OCR.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        r = hit.record
        header = (f"[{i}] {r.filename} | {TIER_NAMES.get(r.source_tier, 'document')}"
                  f" (tier {r.source_tier})"
                  + (f" | {r.published_date}" if r.published_date else "")
                  + f" | pages {r.page_from}-{r.page_to}"
                  + (" | text recovered by OCR" if r.is_ocr else ""))
        if hit.suppressed:
            header += (f"\n    note: another version of this same event exists "
                       f"({', '.join(sorted(set(hit.suppressed)))}); this is the "
                       f"more authoritative rendering")
        blocks.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(blocks)
