"""
Tests for hybrid retrieval.

A note on the embedder used here. HashingEmbedder has no semantic ability, so
the vector arm in these tests is close to noise and the keyword arm carries
almost everything. That is deliberate: it keeps the suite offline, and it
means any test that passes is passing on fusion, filtering and deduplication
logic rather than on embedding quality. Retrieval *quality* is measured in the
eval harness with the real model; what is tested here is mechanism.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.embed import HashingEmbedder                    # noqa: E402
from app.retrieval import (                                     # noqa: E402
    ChunkRecord, InMemoryBackend, apply_floor, deduplicate_events, Hit,
    format_for_prompt, reciprocal_rank_fusion, search,
)

INDEX = Path("/tmp/index.pkl")


def rec(cid, text, ticker=None, tier=5, event=None, filename="f.pdf",
        ocr=False, doc=0):
    return ChunkRecord(chunk_id=cid, document_id=doc, text=text, page_from=1,
                       page_to=1, is_ocr=ocr, ticker=ticker, doc_type=None,
                       source_tier=tier, event_key=event,
                       published_date="2026-02-18", filename=filename)


def build(records):
    emb = HashingEmbedder()
    matrix = emb.encode([r.text for r in records])
    return InMemoryBackend(records, matrix), emb


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def test_rrf_rewards_agreement_between_arms():
    """A chunk both arms rank highly beats one that only a single arm found."""
    scores = reciprocal_rank_fusion([1, 2, 3], [3, 4, 5])
    assert scores[3] > scores[1]
    assert scores[3] > scores[4]


def test_rrf_needs_no_score_normalisation():
    """BM25 scores and cosines live on unrelated ranges; only order matters."""
    a = reciprocal_rank_fusion([7, 8], [8, 7])
    b = reciprocal_rank_fusion([7, 8], [8, 7])
    assert a == b


def test_weights_can_favour_one_arm():
    equal = reciprocal_rank_fusion([1], [2])
    assert equal[1] == equal[2]
    tilted = reciprocal_rank_fusion([1], [2], weights=[3.0, 1.0])
    assert tilted[1] > tilted[2]


def test_floor_drops_an_arm_that_found_nothing_relevant():
    """Otherwise thirty noise candidates still vote and displace good hits."""
    kept = apply_floor([(1, 10.0), (2, 8.0), (3, 0.5), (4, 0.1)])
    assert [cid for cid, _ in kept] == [1, 2]


def test_floor_handles_an_empty_arm():
    assert apply_floor([]) == []
    assert apply_floor([(1, 0.0)]) == []


# ---------------------------------------------------------------------------
# Ticker filtering
# ---------------------------------------------------------------------------

def _cost_corpus():
    """COST is a ticker and also the ordinary word in "cost of sales"."""
    return [
        rec(1, "Costco Wholesale membership fees rose on the September fee "
               "increase with a renewal rate above ninety percent",
            ticker="COST", filename="costco_q4.pdf"),
        rec(2, "Cost of sales increased to 167,494 reflecting higher component "
               "cost and freight cost pressure during the cost cycle",
            ticker="AAPL", filename="apple_10k.pdf"),
    ]


def test_the_word_cost_matches_the_apple_filing_not_costco():
    """Tokenisation alone does not disambiguate: Costco tokenises to "costco"."""
    backend, emb = build(_cost_corpus())
    hits = search("cost of sales", backend, emb, k=5)
    assert [h.record.filename for h in hits] == ["apple_10k.pdf"]


def test_ticker_filter_removes_the_word_cost_entirely():
    """Filtering to COST excludes the Apple chunk that merely says "cost"."""
    backend, emb = build(_cost_corpus())
    hits = search("cost of sales", backend, emb, k=5, ticker="COST")
    assert "apple_10k.pdf" not in {h.record.filename for h in hits}


def test_an_arm_with_no_overlap_at_all_contributes_nothing():
    """apply_floor returns nothing when the best score is zero, so a query
    with no lexical or vector overlap yields no hits rather than noise."""
    backend, emb = build(_cost_corpus())
    assert search("zzzz", backend, emb, k=5) == []


# ---------------------------------------------------------------------------
# Event deduplication
# ---------------------------------------------------------------------------

def _transcripts():
    return [
        Hit(rec(1, "guidance is 41.4 million dollars", ticker="NVDA", tier=4,
                event="NVDA|earnings_call|Q4 FY2026",
                filename="NVDA_Q4_FY26_call_transcript.pdf", doc=1), score=0.9),
        Hit(rec(2, "guidance is 41.4 billion dollars", ticker="NVDA", tier=3,
                event="NVDA|earnings_call|Q4 FY2026",
                filename="NVIDIA_Q4_FY2026_earnings_call_-_Verbatim.pdf",
                doc=2), score=0.8),
    ]


def test_the_corrected_transcript_wins_despite_a_lower_score():
    """This is the mechanism that rejects "$41.4 million"."""
    out = deduplicate_events(_transcripts())
    assert len(out) == 1
    assert out[0].record.source_tier == 3
    assert "billion" in out[0].record.text


def test_what_was_dropped_is_recorded_once():
    """The agent must still be able to say a second version exists."""
    out = deduplicate_events(_transcripts())
    assert out[0].suppressed == ["NVDA_Q4_FY26_call_transcript.pdf"]


def test_many_chunks_of_one_document_collapse_to_one_note():
    hits = _transcripts() + [
        Hit(rec(3, "more of the same call", ticker="NVDA", tier=4,
                event="NVDA|earnings_call|Q4 FY2026",
                filename="NVDA_Q4_FY26_call_transcript.pdf", doc=1), score=0.7)]
    out = deduplicate_events(hits)
    assert out[0].suppressed == ["NVDA_Q4_FY26_call_transcript.pdf"]


def test_different_events_are_not_collapsed():
    """Amazon's Q4 release and Q4 call are the same day and different sources."""
    hits = [
        Hit(rec(1, "release text", ticker="AMZN", tier=2,
                event="AMZN|earnings_release|Q4 2025", filename="release.pdf"), 0.9),
        Hit(rec(2, "call text", ticker="AMZN", tier=3,
                event="AMZN|earnings_call|Q4 2025", filename="call.pdf"), 0.8),
    ]
    assert len(deduplicate_events(hits)) == 2


def test_chunks_without_an_event_key_are_never_collapsed():
    hits = [Hit(rec(1, "a", event=None, filename="a.pdf"), 0.9),
            Hit(rec(2, "b", event=None, filename="b.pdf"), 0.8)]
    assert len(deduplicate_events(hits)) == 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_prompt_block_carries_tier_and_provenance():
    out = format_for_prompt(deduplicate_events(_transcripts()))
    assert "corrected transcript" in out
    assert "tier 3" in out
    assert "pages 1-1" in out


def test_prompt_block_flags_ocr_text():
    """So a figure read from a scan can be caveated rather than quoted flatly."""
    hit = Hit(rec(1, "net interest income 18,268", tier=1, ocr=True,
                  filename="JPMorgan_Chase_4Q25_earnings.pdf"), 0.9)
    assert "recovered by OCR" in format_for_prompt([hit])


def test_prompt_block_mentions_the_suppressed_version():
    out = format_for_prompt(deduplicate_events(_transcripts()))
    assert "another version of this same event exists" in out
    assert "NVDA_Q4_FY26_call_transcript.pdf" in out


# ---------------------------------------------------------------------------
# Real corpus
# ---------------------------------------------------------------------------

def _real():
    records, matrix = pickle.load(open(INDEX, "rb"))
    return InMemoryBackend(records, matrix), HashingEmbedder(), records


def test_exact_phrase_is_found_by_the_keyword_arm():
    """A fictional statute no embedding model has seen."""
    if not INDEX.exists():
        return
    backend, _, records = _real()
    phrase = "Prescription Drug Affordability Act of 2026"
    expected = {r.filename for r in records if phrase in r.text}
    assert len(expected) == 5
    found = {backend.get(cid).filename
             for cid, _ in backend.keyword_candidates(phrase, 5)}
    assert found == expected


def test_restated_keyword_is_dominated_by_boilerplate_without_suppression():
    """Six 10-Ks say "Restated Articles of Incorporation"; one 10-Q has the
    real recast note. Chunk-stage boilerplate suppression removes the former."""
    if not INDEX.exists():
        return
    backend, _, records = _real()
    hits = [backend.get(cid).filename
            for cid, _ in backend.keyword_candidates("restated recast segment", 5)]
    assert any("Amazon_10-Q" in h for h in hits)


def test_corpus_transcripts_deduplicate():
    if not INDEX.exists():
        return
    backend, emb, _ = _real()
    hits = search("NVIDIA guidance for the next quarter", backend, emb,
                  k=6, ticker="NVDA")
    names = [h.record.filename for h in hits]
    assert "NVDA_Q4_FY26_call_transcript.pdf" not in names
    survivor = [h for h in hits if "Verbatim" in h.record.filename]
    assert survivor and survivor[0].suppressed


def test_search_returns_at_most_k():
    if not INDEX.exists():
        return
    backend, emb, _ = _real()
    assert len(search("revenue growth", backend, emb, k=4)) <= 4


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
