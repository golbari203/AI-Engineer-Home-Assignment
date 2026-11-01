"""
What it costs to run.

Two ways to get the number, and this module does both:

  measured    A live eval run records real token counts per turn via
              agent.Turn, and those are the figures to quote.
  estimated   Without an API key, the fixed parts of every prompt can still
              be measured exactly -- the system prompt, the tool schemas and
              the actual JSON each tool returns on this corpus. That gives a
              defensible estimate rather than a guess, and it identifies which
              component to attack first.

Characters are converted at four per token, which is close enough for English
prose and JSON to size a budget. The point is the shape of the bill, not the
third significant figure.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHARS_PER_TOKEN = 4

# Published per-million-token prices vary by model and change over time, so
# they are inputs here rather than constants baked into a claim.
DEFAULT_INPUT_PRICE = 3.00      # USD per million input tokens
DEFAULT_OUTPUT_PRICE = 15.00    # USD per million output tokens
CACHE_READ_DISCOUNT = 0.10      # cached prompt prefix costs about a tenth


@dataclass
class Component:
    name: str
    chars: int
    note: str = ""

    @property
    def tokens(self) -> int:
        return round(self.chars / CHARS_PER_TOKEN)


def measure_fixed() -> list[Component]:
    from app.agent import SYSTEM_PROMPT
    from app.tools import TOOL_SCHEMAS
    return [
        Component("system prompt", len(SYSTEM_PROMPT),
                  "identical every turn, so fully cacheable"),
        Component("tool schemas", len(json.dumps(TOOL_SCHEMAS)),
                  "identical every turn, so fully cacheable"),
    ]


def measure_tool_results(store) -> list[Component]:
    """The real JSON each tool returns on this corpus, not a guess."""
    from app.tools import run_tool

    probes = [
        ("get_holdings (one ticker)", "get_holdings", {"ticker": "COST"}),
        ("get_holdings (whole portfolio)", "get_holdings", {}),
        ("get_transactions", "get_transactions", {"ticker": "NVDA"}),
        ("get_research", "get_research", {"ticker": "LLY"}),
        ("get_consensus", "get_consensus", {"ticker": "UNH"}),
        ("get_data_quality", "get_data_quality", {}),
        ("search_documents (k=8)", "search_documents",
         {"query": "AWS revenue growth and capacity", "ticker": "AMZN"}),
        ("search_documents (k=4)", "search_documents",
         {"query": "AWS revenue growth and capacity", "ticker": "AMZN", "k": 4}),
    ]
    out = []
    for label, name, args in probes:
        payload = json.dumps(run_tool(store, name, args), default=str)
        out.append(Component(label, len(payload)))
    return out


def model_turn(fixed_tokens: int, result_tokens: int, calls: int = 2,
               output_tokens: int = 400) -> dict:
    """Input is re-sent on every iteration, which is the cost that surprises.

    A two-tool question makes three model calls: one to choose the first tool,
    one to choose the second with the first result in context, and one to
    answer with both. The fixed prefix is paid each time.
    """
    iterations = calls + 1
    uncached = fixed_tokens * iterations + result_tokens * (iterations - 1)
    cached = (fixed_tokens * (iterations - 1) * CACHE_READ_DISCOUNT
              + fixed_tokens + result_tokens * (iterations - 1))
    return {"iterations": iterations, "input_uncached": round(uncached),
            "input_cached": round(cached), "output": output_tokens}


def price(tokens_in: int, tokens_out: int,
          input_price: float = DEFAULT_INPUT_PRICE,
          output_price: float = DEFAULT_OUTPUT_PRICE) -> float:
    return (tokens_in * input_price + tokens_out * output_price) / 1_000_000


def report(store, questions_per_month: int = 400) -> str:
    fixed = measure_fixed()
    results = measure_tool_results(store)
    fixed_tokens = sum(c.tokens for c in fixed)
    typical_result = round(
        sum(c.tokens for c in results if "k=4" not in c.name and
            "whole portfolio" not in c.name) / max(1, len(results) - 2))

    lines = ["Fixed prompt, measured"]
    for c in fixed:
        lines.append(f"  {c.name:<34} {c.tokens:>6,} tokens   {c.note}")
    lines.append(f"  {'total fixed prefix':<34} {fixed_tokens:>6,} tokens")

    lines += ["", "Tool results on this corpus, measured"]
    for c in results:
        lines.append(f"  {c.name:<34} {c.tokens:>6,} tokens")

    lines += ["", f"Typical question (2 tool calls, ~{typical_result:,}-token results)"]
    turn = model_turn(fixed_tokens, typical_result)
    for label, key in (("model calls", "iterations"),
                       ("input tokens, no caching", "input_uncached"),
                       ("input tokens, with caching", "input_cached"),
                       ("output tokens", "output")):
        lines.append(f"  {label:<34} {turn[key]:>6,}")

    plain = price(turn["input_uncached"], turn["output"])
    cached = price(turn["input_cached"], turn["output"])
    lines += ["",
              f"  cost per question, no caching   ${plain:.4f}",
              f"  cost per question, with caching ${cached:.4f}  "
              f"({1 - cached / plain:.0%} lower)",
              f"  {questions_per_month} questions/month          "
              f"${cached * questions_per_month:,.2f}"]

    lines += ["", "One-off ingestion",
              "  1 classification call per document, cached by content hash",
              "  1 extraction call per document for statements and facts",
              "  62 documents, so a few dollars once, and free on re-ingest",
              "  embeddings run locally and cost nothing",
              "",
              "Levers, in order of effect",
              "  1. prompt caching on the fixed prefix (above)",
              "  2. a smaller model for classification and extraction",
              "  3. k=4 instead of k=8 roughly halves the retrieval payload",
              "",
              "Prices are inputs, not claims. Check the current rate card and "
              "re-run.\nFor measured rather than estimated figures, run "
              "eval/run_eval.py --live."]
    return "\n".join(lines)


if __name__ == "__main__":
    from app.offline_store import build_store
    print(report(build_store("/mnt/user-data/uploads", with_index=True)))
