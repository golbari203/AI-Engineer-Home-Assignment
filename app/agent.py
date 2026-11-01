"""
The agent: a tool-calling loop plus the answer policy.

Most of the work in this file is the system prompt, and most of the prompt is
there because of something specific in the data. Each rule below traces to a
document in the folder, which is the test of whether a rule earns its place:
a rule with no failing case behind it is decoration.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from app.tools import DISPATCH, TOOL_SCHEMAS, Store, run_tool

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_ITERATIONS = 8

SYSTEM_PROMPT = """You are a research assistant for a family office. The
investor is not technical. They need answers they can act on, and they need to
be able to tell how far to trust each one.

Answer only from the tools. You have no other source of information about these
companies or this portfolio.

1. The documents override anything you think you know.
   These are real companies, but the figures, people and events in these
   documents are the client's own records and may differ from what you learned
   elsewhere. If a document says Apple's chief executive is Daniel Whitford,
   that is the answer. Never correct a document from memory, and never fill a
   gap with outside knowledge.

2. Numbers come from tools, never from your own arithmetic.
   Use get_holdings for anything the client owns, get_transactions for trade
   history, get_facts for reported company figures. Do not add up positions
   yourself and do not compute a total from text you retrieved.

3. State the basis and the period for every company figure.
   "$4.68" is meaningless; "$4.68 non-GAAP for Q1 2026" is an answer. If a
   company reports both GAAP and non-GAAP, say which you are quoting. If a
   figure is guidance rather than a result, say so.

4. When sources disagree, show the disagreement.
   get_facts returns conflicts explicitly and names the authoritative version.
   Lead with that version, then say plainly that another source differs and
   why. Do not average disputed numbers and do not quietly pick one.
   Reliability runs: filings, then company releases, then corrected
   transcripts, then auto-generated transcripts, then presentations, then
   broker notes. A figure the company has restated supersedes the one it
   replaced.

5. For a fact that changed on a date, the most specific dated statement wins.
   A later document that merely repeats stale boilerplate does not overturn an
   explicit, dated announcement. Say what changed and when, and mention that
   later documents still show the old version.

6. Flag what is uncertain, using get_data_quality rather than vague hedging.
   If a figure came from a scanned page, or if the client's own records
   disagree with the custodian, say so in one sentence and keep going.

7. If nothing on file answers the question, say so.
   "There is nothing about Exxon's Q3 2026 results in your documents" is a good
   answer. Inventing one is not. Do not pad with general knowledge about the
   industry.

Style: lead with the answer, then the support. Short sentences. Name the source
document and its date for every figure. No preamble about what you are about to
do. When the answer is a number the investor might act on, make the as-of date
unmissable."""


@dataclass
class Turn:
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    iterations: int = 0

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


def _text_of(content) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", "") == "text")


def _citations_from(tool: str, result: dict) -> list[dict]:
    """Collect what the answer rests on, so the UI can show it.

    An answer the investor cannot trace back to a document is one they have to
    take on faith, which is the opposite of what this system is for.
    """
    if not isinstance(result, dict):
        return []
    if tool == "search_documents":
        return list(result.get("sources") or [])
    out = []
    for position in result.get("positions") or []:
        out.append({"file": f"{position['custodian']} statement",
                    "date": position.get("as_of_date"),
                    "detail": f"{position['ticker']} position"})
    for conflict in result.get("conflicts") or []:
        for version in conflict.get("versions") or []:
            out.append({"file": f"document {version.get('document_id')}",
                        "date": version.get("stated_on"),
                        "detail": f"{conflict['metric']} {conflict['fiscal_period']}"})
    return out


def ask(question: str, store: Store, client=None, model: str = DEFAULT_MODEL,
        history: list[dict] | None = None,
        max_iterations: int = MAX_ITERATIONS) -> tuple[Turn, list[dict]]:
    """Run one turn. Returns the turn and the updated message history.

    Token counts are recorded per turn rather than estimated, so the cost
    figures in DESIGN.md come from measurement.
    """
    if client is None:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["AIEHA_API_KEY"])

    messages = list(history or []) + [{"role": "user", "content": question}]
    turn = Turn()
    started = time.time()

    for _ in range(max_iterations):
        turn.iterations += 1
        response = client.messages.create(
            model=model, max_tokens=2000, system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages)

        usage = getattr(response, "usage", None)
        if usage is not None:
            turn.input_tokens += getattr(usage, "input_tokens", 0)
            turn.output_tokens += getattr(usage, "output_tokens", 0)

        messages.append({"role": "assistant", "content": response.content})

        if getattr(response, "stop_reason", None) != "tool_use":
            turn.answer = _text_of(response.content)
            break

        results = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            result = run_tool(store, block.name, dict(block.input))
            turn.tool_calls.append({"name": block.name, "input": dict(block.input)})
            turn.citations.extend(_citations_from(block.name, result))
            results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": results})
    else:
        turn.answer = ("I could not finish working through that question. "
                       "Please try asking it in smaller parts.")

    turn.latency_ms = int((time.time() - started) * 1000)
    return turn, messages


def log_turn(conn, question: str, turn: Turn) -> None:
    """Persist usage so cost is measured rather than estimated."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO query_log (question, tool_calls, input_tokens, "
        "output_tokens, latency_ms) VALUES (%s, %s, %s, %s, %s)",
        (question, json.dumps(turn.tool_calls), turn.input_tokens,
         turn.output_tokens, turn.latency_ms))
    conn.commit()
