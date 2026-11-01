"""
Evaluation harness.

Three scorers, reported separately because they fail for different reasons and
a single blended number would hide which.

  tool       Does the tool return the right figure? Deterministic, offline,
             and the strictest of the three: 2,300 Costco shares is right or
             it is not.
  retrieval  Does the expected document reach the top k? Offline. Measures the
             index, not the model.
  answer     Does the final answer say the right thing? Needs --live, because
             it runs the agent and grades with a judge.

Reporting per category matters more than the headline. A build that scores 90%
overall while failing every conflict-resolution question is worse than one at
80% that handles them, and an average hides exactly that.

Cost is measured here rather than estimated: the agent records token counts per
turn, and a live run aggregates them into the figures used in DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml                                                     # noqa: E402

from app.filenames import name_key                              # noqa: E402
from app.tools import run_tool                                  # noqa: E402

GOLDEN = Path(__file__).parent / "golden.yaml"
DEFAULT_K = 8


@dataclass
class Result:
    id: str
    category: str
    scorer: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    turns: int = 0

    def add(self, result: Result) -> None:
        self.results.append(result)

    def by(self, key) -> dict[str, tuple[int, int]]:
        buckets: dict[str, list[Result]] = defaultdict(list)
        for r in self.results:
            buckets[key(r)].append(r)
        return {k: (sum(1 for r in v if r.passed), len(v))
                for k, v in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Expectation matching
# ---------------------------------------------------------------------------

def _derive(output: dict, key: str):
    """Map an expectation key onto the tool's actual output shape."""
    if key == "positions_count":
        return len(output.get("positions") or [])
    if key == "error_count":
        return len(output.get("errors") or [])
    if key == "notes_count":
        return len(output.get("notes_on_file") or [])
    if key == "min_transactions":
        return len(output.get("transactions") or [])
    if key == "min_exception_count":
        return output.get("exception_count")
    if key == "has_note":
        return bool(output.get("note"))
    if key == "has_book_disagreement":
        return any(p.get("book_disagreement")
                   for p in output.get("positions") or [])
    if key == "not_yet_reported":
        periods = output.get("periods") or []
        return bool(periods and periods[0].get("status"))
    if key == "target_low":
        return (output.get("target_range") or {}).get("low")
    if key == "target_high":
        return (output.get("target_range") or {}).get("high")
    if key == "total_quantity":
        return output.get("total_quantity", 0)
    return output.get(key)


def check_expectation(output: dict, key: str, expected) -> tuple[bool, str]:
    actual = _derive(output, key)
    if isinstance(expected, dict) and "approx" in expected:
        tol = expected.get("tol", 0.01)
        ok = actual is not None and abs(float(actual) - expected["approx"]) <= tol
        return ok, f"{key}={actual} vs {expected['approx']}±{tol}"
    if key.startswith("min_"):
        ok = actual is not None and actual >= expected
        return ok, f"{key}={actual} (need >= {expected})"
    ok = actual == expected
    return ok, f"{key}={actual!r} vs {expected!r}"


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def score_tool(case: dict, store) -> Result | None:
    spec = case.get("tool")
    if not spec:
        return None
    output = run_tool(store, spec["name"], spec.get("args") or {})
    failures = []
    for key, expected in (case.get("expect") or {}).items():
        ok, detail = check_expectation(output, key, expected)
        if not ok:
            failures.append(detail)
    return Result(case["id"], case["category"], "tool", not failures,
                  "; ".join(failures))


def score_retrieval(case: dict, store, k: int = DEFAULT_K) -> Result | None:
    spec = case.get("retrieval")
    if not spec:
        return None
    output = run_tool(store, "search_documents", {
        "query": spec["query"], "ticker": spec.get("ticker"), "k": k})
    files = {s["file"] for s in output.get("sources") or []}
    # Compared by name_key: the golden file was written against one export's
    # spellings, and a portfolio saved with spaces for underscores is the same
    # set of documents.
    found = {name_key(f) for f in files}

    expected = set(spec.get("expect_any") or [])
    forbidden = set(spec.get("forbid") or [])
    problems = []
    if expected and not (found & {name_key(e) for e in expected}):
        problems.append(f"none of {sorted(expected)} in top {k}; got {sorted(files)}")
    hit = sorted(f for f in files if name_key(f) in {name_key(x) for x in forbidden})
    if hit:
        problems.append(f"forbidden source returned: {hit}")
    return Result(case["id"], case["category"], "retrieval", not problems,
                  "; ".join(problems))


JUDGE_PROMPT = """You are grading an assistant's answer for a family office.

Judge only whether the answer is factually consistent with the expectation and
does not assert anything the expectation contradicts. Style, length and extra
caveats do not matter. Answer with a single word, PASS or FAIL, then a short
reason on the same line."""


def score_answer(case: dict, store, client, model: str,
                 report: Report) -> Result | None:
    spec = case.get("answer")
    if not spec:
        return None
    from app.agent import ask

    turn, _ = ask(case["question"], store, client=client, model=model)
    report.input_tokens += turn.input_tokens
    report.output_tokens += turn.output_tokens
    report.latency_ms += turn.latency_ms
    report.turns += 1

    text = turn.answer.lower()
    problems = [f"missing {phrase!r}" for phrase in spec.get("must_mention") or []
                if phrase.lower() not in text]
    problems += [f"should not say {phrase!r}"
                 for phrase in spec.get("must_not_mention") or []
                 if phrase.lower() in text]

    if not problems and client is not None:
        verdict = _judge(case, turn.answer, client, model)
        if verdict.upper().startswith("FAIL"):
            problems.append(f"judge: {verdict}")
    return Result(case["id"], case["category"], "answer", not problems,
                  "; ".join(problems))


def _judge(case: dict, answer: str, client, model: str) -> str:
    expectation = json.dumps(case.get("answer"), default=str)
    response = client.messages.create(
        model=model, max_tokens=150, system=JUDGE_PROMPT,
        messages=[{"role": "user", "content":
                   f"Question: {case['question']}\n"
                   f"Why this matters: {case.get('why', '')}\n"
                   f"Expectation: {expectation}\n\nAnswer:\n{answer}"}])
    return "".join(b.text for b in response.content
                   if getattr(b, "type", "") == "text").strip()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(cases: list[dict], store, live: bool = False, k: int = DEFAULT_K,
        model: str = "claude-sonnet-5-5") -> Report:
    report = Report()
    client = None
    if live:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["AIEHA_API_KEY"])

    for case in cases:
        for result in (score_tool(case, store),
                       score_retrieval(case, store, k),
                       score_answer(case, store, client, model, report)
                       if live else None):
            if result is not None:
                report.add(result)
    return report


def render(report: Report, verbose: bool = False) -> str:
    lines = []
    width = 24

    lines.append("By category")
    for name, (passed, total) in report.by(lambda r: r.category).items():
        bar = "#" * int(20 * passed / total) if total else ""
        lines.append(f"  {name:<{width}} {passed:>3}/{total:<3} "
                     f"{passed / total:>5.0%}  {bar}")

    lines.append("")
    lines.append("By scorer")
    for name, (passed, total) in report.by(lambda r: r.scorer).items():
        lines.append(f"  {name:<{width}} {passed:>3}/{total:<3} "
                     f"{passed / total:>5.0%}")

    failures = [r for r in report.results if not r.passed]
    if failures:
        lines.append("")
        lines.append(f"Failures ({len(failures)})")
        for r in failures:
            lines.append(f"  [{r.scorer}] {r.id}: {r.detail[:150]}")

    if report.turns:
        avg_in = report.input_tokens / report.turns
        avg_out = report.output_tokens / report.turns
        lines += ["", "Cost per question (measured)",
                  f"  input tokens   {avg_in:,.0f}",
                  f"  output tokens  {avg_out:,.0f}",
                  f"  latency        {report.latency_ms / report.turns / 1000:.1f}s",
                  f"  questions      {report.turns}"]

    passed = sum(1 for r in report.results if r.passed)
    lines += ["", f"Total {passed}/{len(report.results)} "
                  f"({passed / max(1, len(report.results)):.0%})"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="run the agent and grade answers (needs an API key)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--category")
    parser.add_argument("--folder", default="/mnt/user-data/uploads")
    parser.add_argument("--real-embeddings", action="store_true",
                        help="use the real model instead of the offline fallback")
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--mysql", action="store_true",
                        help="read the store from MySQL instead of the files")
    args = parser.parse_args()

    cases = yaml.safe_load(GOLDEN.read_text())
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    if args.mysql:
        from app import db
        store = db.load_store(db.connect())
    else:
        from app.offline_store import attach_index, build_store
        store = build_store(args.folder, with_index=False)
        attach_index(store, args.folder,
                     offline_embedder=not args.real_embeddings)

    started = time.time()
    report = run(cases, store, live=args.live, k=args.k, model=args.model)
    print(render(report))
    print(f"\n{len(cases)} questions, {time.time() - started:.1f}s"
          + ("" if args.real_embeddings else
             "\nnote: offline embedder in use; retrieval scores understate the "
             "real index. Re-run with --real-embeddings before quoting them."))
    return 0 if all(r.passed for r in report.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
