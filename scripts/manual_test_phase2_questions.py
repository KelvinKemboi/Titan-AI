"""
Manual test: drives the real tool-calling loop against all six Phase 2
example questions and checks each one calls a real tool and produces a
sourced final answer.

Requires ANTHROPIC_API_KEY, and a ticker with a completed scan
(`streamlit run app.py` -> Initialize Market Scan, or
`python -m src.analytics.scheduler`) AND at least 2 quarters of ingested
earnings data for that ticker (`python -m scripts.backfill_earnings_transcripts`,
or the calendar scheduler over time - requires EARNINGS_PROVIDER_API_KEY).

Run:
    python -m scripts.manual_test_phase2_questions [TICKER]
"""
import json
import sys

import anthropic
from dotenv import load_dotenv

from src.agents.tools import TOOLS, call_tool
from src.data.db import SessionLocal

load_dotenv()

MODEL = "claude-sonnet-5"

# (question, at least one of these tool names must be called)
QUESTIONS = [
    ("Summarize {ticker}'s latest earnings call.", {"get_earnings_insight"}),
    ("Did {ticker}'s guidance improve or worsen last quarter?", {"get_earnings_insight", "get_qoq_changes"}),
    ("Why did {ticker}'s management sound the way they did on the earnings call?", {"get_earnings_insight"}),
    ("What are {ticker}'s biggest risks right now?", {"get_earnings_insight"}),
    ("What changed for {ticker} from last quarter?", {"get_qoq_changes", "get_earnings_insight"}),
    ("Give me a full analyst report on {ticker}.", {"get_analyst_report"}),
]


def _ask(client, db, question: str, expected_tools: set) -> bool:
    messages = [{"role": "user", "content": question}]
    response = client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

    print(f"\nQ: {question}")
    print(f"  tool calls: {[b.name for b in tool_use_blocks]}")

    if not tool_use_blocks:
        print("  FAIL: no tool called - model may be answering from memory.")
        return False
    if not any(b.name in expected_tools for b in tool_use_blocks):
        print(f"  FAIL: expected one of {expected_tools}, got {[b.name for b in tool_use_blocks]}.")
        return False

    messages.append({"role": "assistant", "content": response.content})
    tool_results = []
    all_sources = []
    for block in tool_use_blocks:
        try:
            result = call_tool(db, block.name, block.input)
            content = result.model_dump_json()
            all_sources.extend(result.sources)
        except ValueError as exc:
            content = json.dumps({"error": str(exc)})
        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
    messages.append({"role": "user", "content": tool_results})

    final = client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
    final_text = "\n".join(b.text for b in final.content if b.type == "text")
    print(f"  answer: {final_text[:300]}")
    print(f"  sources: {[(s.type, s.ticker, s.ref_id) for s in all_sources]}")

    if not all_sources:
        print("  FAIL: tool call produced no source metadata - attribution contract violated.")
        return False

    print("  PASS")
    return True


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    client = anthropic.Anthropic()
    db = SessionLocal()
    try:
        results = [
            _ask(client, db, template.format(ticker=ticker), expected)
            for template, expected in QUESTIONS
        ]
    finally:
        db.close()

    passed = sum(results)
    print(f"\n{passed}/{len(results)} Phase 2 example questions passed for {ticker}.")


if __name__ == "__main__":
    main()
