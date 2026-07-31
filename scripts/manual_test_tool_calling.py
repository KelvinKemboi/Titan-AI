"""
Manual test for docs/issues.md #10: prompt Claude with a comparison
question and confirm it calls `compare_tickers` rather than fabricating
numbers from training data. Requires ANTHROPIC_API_KEY and a Postgres
with at least one completed scan (`streamlit run app.py` -> Initialize
Market Scan, or `python -m src.analytics.scheduler`).

Run:
    python -m scripts.manual_test_tool_calling
"""
import json

import anthropic
from dotenv import load_dotenv

from src.agents.tools.factor_tools import TOOLS, call_tool
from src.data.db import SessionLocal

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
QUESTION = "Compare MSFT and GOOGL's factor scores - which one is stronger and why?"


def main():
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": QUESTION}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

    print(f"stop_reason: {response.stop_reason}")
    print(f"tool calls: {[b.name for b in tool_use_blocks]}")

    if not tool_use_blocks:
        print("FAIL: model did not call a tool at all - it may be answering from memory.")
        for block in response.content:
            if block.type == "text":
                print("model text:", block.text)
        return

    if not any(b.name == "compare_tickers" for b in tool_use_blocks):
        print("FAIL: model called a tool, but not compare_tickers.")
        return

    print("PASS: model called compare_tickers for a comparison question.")

    # Execute the tool call(s) for real and feed results back, to also
    # sanity-check the model's final answer quotes the tool's numbers.
    db = SessionLocal()
    try:
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_use_blocks:
            try:
                result = call_tool(db, block.name, block.input)
                content = result.model_dump_json()
            except ValueError as exc:
                content = json.dumps({"error": str(exc)})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                }
            )
        messages.append({"role": "user", "content": tool_results})

        final = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        for block in final.content:
            if block.type == "text":
                print("\nfinal model response:\n", block.text)
    finally:
        db.close()


if __name__ == "__main__":
    main()
