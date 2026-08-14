"""
Manual test: exercise the Chat/Agent Service (#11) against a real Postgres and
the live Anthropic API. The three checks mirror #11's acceptance criteria:

  1. A question answerable by one tool call gets a correct, sourced response.
  2. The system prompt keeps the model off outside/training knowledge for a
     fact Titan has no tool for (technical-design.md §6) - "what sector is
     Apple in" should be declined rather than answered from training data,
     since there's no company-profile tool yet (#1 isn't built).
  3. An unknown ticker produces a graceful response, not a raw exception.

Requires ANTHROPIC_API_KEY and a Postgres with at least one completed scan
that includes AAPL (`streamlit run app.py` -> Initialize Market Scan, or
`python -m src.analytics.scheduler`).

Run:
    python -m scripts.manual_test_chat_service
"""
from dotenv import load_dotenv

from src.agents.chat_service import answer_question
from src.data.db import SessionLocal

load_dotenv()


def _check_sourced_response(db):
    print("\n--- 1. one tool call -> sourced response ---")
    answer = answer_question(db, "Why is AAPL's factor score what it is?")
    print("response:", answer.response)
    print("sources:", answer.sources)
    if answer.sources:
        print("PASS: response cites at least one source.")
    else:
        print("FAIL: no sources attached - check the response text above for a hallucinated answer.")


def _check_no_outside_knowledge(db):
    print("\n--- 2. no outside knowledge for data Titan doesn't have a tool for ---")
    answer = answer_question(db, "What sector is Apple in?")
    print("response:", answer.response)
    if "technology" in answer.response.lower() or "consumer electronics" in answer.response.lower():
        print("FAIL: model answered from training/general knowledge instead of declining.")
    else:
        print("PASS: model did not state Apple's sector from memory.")


def _check_unknown_ticker_is_graceful(db):
    print("\n--- 3. unknown ticker -> graceful response, not a raw exception ---")
    answer = answer_question(db, "Explain ZZZZNOTREAL's factor score.")
    print("response:", answer.response)
    print("PASS: no exception raised; see response text above for tone check.")


def main():
    db = SessionLocal()
    try:
        _check_sourced_response(db)
        _check_no_outside_knowledge(db)
        _check_unknown_ticker_is_graceful(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
