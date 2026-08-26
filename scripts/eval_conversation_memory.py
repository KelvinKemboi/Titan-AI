"""
Eval harness for conversation memory / entity tracking (technical-design.md
#4): hand-written multi-turn conversations - ask about a ticker, then a
pronoun follow-up, then a comparison - run through the real Chat/Agent
Service (real Postgres persistence between turns, like the /chat route),
checking each turn's *sources* actually cite the ticker(s) the pronoun was
meant to resolve to - not just that the response text "sounds right".

Requires ANTHROPIC_API_KEY and a Postgres with a completed scan that
includes every ticker below (`streamlit run app.py` -> Initialize Market
Scan, or `python -m src.analytics.scheduler`).

Run:
    python -m scripts.eval_conversation_memory
"""
from dotenv import load_dotenv

from src.agents.chat_service import answer_question
from src.agents.entity_tracker import extract_entities
from src.data.chat_repository import add_message, get_or_create_session, get_recent_messages
from src.data.db import SessionLocal

load_dotenv()

# Each conversation: (name, [(question, expected tickers in that turn's sources), ...]).
# Turn shape is fixed per the acceptance criteria: ticker question -> pronoun
# follow-up -> comparison.
CONVERSATIONS = [
    (
        "NVDA -> pronoun momentum -> compare to AMD",
        [
            ("Tell me about NVDA's factor score.", {"NVDA"}),
            ("What's its momentum score?", {"NVDA"}),
            ("How does it compare to AMD?", {"NVDA", "AMD"}),
        ],
    ),
    (
        "MSFT -> pronoun rating -> compare to GOOGL",
        [
            ("Why is MSFT rated the way it is?", {"MSFT"}),
            ("What's driving its quality score?", {"MSFT"}),
            ("Compare it to GOOGL.", {"MSFT", "GOOGL"}),
        ],
    ),
    (
        "AAPL -> pronoun quality -> compare to TSLA",
        [
            ("Explain AAPL's factor score.", {"AAPL"}),
            ("What about its quality score?", {"AAPL"}),
            ("Is it stronger than TSLA?", {"AAPL", "TSLA"}),
        ],
    ),
    (
        "AMZN -> pronoun volatility -> compare to WMT",
        [
            ("What's AMZN's factor score?", {"AMZN"}),
            ("What's driving its volatility rating?", {"AMZN"}),
            ("How does it stack up against WMT?", {"AMZN", "WMT"}),
        ],
    ),
    (
        "META -> pronoun solvency -> compare to GOOGL",
        [
            ("Tell me about META's score.", {"META"}),
            ("Explain its solvency score.", {"META"}),
            ("Compare it with GOOGL.", {"META", "GOOGL"}),
        ],
    ),
]


def run_conversation(db, name, turns):
    print(f"\n=== {name} ===")
    session = get_or_create_session(db, None)
    db.commit()

    all_ok = True
    for question, expected_tickers in turns:
        history = get_recent_messages(db, session.id, limit=10)
        entities_before = extract_entities(history)

        answer = answer_question(db, question, history=history)

        add_message(db, session.id, role="user", content=question)
        add_message(db, session.id, role="assistant", content=answer.response, sources=answer.sources)
        db.commit()

        actual_tickers = {s.ticker for s in answer.sources}
        ok = expected_tickers.issubset(actual_tickers)
        all_ok = all_ok and ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] Q: {question}")
        print(f"    entity state before this turn: {entities_before.model_dump_json()}")
        print(f"    expected tickers in sources: {sorted(expected_tickers)}")
        print(f"    actual tickers in sources:   {sorted(actual_tickers)}")
        print(f"    response: {answer.response}")

    return all_ok


def main():
    db = SessionLocal()
    try:
        results = [run_conversation(db, name, turns) for name, turns in CONVERSATIONS]
        passed = sum(results)
        print(f"\nOverall: {passed}/{len(results)} conversations fully correct.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
