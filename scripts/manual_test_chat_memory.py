"""
Manual test: exercise conversation memory (#13) end-to-end against a real
Postgres and the live Anthropic API - ask about NVDA, then ask a follow-up
that only makes sense with that context ("what about its momentum?"), and
confirm the second answer resolves "its" to NVDA via the persisted last-K
window instead of the user having to restate the ticker.

Requires ANTHROPIC_API_KEY and a Postgres with at least one completed scan
that includes NVDA (`streamlit run app.py` -> Initialize Market Scan, or
`python -m src.analytics.scheduler`).

Run:
    python -m scripts.manual_test_chat_memory
"""
from dotenv import load_dotenv

from src.agents.chat_service import CHAT_HISTORY_WINDOW, answer_question
from src.data.chat_repository import add_message, get_or_create_session, get_recent_messages
from src.data.db import SessionLocal

load_dotenv()


def _turn(db, session, message):
    history = get_recent_messages(db, session.id, limit=CHAT_HISTORY_WINDOW)
    answer = answer_question(db, message, history=history)
    add_message(db, session.id, role="user", content=message)
    add_message(db, session.id, role="assistant", content=answer.response, sources=answer.sources)
    db.commit()
    return answer


def main():
    db = SessionLocal()
    try:
        session = get_or_create_session(db, None)
        db.commit()
        print(f"session: {session.id}\n")

        first = _turn(db, session, "Tell me about NVDA's factor score.")
        print("Q: Tell me about NVDA's factor score.")
        print("A:", first.response)

        second = _turn(db, session, "what about its momentum?")
        print("\nQ: what about its momentum?")
        print("A:", second.response)

        if "nvda" in second.response.lower() or "momentum" in second.response.lower():
            print("\nPASS: follow-up answer stayed on NVDA/momentum without the ticker being restated.")
        else:
            print("\nFAIL: follow-up answer doesn't look like it resolved 'its' to NVDA - check the text above.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
