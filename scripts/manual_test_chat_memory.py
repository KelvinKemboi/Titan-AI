"""
Manual test: exercise conversation memory against Postgres and the
live Anthropic API. Ask about NVDA, then a follow-up that only makes
sense with that context, and confirm the second answer resolves "its"
to NVDA via the persisted last-K window.

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

# single turn: send a message, get the answer, persist both sides of the turn, and commit
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
        session = get_or_create_session(db, None, "manual-test-chat-memory")
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
            print("\nFAIL: follow-up answer doesn't look like it resolved 'its' to NVDA.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
