"""
Manual test: a purely qualitative question ("which companies have deep
competitive moats") should route to the search_memos tool (semantic
search over memo_embeddings), not get_factor_scores/compare_tickers -
there's no single named ticker for those structured tools to key off of.

Requires ANTHROPIC_API_KEY, VOYAGE_API_KEY, and a Postgres with memo
embeddings already indexed for at least one high-margin ticker (a
completed scan indexes memos automatically - see
src/analytics/memo_indexing.py - or run
`python -m scripts.backfill_memo_embeddings` against an existing scan).

Run:
    python -m scripts.manual_test_memo_search
"""
from dotenv import load_dotenv

from src.agents.chat_service import answer_question
from src.data.db import SessionLocal

load_dotenv()


def _check_qualitative_question_uses_memo_search(db):
    print("\n--- qualitative question -> search_memos (not the structured tools) ---")
    answer = answer_question(db, "Which companies have deep competitive moats?")
    print("response:", answer.response)
    print("sources:", answer.sources)

    if not answer.sources:
        print("FAIL: no sources - response may be unsourced/hallucinated.")
        return

    source_types = {s.type for s in answer.sources}
    if source_types == {"memo"}:
        print(f"PASS: answered via search_memos only. Tickers cited: {[s.ticker for s in answer.sources]}")
    else:
        print(f"FAIL: expected only 'memo' sources, got source types {source_types}")


def main():
    db = SessionLocal()
    try:
        _check_qualitative_question_uses_memo_search(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
