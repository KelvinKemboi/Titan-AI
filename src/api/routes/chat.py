from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agents.chat_service import CHAT_HISTORY_WINDOW, answer_question
from src.agents.tools.base import Source
from src.api.deps import get_db
from src.data.chat_repository import add_message, get_or_create_session, get_recent_messages

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    session_id: str


@router.post("/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Exposes the Chat/Agent Service over HTTP, with conversation memory
    (technical-design.md §4): an omitted `session_id` creates a new session;
    a given one (new or existing) is persisted to, and its last
    `CHAT_HISTORY_WINDOW` messages are injected as context so follow-ups
    ("what about its momentum?") resolve against the earlier conversation.
    Every call persists both the user message and the assistant response,
    including its sources.

    Response time budget: every request is a live call chain to the
    Anthropic API, capped at `MAX_TOOL_ITERATIONS` (5) round trips, plus a
    handful of small local Postgres queries/writes for session history that
    are negligible next to that. Budget ~6s p50 / ~12s p95 for the common
    one-tool-call question (2 round trips); worst case (several tool-call
    rounds) budget ~20s. No server-side request timeout is enforced yet -
    this is the baseline #20's response cache should improve on.
    """
    try:
        session = get_or_create_session(db, request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history = get_recent_messages(db, session.id, limit=CHAT_HISTORY_WINDOW)
    answer = answer_question(db, request.message, history=history)

    add_message(db, session.id, role="user", content=request.message)
    add_message(db, session.id, role="assistant", content=answer.response, sources=answer.sources)
    db.commit()

    return ChatResponse(response=answer.response, sources=answer.sources, session_id=str(session.id))
