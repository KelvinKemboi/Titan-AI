import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agents.chat_service import CHAT_HISTORY_WINDOW, answer_question
from src.agents.tools.base import Source
from src.api.auth import get_current_user
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
def post_chat(request: ChatRequest, db: Session = Depends(get_db), user_id: str = Depends(get_current_user),):
    """
    Exposes the Chat/Agent Service over HTTP, with conversation memory:
    an omitted `session_id` creates a new session, owned by the
    authenticated caller (`Authorization: Bearer <key>` - src/api/auth.py);
    a given one (new or existing) is persisted to, and its last
    `CHAT_HISTORY_WINDOW` messages are injected as context so follow-ups
    resolve against the earlier conversation. An existing `session_id`
    owned by a *different* user is rejected (403), not silently served -
    one user must never be able to read or extend another's history this way.
    """
    try:
        session = get_or_create_session(db, request.session_id, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    history = get_recent_messages(db, session.id, limit=CHAT_HISTORY_WINDOW)
    # Minted here so both this turn's chat_messages rows can be labeled
    # with it, tying question and answer to the same trace.
    request_id = str(uuid.uuid4())
    answer = answer_question(db, request.message, history=history, request_id=request_id)

    add_message(db, session.id, role="user", content=request.message, request_id=request_id)
    add_message(
        db, session.id, role="assistant", content=answer.response,
        sources=answer.sources, request_id=request_id,
    )
    db.commit()

    return ChatResponse(response=answer.response, sources=answer.sources, session_id=str(session.id))
