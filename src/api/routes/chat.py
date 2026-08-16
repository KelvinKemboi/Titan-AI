import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agents.chat_service import answer_question
from src.agents.tools.base import Source
from src.api.deps import get_db

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
    Exposes the Chat/Agent Service (#11) over HTTP. An omitted `session_id`
    creates a new one; conversation memory (`chat_sessions`/`chat_messages`,
    #13) isn't wired up yet, so the returned id is just an identifier for the
    client to carry forward - this call is still single-turn under the hood.

    Response time budget (unoptimized - no caching yet; the Redis LLM
    response cache in #20 is the intended fix): every request is a live call
    chain to the Anthropic API, capped at `MAX_TOOL_ITERATIONS` (5) round
    trips. Budget ~6s p50 / ~12s p95 for the common one-tool-call question (2
    round trips); worst case (several tool-call rounds) budget ~20s. No
    server-side request timeout is enforced yet.
    """
    session_id = request.session_id or str(uuid.uuid4())
    answer = answer_question(db, request.message)
    return ChatResponse(response=answer.response, sources=answer.sources, session_id=session_id)
