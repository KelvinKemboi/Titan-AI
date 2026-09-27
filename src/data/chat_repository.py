import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from src.agents.tools.base import Source
from src.data.models import ChatMessage, ChatSession


def get_or_create_session(db: Session, session_id: Optional[str], user_id: str) -> ChatSession:
    """
    Returns the ChatSession for `session_id`, scoped to `user_id`. An
    omitted `session_id` creates a fresh session; a given but unused one
    is created with that id, so a client can pick its own session_id up
    front. Raises ValueError for a malformed `session_id`, or
    PermissionError if it already belongs to a different user.
    """
    if session_id is None:
        session = ChatSession(id=uuid.uuid4(), user_id=user_id)
        db.add(session)
        db.flush()
        return session

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError as exc:
        raise ValueError(f"Invalid session_id '{session_id}' - must be a UUID") from exc

    session = db.get(ChatSession, session_uuid)
    if session is None:
        session = ChatSession(id=session_uuid, user_id=user_id)
        db.add(session)
        db.flush()
        return session

    if session.user_id != user_id:
        raise PermissionError(f"session_id '{session_id}' does not belong to this caller")
    return session


def get_recent_messages(db: Session, session_id: uuid.UUID, limit: int) -> List[ChatMessage]:
    """Last `limit` chat_messages rows for `session_id`, oldest first."""
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def add_message(
    db: Session,
    session_id: uuid.UUID,
    role: str,
    content: str,
    sources: Optional[List[Source]] = None,
    request_id: Optional[str] = None,
) -> ChatMessage:
    """Persists one turn. `sources` mirrors ToolResult.sources; leave it
    unset for user messages. `request_id` ties this row to the
    llm_calls/tool_calls traced under the same id."""
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=[s.model_dump(mode="json") for s in sources] if sources else [],
        request_id=request_id,
    )
    db.add(message)
    return message
