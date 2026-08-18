import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from src.agents.tools.base import Source
from src.data.models import ChatMessage, ChatSession


def get_or_create_session(db: Session, session_id: Optional[str]) -> ChatSession:
    """
    Returns the ChatSession for `session_id`. Omitted -> a fresh session
    (new random id). Provided but not yet used -> created with that id, so a
    client can pick its own session_id up front. Raises ValueError if
    `session_id` isn't a valid UUID.

    A newly created row is flushed immediately: chat_messages.session_id has
    a FK to chat_sessions.id, and (as with ScanRun/FactorScore in
    scanner_service.py) SQLAlchemy only orders flush-time inserts around
    declared `relationship()`s, not bare FK columns - so a later add_message
    in the same transaction would otherwise race the still-pending session
    insert and fail the FK constraint.
    """
    if session_id is None:
        session = ChatSession(id=uuid.uuid4())
        db.add(session)
        db.flush()
        return session

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError as exc:
        raise ValueError(f"Invalid session_id '{session_id}' - must be a UUID") from exc

    session = db.get(ChatSession, session_uuid)
    if session is None:
        session = ChatSession(id=session_uuid)
        db.add(session)
        db.flush()
    return session


def get_recent_messages(db: Session, session_id: uuid.UUID, limit: int) -> List[ChatMessage]:
    """
    Last `limit` chat_messages rows for `session_id`, oldest first - the
    recency window (technical-design.md §4) injected into the next
    Chat/Agent Service call.
    """
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
) -> ChatMessage:
    """Persists one turn. `sources` mirrors ToolResult.sources - pass None
    (or leave the default) for user messages."""
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=[s.model_dump(mode="json") for s in sources] if sources else [],
    )
    db.add(message)
    return message
