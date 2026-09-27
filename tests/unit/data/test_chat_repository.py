import uuid
from unittest.mock import MagicMock

import pytest

from src.agents.tools.base import Source
from src.data.chat_repository import add_message, get_or_create_session, get_recent_messages
from src.data.models import ChatMessage, ChatSession


# omitted session_id creates a new session, owned by the authenticated caller
def test_get_or_create_session_without_id_creates_a_new_session():
    db = MagicMock()

    session = get_or_create_session(db, None, "alice")

    assert isinstance(session.id, uuid.UUID)
    assert session.user_id == "alice"
    db.add.assert_called_once_with(session)
    # a new session must be flushed before any dependent chat_messages insert
    db.flush.assert_called_once()


def test_get_or_create_session_with_known_id_owned_by_caller_returns_the_existing_row():
    db = MagicMock()
    session_uuid = uuid.uuid4()
    existing = ChatSession(id=session_uuid, user_id="alice")
    db.get.return_value = existing

    session = get_or_create_session(db, str(session_uuid), "alice")

    assert session is existing
    db.get.assert_called_once_with(ChatSession, session_uuid)
    db.add.assert_not_called()
    db.flush.assert_not_called()


# a session_id that exists but belongs to a different user must never be handed
# back (or extended) - that would leak one user's conversation into another's
def test_get_or_create_session_owned_by_another_user_raises_permission_error():
    db = MagicMock()
    session_uuid = uuid.uuid4()
    db.get.return_value = ChatSession(id=session_uuid, user_id="alice")

    with pytest.raises(PermissionError):
        get_or_create_session(db, str(session_uuid), "bob")

    db.add.assert_not_called()


def test_get_or_create_session_with_unused_id_creates_it_owned_by_the_caller():
    db = MagicMock()
    session_uuid = uuid.uuid4()
    db.get.return_value = None

    session = get_or_create_session(db, str(session_uuid), "alice")

    assert session.id == session_uuid
    assert session.user_id == "alice"
    db.add.assert_called_once_with(session)
    db.flush.assert_called_once()


def test_get_or_create_session_rejects_a_non_uuid_session_id():
    db = MagicMock()

    with pytest.raises(ValueError):
        get_or_create_session(db, "not-a-uuid", "alice")

    db.add.assert_not_called()


# recency window: last K messages, oldest first (so they replay as valid conversation history)
def test_get_recent_messages_returns_them_oldest_first():
    db = MagicMock()
    session_uuid = uuid.uuid4()
    newest_first = [
        ChatMessage(id=3, session_id=session_uuid, role="assistant", content="c"),
        ChatMessage(id=2, session_id=session_uuid, role="user", content="b"),
        ChatMessage(id=1, session_id=session_uuid, role="user", content="a"),
    ]
    db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
        newest_first
    )

    result = get_recent_messages(db, session_uuid, limit=5)

    assert [m.content for m in result] == ["a", "b", "c"]
    db.query.return_value.filter.return_value.order_by.return_value.limit.assert_called_once_with(5)


# every /chat call persists both the user message and assistant response, including sources
def test_add_message_serializes_sources():
    db = MagicMock()
    session_uuid = uuid.uuid4()
    source = Source(type="factor_score", ticker="AAPL", ref_id=1)

    message = add_message(db, session_uuid, role="assistant", content="AAPL scores 90.", sources=[source])

    assert message.role == "assistant"
    assert message.content == "AAPL scores 90."
    assert message.sources == [source.model_dump(mode="json")]
    db.add.assert_called_once_with(message)


def test_add_message_defaults_to_no_sources():
    db = MagicMock()
    session_uuid = uuid.uuid4()

    message = add_message(db, session_uuid, role="user", content="hello")

    assert message.sources == []


# request_id ties this turn to its llm_calls/tool_calls trace
def test_add_message_persists_the_request_id():
    db = MagicMock()
    session_uuid = uuid.uuid4()

    message = add_message(db, session_uuid, role="assistant", content="AAPL scores 90.", request_id="req-123")

    assert message.request_id == "req-123"


def test_add_message_defaults_request_id_to_none():
    db = MagicMock()
    session_uuid = uuid.uuid4()

    message = add_message(db, session_uuid, role="user", content="hello")

    assert message.request_id is None
