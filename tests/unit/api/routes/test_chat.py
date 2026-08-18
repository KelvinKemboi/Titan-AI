import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agents.chat_service import ChatAnswer
from src.agents.tools.base import Source
from src.api.deps import get_db
from src.api.main import app
from src.data.models import ChatMessage, ChatSession


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()

# omitted session_id creates a new session
def _stub_session_lookup(monkeypatch, session_uuid, history=None):
    mock_get_or_create = MagicMock(return_value=ChatSession(id=session_uuid))
    mock_get_recent = MagicMock(return_value=history or [])
    mock_add_message = MagicMock()
    monkeypatch.setattr("src.api.routes.chat.get_or_create_session", mock_get_or_create)
    monkeypatch.setattr("src.api.routes.chat.get_recent_messages", mock_get_recent)
    monkeypatch.setattr("src.api.routes.chat.add_message", mock_add_message)
    return mock_get_or_create, mock_get_recent, mock_add_message


# POST /chat with {session_id?, message} returns {response, sources, session_id}
def test_chat_returns_response_sources_and_session_id(monkeypatch, client):
    session_uuid = uuid.uuid4()
    _stub_session_lookup(monkeypatch, session_uuid)
    answer = ChatAnswer(
        response="AAPL scores 90/100 (STRONG BUY).",
        sources=[Source(type="factor_score", ticker="AAPL", ref_id=1)],
    )
    monkeypatch.setattr("src.api.routes.chat.answer_question", MagicMock(return_value=answer))

    resp = client.post("/chat", json={"session_id": str(session_uuid), "message": "Why is AAPL ranked so high?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "AAPL scores 90/100 (STRONG BUY)."
    assert body["sources"] == [{"type": "factor_score", "ticker": "AAPL", "ref_id": 1, "as_of": None}]
    assert body["session_id"] == str(session_uuid)


# omitted session_id creates a new session
def test_chat_without_session_id_creates_a_new_session(monkeypatch, client):
    session_uuid = uuid.uuid4()
    mock_get_or_create, _, _ = _stub_session_lookup(monkeypatch, session_uuid)
    answer = ChatAnswer(response="Hi there.", sources=[])
    monkeypatch.setattr("src.api.routes.chat.answer_question", MagicMock(return_value=answer))

    resp = client.post("/chat", json={"message": "hello"})

    assert resp.status_code == 200
    assert resp.json()["session_id"] == str(session_uuid)
    # session_id was omitted, so the repository is asked to create a fresh one
    assert mock_get_or_create.call_args.args[1] is None


# every /chat call persists both the user message and assistant response, including sources
def test_chat_persists_user_and_assistant_messages(monkeypatch, client, db):
    session_uuid = uuid.uuid4()
    _, _, mock_add_message = _stub_session_lookup(monkeypatch, session_uuid)
    source = Source(type="factor_score", ticker="AAPL", ref_id=1)
    answer = ChatAnswer(response="AAPL scores 90/100.", sources=[source])
    monkeypatch.setattr("src.api.routes.chat.answer_question", MagicMock(return_value=answer))

    resp = client.post("/chat", json={"session_id": str(session_uuid), "message": "Explain AAPL"})

    assert resp.status_code == 200
    assert mock_add_message.call_count == 2

    user_call, assistant_call = mock_add_message.call_args_list
    assert user_call.kwargs["role"] == "user"
    assert user_call.kwargs["content"] == "Explain AAPL"

    assert assistant_call.kwargs["role"] == "assistant"
    assert assistant_call.kwargs["content"] == "AAPL scores 90/100."
    assert assistant_call.kwargs["sources"] == [source]

    db.commit.assert_called_once()


# the last-K window is loaded from history and passed into the Chat/Agent Service
# call, so a follow-up question resolves context from prior turns
def test_chat_injects_recent_history_into_the_chat_service_call(monkeypatch, client):
    session_uuid = uuid.uuid4()
    prior_turns = [
        ChatMessage(session_id=session_uuid, role="user", content="Tell me about NVDA"),
        ChatMessage(session_id=session_uuid, role="assistant", content="NVDA scores 88/100."),
    ]
    _, mock_get_recent, _ = _stub_session_lookup(monkeypatch, session_uuid, history=prior_turns)
    mock_answer_question = MagicMock(return_value=ChatAnswer(response="Its momentum is strong.", sources=[]))
    monkeypatch.setattr("src.api.routes.chat.answer_question", mock_answer_question)

    resp = client.post("/chat", json={"session_id": str(session_uuid), "message": "what about its momentum?"})

    assert resp.status_code == 200
    mock_get_recent.assert_called_once()
    assert mock_answer_question.call_args.kwargs["history"] == prior_turns


def test_chat_requires_a_message(client):
    resp = client.post("/chat", json={"session_id": "abc"})

    assert resp.status_code == 422


# an invalid (non-UUID) session_id is a graceful 400, not a raw exception
def test_chat_with_invalid_session_id_returns_400(monkeypatch, client):
    monkeypatch.setattr(
        "src.api.routes.chat.get_or_create_session",
        MagicMock(side_effect=ValueError("Invalid session_id 'not-a-uuid' - must be a UUID")),
    )

    resp = client.post("/chat", json={"session_id": "not-a-uuid", "message": "hello"})

    assert resp.status_code == 400
