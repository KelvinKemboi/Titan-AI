import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agents.chat_service import ChatAnswer
from src.agents.tools.base import Source
from src.api.deps import get_db
from src.api.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    yield TestClient(app)
    app.dependency_overrides.clear()


# POST /chat with {session_id?, message} returns {response, sources, session_id}
def test_chat_returns_response_sources_and_session_id(monkeypatch, client):
    answer = ChatAnswer(
        response="AAPL scores 90/100 (STRONG BUY).",
        sources=[Source(type="factor_score", ticker="AAPL", ref_id=1)],
    )
    monkeypatch.setattr("src.api.routes.chat.answer_question", MagicMock(return_value=answer))

    resp = client.post("/chat", json={"session_id": "existing-session", "message": "Why is AAPL ranked so high?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "AAPL scores 90/100 (STRONG BUY)."
    assert body["sources"] == [{"type": "factor_score", "ticker": "AAPL", "ref_id": 1, "as_of": None}]
    assert body["session_id"] == "existing-session"


# omitted session_id creates a new session
def test_chat_without_session_id_generates_a_new_one(monkeypatch, client):
    answer = ChatAnswer(response="Hi there.", sources=[])
    monkeypatch.setattr("src.api.routes.chat.answer_question", MagicMock(return_value=answer))

    resp = client.post("/chat", json={"message": "hello"})

    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    assert uuid.UUID(session_id)  # a real, parseable session id was generated


def test_chat_generates_a_distinct_session_id_per_request(monkeypatch, client):
    answer = ChatAnswer(response="Hi there.", sources=[])
    monkeypatch.setattr("src.api.routes.chat.answer_question", MagicMock(return_value=answer))

    first = client.post("/chat", json={"message": "hello"}).json()["session_id"]
    second = client.post("/chat", json={"message": "hello again"}).json()["session_id"]

    assert first != second


def test_chat_requires_a_message(client):
    resp = client.post("/chat", json={"session_id": "abc"})

    assert resp.status_code == 422
