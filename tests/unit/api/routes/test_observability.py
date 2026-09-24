from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.auth import get_current_user
from src.api.deps import get_db
from src.api.main import app
from src.observability.spend import LLMCallDetail, RequestTrace, SpendSummary

TEST_USER_ID = "test-user"


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: TEST_USER_ID
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def unauthenticated_client(db):
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_get_aggregate_spend_returns_the_summary(monkeypatch, client):
    summary = SpendSummary(call_count=3, input_tokens=100, output_tokens=20, cost_usd=0.01)
    monkeypatch.setattr("src.api.routes.observability.get_aggregate_spend", MagicMock(return_value=summary))

    resp = client.get("/observability/spend")

    assert resp.status_code == 200
    assert resp.json()["call_count"] == 3


def test_get_aggregate_spend_passes_since_days_through(monkeypatch, client):
    mock_get_spend = MagicMock(return_value=SpendSummary(call_count=0, input_tokens=0, output_tokens=0))
    monkeypatch.setattr("src.api.routes.observability.get_aggregate_spend", mock_get_spend)

    resp = client.get("/observability/spend?since_days=7")

    assert resp.status_code == 200
    assert mock_get_spend.call_args.kwargs["since"] is not None


def test_get_top_sessions_returns_the_list(monkeypatch, client):
    rows = [{"session_id": "s1", "cost_usd": 1.0, "input_tokens": 10, "output_tokens": 2, "call_count": 1}]
    monkeypatch.setattr("src.api.routes.observability.get_top_sessions_by_spend", MagicMock(return_value=rows))

    resp = client.get("/observability/spend/top-sessions")

    assert resp.status_code == 200
    assert resp.json() == rows


def test_get_session_spend_returns_the_summary(monkeypatch, client):
    summary = SpendSummary(call_count=1, input_tokens=10, output_tokens=2, cost_usd=0.001)
    monkeypatch.setattr("src.api.routes.observability.get_session_spend", MagicMock(return_value=summary))

    resp = client.get("/observability/spend/11111111-1111-1111-1111-111111111111")

    assert resp.status_code == 200
    assert resp.json()["call_count"] == 1


def test_get_session_spend_with_a_malformed_session_id_returns_404(client):
    resp = client.get("/observability/spend/not-a-uuid")

    assert resp.status_code == 404


def test_top_sessions_route_is_not_shadowed_by_the_session_id_route(monkeypatch, client):
    # "top-sessions" must resolve to the dedicated route, not be parsed as a session_id
    mock_top = MagicMock(return_value=[])
    mock_session = MagicMock()
    monkeypatch.setattr("src.api.routes.observability.get_top_sessions_by_spend", mock_top)
    monkeypatch.setattr("src.api.routes.observability.get_session_spend", mock_session)

    resp = client.get("/observability/spend/top-sessions")

    assert resp.status_code == 200
    mock_top.assert_called_once()
    mock_session.assert_not_called()


def test_get_request_trace_returns_the_trace(monkeypatch, client):
    trace = RequestTrace(
        request_id="req-1",
        llm_calls=[LLMCallDetail(call_type="chat_generation", model="claude-sonnet-5")],
        tool_calls=[],
    )
    monkeypatch.setattr("src.api.routes.observability.get_request_trace", MagicMock(return_value=trace))

    resp = client.get("/observability/trace/req-1")

    assert resp.status_code == 200
    assert resp.json()["request_id"] == "req-1"


def test_get_request_trace_for_an_unknown_request_id_returns_404(monkeypatch, client):
    empty_trace = RequestTrace(request_id="never-traced", llm_calls=[], tool_calls=[])
    monkeypatch.setattr("src.api.routes.observability.get_request_trace", MagicMock(return_value=empty_trace))

    resp = client.get("/observability/trace/never-traced")

    assert resp.status_code == 404


def test_observability_routes_require_auth(unauthenticated_client):
    assert unauthenticated_client.get("/observability/spend").status_code == 401
    assert unauthenticated_client.get("/observability/spend/top-sessions").status_code == 401
    assert unauthenticated_client.get("/observability/spend/11111111-1111-1111-1111-111111111111").status_code == 401
    assert unauthenticated_client.get("/observability/trace/req-1").status_code == 401
