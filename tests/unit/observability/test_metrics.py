from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.api.deps import get_db
from src.api.main import app
from src.observability.tracing import record_llm_call, record_tool_call


def _sample(name, labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_exposes_prometheus_text():
    resp = TestClient(app).get("/metrics")
    assert resp.status_code == 200
    assert "titan_http_requests_total" in resp.text


def test_http_requests_are_labelled_by_route_template_not_raw_path():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    try:
        client = TestClient(app, raise_server_exceptions=False)
        labels = {"method": "GET", "route": "/company/{ticker}"}
        before = _sample("titan_http_request_duration_seconds_count", labels)
        client.get("/company/ZZZZ")
        assert _sample("titan_http_request_duration_seconds_count", labels) == before + 1
    finally:
        app.dependency_overrides.clear()


def test_record_llm_call_updates_token_and_cost_metrics_even_without_db():
    labels = {"model": "claude-sonnet-5", "call_type": "metrics_test", "direction": "input"}
    before = _sample("titan_llm_tokens_total", labels)
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=1000, output_tokens=200))

    record_llm_call(None, request_id="r1", call_type="metrics_test", model="claude-sonnet-5",
                    response=response, latency_ms=1500)

    assert _sample("titan_llm_tokens_total", labels) == before + 1000
    assert _sample("titan_llm_cost_usd_total", {"model": "claude-sonnet-5", "call_type": "metrics_test"}) > 0


def test_record_tool_call_counts_success_and_failure():
    before = _sample("titan_tool_calls_total", {"tool": "metrics_tool", "success": "false"})
    record_tool_call(MagicMock(), request_id="r1", tool_name="metrics_tool", tool_input={}, success=False)
    assert _sample("titan_tool_calls_total", {"tool": "metrics_tool", "success": "false"}) == before + 1
