from types import SimpleNamespace
from unittest.mock import MagicMock

from src.observability.tracing import (
    MODEL_PRICING_PER_MILLION_TOKENS,
    estimate_cost_usd,
    record_llm_call,
    record_tool_call,
)


# estimate_cost_usd
def test_estimate_cost_usd_computes_input_and_output_cost():
    pricing = MODEL_PRICING_PER_MILLION_TOKENS["claude-sonnet-5"]
    cost = estimate_cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)

    assert cost == pricing["input"] + pricing["output"]


def test_estimate_cost_usd_returns_none_for_an_unpriced_model():
    assert estimate_cost_usd("some-future-model", input_tokens=100, output_tokens=50) is None


def test_estimate_cost_usd_returns_none_when_token_counts_are_missing():
    assert estimate_cost_usd("claude-sonnet-5", input_tokens=None, output_tokens=50) is None
    assert estimate_cost_usd("claude-sonnet-5", input_tokens=100, output_tokens=None) is None


def test_estimate_cost_usd_zero_tokens_is_zero_cost_not_none():
    assert estimate_cost_usd("claude-sonnet-5", input_tokens=0, output_tokens=0) == 0.0


# record_llm_call
def test_record_llm_call_persists_usage_and_computed_cost():
    db = MagicMock()
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=1000, output_tokens=200))

    record_llm_call(db, request_id="req-1", call_type="chat_generation", model="claude-sonnet-5",
                     response=response, latency_ms=250)

    db.add.assert_called_once()
    row = db.add.call_args.args[0]
    assert row.request_id == "req-1"
    assert row.call_type == "chat_generation"
    assert row.model == "claude-sonnet-5"
    assert row.input_tokens == 1000
    assert row.output_tokens == 200
    assert row.latency_ms == 250
    assert row.cost_usd == estimate_cost_usd("claude-sonnet-5", 1000, 200)


def test_record_llm_call_is_a_noop_when_db_is_none():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=1000, output_tokens=200))

    record_llm_call(None, request_id="req-1", call_type="chat_generation", model="claude-sonnet-5", response=response)
    # no exception, nothing to assert against - the point is it doesn't crash


def test_record_llm_call_handles_a_response_with_no_usage_attribute():
    db = MagicMock()
    response = SimpleNamespace()  # no .usage at all - e.g. a bare mocked test response

    record_llm_call(db, request_id="req-1", call_type="chat_generation", model="claude-sonnet-5", response=response)

    row = db.add.call_args.args[0]
    assert row.input_tokens is None
    assert row.output_tokens is None
    assert row.cost_usd is None


def test_record_llm_call_swallows_a_db_failure_instead_of_raising(caplog):
    db = MagicMock()
    db.add.side_effect = RuntimeError("db is down")
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    with caplog.at_level("ERROR", logger="src.observability.tracing"):
        record_llm_call(db, request_id="req-1", call_type="chat_generation", model="claude-sonnet-5", response=response)

    assert any("req-1" in record.message for record in caplog.records)


# record_tool_call
def test_record_tool_call_persists_success_with_sources():
    db = MagicMock()
    sources = [{"type": "factor_score", "ticker": "AAPL", "ref_id": 1}]

    record_tool_call(db, request_id="req-1", tool_name="get_factor_scores", tool_input={"ticker": "AAPL"},
                      success=True, sources=sources)

    row = db.add.call_args.args[0]
    assert row.request_id == "req-1"
    assert row.tool_name == "get_factor_scores"
    assert row.tool_input == {"ticker": "AAPL"}
    assert row.success is True
    assert row.error is None
    assert row.sources == sources


def test_record_tool_call_persists_failure_with_error_and_empty_sources():
    db = MagicMock()

    record_tool_call(db, request_id="req-1", tool_name="get_factor_scores", tool_input={"ticker": "ZZZZ"},
                      success=False, error="No factor_scores found for ticker 'ZZZZ'")

    row = db.add.call_args.args[0]
    assert row.success is False
    assert row.error == "No factor_scores found for ticker 'ZZZZ'"
    assert row.sources == []


def test_record_tool_call_is_a_noop_when_db_is_none():
    record_tool_call(None, request_id="req-1", tool_name="get_factor_scores", tool_input={}, success=True)


def test_record_tool_call_swallows_a_db_failure_instead_of_raising(caplog):
    db = MagicMock()
    db.add.side_effect = RuntimeError("db is down")

    with caplog.at_level("ERROR", logger="src.observability.tracing"):
        record_tool_call(db, request_id="req-1", tool_name="get_factor_scores", tool_input={}, success=True)

    assert any("req-1" in record.message for record in caplog.records)
