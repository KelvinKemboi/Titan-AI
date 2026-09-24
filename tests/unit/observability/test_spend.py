from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.data.models import ChatMessage, LLMCall, ToolCall
from src.observability.spend import (
    get_aggregate_spend,
    get_request_trace,
    get_session_spend,
    get_top_sessions_by_spend,
)


def _llm_call(call_type="chat_generation", model="claude-sonnet-5", input_tokens=100,
              output_tokens=20, cost_usd=0.001, created_at=None):
    return LLMCall(
        request_id="req-1", call_type=call_type, model=model, input_tokens=input_tokens,
        output_tokens=output_tokens, cost_usd=cost_usd, latency_ms=200,
        created_at=created_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _tool_call(tool_name="get_factor_scores", success=True, error=None, sources=None):
    return ToolCall(
        request_id="req-1", tool_name=tool_name, tool_input={"ticker": "AAPL"},
        success=success, error=error, sources=sources or [],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# get_aggregate_spend
def test_aggregate_spend_sums_tokens_and_cost_across_calls():
    db = MagicMock()
    db.query.return_value.all.return_value = [
        _llm_call(input_tokens=100, output_tokens=20, cost_usd=0.001),
        _llm_call(input_tokens=200, output_tokens=40, cost_usd=0.002),
    ]

    summary = get_aggregate_spend(db)

    assert summary.call_count == 2
    assert summary.input_tokens == 300
    assert summary.output_tokens == 60
    assert summary.cost_usd == 0.003


def test_aggregate_spend_breaks_down_by_call_type():
    db = MagicMock()
    db.query.return_value.all.return_value = [
        _llm_call(call_type="intent_classification", input_tokens=10, output_tokens=2, cost_usd=0.0001),
        _llm_call(call_type="chat_generation", input_tokens=500, output_tokens=100, cost_usd=0.005),
        _llm_call(call_type="chat_generation", input_tokens=500, output_tokens=100, cost_usd=0.005),
    ]

    summary = get_aggregate_spend(db)

    by_type = {b.call_type: b for b in summary.by_call_type}
    assert by_type["intent_classification"].call_count == 1
    assert by_type["chat_generation"].call_count == 2
    assert by_type["chat_generation"].input_tokens == 1000


def test_aggregate_spend_with_no_calls_is_all_zero_not_an_error():
    db = MagicMock()
    db.query.return_value.all.return_value = []

    summary = get_aggregate_spend(db)

    assert summary.call_count == 0
    assert summary.input_tokens == 0
    assert summary.cost_usd is None
    assert summary.by_call_type == []


def test_aggregate_spend_handles_a_call_with_no_cost_estimate():
    db = MagicMock()
    db.query.return_value.all.return_value = [_llm_call(cost_usd=None), _llm_call(cost_usd=0.002)]

    summary = get_aggregate_spend(db)

    # unpriced calls don't silently zero out the total - they're just excluded from it
    assert summary.cost_usd == 0.002


def test_aggregate_spend_filters_by_since(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []

    get_aggregate_spend(db, since=datetime(2026, 1, 1, tzinfo=timezone.utc))

    db.query.return_value.filter.assert_called_once()


# get_session_spend
def test_session_spend_joins_through_chat_message_request_ids():
    db = MagicMock()
    db.query.return_value.filter.return_value.distinct.return_value.all.return_value = [("req-1",), ("req-2",)]
    db.query.return_value.filter.return_value.all.return_value = [_llm_call(), _llm_call()]

    summary = get_session_spend(db, session_id="11111111-1111-1111-1111-111111111111")

    assert summary.call_count == 2


def test_session_spend_with_no_traced_requests_is_all_zero_not_an_error():
    db = MagicMock()
    db.query.return_value.filter.return_value.distinct.return_value.all.return_value = []

    summary = get_session_spend(db, session_id="11111111-1111-1111-1111-111111111111")

    assert summary.call_count == 0
    assert summary.cost_usd is None


# get_top_sessions_by_spend
def test_top_sessions_by_spend_shapes_each_row():
    db = MagicMock()
    row = MagicMock(session_id="s1", cost_usd=1.23, input_tokens=1000, output_tokens=200, call_count=5)
    db.query.return_value.join.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [row]

    result = get_top_sessions_by_spend(db, limit=10)

    assert result == [{"session_id": "s1", "cost_usd": 1.23, "input_tokens": 1000, "output_tokens": 200, "call_count": 5}]


def test_top_sessions_by_spend_handles_a_null_cost_sum():
    db = MagicMock()
    row = MagicMock(session_id="s1", cost_usd=None, input_tokens=None, output_tokens=None, call_count=1)
    db.query.return_value.join.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [row]

    result = get_top_sessions_by_spend(db, limit=10)

    assert result[0]["cost_usd"] is None
    assert result[0]["input_tokens"] == 0
    assert result[0]["output_tokens"] == 0


# regression: joining llm_calls straight against chat_messages (not a
# DISTINCT request_id/session_id subquery first) double-counts every
# sum, since one /chat turn writes 2 chat_messages rows (user +
# assistant) sharing one request_id - this must query the deduplicated
# subquery, not the raw table, to avoid that fan-out.
def test_top_sessions_by_spend_uses_a_distinct_subquery_not_the_raw_chat_messages_table():
    db = MagicMock()
    db.query.return_value.join.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

    get_top_sessions_by_spend(db, limit=10)

    # the first db.query(...) call builds the distinct subquery
    first_call_args = db.query.call_args_list[0].args
    assert first_call_args == (ChatMessage.request_id, ChatMessage.session_id)
    db.query.return_value.filter.return_value.distinct.return_value.subquery.assert_called_once()


# get_request_trace
def test_request_trace_returns_llm_calls_and_tool_calls_for_the_request():
    db = MagicMock()
    llm_calls = [_llm_call()]
    tool_calls = [_tool_call()]
    db.query.return_value.filter.return_value.order_by.return_value.all.side_effect = [llm_calls, tool_calls]

    trace = get_request_trace(db, "req-1")

    assert trace.request_id == "req-1"
    assert len(trace.llm_calls) == 1
    assert trace.llm_calls[0].call_type == "chat_generation"
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "get_factor_scores"


def test_request_trace_carries_the_error_for_a_failed_tool_call():
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.side_effect = [
        [], [_tool_call(success=False, error="No factor_scores found for ticker 'ZZZZ'")],
    ]

    trace = get_request_trace(db, "req-1")

    assert trace.tool_calls[0].success is False
    assert "ZZZZ" in trace.tool_calls[0].error


def test_request_trace_for_an_unknown_request_id_is_empty_not_an_error():
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.side_effect = [[], []]

    trace = get_request_trace(db, "never-traced")

    assert trace.llm_calls == []
    assert trace.tool_calls == []
