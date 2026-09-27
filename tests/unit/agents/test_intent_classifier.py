from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx

from src.agents.intent_classifier import (
    CLASSIFIER_MODEL,
    INTENT_HINTS,
    INTENTS,
    classify_intent,
)


def _tool_use_block(intent_value):
    return SimpleNamespace(type="tool_use", id="t1", name="classify_intent", input={"intent": intent_value})


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _client_returning(content, usage=None):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=content, usage=usage)
    return client


# whatever intent the model's tool call carries is returned
def test_classify_intent_returns_the_tool_calls_intent():
    client = _client_returning([_tool_use_block("comparison")])

    assert classify_intent("Compare MSFT and GOOGL", client=client) == "comparison"


# forces the model to call classify_intent (not reply with free text) with the model
def test_classify_intent_forces_the_classify_tool_on_the_cheap_model():
    client = _client_returning([_tool_use_block("structured")])

    classify_intent("Why is AAPL rated a BUY?", client=client)

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == CLASSIFIER_MODEL
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "classify_intent"}
    assert call_kwargs["messages"] == [{"role": "user", "content": "Why is AAPL rated a BUY?"}]


# an Anthropic API error (rate limit, invalid/expired key, etc.) degrades to "no hint"
def test_classify_intent_returns_none_on_anthropic_api_error():
    client = MagicMock()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = anthropic.APIConnectionError(request=request)

    assert classify_intent("Why is AAPL rated a BUY?", client=client) is None


# a missing API key raises a plain TypeError from the SDK
def test_classify_intent_returns_none_on_missing_api_key_type_error():
    client = MagicMock()
    client.messages.create.side_effect = TypeError("Could not resolve authentication method")

    assert classify_intent("Why is AAPL rated a BUY?", client=client) is None


# the model replying with only text (no tool_use), despite tool_choice forcing one
def test_classify_intent_returns_none_if_no_tool_use_block():
    client = _client_returning([_text_block("I'm not sure")])

    assert classify_intent("Why is AAPL rated a BUY?", client=client) is None


# an intent value outside the enum must not be passed through as if valid
def test_classify_intent_returns_none_for_a_value_outside_the_enum():
    client = _client_returning([_tool_use_block("something_else")])

    assert classify_intent("Why is AAPL rated a BUY?", client=client) is None


def test_intent_hints_cover_every_intent():
    assert set(INTENT_HINTS.keys()) == set(INTENTS)


# --- tracing ---

def test_classify_intent_traces_the_call_when_db_and_request_id_are_given(monkeypatch):
    usage = SimpleNamespace(input_tokens=12, output_tokens=3)
    client = _client_returning([_tool_use_block("comparison")], usage=usage)
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.intent_classifier.record_llm_call", mock_record)
    db = MagicMock()

    classify_intent("Compare MSFT and GOOGL", client=client, db=db, request_id="req-1")

    mock_record.assert_called_once()
    call_kwargs = mock_record.call_args.kwargs
    assert call_kwargs["request_id"] == "req-1"
    assert call_kwargs["call_type"] == "intent_classification"
    assert call_kwargs["model"] == CLASSIFIER_MODEL
    assert call_kwargs["response"].usage is usage
    assert isinstance(call_kwargs["latency_ms"], int)


def test_classify_intent_does_not_trace_without_db_and_request_id(monkeypatch):
    client = _client_returning([_tool_use_block("comparison")])
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.intent_classifier.record_llm_call", mock_record)

    classify_intent("Compare MSFT and GOOGL", client=client)

    mock_record.assert_not_called()


def test_classify_intent_does_not_trace_with_only_db_or_only_request_id(monkeypatch):
    client = _client_returning([_tool_use_block("comparison")])
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.intent_classifier.record_llm_call", mock_record)

    classify_intent("Compare MSFT and GOOGL", client=client, db=MagicMock())
    classify_intent("Compare MSFT and GOOGL", client=client, request_id="req-1")

    mock_record.assert_not_called()


# tracing must never happen for a call that itself failed - there's no response to trace
def test_classify_intent_does_not_trace_on_an_api_error(monkeypatch):
    client = MagicMock()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = anthropic.APIConnectionError(request=request)
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.intent_classifier.record_llm_call", mock_record)

    classify_intent("Why is AAPL rated a BUY?", client=client, db=MagicMock(), request_id="req-1")

    mock_record.assert_not_called()
