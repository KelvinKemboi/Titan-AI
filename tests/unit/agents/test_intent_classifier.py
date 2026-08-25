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


def _client_returning(content):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=content)
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
