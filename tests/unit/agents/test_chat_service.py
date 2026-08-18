from types import SimpleNamespace
from unittest.mock import MagicMock # testing utility for creating mock objects

import anthropic
import httpx

from src.agents.chat_service import (
    CHAT_HISTORY_WINDOW,
    FALLBACK_RESPONSE,
    INCONCLUSIVE_RESPONSE,
    MAX_TOOL_ITERATIONS,
    SYSTEM_PROMPT,
    answer_question,
)
from src.agents.tools.base import Source, ToolResult
from src.data.models import ChatMessage


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def _response(content):
    return SimpleNamespace(content=content)


def _client_with_responses(*responses):
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return client


def _factor_score_result(ticker="AAPL", ref_id=1):
    return ToolResult(
        data={"ticker": ticker, "composite_score": 90.0},
        sources=[Source(type="factor_score", ticker=ticker, ref_id=ref_id)],
    )


# a question answerable by one tool call returns a correct, sourced response end-to-end
def test_single_tool_call_returns_sourced_response(monkeypatch):
    tool_result = _factor_score_result()
    mock_call_tool = MagicMock(return_value=tool_result)
    monkeypatch.setattr("src.agents.chat_service.call_tool", mock_call_tool)

    client = _client_with_responses(
        _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "AAPL"})]),
        _response([_text_block("AAPL scores 90/100 (STRONG BUY).")]),
    )

    result = answer_question(db=MagicMock(), question="Why is AAPL ranked so high?", client=client)

    assert result.response == "AAPL scores 90/100 (STRONG BUY)."
    assert result.sources == tool_result.sources
    assert mock_call_tool.call_args.args[1:] == ("get_factor_scores", {"ticker": "AAPL"})
    assert client.messages.create.call_count == 2


# tool call failures (e.g. unknown ticker) produce a graceful response, not a raw exception
def test_tool_call_failure_produces_graceful_response(monkeypatch):
    mock_call_tool = MagicMock(side_effect=ValueError("No factor_scores found for ticker 'ZZZZ'"))
    monkeypatch.setattr("src.agents.chat_service.call_tool", mock_call_tool)

    client = _client_with_responses(
        _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "ZZZZ"})]),
        _response([_text_block("I couldn't find any data for ZZZZ - could you double-check the ticker?")]),
    )

    result = answer_question(db=MagicMock(), question="Explain ZZZZ's score", client=client)

    assert result.response == "I couldn't find any data for ZZZZ - could you double-check the ticker?"
    assert result.sources == []

    # the error must have been fed back to the model as an is_error tool_result,
    # not raised out of answer_question
    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_block = second_call_messages[-1]["content"][0]
    assert tool_result_block["is_error"] is True
    assert "ZZZZ" in tool_result_block["content"]


# multiple tool_use blocks in one assistant turn are all executed and their
# results returned together in a single user message (parallel tool use contract)
def test_multiple_tool_calls_in_one_turn_are_batched(monkeypatch):
    results = [_factor_score_result("MSFT", 1), _factor_score_result("GOOGL", 1)]
    mock_call_tool = MagicMock(side_effect=results)
    monkeypatch.setattr("src.agents.chat_service.call_tool", mock_call_tool)

    client = _client_with_responses(
        _response(
            [
                _tool_use_block("t1", "get_factor_scores", {"ticker": "MSFT"}),
                _tool_use_block("t2", "get_factor_scores", {"ticker": "GOOGL"}),
            ]
        ),
        _response([_text_block("MSFT edges out GOOGL.")]),
    )

    result = answer_question(db=MagicMock(), question="Compare MSFT and GOOGL", client=client)

    assert result.response == "MSFT edges out GOOGL."
    assert len(result.sources) == 2

    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_results_sent = second_call_messages[-1]["content"]
    assert len(tool_results_sent) == 2
    assert {b["tool_use_id"] for b in tool_results_sent} == {"t1", "t2"}


# an anthropic API failure (network/rate limit/etc.) must not surface as a raw exception
def test_anthropic_api_error_produces_graceful_response():
    client = MagicMock()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = anthropic.APIConnectionError(request=request)

    result = answer_question(db=MagicMock(), question="Explain AAPL's score", client=client)

    assert result.response == FALLBACK_RESPONSE
    assert result.sources == []


# a pathological loop that never stops calling tools terminates instead of hanging
def test_runaway_tool_loop_terminates_gracefully(monkeypatch):
    tool_result = _factor_score_result()
    mock_call_tool = MagicMock(return_value=tool_result)
    monkeypatch.setattr("src.agents.chat_service.call_tool", mock_call_tool)

    always_tool_use = _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "AAPL"})])
    client = _client_with_responses(*([always_tool_use] * MAX_TOOL_ITERATIONS))

    result = answer_question(db=MagicMock(), question="Explain AAPL's score", client=client)

    assert result.response == INCONCLUSIVE_RESPONSE
    assert client.messages.create.call_count == MAX_TOOL_ITERATIONS
    assert mock_call_tool.call_count == MAX_TOOL_ITERATIONS


# system prompt enforces "don't use outside knowledge for anything Titan has data for"
def test_system_prompt_forbids_outside_knowledge():
    assert "Do not answer from your general knowledge or training data" in SYSTEM_PROMPT
    assert "rather than answering from general knowledge" in SYSTEM_PROMPT


# prior turns are injected ahead of the new question, 
# so a follow-up carries the earlier conversation into the model call
def test_history_is_injected_ahead_of_the_new_question():
    session_id = "11111111-1111-1111-1111-111111111111"
    history = [
        ChatMessage(session_id=session_id, role="user", content="Tell me about NVDA"),
        ChatMessage(session_id=session_id, role="assistant", content="NVDA scores 88/100."),
    ]
    client = _client_with_responses(_response([_text_block("Its momentum score is 92/100.")]))

    result = answer_question(
        db=MagicMock(), question="what about its momentum?", history=history, client=client
    )

    assert result.response == "Its momentum score is 92/100."
    sent_messages = client.messages.create.call_args.kwargs["messages"]
    assert sent_messages == [
        {"role": "user", "content": "Tell me about NVDA"},
        {"role": "assistant", "content": "NVDA scores 88/100."},
        {"role": "user", "content": "what about its momentum?"},
    ]


# omitting history behaves exactly as a fresh conversation
def test_no_history_starts_a_fresh_conversation():
    client = _client_with_responses(_response([_text_block("Hi there.")]))

    answer_question(db=MagicMock(), question="hello", client=client)

    sent_messages = client.messages.create.call_args.kwargs["messages"]
    assert sent_messages == [{"role": "user", "content": "hello"}]


# window size (K) is a config value
def test_history_window_is_configurable_via_env():
    # runs in a fresh subprocess so it can't leak module-reload state into the rest of the suite
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "CHAT_HISTORY_WINDOW": "3"}
    result = subprocess.run(
        [sys.executable, "-c", "from src.agents.chat_service import CHAT_HISTORY_WINDOW; print(CHAT_HISTORY_WINDOW)"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "3"


def test_default_history_window_is_ten():
    assert CHAT_HISTORY_WINDOW == 10
