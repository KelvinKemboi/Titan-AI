import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from src.agents.chat_service import (
    CHAT_HISTORY_WINDOW,
    FALLBACK_RESPONSE,
    INCONCLUSIVE_RESPONSE,
    MAX_TOOL_ITERATIONS,
    SYSTEM_PROMPT,
    ChatAnswer,
    answer_question,
)
from src.agents.tools import TOOLS
from src.agents.tools.base import Source, ToolResult
from src.data.cache import chat_cache_key
from src.data.models import ChatMessage


# automatically patches classify_intent to return None for all tests in this module
@pytest.fixture(autouse=True)
def _no_intent_classification(monkeypatch):
    monkeypatch.setattr("src.agents.chat_service.classify_intent", MagicMock(return_value=None))


# Every test below exercises the tool-calling loop itself, not the FAQ
# response cache - default cache_get to a miss and neutralize cache_set so
# none of them incidentally depend on (or attempt to reach) real Redis.
# Tests that specifically cover the cache override these within their own body.
@pytest.fixture(autouse=True)
def _no_chat_response_cache(monkeypatch):
    monkeypatch.setattr("src.agents.chat_service.cache_get", MagicMock(return_value=None))
    monkeypatch.setattr("src.agents.chat_service.cache_set", MagicMock())


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def _response(content, usage=None):
    return SimpleNamespace(content=content, usage=usage)


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

    # the error must have been fed back to the model as an is_error tool_result
    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_block = second_call_messages[-1]["content"][0]
    assert tool_result_block["is_error"] is True
    assert "ZZZZ" in tool_result_block["content"]


# an unexpected (non-ValueError) tool failure
def test_unexpected_tool_exception_produces_graceful_response_not_a_raw_error(monkeypatch):
    mock_call_tool = MagicMock(side_effect=RuntimeError("Voyage API unreachable"))
    monkeypatch.setattr("src.agents.chat_service.call_tool", mock_call_tool)

    client = _client_with_responses(
        _response([_tool_use_block("t1", "search_memos", {"query": "deep competitive moats"})]),
        _response([_text_block("I ran into a problem searching for that - please try again.")]),
    )

    result = answer_question(db=MagicMock(), question="Which companies have deep moats?", client=client)

    assert result.response == "I ran into a problem searching for that - please try again."
    assert result.sources == []

    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_block = second_call_messages[-1]["content"][0]
    assert tool_result_block["is_error"] is True
    assert "Voyage API unreachable" in tool_result_block["content"]


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


# a classified intent is injected into the system prompt as a routing hint
def test_classified_intent_adds_a_hint_to_the_system_prompt(monkeypatch):
    monkeypatch.setattr("src.agents.chat_service.classify_intent", MagicMock(return_value="qualitative"))
    client = _client_with_responses(_response([_text_block("Answer.")]))

    answer_question(db=MagicMock(), question="Which companies have deep moats?", client=client)

    system_sent = client.messages.create.call_args.kwargs["system"]
    assert "search_memos" in system_sent
    assert "hint, not a restriction" in system_sent


# a classified intent that is not one of the known intents does not add a hint
def test_all_tools_remain_available_regardless_of_classified_intent(monkeypatch):
    for guessed_intent in ["structured", "qualitative", "comparison", None]:
        monkeypatch.setattr(
            "src.agents.chat_service.classify_intent", MagicMock(return_value=guessed_intent)
        )
        client = _client_with_responses(_response([_text_block("Answer.")]))

        answer_question(db=MagicMock(), question="Explain AAPL's score", client=client)

        tools_sent = client.messages.create.call_args.kwargs["tools"]
        assert tools_sent == TOOLS
        assert {t["name"] for t in tools_sent} == {
            "get_factor_scores", "compare_tickers", "search_memos", "get_earnings_insight",
            "get_qoq_changes", "search_earnings", "get_analyst_report",
        }


# a classifier that returns nothing- falls back to the unmodified system prompt
def test_no_classified_intent_leaves_system_prompt_unmodified(monkeypatch):
    monkeypatch.setattr("src.agents.chat_service.classify_intent", MagicMock(return_value=None))
    client = _client_with_responses(_response([_text_block("Answer.")]))

    answer_question(db=MagicMock(), question="Explain AAPL's score", client=client)

    assert client.messages.create.call_args.kwargs["system"] == SYSTEM_PROMPT


# a ticker cited in an earlier turn's sources grounds a pronoun follow-up via the
# system prompt's entity tracking, not just the raw history
def test_entity_from_prior_sourced_turn_adds_a_hint_to_the_system_prompt():
    session_id = "22222222-2222-2222-2222-222222222222"
    history = [
        ChatMessage(session_id=session_id, role="user", content="Tell me about NVDA"),
        ChatMessage(
            session_id=session_id, role="assistant", content="NVDA scores 88/100.",
            sources=[{"type": "factor_score", "ticker": "NVDA", "ref_id": 1, "as_of": None}],
        ),
    ]
    client = _client_with_responses(_response([_text_block("Its momentum is 92/100.")]))

    answer_question(db=MagicMock(), question="what about its momentum?", history=history, client=client)

    system_sent = client.messages.create.call_args.kwargs["system"]
    assert "Most recently discussed ticker this session: NVDA" in system_sent
    assert "pronoun" in system_sent


# history with no sourced turns (e.g. the model answered from memory, or nothing asked
# yet) has nothing to ground a pronoun in, so no entity hint is added
def test_no_entity_hint_when_no_prior_turn_has_sources():
    history = [
        ChatMessage(session_id="s", role="user", content="Tell me about NVDA"),
        ChatMessage(session_id="s", role="assistant", content="NVDA scores 88/100.", sources=[]),
    ]
    client = _client_with_responses(_response([_text_block("Answer.")]))

    answer_question(db=MagicMock(), question="what about its momentum?", history=history, client=client)

    assert client.messages.create.call_args.kwargs["system"] == SYSTEM_PROMPT


# entity state is logged every turn so an incorrect resolution is debuggable after the
# fact - what did the system think "it" referred to, and why
def test_entity_state_is_logged_for_debugging(caplog):
    session_id = "33333333-3333-3333-3333-333333333333"
    history = [
        ChatMessage(
            session_id=session_id, role="assistant", content="NVDA scores 88/100.",
            sources=[{"type": "factor_score", "ticker": "NVDA", "ref_id": 1, "as_of": None}],
        ),
    ]
    client = _client_with_responses(_response([_text_block("Answer.")]))

    with caplog.at_level("INFO", logger="src.agents.chat_service"):
        answer_question(db=MagicMock(), question="what about its momentum?", history=history, client=client)

    assert any("NVDA" in record.message for record in caplog.records)


# --- FAQ-style /chat response cache ---

def _db_with_scan_run_id(scan_run_id):
    db = MagicMock()
    db.query.return_value.scalar.return_value = scan_run_id
    return db


# a cache hit for a context-free (no history) question short-circuits before the
# classifier, entity extraction, or the model itself are ever invoked
def test_cache_hit_for_context_free_question_skips_the_model_entirely(monkeypatch):
    cached_answer = ChatAnswer(
        response="Cached answer.", sources=[Source(type="factor_score", ticker="AAPL", ref_id=1)]
    )
    monkeypatch.setattr(
        "src.agents.chat_service.cache_get", MagicMock(return_value=cached_answer.model_dump_json())
    )
    mock_classify = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.classify_intent", mock_classify)
    client = MagicMock()

    result = answer_question(db=_db_with_scan_run_id(42), question="Explain AAPL's score", client=client)

    assert result == cached_answer
    client.messages.create.assert_not_called()
    mock_classify.assert_not_called()


# a successful, model-completed answer to a context-free question is written to the cache
def test_successful_context_free_answer_is_written_to_the_cache(monkeypatch):
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.cache_set", mock_cache_set)
    tool_result = _factor_score_result()
    monkeypatch.setattr("src.agents.chat_service.call_tool", MagicMock(return_value=tool_result))
    client = _client_with_responses(
        _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "AAPL"})]),
        _response([_text_block("AAPL scores 90/100.")]),
    )

    result = answer_question(db=_db_with_scan_run_id(42), question="Explain AAPL's score", client=client)

    mock_cache_set.assert_called_once()
    cache_key, cached_json = mock_cache_set.call_args.args
    assert cache_key == chat_cache_key("Explain AAPL's score", 42)
    assert ChatAnswer.model_validate_json(cached_json) == result


# the latest scan_run_id (data_version) is part of the key - a later scan must not
# accidentally reuse an answer cached against older data
def test_cache_key_incorporates_the_latest_scan_run_id(monkeypatch):
    mock_cache_get = MagicMock(return_value=None)
    monkeypatch.setattr("src.agents.chat_service.cache_get", mock_cache_get)
    client = _client_with_responses(_response([_text_block("Answer.")]))

    answer_question(db=_db_with_scan_run_id(99), question="What are the strongest momentum stocks?", client=client)

    mock_cache_get.assert_called_once_with(chat_cache_key("What are the strongest momentum stocks?", 99))


# a question submitted WITH history must never touch the cache
def test_questions_with_history_never_touch_the_cache(monkeypatch):
    mock_cache_get = MagicMock()
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.cache_get", mock_cache_get)
    monkeypatch.setattr("src.agents.chat_service.cache_set", mock_cache_set)
    history = [ChatMessage(session_id="s", role="user", content="Tell me about NVDA")]
    client = _client_with_responses(_response([_text_block("Its momentum is 92/100.")]))

    answer_question(db=MagicMock(), question="what about its momentum?", history=history, client=client)

    mock_cache_get.assert_not_called()
    mock_cache_set.assert_not_called()


# a transient API failure must never be cached as if it were a real answer
def test_fallback_response_is_not_cached(monkeypatch):
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.cache_set", mock_cache_set)
    client = MagicMock()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = anthropic.APIConnectionError(request=request)

    result = answer_question(db=_db_with_scan_run_id(42), question="Explain AAPL's score", client=client)

    assert result.response == FALLBACK_RESPONSE
    mock_cache_set.assert_not_called()


# a loop that never converges isn't a reliable "answer" either - must not be cached
def test_inconclusive_response_is_not_cached(monkeypatch):
    mock_cache_set = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.cache_set", mock_cache_set)
    tool_result = _factor_score_result()
    monkeypatch.setattr("src.agents.chat_service.call_tool", MagicMock(return_value=tool_result))
    always_tool_use = _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "AAPL"})])
    client = _client_with_responses(*([always_tool_use] * MAX_TOOL_ITERATIONS))

    result = answer_question(db=_db_with_scan_run_id(42), question="Explain AAPL's score", client=client)

    assert result.response == INCONCLUSIVE_RESPONSE
    mock_cache_set.assert_not_called()


# --- request tracing ---

def test_answer_question_generates_a_fresh_request_id_when_none_given(monkeypatch):
    mock_classify = MagicMock(return_value=None)
    monkeypatch.setattr("src.agents.chat_service.classify_intent", mock_classify)
    client = _client_with_responses(_response([_text_block("Answer.")]))

    answer_question(db=MagicMock(), question="Explain AAPL's score", client=client)

    request_id = mock_classify.call_args.kwargs["request_id"]
    assert uuid.UUID(request_id)


def test_answer_question_uses_a_given_request_id_instead_of_generating_one(monkeypatch):
    mock_classify = MagicMock(return_value=None)
    monkeypatch.setattr("src.agents.chat_service.classify_intent", mock_classify)
    client = _client_with_responses(_response([_text_block("Answer.")]))

    answer_question(db=MagicMock(), question="Explain AAPL's score", client=client, request_id="my-request-id")

    assert mock_classify.call_args.kwargs["request_id"] == "my-request-id"


def test_classify_intent_is_called_with_db_and_request_id(monkeypatch):
    mock_classify = MagicMock(return_value=None)
    monkeypatch.setattr("src.agents.chat_service.classify_intent", mock_classify)
    client = _client_with_responses(_response([_text_block("Answer.")]))
    db = MagicMock()

    answer_question(db=db, question="Explain AAPL's score", client=client, request_id="req-1")

    assert mock_classify.call_args.kwargs["db"] is db
    assert mock_classify.call_args.kwargs["request_id"] == "req-1"


def test_generation_call_is_traced_with_request_id_and_call_type(monkeypatch):
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.record_llm_call", mock_record)
    usage = SimpleNamespace(input_tokens=100, output_tokens=20)
    client = _client_with_responses(_response([_text_block("Answer.")], usage=usage))
    db = MagicMock()

    answer_question(db=db, question="Explain AAPL's score", client=client, request_id="req-1")

    mock_record.assert_called_once()
    call_kwargs = mock_record.call_args.kwargs
    assert call_kwargs["request_id"] == "req-1"
    assert call_kwargs["call_type"] == "chat_generation"
    assert call_kwargs["response"].usage is usage
    assert isinstance(call_kwargs["latency_ms"], int)


def test_every_generation_iteration_is_traced_separately(monkeypatch):
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.record_llm_call", mock_record)
    tool_result = _factor_score_result()
    monkeypatch.setattr("src.agents.chat_service.call_tool", MagicMock(return_value=tool_result))
    client = _client_with_responses(
        _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "AAPL"})]),
        _response([_text_block("AAPL scores 90.")]),
    )

    answer_question(db=MagicMock(), question="Explain AAPL's score", client=client, request_id="req-1")

    assert mock_record.call_count == 2
    assert all(c.kwargs["request_id"] == "req-1" for c in mock_record.call_args_list)


def test_a_failed_generation_call_is_not_traced(monkeypatch):
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.record_llm_call", mock_record)
    client = MagicMock()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = anthropic.APIConnectionError(request=request)

    answer_question(db=MagicMock(), question="Explain AAPL's score", client=client)

    mock_record.assert_not_called()


def test_a_cache_hit_traces_nothing_at_all(monkeypatch):
    mock_record_llm = MagicMock()
    mock_record_tool = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.record_llm_call", mock_record_llm)
    monkeypatch.setattr("src.agents.chat_service.record_tool_call", mock_record_tool)
    cached_answer = ChatAnswer(response="Cached answer.", sources=[])
    monkeypatch.setattr(
        "src.agents.chat_service.cache_get", MagicMock(return_value=cached_answer.model_dump_json())
    )

    answer_question(db=_db_with_scan_run_id(42), question="Explain AAPL's score", client=MagicMock())

    mock_record_llm.assert_not_called()
    mock_record_tool.assert_not_called()


# --- tool call tracing ---

def test_successful_tool_call_is_traced_with_its_input_and_sources(monkeypatch):
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.record_tool_call", mock_record)
    tool_result = _factor_score_result("AAPL", ref_id=1)
    monkeypatch.setattr("src.agents.chat_service.call_tool", MagicMock(return_value=tool_result))
    client = _client_with_responses(
        _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "AAPL"})]),
        _response([_text_block("AAPL scores 90.")]),
    )

    answer_question(db=MagicMock(), question="Explain AAPL's score", client=client, request_id="req-1")

    mock_record.assert_called_once()
    call_kwargs = mock_record.call_args.kwargs
    assert call_kwargs["request_id"] == "req-1"
    assert call_kwargs["tool_name"] == "get_factor_scores"
    assert call_kwargs["tool_input"] == {"ticker": "AAPL"}
    assert call_kwargs["success"] is True
    assert call_kwargs["sources"] == [s.model_dump(mode="json") for s in tool_result.sources]


def test_failed_tool_call_is_traced_with_success_false_and_the_error(monkeypatch):
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.record_tool_call", mock_record)
    monkeypatch.setattr(
        "src.agents.chat_service.call_tool", MagicMock(side_effect=ValueError("No factor_scores found for ticker 'ZZZZ'"))
    )
    client = _client_with_responses(
        _response([_tool_use_block("t1", "get_factor_scores", {"ticker": "ZZZZ"})]),
        _response([_text_block("I couldn't find ZZZZ.")]),
    )

    answer_question(db=MagicMock(), question="Explain ZZZZ's score", client=client, request_id="req-1")

    mock_record.assert_called_once()
    call_kwargs = mock_record.call_args.kwargs
    assert call_kwargs["request_id"] == "req-1"
    assert call_kwargs["success"] is False
    assert "ZZZZ" in call_kwargs["error"]


def test_multiple_tool_calls_in_one_turn_are_each_traced_separately(monkeypatch):
    mock_record = MagicMock()
    monkeypatch.setattr("src.agents.chat_service.record_tool_call", mock_record)
    results = [_factor_score_result("MSFT", 1), _factor_score_result("GOOGL", 1)]
    monkeypatch.setattr("src.agents.chat_service.call_tool", MagicMock(side_effect=results))
    client = _client_with_responses(
        _response(
            [
                _tool_use_block("t1", "get_factor_scores", {"ticker": "MSFT"}),
                _tool_use_block("t2", "get_factor_scores", {"ticker": "GOOGL"}),
            ]
        ),
        _response([_text_block("MSFT edges out GOOGL.")]),
    )

    answer_question(db=MagicMock(), question="Compare MSFT and GOOGL", client=client, request_id="req-1")

    assert mock_record.call_count == 2
    tickers_traced = {c.kwargs["tool_input"]["ticker"] for c in mock_record.call_args_list}
    assert tickers_traced == {"MSFT", "GOOGL"}
