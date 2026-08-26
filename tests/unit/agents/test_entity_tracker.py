from src.agents.entity_tracker import EntityState, entity_hint, extract_entities
from src.data.models import ChatMessage

SESSION_ID = "11111111-1111-1111-1111-111111111111"


def _user(content):
    return ChatMessage(session_id=SESSION_ID, role="user", content=content)


def _assistant(content, sources=None):
    return ChatMessage(session_id=SESSION_ID, role="assistant", content=content, sources=sources or [])


def _source(ticker, ref_id=1):
    return {"type": "factor_score", "ticker": ticker, "ref_id": ref_id, "as_of": None}


def test_empty_history_returns_empty_state():
    state = extract_entities([])

    assert state == EntityState(tickers_mentioned=[], factors_mentioned=[], last_ticker=None, last_factor=None)


# tickers come from the assistant turn's already-verified `sources`, not from parsing
# free text - a ticker the user typed but Titan never actually looked up isn't tracked
def test_tickers_are_extracted_from_assistant_sources_not_user_text():
    history = [
        _user("Tell me about NVDA"),
        _assistant("NVDA scores 88/100.", sources=[_source("NVDA")]),
    ]

    state = extract_entities(history)

    assert state.tickers_mentioned == ["NVDA"]
    assert state.last_ticker == "NVDA"


# user messages never carry sources (chat_repository.add_message only attaches them to
# assistant turns) - a ticker mentioned only in a user message must not be tracked
def test_ticker_named_only_in_user_text_is_not_tracked():
    history = [_user("What about AAPL?")]

    state = extract_entities(history)

    assert state.tickers_mentioned == []
    assert state.last_ticker is None


# re-mentioning an earlier ticker bumps it back to "most recent", not just first-seen order
def test_rementioning_a_ticker_bumps_it_to_most_recent():
    history = [
        _assistant("NVDA scores 88.", sources=[_source("NVDA")]),
        _assistant("AMD scores 75.", sources=[_source("AMD")]),
        _assistant("NVDA again: 88.", sources=[_source("NVDA")]),
    ]

    state = extract_entities(history)

    assert state.tickers_mentioned == ["AMD", "NVDA"]
    assert state.last_ticker == "NVDA"


# multiple sources in one turn (e.g. compare_tickers) are all tracked, in order
def test_multiple_sources_in_one_turn_are_all_tracked():
    history = [_assistant("MSFT edges out GOOGL.", sources=[_source("MSFT"), _source("GOOGL")])]

    state = extract_entities(history)

    assert state.tickers_mentioned == ["MSFT", "GOOGL"]
    assert state.last_ticker == "GOOGL"


# factors are matched by name (the closed WEIGHTS vocabulary) against any turn's text,
# case-insensitively, regardless of role
def test_factors_are_extracted_by_keyword_match():
    history = [
        _user("What's its momentum score?"),
        _assistant("Momentum is 92/100, driven by strong RSI."),
    ]

    state = extract_entities(history)

    assert state.factors_mentioned == ["Momentum"]
    assert state.last_factor == "Momentum"


def test_rementioning_a_factor_bumps_it_to_most_recent():
    history = [
        _assistant("Quality is strong."),
        _assistant("Momentum is strong too."),
        _assistant("Back to quality: still strong."),
    ]

    state = extract_entities(history)

    assert state.factors_mentioned == ["Momentum", "Quality"]
    assert state.last_factor == "Quality"


# --- entity_hint ---

def test_entity_hint_is_none_when_nothing_has_been_discussed():
    assert entity_hint(EntityState()) is None


def test_entity_hint_names_the_last_ticker_and_factor():
    state = EntityState(tickers_mentioned=["NVDA"], factors_mentioned=["Momentum"], last_ticker="NVDA", last_factor="Momentum")

    hint = entity_hint(state)

    assert "NVDA" in hint
    assert "Momentum" in hint
    assert "pronoun" in hint


def test_entity_hint_omits_factor_clause_when_no_factor_discussed_yet():
    state = EntityState(tickers_mentioned=["NVDA"], factors_mentioned=[], last_ticker="NVDA", last_factor=None)

    hint = entity_hint(state)

    assert "NVDA" in hint
    assert "Most recently discussed factor" not in hint
