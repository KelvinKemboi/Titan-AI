from unittest.mock import MagicMock

import pytest
from requests.adapters import HTTPAdapter

from src.earnings.provider_client import (
    EarningsProviderAuthError,
    EarningsProviderError,
    Transcript,
    TranscriptSearchResult,
    _session_with_retries,
    get_transcript,
    search_transcripts,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("EARNINGS_PROVIDER_API_KEY", "test-provider-key")


def _response(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body
    resp.text = text
    return resp


# auth
def test_get_transcript_raises_when_api_key_is_unset(monkeypatch):
    monkeypatch.delenv("EARNINGS_PROVIDER_API_KEY", raising=False)

    with pytest.raises(EarningsProviderAuthError):
        get_transcript("AAPL", session=MagicMock())


def test_get_transcript_sends_the_configured_key_as_a_header():
    session = MagicMock()
    session.get.return_value = _response(json_body={
        "ticker": "AAPL", "year": "2024", "quarter": "2", "date": "2024-05-02", "transcript": "hello",
    })

    get_transcript("AAPL", session=session)

    assert session.get.call_args.kwargs["headers"] == {"X-Api-Key": "test-provider-key"}


def test_get_transcript_raises_auth_error_when_provider_rejects_the_key():
    session = MagicMock()
    session.get.return_value = _response(status_code=400, text="Invalid API Key")

    with pytest.raises(EarningsProviderAuthError):
        get_transcript("AAPL", session=session)


def test_get_transcript_raises_a_plain_error_for_an_unrelated_400():
    session = MagicMock()
    session.get.return_value = _response(status_code=400, text="Bad ticker format")

    with pytest.raises(EarningsProviderError):
        get_transcript("AAPL", session=session)


def test_get_transcript_raises_for_a_server_error():
    session = MagicMock()
    session.get.return_value = _response(status_code=500, text="oops")

    with pytest.raises(EarningsProviderError):
        get_transcript("AAPL", session=session)


# get_transcript mapping
def test_get_transcript_maps_the_provider_payload_into_our_own_shape():
    session = MagicMock()
    session.get.return_value = _response(json_body={
        "ticker": "AAPL", "year": "2024", "quarter": "2", "date": "2024-05-02",
        "transcript": "Operator: Welcome to the call...",
    })

    result = get_transcript("AAPL", year=2024, quarter=2, session=session)

    assert result == Transcript(
        ticker="AAPL", fiscal_year=2024, fiscal_quarter="Q2",
        call_date="2024-05-02", raw_text="Operator: Welcome to the call...",
    )


def test_get_transcript_normalizes_and_forwards_ticker_year_quarter():
    session = MagicMock()
    session.get.return_value = _response(json_body={
        "ticker": "MSFT", "year": "2025", "quarter": "1", "date": "2025-01-28", "transcript": "text",
    })

    get_transcript(" msft ", year=2025, quarter=1, session=session)

    params = session.get.call_args.kwargs["params"]
    assert params == {"ticker": "MSFT", "year": 2025, "quarter": 1}


def test_get_transcript_omits_year_and_quarter_when_not_given():
    session = MagicMock()
    session.get.return_value = _response(json_body={
        "ticker": "MSFT", "year": "2025", "quarter": "1", "date": "2025-01-28", "transcript": "text",
    })

    get_transcript("MSFT", session=session)

    assert session.get.call_args.kwargs["params"] == {"ticker": "MSFT"}


# graceful "no transcript available"
def test_get_transcript_returns_none_for_an_empty_body():
    session = MagicMock()
    session.get.return_value = _response(json_body={})

    assert get_transcript("ZZZZ", session=session) is None


def test_get_transcript_returns_none_when_transcript_text_is_missing():
    session = MagicMock()
    session.get.return_value = _response(json_body={"ticker": "ZZZZ", "year": "2024", "quarter": "1"})

    assert get_transcript("ZZZZ", session=session) is None


# search_transcripts pagination
def test_search_transcripts_yields_a_single_short_page_without_a_second_call():
    session = MagicMock()
    session.get.return_value = _response(json_body=[
        {"ticker": "AAPL", "year": "2024", "quarter": "2", "date": "2024-05-02"},
        {"ticker": "AAPL", "year": "2024", "quarter": "1", "date": "2024-02-01"},
    ])

    results = list(search_transcripts("AAPL", session=session))

    assert session.get.call_count == 1
    assert results == [
        TranscriptSearchResult(ticker="AAPL", fiscal_year=2024, fiscal_quarter="Q2", call_date="2024-05-02"),
        TranscriptSearchResult(ticker="AAPL", fiscal_year=2024, fiscal_quarter="Q1", call_date="2024-02-01"),
    ]


def test_search_transcripts_walks_every_full_page_until_a_short_one(monkeypatch):
    from src.earnings import provider_client

    monkeypatch.setattr(provider_client, "_SEARCH_PAGE_SIZE", 2)
    full_page = [
        {"ticker": "AAPL", "year": "2024", "quarter": "2", "date": "2024-05-02"},
        {"ticker": "AAPL", "year": "2024", "quarter": "1", "date": "2024-02-01"},
    ]
    short_page = [{"ticker": "AAPL", "year": "2023", "quarter": "4", "date": "2023-11-01"}]
    session = MagicMock()
    session.get.side_effect = [_response(json_body=full_page), _response(json_body=short_page)]

    results = list(search_transcripts("AAPL", session=session))

    assert session.get.call_count == 2
    first_call_params = session.get.call_args_list[0].kwargs["params"]
    second_call_params = session.get.call_args_list[1].kwargs["params"]
    assert first_call_params == {"ticker": "AAPL", "offset": 0, "limit": 2}
    assert second_call_params == {"ticker": "AAPL", "offset": 2, "limit": 2}
    assert [r.fiscal_quarter for r in results] == ["Q2", "Q1", "Q4"]


def test_search_transcripts_yields_nothing_for_a_ticker_with_no_transcripts():
    session = MagicMock()
    session.get.return_value = _response(json_body=[])

    assert list(search_transcripts("ZZZZ", session=session)) == []


# rate limiting is actually wired into the shared session
def test_shared_session_retries_rate_limits_and_transient_server_errors():
    session = _session_with_retries()

    adapter = session.get_adapter("https://api.api-ninjas.com")
    assert isinstance(adapter, HTTPAdapter)
    assert 429 in adapter.max_retries.status_forcelist
    assert adapter.max_retries.total == 5
