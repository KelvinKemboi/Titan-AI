from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.tools.earnings_tools import DISPATCH, TOOLS, get_earnings_insight, get_qoq_changes, search_earnings
from src.analytics.earnings_search import EarningsChunkHit
from src.api.routes.earnings import EarningsInsightDetail, EarningsTranscriptDetail


def _transcript_detail(fiscal_year=2024, fiscal_quarter="Q2", transcript_id=1, insight=None, ingested_at=None):
    return EarningsTranscriptDetail(
        transcript_id=transcript_id, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        source_url="https://api.api-ninjas.com/v1/earningstranscript",
        ingested_at=ingested_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
        insight=insight,
    )


def _insight_detail(summary="a summary", guidance_direction="raised", guidance_quote="a quote",
                     sentiment_score=0.5, risks=None, qoq_changes=None):
    return EarningsInsightDetail(
        summary=summary, guidance_direction=guidance_direction, guidance_quote=guidance_quote,
        sentiment_score=sentiment_score, risks=risks or [], qoq_changes=qoq_changes,
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _hit(ticker="AAPL", transcript_id=1, chunk_type="qna", chunk_text="text", as_of=None):
    return EarningsChunkHit(
        ticker=ticker, transcript_id=transcript_id, fiscal_year=2024, fiscal_quarter="Q2",
        chunk_type=chunk_type, chunk_text=chunk_text, as_of=as_of or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# get_earnings_insight
def test_defaults_to_the_latest_quarter_when_none_given(monkeypatch):
    latest = _transcript_detail(fiscal_year=2024, fiscal_quarter="Q2", transcript_id=2, insight=_insight_detail())
    older = _transcript_detail(fiscal_year=2024, fiscal_quarter="Q1", transcript_id=1)
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker", MagicMock(return_value=[latest, older]),
    )

    result = get_earnings_insight(db=MagicMock(), ticker="AAPL")

    assert result.data["fiscal_quarter"] == "Q2"
    assert result.sources[0].ref_id == 2


def test_honors_an_explicit_quarter(monkeypatch):
    latest = _transcript_detail(fiscal_year=2024, fiscal_quarter="Q2", transcript_id=2)
    older = _transcript_detail(fiscal_year=2023, fiscal_quarter="Q1", transcript_id=1, insight=_insight_detail())
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker", MagicMock(return_value=[latest, older]),
    )

    result = get_earnings_insight(db=MagicMock(), ticker="AAPL", quarter=1)

    assert result.data["fiscal_quarter"] == "Q1"
    assert result.sources[0].ref_id == 1


# source metadata includes transcript_id (acceptance criterion)
def test_source_type_and_transcript_id_ref(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(transcript_id=7, insight=_insight_detail())]),
    )

    result = get_earnings_insight(db=MagicMock(), ticker="AAPL")

    assert result.sources[0].type == "earnings_insight"
    assert result.sources[0].ticker == "AAPL"
    assert result.sources[0].ref_id == 7


def test_detail_carries_the_insight_fields(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(insight=_insight_detail(summary="AAPL had a great quarter."))]),
    )

    result = get_earnings_insight(db=MagicMock(), ticker="AAPL")

    assert result.sources[0].detail["summary"] == "AAPL had a great quarter."
    assert result.sources[0].detail["guidance_direction"] == "raised"
    assert result.data["summary"] == "AAPL had a great quarter."


def test_with_no_insight_generated_yet_returns_none_fields_not_an_error(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(insight=None)]),
    )

    result = get_earnings_insight(db=MagicMock(), ticker="AAPL")

    assert result.data["summary"] is None
    assert result.data["risks"] == []
    assert result.data["qoq_changes"] is None


def test_qoq_changes_is_included_in_data_and_detail(monkeypatch):
    qoq = {"status": "ok", "guidance_direction": {"prior": "maintained", "current": "raised", "changed": True}}
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(insight=_insight_detail(qoq_changes=qoq))]),
    )

    result = get_earnings_insight(db=MagicMock(), ticker="AAPL")

    assert result.data["qoq_changes"] == qoq
    assert result.sources[0].detail["qoq_changes"] == qoq


def test_insufficient_history_qoq_changes_is_passed_through_as_is(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(insight=_insight_detail(qoq_changes={"status": "insufficient_history"}))]),
    )

    result = get_earnings_insight(db=MagicMock(), ticker="AAPL")

    assert result.data["qoq_changes"] == {"status": "insufficient_history"}


def test_raises_for_a_ticker_with_no_ingested_transcripts(monkeypatch):
    monkeypatch.setattr("src.agents.tools.earnings_tools._get_earnings_for_ticker", MagicMock(return_value=[]))

    with pytest.raises(ValueError, match="ZZZZ"):
        get_earnings_insight(db=MagicMock(), ticker="ZZZZ")


def test_raises_for_a_quarter_that_was_never_ingested(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(fiscal_quarter="Q2")]),
    )

    with pytest.raises(ValueError, match="Q4"):
        get_earnings_insight(db=MagicMock(), ticker="AAPL", quarter=4)


def test_ticker_is_normalized(monkeypatch):
    mock_get = MagicMock(return_value=[_transcript_detail(insight=_insight_detail())])
    monkeypatch.setattr("src.agents.tools.earnings_tools._get_earnings_for_ticker", mock_get)

    get_earnings_insight(db=MagicMock(), ticker=" aapl ")

    assert mock_get.call_args.args[1] == "AAPL"


# get_qoq_changes
def test_qoq_changes_surfaces_the_already_persisted_field(monkeypatch):
    qoq = {"status": "ok", "guidance_direction": {"prior": "maintained", "current": "raised", "changed": True}}
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(transcript_id=9, insight=_insight_detail(qoq_changes=qoq))]),
    )

    result = get_qoq_changes(db=MagicMock(), ticker="AAPL")

    assert result.data["qoq_changes"] == qoq
    assert result.sources[0].type == "earnings_insight"
    assert result.sources[0].ticker == "AAPL"
    assert result.sources[0].ref_id == 9
    assert result.sources[0].detail["qoq_changes"] == qoq


def test_qoq_changes_always_reads_the_latest_quarter_ignoring_older_ones(monkeypatch):
    latest = _transcript_detail(
        fiscal_year=2024, fiscal_quarter="Q2", transcript_id=2,
        insight=_insight_detail(qoq_changes={"status": "ok"}),
    )
    older = _transcript_detail(
        fiscal_year=2024, fiscal_quarter="Q1", transcript_id=1,
        insight=_insight_detail(qoq_changes={"status": "insufficient_history"}),
    )
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker", MagicMock(return_value=[latest, older]),
    )

    result = get_qoq_changes(db=MagicMock(), ticker="AAPL")

    assert result.data["fiscal_quarter"] == "Q2"
    assert result.data["qoq_changes"] == {"status": "ok"}


def test_qoq_changes_insufficient_history_is_passed_through_as_is(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(insight=_insight_detail(qoq_changes={"status": "insufficient_history"}))]),
    )

    result = get_qoq_changes(db=MagicMock(), ticker="AAPL")

    assert result.data["qoq_changes"] == {"status": "insufficient_history"}


def test_qoq_changes_with_no_insight_generated_yet_returns_none_not_an_error(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._get_earnings_for_ticker",
        MagicMock(return_value=[_transcript_detail(insight=None)]),
    )

    result = get_qoq_changes(db=MagicMock(), ticker="AAPL")

    assert result.data["qoq_changes"] is None


def test_qoq_changes_raises_for_a_ticker_with_no_ingested_transcripts(monkeypatch):
    monkeypatch.setattr("src.agents.tools.earnings_tools._get_earnings_for_ticker", MagicMock(return_value=[]))

    with pytest.raises(ValueError, match="ZZZZ"):
        get_qoq_changes(db=MagicMock(), ticker="ZZZZ")


def test_qoq_changes_ticker_is_normalized(monkeypatch):
    mock_get = MagicMock(return_value=[_transcript_detail(insight=_insight_detail())])
    monkeypatch.setattr("src.agents.tools.earnings_tools._get_earnings_for_ticker", mock_get)

    get_qoq_changes(db=MagicMock(), ticker=" aapl ")

    assert mock_get.call_args.args[1] == "AAPL"


def test_dispatch_maps_get_qoq_changes(monkeypatch):
    db = MagicMock()
    mock_get = MagicMock(return_value=[_transcript_detail(insight=_insight_detail())])
    monkeypatch.setattr("src.agents.tools.earnings_tools._get_earnings_for_ticker", mock_get)

    DISPATCH["get_qoq_changes"](db, {"ticker": "AAPL"})

    mock_get.assert_called_once_with(db, "AAPL")


# search_earnings
def test_search_earnings_returns_sourced_results(monkeypatch):
    hits = [_hit("AAPL", 1), _hit("MSFT", 2)]
    monkeypatch.setattr("src.agents.tools.earnings_tools._search_earnings", MagicMock(return_value=hits))

    result = search_earnings(db=MagicMock(), query="AI capex")

    assert [s.type for s in result.sources] == ["earnings_chunk", "earnings_chunk"]
    assert [s.ticker for s in result.sources] == ["AAPL", "MSFT"]
    assert [s.ref_id for s in result.sources] == [1, 2]


def test_search_earnings_detail_carries_chunk_type_and_text(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.earnings_tools._search_earnings",
        MagicMock(return_value=[_hit(chunk_type="qna", chunk_text="We are seeing strong AI demand.")]),
    )

    result = search_earnings(db=MagicMock(), query="AI capex")

    assert result.sources[0].detail == {"chunk_type": "qna", "chunk_text": "We are seeing strong AI demand."}


def test_search_earnings_with_no_hits_returns_empty_result(monkeypatch):
    monkeypatch.setattr("src.agents.tools.earnings_tools._search_earnings", MagicMock(return_value=[]))

    result = search_earnings(db=MagicMock(), query="nothing matches this")

    assert result.data == {"results": []}
    assert result.sources == []


# dispatch + schema registration
def test_dispatch_maps_get_earnings_insight_with_a_quarter(monkeypatch):
    db = MagicMock()
    mock_get = MagicMock(return_value=[_transcript_detail(insight=_insight_detail())])
    monkeypatch.setattr("src.agents.tools.earnings_tools._get_earnings_for_ticker", mock_get)

    DISPATCH["get_earnings_insight"](db, {"ticker": "AAPL", "quarter": 2})

    mock_get.assert_called_once_with(db, "AAPL")


def test_dispatch_maps_get_earnings_insight_without_a_quarter(monkeypatch):
    db = MagicMock()
    mock_get = MagicMock(return_value=[_transcript_detail(insight=_insight_detail())])
    monkeypatch.setattr("src.agents.tools.earnings_tools._get_earnings_for_ticker", mock_get)

    DISPATCH["get_earnings_insight"](db, {"ticker": "AAPL"})

    mock_get.assert_called_once_with(db, "AAPL")


def test_dispatch_maps_search_earnings_to_query_input(monkeypatch):
    db = MagicMock()
    mock_search = MagicMock(return_value=[])
    monkeypatch.setattr("src.agents.tools.earnings_tools._search_earnings", mock_search)

    DISPATCH["search_earnings"](db, {"query": "AI capex"})

    mock_search.assert_called_once_with(db, "AI capex")


def test_schemas_are_registered_in_tools():
    assert {t["name"] for t in TOOLS} == {"get_earnings_insight", "get_qoq_changes", "search_earnings"}
