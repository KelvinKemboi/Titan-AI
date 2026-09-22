from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.tools.report_tools import DISPATCH, TOOLS, get_analyst_report
from src.analytics.report import AnalystReport
from src.api.routes.earnings import EarningsTranscriptDetail


def _report(ticker="AAPL", scan_run_id=42, computed_at=None, composite_score=90.75,
            rating="STRONG BUY", memo="the memo text", has_earnings_data=False):
    return AnalystReport(
        ticker=ticker, scan_run_id=scan_run_id, computed_at=computed_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
        composite_score=composite_score, rating=rating, memo=memo, has_earnings_data=has_earnings_data,
    )


def _transcript_detail(transcript_id=7, fiscal_year=2024, fiscal_quarter="Q2", ingested_at=None):
    return EarningsTranscriptDetail(
        transcript_id=transcript_id, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        source_url="https://api.api-ninjas.com/v1/earningstranscript",
        ingested_at=ingested_at or datetime(2026, 1, 2, tzinfo=timezone.utc), insight=None,
    )


def _patch_report(monkeypatch, report):
    monkeypatch.setattr("src.agents.tools.report_tools.generate_ticker_report", MagicMock(return_value=report))


def _patch_transcripts(monkeypatch, transcripts):
    mock = MagicMock(return_value=transcripts)
    monkeypatch.setattr("src.agents.tools.report_tools._get_earnings_for_ticker", mock)
    return mock


def test_data_matches_the_report_verbatim(monkeypatch):
    report = _report(memo="AAPL is a STRONG BUY.")
    _patch_report(monkeypatch, report)

    result = get_analyst_report(db=MagicMock(), ticker="AAPL")

    assert result.data == report.model_dump(mode="json")


def test_without_earnings_data_returns_a_single_factor_score_source(monkeypatch):
    report = _report(has_earnings_data=False)
    _patch_report(monkeypatch, report)
    mock_transcripts = _patch_transcripts(monkeypatch, [_transcript_detail()])

    result = get_analyst_report(db=MagicMock(), ticker="AAPL")

    assert len(result.sources) == 1
    assert result.sources[0].type == "factor_score"
    assert result.sources[0].ticker == "AAPL"
    assert result.sources[0].ref_id == report.scan_run_id
    assert result.sources[0].as_of == report.computed_at
    assert result.sources[0].detail["memo"] == report.memo
    mock_transcripts.assert_not_called()  # no need to look up earnings data at all


def test_with_earnings_data_adds_a_second_earnings_insight_source(monkeypatch):
    report = _report(has_earnings_data=True)
    _patch_report(monkeypatch, report)
    _patch_transcripts(monkeypatch, [_transcript_detail(transcript_id=7, fiscal_year=2024, fiscal_quarter="Q2")])

    result = get_analyst_report(db=MagicMock(), ticker="AAPL")

    assert len(result.sources) == 2
    assert result.sources[0].type == "factor_score"
    assert result.sources[1].type == "earnings_insight"
    assert result.sources[1].ticker == "AAPL"
    assert result.sources[1].ref_id == 7
    assert result.sources[1].detail == {"fiscal_year": 2024, "fiscal_quarter": "Q2"}


def test_with_earnings_data_but_no_transcripts_found_falls_back_to_one_source(monkeypatch):
    # defensive: has_earnings_data=True but the lookup somehow finds nothing
    report = _report(has_earnings_data=True)
    _patch_report(monkeypatch, report)
    _patch_transcripts(monkeypatch, [])

    result = get_analyst_report(db=MagicMock(), ticker="AAPL")

    assert len(result.sources) == 1


def test_propagates_value_error_for_a_ticker_with_no_factor_scores(monkeypatch):
    monkeypatch.setattr(
        "src.agents.tools.report_tools.generate_ticker_report",
        MagicMock(side_effect=ValueError("No factor_scores found for ticker 'ZZZZ'")),
    )

    with pytest.raises(ValueError, match="ZZZZ"):
        get_analyst_report(db=MagicMock(), ticker="ZZZZ")


def test_ticker_argument_passes_through_to_generate_ticker_report(monkeypatch):
    mock_generate = MagicMock(return_value=_report())
    monkeypatch.setattr("src.agents.tools.report_tools.generate_ticker_report", mock_generate)
    _patch_transcripts(monkeypatch, [])
    db = MagicMock()

    get_analyst_report(db=db, ticker="AAPL")

    mock_generate.assert_called_once_with(db, "AAPL")


def test_dispatch_maps_get_analyst_report(monkeypatch):
    db = MagicMock()
    mock_generate = MagicMock(return_value=_report())
    monkeypatch.setattr("src.agents.tools.report_tools.generate_ticker_report", mock_generate)
    _patch_transcripts(monkeypatch, [])

    DISPATCH["get_analyst_report"](db, {"ticker": "AAPL"})

    mock_generate.assert_called_once_with(db, "AAPL")


def test_schema_is_registered_in_tools():
    assert [t["name"] for t in TOOLS] == ["get_analyst_report"]
    assert "ticker" in TOOLS[0]["input_schema"]["properties"]
