from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.analytics.report import generate_report, generate_ticker_report
from src.data.models import EarningsInsight, EarningsTranscript, FactorScore


def _factor_score(ticker="AAPL", scan_run_id=42, composite_score=90.75):
    return FactorScore(
        ticker=ticker, scan_run_id=scan_run_id, composite_score=composite_score,
        computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        raw_metrics={
            "Price": 190.5, "RSI": 65.0, "Trend": "Bullish",
            "Val_Metric": 1.5, "Val_Type": "PEG",
            "Margin": 0.25, "Debt": 40.0, "Beta": 0.9,
            "Scores": [75.0, 100.0, 100.0, 80.0, 90.0],
        },
    )


def _insight(summary="Record revenue.", guidance_direction="raised", risks=None):
    return EarningsInsight(
        transcript_id=1, summary=summary, guidance_direction=guidance_direction,
        risks=risks if risks is not None else [{"risk": "Supply chain", "quote": "q"}],
    )


def _transcript(id=1, ticker="AAPL"):
    return EarningsTranscript(
        id=id, ticker=ticker, fiscal_year=2024, fiscal_quarter="Q2",
        raw_text="irrelevant", ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_no_earnings_insight_produces_a_memo_with_no_recent_earnings_section():
    report = generate_report(_factor_score())

    assert report.has_earnings_data is False
    assert "Recent Earnings" not in report.memo
    assert report.ticker == "AAPL"
    assert report.rating == "STRONG BUY"
    assert report.composite_score == pytest.approx(90.75)


def test_earnings_insight_produces_a_memo_with_a_recent_earnings_section():
    report = generate_report(_factor_score(), _insight())

    assert report.has_earnings_data is True
    assert "**Recent Earnings:**" in report.memo
    assert "Record revenue." in report.memo
    assert "**Guidance:** Raised" in report.memo
    assert "Supply chain" in report.memo


def test_null_composite_score_raises_value_error():
    with pytest.raises(ValueError):
        generate_report(_factor_score(composite_score=None))


def test_generate_ticker_report_combines_latest_factor_score_and_latest_insight():
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = _factor_score()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [_transcript()]
    db.query.return_value.filter.return_value.one_or_none.return_value = _insight()

    report = generate_ticker_report(db, "aapl")

    assert report.ticker == "AAPL"
    assert report.has_earnings_data is True
    assert "**Recent Earnings:**" in report.memo


def test_generate_ticker_report_with_no_transcripts_has_no_earnings_section():
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = _factor_score()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

    report = generate_ticker_report(db, "AAPL")

    assert report.has_earnings_data is False
    assert "Recent Earnings" not in report.memo


def test_generate_ticker_report_raises_when_ticker_has_no_factor_score():
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    with pytest.raises(ValueError):
        generate_ticker_report(db, "ZZZZNOTREAL")
