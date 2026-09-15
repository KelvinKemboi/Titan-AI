from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from src.api.routes.earnings import EarningsInsightDetail, EarningsTranscriptDetail
from src.earnings import qoq as qoq_module
from src.earnings.qoq import compute_qoq_changes, generate_qoq_changes


def _transcript_detail(transcript_id, fiscal_year, fiscal_quarter, insight=None):
    return EarningsTranscriptDetail(
        transcript_id=transcript_id, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        source_url="https://api.api-ninjas.com/v1/earningstranscript",
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc), insight=insight,
    )


def _insight_detail(guidance_direction="maintained", sentiment_score=0.5, risks=None):
    return EarningsInsightDetail(
        summary="a summary", guidance_direction=guidance_direction, guidance_quote="a quote",
        sentiment_score=sentiment_score, risks=risks or [],
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _bound_params(stmt):
    return stmt.compile(dialect=postgresql.dialect()).params


def _patch_transcripts(monkeypatch, transcripts):
    monkeypatch.setattr("src.earnings.qoq._get_earnings_for_ticker", MagicMock(return_value=transcripts))


# insufficient_history
def test_first_quarter_ingested_returns_insufficient_history(monkeypatch):
    current = _transcript_detail(1, 2024, "Q2", insight=_insight_detail())
    _patch_transcripts(monkeypatch, [current])  # no prior transcript at all

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=1)

    assert result == {"status": "insufficient_history"}


def test_prior_transcript_with_no_insight_yet_is_insufficient_history(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail())
    prior = _transcript_detail(1, 2024, "Q1", insight=None)
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result == {"status": "insufficient_history"}


def test_current_transcript_with_no_insight_yet_is_insufficient_history(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=None)
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail())
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result == {"status": "insufficient_history"}


def test_unknown_transcript_id_raises(monkeypatch):
    _patch_transcripts(monkeypatch, [_transcript_detail(1, 2024, "Q2", insight=_insight_detail())])

    with pytest.raises(ValueError, match="99"):
        compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=99)


# guidance direction change
def test_guidance_direction_change_is_flagged(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail(guidance_direction="raised"))
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail(guidance_direction="maintained"))
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result["status"] == "ok"
    assert result["guidance_direction"] == {"prior": "maintained", "current": "raised", "changed": True}


def test_unchanged_guidance_direction_is_not_flagged(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail(guidance_direction="maintained"))
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail(guidance_direction="maintained"))
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result["guidance_direction"]["changed"] is False


# sentiment delta 
def test_sentiment_delta_is_current_minus_prior(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail(sentiment_score=0.8))
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail(sentiment_score=0.3))
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result["sentiment_delta"] == pytest.approx(0.5)


def test_sentiment_delta_is_none_when_either_score_is_missing(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail(sentiment_score=None))
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail(sentiment_score=0.3))
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result["sentiment_delta"] is None


# risk set difference
def test_new_and_resolved_risks_are_set_differenced(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail(risks=[
        {"risk": "Supply chain constraints", "quote": "q1"},
        {"risk": "Currency headwinds", "quote": "q2"},
    ]))
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail(risks=[
        {"risk": "Currency headwinds", "quote": "q2"},
        {"risk": "Regulatory scrutiny", "quote": "q3"},
    ]))
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result["new_risks"] == [{"risk": "Supply chain constraints", "quote": "q1"}]
    assert result["resolved_risks"] == [{"risk": "Regulatory scrutiny", "quote": "q3"}]


def test_risk_matching_ignores_case_and_surrounding_whitespace(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail(
        risks=[{"risk": "  Currency Headwinds  ", "quote": "q2-new"}],
    ))
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail(
        risks=[{"risk": "currency headwinds", "quote": "q2-old"}],
    ))
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result["new_risks"] == []
    assert result["resolved_risks"] == []


def test_prior_transcript_id_and_quarter_are_included(monkeypatch):
    current = _transcript_detail(2, 2024, "Q2", insight=_insight_detail())
    prior = _transcript_detail(1, 2024, "Q1", insight=_insight_detail())
    _patch_transcripts(monkeypatch, [current, prior])

    result = compute_qoq_changes(db=MagicMock(), ticker="AAPL", transcript_id=2)

    assert result["prior_transcript_id"] == 1
    assert result["prior_fiscal_year"] == 2024
    assert result["prior_fiscal_quarter"] == "Q1"


# generate_qoq_changes: persistence 
def test_generate_qoq_changes_upserts_on_transcript_id(monkeypatch):
    computed = {"status": "ok", "guidance_direction": {"prior": "maintained", "current": "raised", "changed": True}}
    monkeypatch.setattr(qoq_module, "compute_qoq_changes", MagicMock(return_value=computed))
    db = MagicMock()
    db.execute.return_value.scalar_one.return_value = 4
    persisted_row = object()
    db.get.return_value = persisted_row

    result = generate_qoq_changes(db, transcript_id=4, ticker="AAPL")

    assert result is persisted_row
    db.get.assert_called_once()
    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["transcript_id"] == 4
    assert params["qoq_changes"] == computed
    assert set(stmt._post_values_clause.inferred_target_elements) == {"transcript_id"}


def test_generate_qoq_changes_persists_insufficient_history(monkeypatch):
    monkeypatch.setattr(qoq_module, "compute_qoq_changes", MagicMock(return_value={"status": "insufficient_history"}))
    db = MagicMock()
    db.execute.return_value.scalar_one.return_value = 1

    generate_qoq_changes(db, transcript_id=1, ticker="AAPL")

    stmt = db.execute.call_args.args[0]
    assert _bound_params(stmt)["qoq_changes"] == {"status": "insufficient_history"}


def test_generate_qoq_changes_swallows_a_failure(monkeypatch, caplog):
    monkeypatch.setattr(qoq_module, "compute_qoq_changes", MagicMock(side_effect=RuntimeError("bug")))
    db = MagicMock()

    with caplog.at_level("ERROR", logger="src.earnings.qoq"):
        result = generate_qoq_changes(db, transcript_id=5, ticker="AAPL")  # must not raise

    assert result is None
    db.execute.assert_not_called()
    assert any("5" in record.message for record in caplog.records)
