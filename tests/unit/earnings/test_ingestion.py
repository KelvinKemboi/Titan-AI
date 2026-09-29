from datetime import date
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from src.data.models import EarningsTranscript
from src.earnings.ingestion import ingest_transcript
from src.earnings.provider_client import EarningsProviderAuthError, EarningsProviderError, Transcript


def _bound_params(stmt):
    """Compiles a pg_insert(...).on_conflict_do_nothing(...) statement to inspect the values it was built with."""
    return stmt.compile(dialect=postgresql.dialect()).params


def _transcript(**overrides):
    defaults = dict(
        ticker="AAPL", fiscal_year=2024, fiscal_quarter="Q2", call_date=date(2024, 5, 2),
        raw_text="Operator: Welcome to the call...",
        source_url="https://api.api-ninjas.com/v1/earningstranscript?ticker=AAPL&year=2024&quarter=2",
    )
    defaults.update(overrides)
    return Transcript(**defaults)


def _db_with_no_existing_row():
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    return db


# Chunk indexing is exercised on its own in test_chunk_indexing.py
@pytest.fixture(autouse=True)
def _no_chunk_indexing(monkeypatch):
    mock_index = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.index_transcript_chunks", mock_index)
    return mock_index


# Summary generation is exercised on its own in test_summary.py
@pytest.fixture(autouse=True)
def _no_summary_generation(monkeypatch):
    mock_generate = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.generate_summary", mock_generate)
    return mock_generate


# Guidance extraction is exercised on its own in test_guidance.py - neutralize it
# here too so these tests stay fast/offline regardless of environment state.
@pytest.fixture(autouse=True)
def _no_guidance_generation(monkeypatch):
    mock_generate = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.generate_guidance", mock_generate)
    return mock_generate


# Sentiment scoring is exercised on its own in test_sentiment.py - same reasoning.
@pytest.fixture(autouse=True)
def _no_sentiment_generation(monkeypatch):
    mock_generate = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.generate_sentiment", mock_generate)
    return mock_generate


# Risk extraction is exercised on its own in test_risk.py - same reasoning.
@pytest.fixture(autouse=True)
def _no_risk_generation(monkeypatch):
    mock_generate = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.generate_risks", mock_generate)
    return mock_generate


# QoQ diffing is exercised on its own in test_qoq.py - same reasoning.
@pytest.fixture(autouse=True)
def _no_qoq_generation(monkeypatch):
    mock_generate = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.generate_qoq_changes", mock_generate)
    return mock_generate


# idempotency: an already-ingested (ticker, fiscal_year, fiscal_quarter)
def test_returns_existing_row_without_calling_the_provider(monkeypatch):
    mock_get_transcript = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = MagicMock()
    existing_row = object()
    db.query.return_value.filter.return_value.one_or_none.return_value = existing_row

    result = ingest_transcript(db, "AAPL", year=2024, quarter=2)

    assert result is existing_row
    mock_get_transcript.assert_not_called()
    db.execute.assert_not_called()


# successful fetch creates one row with raw text + source URL
def test_successful_fetch_inserts_a_row_with_raw_text_and_source_url(monkeypatch):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 42
    persisted_row = object()
    db.get.return_value = persisted_row

    result = ingest_transcript(db, "AAPL", year=2024, quarter=2)

    assert result is persisted_row
    db.get.assert_called_once_with(EarningsTranscript, 42)

    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["ticker"] == "AAPL"
    assert params["fiscal_year"] == 2024
    assert params["fiscal_quarter"] == "Q2"
    assert params["raw_text"] == "Operator: Welcome to the call..."
    assert params["source_url"] == "https://api.api-ninjas.com/v1/earningstranscript?ticker=AAPL&year=2024&quarter=2"


# wiring: a freshly-inserted transcript's chunks get indexed
def test_a_fresh_insert_triggers_chunk_indexing(monkeypatch, _no_chunk_indexing):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 42

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_chunk_indexing.assert_called_once_with(db, 42, "Operator: Welcome to the call...")


def test_an_already_existing_row_does_not_re_trigger_chunk_indexing(monkeypatch, _no_chunk_indexing):
    mock_get_transcript = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = object()

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_chunk_indexing.assert_not_called()


def test_losing_the_insert_race_does_not_re_trigger_chunk_indexing(monkeypatch, _no_chunk_indexing):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.one.return_value = object()
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - lost the race

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_chunk_indexing.assert_not_called()


# wiring: a freshly-inserted transcript's summary gets generated
def test_a_fresh_insert_triggers_summary_generation(monkeypatch, _no_summary_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 42

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_summary_generation.assert_called_once_with(db, 42, "Operator: Welcome to the call...")


def test_an_already_existing_row_does_not_re_trigger_summary_generation(monkeypatch, _no_summary_generation):
    mock_get_transcript = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = object()

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_summary_generation.assert_not_called()


def test_losing_the_insert_race_does_not_re_trigger_summary_generation(monkeypatch, _no_summary_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.one.return_value = object()
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - lost the race

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_summary_generation.assert_not_called()


# wiring: a freshly-inserted transcript's guidance gets extracted
def test_a_fresh_insert_triggers_guidance_generation(monkeypatch, _no_guidance_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 42

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_guidance_generation.assert_called_once_with(db, 42, "Operator: Welcome to the call...")


def test_an_already_existing_row_does_not_re_trigger_guidance_generation(monkeypatch, _no_guidance_generation):
    mock_get_transcript = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = object()

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_guidance_generation.assert_not_called()


def test_losing_the_insert_race_does_not_re_trigger_guidance_generation(monkeypatch, _no_guidance_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.one.return_value = object()
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - lost the race

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_guidance_generation.assert_not_called()


# wiring: a freshly-inserted transcript's sentiment gets scored
def test_a_fresh_insert_triggers_sentiment_generation(monkeypatch, _no_sentiment_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 42

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_sentiment_generation.assert_called_once_with(db, 42, "Operator: Welcome to the call...")


def test_an_already_existing_row_does_not_re_trigger_sentiment_generation(monkeypatch, _no_sentiment_generation):
    mock_get_transcript = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = object()

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_sentiment_generation.assert_not_called()


def test_losing_the_insert_race_does_not_re_trigger_sentiment_generation(monkeypatch, _no_sentiment_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.one.return_value = object()
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - lost the race

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_sentiment_generation.assert_not_called()


# a freshly-inserted transcript's risks get extracted
def test_a_fresh_insert_triggers_risk_generation(monkeypatch, _no_risk_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 42

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_risk_generation.assert_called_once_with(db, 42, "Operator: Welcome to the call...")


def test_an_already_existing_row_does_not_re_trigger_risk_generation(monkeypatch, _no_risk_generation):
    mock_get_transcript = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = object()

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_risk_generation.assert_not_called()


def test_losing_the_insert_race_does_not_re_trigger_risk_generation(monkeypatch, _no_risk_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.one.return_value = object()
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - lost the race

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_risk_generation.assert_not_called()


# wiring: a freshly-inserted transcript's QoQ diff gets computed
def test_a_fresh_insert_triggers_qoq_generation(monkeypatch, _no_qoq_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 42

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_qoq_generation.assert_called_once_with(db, 42, "AAPL")


def test_an_already_existing_row_does_not_re_trigger_qoq_generation(monkeypatch, _no_qoq_generation):
    mock_get_transcript = MagicMock()
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = object()

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_qoq_generation.assert_not_called()


def test_losing_the_insert_race_does_not_re_trigger_qoq_generation(monkeypatch, _no_qoq_generation):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.one.return_value = object()
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - lost the race

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    _no_qoq_generation.assert_not_called()


def test_insert_conflict_target_is_ticker_fiscal_year_fiscal_quarter(monkeypatch):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 1

    ingest_transcript(db, "AAPL", year=2024, quarter=2)

    stmt = db.execute.call_args.args[0]
    assert set(stmt._post_values_clause.inferred_target_elements) == {"ticker", "fiscal_year", "fiscal_quarter"}


def test_ticker_is_normalized_before_being_sent_to_the_provider(monkeypatch):
    mock_get_transcript = MagicMock(return_value=_transcript())
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", mock_get_transcript)
    db = _db_with_no_existing_row()
    db.execute.return_value.scalar_one_or_none.return_value = 1

    ingest_transcript(db, " aapl ", year=2024, quarter=2)

    mock_get_transcript.assert_called_once_with("AAPL", year=2024, quarter=2)


def test_a_lost_insert_race_returns_the_row_that_won_instead_of_none(monkeypatch):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=_transcript()))
    db = MagicMock()
    winning_row = object()
    # First .query(...).filter(...).one_or_none() call is the pre-check (no
    # existing row); after losing the insert race, the fallback query uses
    # .one() (a row must exist now) rather than .one_or_none()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.one.return_value = winning_row
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - lost the race

    result = ingest_transcript(db, "AAPL", year=2024, quarter=2)

    assert result is winning_row
    db.get.assert_not_called()


# missing/unavailable transcript: graceful, not a hard failure
def test_no_transcript_available_returns_none_without_touching_the_db(monkeypatch, caplog):
    monkeypatch.setattr("src.earnings.ingestion.get_transcript", MagicMock(return_value=None))
    db = _db_with_no_existing_row()

    with caplog.at_level("WARNING", logger="src.earnings.ingestion"):
        result = ingest_transcript(db, "ZZZZ", year=2024, quarter=1)

    assert result is None
    db.execute.assert_not_called()
    assert any("ZZZZ" in record.message for record in caplog.records)


# an ordinary provider failure is logged and swallowed, not a hard failure
def test_provider_error_is_logged_and_swallowed(monkeypatch, caplog):
    monkeypatch.setattr(
        "src.earnings.ingestion.get_transcript", MagicMock(side_effect=EarningsProviderError("503 from provider"))
    )
    db = _db_with_no_existing_row()

    with caplog.at_level("WARNING", logger="src.earnings.ingestion"):
        result = ingest_transcript(db, "AAPL", year=2024, quarter=2)  # must not raise

    assert result is None
    db.execute.assert_not_called()
    assert any("AAPL" in record.message for record in caplog.records)


# a bad/missing API key is a config problem, not a per-ticker coverage gap 
def test_auth_error_propagates_instead_of_being_swallowed(monkeypatch):
    monkeypatch.setattr(
        "src.earnings.ingestion.get_transcript", MagicMock(side_effect=EarningsProviderAuthError("bad key"))
    )
    db = _db_with_no_existing_row()

    with pytest.raises(EarningsProviderAuthError):
        ingest_transcript(db, "AAPL", year=2024, quarter=2)

    db.execute.assert_not_called()
