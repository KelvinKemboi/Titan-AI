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


#an ordinary provider failure is logged and swallowed, not a hard failure
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
