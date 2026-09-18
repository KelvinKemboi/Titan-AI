from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from src.earnings.earnings_scheduler import (
    BACKOFF_BASE,
    BACKOFF_MAX,
    MAX_ATTEMPTS,
    MAX_WINDOW,
    _backoff_delay,
    discover_due_earnings_jobs,
    process_due_earnings_jobs,
)
from src.earnings.provider_client import EarningsProviderAuthError, EarningsProviderError, TranscriptSearchResult
from src.data.models import EarningsIngestionJob


def _bound_params(stmt):
    return stmt.compile(dialect=postgresql.dialect()).params


# Recent-but-not-today, relative to whenever these tests actually run -
# a fixed calendar date would silently drift out of MAX_WINDOW over time
# and start failing every "still within window" test below for reasons
# unrelated to what they're testing.
_RECENT_EARNINGS_DATE = (datetime.now(timezone.utc) - timedelta(days=2)).date()


def _job(id=1, ticker="AAPL", earnings_date=None, status="pending", attempt_count=0, next_attempt_at=None):
    return EarningsIngestionJob(
        id=id, ticker=ticker, earnings_date=earnings_date or _RECENT_EARNINGS_DATE,
        status=status, attempt_count=attempt_count,
        next_attempt_at=next_attempt_at or datetime.now(timezone.utc),
    )


def _search_result(fiscal_year=2026, fiscal_quarter="Q3", call_date=None, ticker="AAPL"):
    return TranscriptSearchResult(
        ticker=ticker, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        call_date=call_date or _RECENT_EARNINGS_DATE,
    )


# discover_due_earnings_jobs
def test_discover_opens_a_job_for_a_ticker_with_a_recent_reported_date(monkeypatch):
    monkeypatch.setattr(
        "src.earnings.earnings_scheduler.latest_reported_earnings_date",
        MagicMock(return_value=_RECENT_EARNINGS_DATE),
    )
    db = MagicMock()
    db.query.return_value.all.return_value = [("AAPL",)]
    db.execute.return_value.scalar_one_or_none.return_value = 1  # a new row was inserted

    opened = discover_due_earnings_jobs(db)

    assert opened == 1
    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["ticker"] == "AAPL"
    assert params["earnings_date"] == _RECENT_EARNINGS_DATE
    assert params["status"] == "pending"
    db.commit.assert_called_once()


def test_discover_skips_a_ticker_with_no_reported_date(monkeypatch):
    monkeypatch.setattr(
        "src.earnings.earnings_scheduler.latest_reported_earnings_date", MagicMock(return_value=None),
    )
    db = MagicMock()
    db.query.return_value.all.return_value = [("AAPL",)]

    opened = discover_due_earnings_jobs(db)

    assert opened == 0
    db.execute.assert_not_called()


def test_discover_skips_a_reported_date_older_than_max_window(monkeypatch):
    stale_date = (datetime.now(timezone.utc) - MAX_WINDOW - timedelta(days=1)).date()
    monkeypatch.setattr(
        "src.earnings.earnings_scheduler.latest_reported_earnings_date", MagicMock(return_value=stale_date),
    )
    db = MagicMock()
    db.query.return_value.all.return_value = [("AAPL",)]

    opened = discover_due_earnings_jobs(db)

    assert opened == 0
    db.execute.assert_not_called()


# on_conflict_do_nothing means a re-discovered event is a no-op, not a duplicate
def test_discover_does_not_count_an_already_discovered_event(monkeypatch):
    monkeypatch.setattr(
        "src.earnings.earnings_scheduler.latest_reported_earnings_date",
        MagicMock(return_value=_RECENT_EARNINGS_DATE),
    )
    db = MagicMock()
    db.query.return_value.all.return_value = [("AAPL",)]
    db.execute.return_value.scalar_one_or_none.return_value = None  # conflict - already exists

    opened = discover_due_earnings_jobs(db)

    assert opened == 0


def test_discover_checks_every_known_ticker(monkeypatch):
    mock_calendar = MagicMock(side_effect=[_RECENT_EARNINGS_DATE, None, _RECENT_EARNINGS_DATE - timedelta(days=1)])
    monkeypatch.setattr("src.earnings.earnings_scheduler.latest_reported_earnings_date", mock_calendar)
    db = MagicMock()
    db.query.return_value.all.return_value = [("AAPL",), ("ZZZZ",), ("MSFT",)]
    db.execute.return_value.scalar_one_or_none.return_value = 1

    opened = discover_due_earnings_jobs(db)

    assert mock_calendar.call_args_list == [(("AAPL",),), (("ZZZZ",),), (("MSFT",),)]
    assert opened == 2


# process_due_earnings_jobs / _process_one_job
def _patch_search(monkeypatch, results=None, exc=None):
    mock = MagicMock(side_effect=exc) if exc else MagicMock(return_value=results or [])
    monkeypatch.setattr("src.earnings.earnings_scheduler.search_transcripts", mock)
    return mock


def _patch_ingest(monkeypatch, result=None, exc=None):
    mock = MagicMock(side_effect=exc) if exc else MagicMock(return_value=result)
    monkeypatch.setattr("src.earnings.earnings_scheduler.ingest_transcript", mock)
    return mock


def _db_with_due_jobs(jobs):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = jobs
    return db


def test_a_ready_transcript_is_ingested_and_job_marked_succeeded(monkeypatch):
    job = _job()
    _patch_search(monkeypatch, results=[_search_result(fiscal_year=2026, fiscal_quarter="Q3")])
    mock_ingest = _patch_ingest(monkeypatch, result=object())
    db = _db_with_due_jobs([job])

    process_due_earnings_jobs(db)

    mock_ingest.assert_called_once_with(db, "AAPL", 2026, 3)
    assert job.status == "succeeded"
    assert job.fiscal_year == 2026
    assert job.fiscal_quarter == "Q3"
    assert job.attempt_count == 1
    db.commit.assert_called_once()


def test_provider_has_no_transcripts_listed_yet_backs_off(monkeypatch):
    job = _job(attempt_count=0)
    _patch_search(monkeypatch, results=[])
    mock_ingest = _patch_ingest(monkeypatch)
    db = _db_with_due_jobs([job])

    process_due_earnings_jobs(db)

    mock_ingest.assert_not_called()
    assert job.status == "pending"
    assert job.attempt_count == 1
    assert job.next_attempt_at > datetime.now(timezone.utc)  # rescheduled into the future


def test_providers_latest_listing_still_predates_the_earnings_date_backs_off(monkeypatch):
    job = _job()
    # provider's search still only shows the PRIOR quarter's call
    _patch_search(monkeypatch, results=[_search_result(fiscal_year=2026, fiscal_quarter="Q2", call_date=_RECENT_EARNINGS_DATE - timedelta(days=90))])
    mock_ingest = _patch_ingest(monkeypatch)
    db = _db_with_due_jobs([job])

    process_due_earnings_jobs(db)

    mock_ingest.assert_not_called()
    assert job.status == "pending"
    assert job.attempt_count == 1


def test_ingest_transcript_returning_none_backs_off_rather_than_succeeding(monkeypatch):
    job = _job()
    _patch_search(monkeypatch, results=[_search_result()])
    _patch_ingest(monkeypatch, result=None)
    db = _db_with_due_jobs([job])

    process_due_earnings_jobs(db)

    assert job.status == "pending"
    assert job.attempt_count == 1


def test_provider_error_backs_off_rather_than_failing_permanently(monkeypatch):
    job = _job()
    _patch_search(monkeypatch, exc=EarningsProviderError("503"))
    mock_ingest = _patch_ingest(monkeypatch)
    db = _db_with_due_jobs([job])

    process_due_earnings_jobs(db)

    mock_ingest.assert_not_called()
    assert job.status == "pending"
    assert job.attempt_count == 1
    assert job.status != "exhausted"


def test_auth_error_propagates_and_stops_the_whole_run(monkeypatch):
    job = _job()
    _patch_search(monkeypatch, exc=EarningsProviderAuthError("bad key"))
    db = _db_with_due_jobs([job])

    with pytest.raises(EarningsProviderAuthError):
        process_due_earnings_jobs(db)


def test_job_is_exhausted_after_max_attempts(monkeypatch):
    job = _job(attempt_count=MAX_ATTEMPTS - 1)
    _patch_search(monkeypatch, results=[])
    db = _db_with_due_jobs([job])

    process_due_earnings_jobs(db)

    assert job.status == "exhausted"
    assert job.attempt_count == MAX_ATTEMPTS


def test_job_is_exhausted_once_past_max_window_regardless_of_attempt_count(monkeypatch):
    old_earnings_date = (datetime.now(timezone.utc) - MAX_WINDOW - timedelta(days=1)).date()
    job = _job(earnings_date=old_earnings_date, attempt_count=0)
    _patch_search(monkeypatch, results=[])
    db = _db_with_due_jobs([job])

    process_due_earnings_jobs(db)

    assert job.status == "exhausted"


def test_only_pending_jobs_due_now_are_queried():
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []

    process_due_earnings_jobs(db)

    db.query.assert_called_once_with(EarningsIngestionJob)
    filter_args = db.query.return_value.filter.call_args.args
    # two filter clauses: status == "pending" and next_attempt_at <= now
    assert len(filter_args) == 2


# backoff schedule shape 
def test_backoff_delay_grows_exponentially_and_is_capped():
    assert _backoff_delay(0) == BACKOFF_BASE
    assert _backoff_delay(1) == BACKOFF_BASE * 2
    assert _backoff_delay(2) == BACKOFF_BASE * 4
    assert _backoff_delay(10) == BACKOFF_MAX  # capped, not unbounded
