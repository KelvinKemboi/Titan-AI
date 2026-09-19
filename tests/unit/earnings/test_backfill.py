from datetime import date
from unittest.mock import MagicMock, call

import pytest
from sqlalchemy.dialects import postgresql

from src.earnings import backfill as backfill_module
from src.earnings.backfill import (
    MAX_TRAILING_QUARTERS,
    REQUEST_INTERVAL_SECONDS,
    _throttle,
    backfill_ticker,
    backfill_universe,
)
from src.earnings.provider_client import EarningsProviderAuthError, EarningsProviderError, TranscriptSearchResult
from src.data.models import Company, EarningsBackfillProgress


def _bound_params(stmt):
    return stmt.compile(dialect=postgresql.dialect()).params


def _result(fiscal_year=2024, fiscal_quarter="Q2", call_date=None, ticker="AAPL"):
    return TranscriptSearchResult(
        ticker=ticker, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        call_date=call_date or date(2024, 5, 2),
    )


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    """Every backfill_ticker/backfill_universe test replaces the real
    throttle with a no-op so tests don't actually sleep - _throttle's own
    pacing behavior is exercised directly in its own tests below."""
    monkeypatch.setattr(backfill_module, "_throttle", MagicMock())


def _patch_search(monkeypatch, results=None, exc=None):
    mock = MagicMock(side_effect=exc) if exc else MagicMock(return_value=results or [])
    monkeypatch.setattr("src.earnings.backfill.search_transcripts", mock)
    return mock


def _patch_ingest(monkeypatch, side_effect=None, return_value=None):
    mock = MagicMock(side_effect=side_effect) if side_effect else MagicMock(return_value=return_value)
    monkeypatch.setattr("src.earnings.backfill.ingest_transcript", mock)
    return mock


# --- backfill_ticker ---


def test_ingests_every_result_up_to_the_trailing_quarter_cap(monkeypatch):
    results = [_result(fiscal_year=2024, fiscal_quarter=f"Q{q}") for q in (4, 3, 2, 1)]
    _patch_search(monkeypatch, results=results)
    mock_ingest = _patch_ingest(monkeypatch, return_value=object())
    db = MagicMock()

    outcome = backfill_ticker(db, "AAPL")

    assert outcome.status == "done"
    assert outcome.quarters_ingested == 4
    assert outcome.error is None
    assert mock_ingest.call_count == 4
    mock_ingest.assert_any_call(db, "AAPL", 2024, 4)
    mock_ingest.assert_any_call(db, "AAPL", 2024, 1)


def test_only_the_most_recent_max_trailing_quarters_are_attempted(monkeypatch):
    # search_transcripts yields most-recent-call-first; 6 results available, cap is 4
    results = [_result(fiscal_year=2024, fiscal_quarter=f"Q{q}") for q in (2, 1)]
    results += [_result(fiscal_year=2023, fiscal_quarter=f"Q{q}") for q in (4, 3, 2, 1)]
    assert len(results) == 6
    _patch_search(monkeypatch, results=results)
    mock_ingest = _patch_ingest(monkeypatch, return_value=object())
    db = MagicMock()

    outcome = backfill_ticker(db, "AAPL")

    assert mock_ingest.call_count == MAX_TRAILING_QUARTERS == 4
    # the two oldest (2023 Q2, 2023 Q1) were never attempted
    called_args = {c.args[2:] for c in mock_ingest.call_args_list}
    assert (2023, 2) not in called_args
    assert (2023, 1) not in called_args
    assert outcome.quarters_ingested == 4


def test_search_transcripts_failure_marks_the_ticker_failed_for_retry(monkeypatch):
    _patch_search(monkeypatch, exc=EarningsProviderError("503"))
    mock_ingest = _patch_ingest(monkeypatch)
    db = MagicMock()

    outcome = backfill_ticker(db, "AAPL")

    assert outcome.status == "failed"
    assert outcome.quarters_ingested == 0
    mock_ingest.assert_not_called()


def test_search_transcripts_auth_error_propagates(monkeypatch):
    _patch_search(monkeypatch, exc=EarningsProviderAuthError("bad key"))
    db = MagicMock()

    with pytest.raises(EarningsProviderAuthError):
        backfill_ticker(db, "AAPL")


# a miss on one quarter doesn't abort the ticker - still "done", just fewer quarters ingested
def test_a_missing_quarter_does_not_fail_the_whole_ticker(monkeypatch):
    results = [_result(fiscal_year=2024, fiscal_quarter="Q2"), _result(fiscal_year=2024, fiscal_quarter="Q1")]
    _patch_search(monkeypatch, results=results)
    _patch_ingest(monkeypatch, return_value=None)  # provider has neither
    db = MagicMock()

    outcome = backfill_ticker(db, "AAPL")

    assert outcome.status == "done"
    assert outcome.quarters_ingested == 0
    assert outcome.error is not None


def test_an_ingest_provider_error_is_logged_and_does_not_abort_remaining_quarters(monkeypatch):
    results = [_result(fiscal_year=2024, fiscal_quarter="Q2"), _result(fiscal_year=2024, fiscal_quarter="Q1")]
    _patch_search(monkeypatch, results=results)
    mock_ingest = MagicMock(side_effect=[EarningsProviderError("timeout"), object()])
    monkeypatch.setattr("src.earnings.backfill.ingest_transcript", mock_ingest)
    db = MagicMock()

    outcome = backfill_ticker(db, "AAPL")

    assert outcome.status == "done"
    assert outcome.quarters_ingested == 1  # only the second call succeeded
    assert mock_ingest.call_count == 2


def test_ingest_auth_error_propagates_and_stops_further_quarters(monkeypatch):
    results = [_result(fiscal_year=2024, fiscal_quarter="Q2"), _result(fiscal_year=2024, fiscal_quarter="Q1")]
    _patch_search(monkeypatch, results=results)
    mock_ingest = MagicMock(side_effect=EarningsProviderAuthError("bad key"))
    monkeypatch.setattr("src.earnings.backfill.ingest_transcript", mock_ingest)
    db = MagicMock()

    with pytest.raises(EarningsProviderAuthError):
        backfill_ticker(db, "AAPL")

    mock_ingest.assert_called_once()  # stopped after the first quarter, didn't try the second


def test_no_results_at_all_is_done_with_zero_quarters(monkeypatch):
    _patch_search(monkeypatch, results=[])
    mock_ingest = _patch_ingest(monkeypatch)
    db = MagicMock()

    outcome = backfill_ticker(db, "AAPL")

    assert outcome.status == "done"
    assert outcome.quarters_ingested == 0
    mock_ingest.assert_not_called()


# --- backfill_universe ---


def _db_with(company_present=True, progress=None):
    db = MagicMock()

    def _get(model, ticker):
        if model is Company:
            return Company(ticker=ticker) if company_present else None
        if model is EarningsBackfillProgress:
            return progress
        return None

    db.get.side_effect = _get
    return db


def test_skips_a_ticker_already_marked_done(monkeypatch):
    mock_backfill = MagicMock()
    monkeypatch.setattr("src.earnings.backfill.backfill_ticker", mock_backfill)
    done_row = EarningsBackfillProgress(ticker="AAPL", status="done", quarters_ingested=4)
    db = _db_with(progress=done_row)

    summary = backfill_universe(db, ["AAPL"])

    mock_backfill.assert_not_called()
    assert summary["skipped_already_done"] == 1
    assert summary["done"] == 0


def test_retries_a_ticker_previously_marked_failed(monkeypatch):
    outcome = backfill_module.BackfillOutcome(status="done", quarters_ingested=2)
    mock_backfill = MagicMock(return_value=outcome)
    monkeypatch.setattr("src.earnings.backfill.backfill_ticker", mock_backfill)
    failed_row = EarningsBackfillProgress(ticker="AAPL", status="failed", quarters_ingested=0)
    db = _db_with(progress=failed_row)

    summary = backfill_universe(db, ["AAPL"])

    mock_backfill.assert_called_once()
    assert summary["done"] == 1


def test_skips_a_ticker_with_no_companies_row(monkeypatch):
    mock_backfill = MagicMock()
    monkeypatch.setattr("src.earnings.backfill.backfill_ticker", mock_backfill)
    db = _db_with(company_present=False, progress=None)

    summary = backfill_universe(db, ["ZZZZNOTREAL"])

    mock_backfill.assert_not_called()
    assert summary["skipped_unscanned"] == 1


def test_processes_a_new_ticker_and_persists_progress(monkeypatch):
    outcome = backfill_module.BackfillOutcome(status="done", quarters_ingested=3, error=None)
    monkeypatch.setattr("src.earnings.backfill.backfill_ticker", MagicMock(return_value=outcome))
    db = _db_with(progress=None)

    summary = backfill_universe(db, ["AAPL"])

    assert summary["done"] == 1
    assert summary["quarters_ingested"] == 3
    db.execute.assert_called_once()
    stmt = db.execute.call_args.args[0]
    params = _bound_params(stmt)
    assert params["ticker"] == "AAPL"
    assert params["status"] == "done"
    assert params["quarters_ingested"] == 3
    db.commit.assert_called_once()


def test_progress_is_committed_after_every_ticker_not_just_at_the_end(monkeypatch):
    outcome = backfill_module.BackfillOutcome(status="done", quarters_ingested=1)
    monkeypatch.setattr("src.earnings.backfill.backfill_ticker", MagicMock(return_value=outcome))
    db = _db_with(progress=None)

    backfill_universe(db, ["AAPL", "MSFT", "NVDA"])

    assert db.commit.call_count == 3


def test_ticker_is_normalized_before_lookups(monkeypatch):
    outcome = backfill_module.BackfillOutcome(status="done", quarters_ingested=0)
    mock_backfill = MagicMock(return_value=outcome)
    monkeypatch.setattr("src.earnings.backfill.backfill_ticker", mock_backfill)
    db = _db_with(progress=None)

    backfill_universe(db, [" aapl "])

    mock_backfill.assert_called_once_with(db, "AAPL")


def test_auth_error_stops_the_whole_run_rather_than_being_swallowed_per_ticker(monkeypatch):
    mock_backfill = MagicMock(side_effect=EarningsProviderAuthError("bad key"))
    monkeypatch.setattr("src.earnings.backfill.backfill_ticker", mock_backfill)
    db = _db_with(progress=None)

    with pytest.raises(EarningsProviderAuthError):
        backfill_universe(db, ["AAPL", "MSFT", "NVDA"])

    mock_backfill.assert_called_once()  # never reached MSFT/NVDA


def test_summary_accumulates_across_multiple_tickers(monkeypatch):
    outcomes = {
        "AAPL": backfill_module.BackfillOutcome(status="done", quarters_ingested=4),
        "MSFT": backfill_module.BackfillOutcome(status="failed", quarters_ingested=0),
    }
    monkeypatch.setattr(
        "src.earnings.backfill.backfill_ticker", MagicMock(side_effect=lambda db, t: outcomes[t]),
    )
    db = _db_with(progress=None)

    summary = backfill_universe(db, ["AAPL", "MSFT"])

    assert summary["total"] == 2
    assert summary["done"] == 1
    assert summary["failed"] == 1
    assert summary["quarters_ingested"] == 4


# --- _throttle ---


@pytest.fixture(autouse=False)
def _reset_last_request_at():
    """_throttle's pacing state is a module global (deliberately, so it
    persists across every provider call in a run, not just within one
    ticker) - reset it around these tests so they don't depend on each
    other's execution order."""
    previous = backfill_module._last_request_at
    backfill_module._last_request_at = None
    yield
    backfill_module._last_request_at = previous


def test_throttle_does_not_sleep_on_the_very_first_call(monkeypatch, _reset_last_request_at):
    mock_sleep = MagicMock()
    monkeypatch.setattr(backfill_module.time, "sleep", mock_sleep)

    _throttle()

    mock_sleep.assert_not_called()


def test_throttle_sleeps_the_remaining_interval_when_called_too_soon(monkeypatch, _reset_last_request_at):
    # both of _throttle's own time.monotonic() reads return the same fixed
    # "now" - only _last_request_at (set below, simulating an earlier call)
    # varies, which is all the wait calculation depends on.
    monkeypatch.setattr(backfill_module.time, "monotonic", lambda: 100.5)
    mock_sleep = MagicMock()
    monkeypatch.setattr(backfill_module.time, "sleep", mock_sleep)
    backfill_module._last_request_at = 100.0  # previous call was 0.5s ago

    _throttle()

    mock_sleep.assert_called_once_with(pytest.approx(REQUEST_INTERVAL_SECONDS - 0.5))


def test_throttle_does_not_sleep_once_enough_time_has_already_passed(monkeypatch, _reset_last_request_at):
    monkeypatch.setattr(backfill_module.time, "monotonic", lambda: 100.0 + REQUEST_INTERVAL_SECONDS + 10)
    mock_sleep = MagicMock()
    monkeypatch.setattr(backfill_module.time, "sleep", mock_sleep)
    backfill_module._last_request_at = 100.0

    _throttle()

    mock_sleep.assert_not_called()
