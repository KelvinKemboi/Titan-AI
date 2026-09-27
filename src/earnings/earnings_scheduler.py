"""
Calendar-driven auto-ingestion scheduler: two recurring jobs, tracked in
earnings_ingestion_jobs. `discover_due_earnings_jobs` (daily) checks
every known company's earnings calendar and opens one job per newly-
reported earnings date. `process_due_earnings_jobs` (hourly) attempts
every due job, backing off exponentially on a miss rather than failing
permanently, until it's marked "exhausted" after enough attempts or too
much time elapsed.

Standalone process:
    python -m src.earnings.earnings_scheduler
"""
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.db import SessionLocal
from src.data.models import Company, EarningsIngestionJob
from src.earnings.calendar import latest_reported_earnings_date
from src.earnings.ingestion import ingest_transcript
from src.earnings.provider_client import (
    EarningsProviderAuthError,
    EarningsProviderError,
    search_transcripts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DISCOVERY_INTERVAL_SECONDS = 24 * 3600
PROCESSING_INTERVAL_SECONDS = 3600

# How long after a reported earnings date this system keeps trying before giving up on that quarter.
MAX_WINDOW = timedelta(days=21)

# Exponential backoff between attempts once a job is discovered:
# 6h, 12h, 24h, 48h, 72h, 72h, ... capped at BACKOFF_MAX.
BACKOFF_BASE = timedelta(hours=6)
BACKOFF_FACTOR = 2
BACKOFF_MAX = timedelta(days=3)
MAX_ATTEMPTS = 10


def _backoff_delay(attempt_count: int) -> timedelta:
    return min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt_count), BACKOFF_MAX)


def discover_due_earnings_jobs(db: Session) -> int:
    """
    Opens one earnings_ingestion_jobs row per (ticker, earnings_date)
    reporting event this system hasn't already seen, for every company
    with a reported earnings date within MAX_WINDOW. Returns the number
    of new jobs opened. Idempotent - the table's unique index turns a
    re-discovered event into a no-op, not a duplicate.
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - MAX_WINDOW).date()
    opened = 0

    tickers = [row[0] for row in db.query(Company.ticker).all()]
    for ticker in tickers:
        earnings_date = latest_reported_earnings_date(ticker)
        if earnings_date is None or earnings_date < cutoff:
            continue

        stmt = (
            pg_insert(EarningsIngestionJob)
            .values(
                ticker=ticker,
                earnings_date=earnings_date,
                status="pending",
                attempt_count=0,
                next_attempt_at=now,
            )
            .on_conflict_do_nothing(index_elements=["ticker", "earnings_date"])
            .returning(EarningsIngestionJob.id)
        )
        if db.execute(stmt).scalar_one_or_none() is not None:
            opened += 1

    db.commit()
    return opened


def _reschedule(job: EarningsIngestionJob, now: datetime, *, error: str) -> None:
    """Records a miss and either backs off to a later next_attempt_at, or
    gives up (status="exhausted") if this job is out of budget - never
    raises, and never leaves the job silently stuck as "pending" forever."""
    job.attempt_count += 1
    job.last_attempted_at = now
    job.last_error = error

    out_of_attempts = job.attempt_count >= MAX_ATTEMPTS
    out_of_window = (now.date() - job.earnings_date) > MAX_WINDOW
    if out_of_attempts or out_of_window:
        job.status = "exhausted"
        logger.warning(
            "Giving up on %s earnings_date=%s after %d attempt(s) (%s): %s",
            job.ticker, job.earnings_date, job.attempt_count,
            "out of attempts" if out_of_attempts else "past MAX_WINDOW", error,
        )
    else:
        job.next_attempt_at = now + _backoff_delay(job.attempt_count)
        logger.info(
            "%s earnings_date=%s not ready (%s); retrying at %s (attempt %d/%d)",
            job.ticker, job.earnings_date, error, job.next_attempt_at, job.attempt_count, MAX_ATTEMPTS,
        )


def _process_one_job(db: Session, job: EarningsIngestionJob, now: datetime) -> None:
    """
    Resolves `job`'s reporting event to a (fiscal_year, fiscal_quarter)
    via the provider's own transcript listing, since the calendar source
    only gives a date, then delegates to the existing ingest_transcript.
    Raises EarningsProviderAuthError so the caller can stop the whole
    run; every other miss is recorded via _reschedule, not raised.
    """
    try:
        results = list(search_transcripts(job.ticker))
    except EarningsProviderAuthError:
        raise
    except EarningsProviderError as exc:
        _reschedule(job, now, error=f"search_transcripts failed: {exc}")
        return

    if not results:
        _reschedule(job, now, error="provider has no transcripts listed for this ticker yet")
        return

    latest = results[0]  # search_transcripts yields most-recent-call-first
    if latest.call_date is not None and latest.call_date < job.earnings_date:
        # The provider's own listing hasn't caught up to this reporting
        # event yet (its "latest" is still an older quarter) - not ready.
        _reschedule(job, now, error="provider's latest listed transcript predates this earnings_date")
        return

    quarter = int(latest.fiscal_quarter[1:])
    try:
        result = ingest_transcript(db, job.ticker, latest.fiscal_year, quarter)
    except EarningsProviderAuthError:
        raise
    except EarningsProviderError as exc:
        _reschedule(job, now, error=f"ingest_transcript failed: {exc}")
        return

    if result is None:
        _reschedule(job, now, error="ingest_transcript returned no transcript")
        return

    job.status = "succeeded"
    job.fiscal_year = latest.fiscal_year
    job.fiscal_quarter = latest.fiscal_quarter
    job.attempt_count += 1
    job.last_attempted_at = now
    job.last_error = None


def process_due_earnings_jobs(db: Session) -> None:
    """
    Attempts every pending earnings_ingestion_jobs row whose
    next_attempt_at has arrived, committing each job's outcome
    independently. An EarningsProviderAuthError stops the whole run
    rather than being swallowed per job, since a bad key affects every
    remaining job identically.
    """
    now = datetime.now(timezone.utc)
    due_jobs = (
        db.query(EarningsIngestionJob)
        .filter(
            EarningsIngestionJob.status == "pending",
            EarningsIngestionJob.next_attempt_at <= now,
        )
        .all()
    )

    for job in due_jobs:
        _process_one_job(db, job, now)
        db.commit()


def scheduled_discovery():
    db = SessionLocal()
    try:
        opened = discover_due_earnings_jobs(db)
        logger.info("Earnings calendar discovery complete: %d new job(s) opened", opened)
    except Exception:
        logger.exception("Earnings calendar discovery failed")
    finally:
        db.close()


def scheduled_processing():
    db = SessionLocal()
    try:
        process_due_earnings_jobs(db)
    except EarningsProviderAuthError:
        logger.exception("Earnings ingestion processing stopped: provider auth error")
    except Exception:
        logger.exception("Earnings ingestion processing failed")
    finally:
        db.close()


def main():
    scheduler = BlockingScheduler()
    scheduler.add_job(
        scheduled_discovery,
        trigger=IntervalTrigger(seconds=DISCOVERY_INTERVAL_SECONDS),
        id="earnings_calendar_discovery",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(),
    )
    scheduler.add_job(
        scheduled_processing,
        trigger=IntervalTrigger(seconds=PROCESSING_INTERVAL_SECONDS),
        id="earnings_ingestion_processing",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(),
    )
    logger.info(
        "Earnings scheduler started: discovery every %ds, processing every %ds",
        DISCOVERY_INTERVAL_SECONDS, PROCESSING_INTERVAL_SECONDS,
    )
    scheduler.start()


if __name__ == "__main__":
    main()
