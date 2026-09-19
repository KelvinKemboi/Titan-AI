"""
One-time historical earnings backfill (technical-design.md §15): ingests
up to MAX_TRAILING_QUARTERS of transcripts per ticker across the full
S&P 500 universe, so QoQ comparison (#12) and "what changed last
quarter" chat questions work immediately for tickers that predate
src/earnings/earnings_scheduler.py's calendar-driven scheduling, instead
of waiting 1-2 real quarters for that scheduler to accumulate history
naturally.

Run via scripts/backfill_earnings_transcripts.py.
"""
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.models import Company, EarningsBackfillProgress
from src.earnings.ingestion import ingest_transcript
from src.earnings.provider_client import (
    EarningsProviderAuthError,
    EarningsProviderError,
    search_transcripts,
)

logger = logging.getLogger(__name__)

# "up to 4 trailing quarters" - acceptance criteria.
MAX_TRAILING_QUARTERS = 4

# Minimum seconds between provider requests (search_transcripts and every
# ingest_transcript call alike) - api-ninjas.com's free tier is 100
# requests/hour (technical-design.md §7); 45s keeps this comfortably
# under that even with src/earnings/earnings_scheduler.py's own hourly
# processing tick sharing the same quota concurrently. A full-universe,
# from-scratch run (worst case 1 search + 4 ingests per ticker, 500
# tickers) is therefore slow by design - several hours to a day+ - which
# is the real cost of respecting a free-tier rate limit, not a bug to
# work around here (see technical-design.md §15's note on the paid tier).
REQUEST_INTERVAL_SECONDS = 45

_last_request_at: Optional[float] = None


def _throttle() -> None:
    """Blocks just long enough since the last provider request to keep
    requests at least REQUEST_INTERVAL_SECONDS apart - not a flat sleep
    per call (which would needlessly stack on top of however long the
    request itself took)."""
    global _last_request_at
    now = time.monotonic()
    if _last_request_at is not None:
        wait = REQUEST_INTERVAL_SECONDS - (now - _last_request_at)
        if wait > 0:
            time.sleep(wait)
    _last_request_at = time.monotonic()


class BackfillOutcome:
    def __init__(self, status: str, quarters_ingested: int, error: Optional[str] = None):
        self.status = status  # done | failed
        self.quarters_ingested = quarters_ingested
        self.error = error


def backfill_ticker(db: Session, ticker: str) -> BackfillOutcome:
    """
    Ingests up to MAX_TRAILING_QUARTERS for `ticker`, most recent first.
    Raises EarningsProviderAuthError (bad/missing key - a config problem
    every remaining ticker would hit identically) so the caller can stop
    the whole run; every other miss is recorded on the returned outcome,
    never raised.

    status="failed" only when the initial search_transcripts call itself
    fails - we don't yet know what's available, worth retrying on the
    next resumed run. status="done" once search_transcripts succeeds and
    every quarter it found has been attempted, even if some individual
    quarter didn't ingest (already logged there): re-running search
    wouldn't surface anything new for those, so retrying provides no
    value and would just spend the request quota again for nothing.
    """
    _throttle()
    try:
        results = list(search_transcripts(ticker))
    except EarningsProviderAuthError:
        raise
    except EarningsProviderError as exc:
        logger.warning("%s: search_transcripts failed, will retry on next run: %s", ticker, exc)
        return BackfillOutcome(status="failed", quarters_ingested=0, error=str(exc))

    trailing = results[:MAX_TRAILING_QUARTERS]  # most-recent-call-first, per search_transcripts
    ingested = 0
    errors: List[str] = []
    for result in trailing:
        quarter = int(result.fiscal_quarter[1:])
        _throttle()
        try:
            transcript = ingest_transcript(db, ticker, result.fiscal_year, quarter)
        except EarningsProviderAuthError:
            raise
        except EarningsProviderError as exc:
            logger.warning(
                "%s %s %d: ingest_transcript failed: %s", ticker, result.fiscal_quarter, result.fiscal_year, exc,
            )
            errors.append(f"{result.fiscal_quarter} {result.fiscal_year}: {exc}")
            continue

        if transcript is None:
            logger.warning(
                "%s %s %d: no transcript available from provider", ticker, result.fiscal_quarter, result.fiscal_year,
            )
            errors.append(f"{result.fiscal_quarter} {result.fiscal_year}: not available")
            continue

        ingested += 1

    logger.info(
        "%s: backfilled %d/%d quarter(s)%s",
        ticker, ingested, len(trailing), f" - {len(errors)} miss(es)" if errors else "",
    )
    return BackfillOutcome(
        status="done", quarters_ingested=ingested, error="; ".join(errors) if errors else None,
    )


def _record_progress(db: Session, ticker: str, outcome: BackfillOutcome) -> None:
    now = datetime.now(timezone.utc)
    stmt = (
        pg_insert(EarningsBackfillProgress)
        .values(
            ticker=ticker,
            status=outcome.status,
            quarters_ingested=outcome.quarters_ingested,
            attempted_at=now,
            error=outcome.error,
        )
        .on_conflict_do_update(
            index_elements=["ticker"],
            set_={
                "status": outcome.status,
                "quarters_ingested": outcome.quarters_ingested,
                "attempted_at": now,
                "error": outcome.error,
            },
        )
    )
    db.execute(stmt)


def backfill_universe(db: Session, tickers: List[str]) -> dict:
    """
    Runs backfill_ticker for every ticker in `tickers` not already marked
    "done" in earnings_backfill_progress, committing after each ticker so
    a killed/crashed run's completed work is durable - a resumed run
    (same call, same `tickers`) picks up from the first not-yet-done
    ticker rather than restarting from the first ticker in the list.

    A ticker with no `companies` row yet (never scanned - see
    src/analytics/scanner_service.py) is skipped and logged rather than
    raising, since every earnings table's ticker FK requires one to
    exist first, same precondition src/earnings/earnings_scheduler.py's
    discovery already assumes.

    Returns a summary dict: {"total", "skipped_already_done",
    "skipped_unscanned", "done", "failed", "quarters_ingested"}.
    """
    summary = {
        "total": len(tickers), "skipped_already_done": 0, "skipped_unscanned": 0,
        "done": 0, "failed": 0, "quarters_ingested": 0,
    }

    for ticker in tickers:
        ticker = ticker.strip().upper()

        progress = db.get(EarningsBackfillProgress, ticker)
        if progress is not None and progress.status == "done":
            logger.info("%s: already backfilled (%d quarters) - skipping", ticker, progress.quarters_ingested)
            summary["skipped_already_done"] += 1
            continue

        if db.get(Company, ticker) is None:
            logger.warning("%s: no companies row yet (not scanned) - skipping until a scan run adds it", ticker)
            summary["skipped_unscanned"] += 1
            continue

        outcome = backfill_ticker(db, ticker)
        _record_progress(db, ticker, outcome)
        db.commit()

        summary[outcome.status] += 1
        summary["quarters_ingested"] += outcome.quarters_ingested

    return summary
