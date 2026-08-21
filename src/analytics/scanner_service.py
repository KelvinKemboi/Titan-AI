import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql import func

from src.analytics.memo_indexing import index_memos
from src.data.db import SessionLocal
from src.data.models import Company, FactorScore, ScanRun
from titan.analyst import RoboAnalyst

# A `running` scan_runs row older than this is assumed to belong to a
# crashed process, not an in-progress scan, so it no longer blocks new runs.
STALE_RUN_THRESHOLD = timedelta(hours=2)


class ScanAlreadyRunningError(Exception):
    """Raised when a scan is requested while another one is still in progress."""


def _analyze_ticker(ticker):
    # Delay to prevent IP Bans (Dynamic Throttling)
    time.sleep(random.uniform(0.1, 1.0))

    analyst = RoboAnalyst(ticker)
    if analyst.analyze():
        analyst.generate_memo()
        return analyst
    return None


def run_scan(tickers, concurrency=5, on_progress=None):
    """
    Runs the factor-scoring scan over `tickers` and returns the valid
    RoboAnalyst results, unsorted.

    UI-agnostic: callers drive their own progress display via
    `on_progress(index, total)`, invoked with the same 0-based index and
    total count for every ticker as it completes.
    """
    results = []

    # Run the analysis concurrently with a thread pool
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_analyze_ticker, t) for t in tickers] # Submit each ticker to the thread pool for analysis

        # Process the results as they complete
        for i, future in enumerate(futures):
            res = future.result()
            if res and res.valid:
                results.append(res)

            # on_progress callback for UI updates, if provided
            if on_progress:
                on_progress(i, len(futures))

    return results

# compare the results of the scan with the database and persist the new data - update then insert if not exists
def _upsert_company(session, result):
    """Insert or refresh a companies row from the fundamentals RoboAnalyst.analyze() fetched."""
    info = result.info or {}
    # Upsert the company data into the database using PostgreSQL's ON CONFLICT clause
    stmt = pg_insert(Company).values(
        ticker=result.ticker,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        description=info.get("longBusinessSummary"),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={
            "name": stmt.excluded.name,
            "sector": stmt.excluded.sector,
            "industry": stmt.excluded.industry,
            "description": stmt.excluded.description,
            "updated_at": func.now(),
        },
    )
    session.execute(stmt)

# save the factor score for a ticker in a scan run to the database
def _save_factor_score(session, scan_run_id, result):
    """Insert one factor_scores row from a completed RoboAnalyst result."""
    # Unpack the individual factor scores from the result metrics
    value_score, momentum_score, quality_score, solvency_score, volatility_score = (
        result.metrics["Scores"]
    )
    # add the factor score to the database
    session.add(FactorScore(
        ticker=result.ticker,
        scan_run_id=scan_run_id,
        value_score=value_score,
        momentum_score=momentum_score,
        quality_score=quality_score,
        solvency_score=solvency_score,
        volatility_score=volatility_score,
        composite_score=result.score,
        rating=result.rating,
        raw_metrics=result.metrics,
    ))

def _blocking_run(session):
    """
    Returns the currently in-progress scan_runs row, or None if there
    isn't one (or the only `running` row is stale. Its process crashed
    without ever reaching a terminal status).
    """
    running = (
        session.query(ScanRun)
        .filter(ScanRun.status == "running")
        .order_by(ScanRun.started_at.desc())
        .first()
    )
    if running is None or running.started_at is None:
        return running

    started_at = running.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - started_at > STALE_RUN_THRESHOLD:
        return None
    return running


# scan the tickers and persist the results to the database, handling concurrency and progress updates
def run_scan_and_persist(tickers, concurrency=5, on_progress=None):
    """
    Runs `run_scan` and persists the outcome to Postgres: one `scan_runs`
    row for the run, one `factor_scores` row per successfully-analyzed
    ticker, and an upserted `companies` row per ticker from the
    fundamentals fetched during analysis.

    A run where every ticker fails `analyze()` is marked `failed`; a run
    where some (but not all) tickers fail is marked `partial`, so a
    partial failure never fails the whole run. Raises
    `ScanAlreadyRunningError` instead of starting a new run if one is
    already in progress. Returns the same `run_scan` results, unsorted.
    """
    session = SessionLocal()
    try:
        blocking = _blocking_run(session)
        if blocking is not None:
            raise ScanAlreadyRunningError(
                f"scan_runs id={blocking.id} has been running since "
                f"{blocking.started_at}; refusing to start a new scan"
            )

        scan_run = ScanRun(status="running", universe_size=len(tickers)) # create a new scan run record in the database with the status "running" and the total number of tickers to be scanned
        session.add(scan_run)
        session.commit()

        try:
            # Run the scan
            results = run_scan(tickers, concurrency=concurrency, on_progress=on_progress)
            # Persist the results to the database
            for result in results:
                _upsert_company(session, result)
                _save_factor_score(session, scan_run.id, result)
            # Embeds + persists each memo for semantic search (search_memos tool).
            # Failures inside are logged and swallowed there, not raised - a scan's
            # factor scores must still count as persisted even if this doesn't.
            index_memos(session, scan_run.id, results)
        except Exception:
            session.rollback() # Rollback the session in case of an exception to avoid partial commits
            scan_run.status = "failed"
            scan_run.completed_at = func.now()
            session.commit()
            raise

        if not results:
            scan_run.status = "failed"
        elif len(results) < len(tickers):
            scan_run.status = "partial"
        else:
            scan_run.status = "complete"
        scan_run.completed_at = func.now()
        session.commit()

        return results
    finally:
        session.close()
