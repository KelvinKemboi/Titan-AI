import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql import func

from src.analytics.memo_indexing import index_memos
from src.data.cache import invalidate_scan_caches
from src.data.db import SessionLocal
from src.data.models import Company, FactorScore, ScanRun
from titan.analyst import RoboAnalyst

# A `running` scan_runs row older than this is assumed to belong to a
# crashed process, not an in-progress scan, so it no longer blocks new runs.
STALE_RUN_THRESHOLD = timedelta(hours=2)

# Postgres advisory-lock key for the "is a scan already running" check.
# Held only across _blocking_run()'s check and the new ScanRun insert, so
# two near-simultaneous callers can't both pass the check before either commits.
_SCAN_RUN_LOCK_KEY = 727100


class ScanAlreadyRunningError(Exception):
    """Raised when a scan is requested while another one is still in progress."""


def _analyze_ticker(ticker):
    # Randomized delay so concurrent requests don't look like a burst to the data provider.
    time.sleep(random.uniform(0.1, 1.0))

    analyst = RoboAnalyst(ticker)
    if analyst.analyze():
        analyst.generate_memo()
        return analyst
    return None


def run_scan(tickers, concurrency=5, on_progress=None):
    """
    Runs the factor-scoring scan over `tickers` and returns the valid
    RoboAnalyst results, unsorted. UI-agnostic: callers drive their own
    progress display via `on_progress(index, total)`.
    """
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_analyze_ticker, t) for t in tickers]

        for i, future in enumerate(futures):
            res = future.result()
            if res and res.valid:
                results.append(res)
            if on_progress:
                on_progress(i, len(futures))

    return results

def _upsert_company(session, result):
    """Insert or refresh a companies row from the fundamentals RoboAnalyst.analyze() fetched."""
    info = result.info or {}
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

def _save_factor_score(session, scan_run_id, result):
    """Insert one factor_scores row from a completed RoboAnalyst result."""
    value_score, momentum_score, quality_score, solvency_score, volatility_score = (
        result.metrics["Scores"]
    )
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
    """Returns the currently in-progress scan_runs row, or None if there
    isn't one, or the only `running` row belongs to a crashed process."""
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


def run_scan_and_persist(tickers, concurrency=5, on_progress=None):
    """
    Runs `run_scan` and persists the outcome to Postgres: one `scan_runs`
    row for the run, one `factor_scores` row per successfully-analyzed
    ticker, and an upserted `companies` row per ticker. A run where every
    ticker fails is marked `failed`; a partial failure is marked
    `partial`. Raises `ScanAlreadyRunningError` if one is already in
    progress.
    """
    session = SessionLocal()
    try:
        # Held through the commit below so the check-then-insert is atomic across concurrent callers.
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SCAN_RUN_LOCK_KEY})

        blocking = _blocking_run(session)
        if blocking is not None:
            raise ScanAlreadyRunningError(
                f"scan_runs id={blocking.id} has been running since "
                f"{blocking.started_at}; refusing to start a new scan"
            )

        scan_run = ScanRun(status="running", universe_size=len(tickers))
        session.add(scan_run)
        session.commit()

        try:
            results = run_scan(tickers, concurrency=concurrency, on_progress=on_progress)
            for result in results:
                _upsert_company(session, result)
                _save_factor_score(session, scan_run.id, result)
            # Failures here are logged and swallowed inside index_memos, not raised.
            index_memos(session, scan_run.id, results)
        except Exception:
            session.rollback()
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

        # Invalidate only after the commit above, so cached reads never see stale data.
        if results:
            invalidate_scan_caches([r.ticker for r in results])

        return results
    finally:
        session.close()
