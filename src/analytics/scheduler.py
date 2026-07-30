"""
Runs the Scanner Service on a recurring schedule instead of only on
manual trigger. Standalone process:

    python -m src.analytics.scheduler
"""
import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.analytics.scanner_service import ScanAlreadyRunningError, run_scan_and_persist
from titan.data import get_sp500_tickers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Run the scan every hour (3600 seconds)
SCAN_INTERVAL_SECONDS = 3600


# function that will be called by the scheduler to run the scan and persist the results to the database. It handles exceptions and logs the results.
def scheduled_scan():
    tickers = get_sp500_tickers()
    logger.info("Starting scheduled scan of %d tickers", len(tickers))

    try:
        results = run_scan_and_persist(tickers)
    except ScanAlreadyRunningError as exc:
        # APScheduler's max_instances=1 already prevents this within a single
        # process; this catches the cross-process case (e.g. a manual scan
        # from app.py overlapping a scheduled one)
        logger.warning("Skipping scheduled scan: %s", exc)
        return
    except Exception:
        # Logged with a full traceback, so one bad run doesn't kill the scheduler process and future runs still fire
        logger.exception("Scheduled scan failed")
        return

    logger.info(
        "Scheduled scan complete: %d/%d tickers succeeded",
        len(results), len(tickers),
    )


def main():
    scheduler = BlockingScheduler() # Create a blocking scheduler that runs in the foreground and blocks the main thread
    scheduler.add_job(
        scheduled_scan,
        trigger=IntervalTrigger(seconds=SCAN_INTERVAL_SECONDS),
        id="scanner_service_hourly_scan",
        max_instances=1, # a still-running scan blocks the next tick
        coalesce=True, # if ticks were missed (e.g. process was down), run once, not once-per-missed-tick
        next_run_time=datetime.now(), # scan immediately on startup, then hourly
    )
    logger.info("Scheduler started: scanning every %d seconds", SCAN_INTERVAL_SECONDS)
    scheduler.start()


if __name__ == "__main__":
    main()
