"""
One-time historical earnings backfill: ingests up to 4 trailing
quarters of transcripts per ticker across the full S&P 500 universe, so
QoQ comparison and "what changed last quarter" chat questions work
immediately for the existing universe rather than only going forward
from the calendar-driven scheduler.

Throttled to respect api-ninjas.com's free-tier rate limit (one request
every 45s - src/earnings/backfill.py's REQUEST_INTERVAL_SECONDS) - a
full-universe run from scratch is several hours to a day+, by design.

Resumable: progress per ticker is persisted to earnings_backfill_progress
as it goes (committed after every ticker, not just at the end), so
killing this script (Ctrl-C, a machine restart, running out of API quota
for the day) and re-running the exact same command later skips every
ticker already marked "done" instead of restarting from the first
ticker in the list.

Requires EARNINGS_PROVIDER_API_KEY (see src/earnings/provider_client.py).

Run:
    python -m scripts.backfill_earnings_transcripts
"""
import logging

from dotenv import load_dotenv

from src.data.db import SessionLocal
from src.earnings.backfill import backfill_universe
from src.earnings.provider_client import EarningsProviderAuthError
from titan.data import get_sp500_tickers

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    tickers = get_sp500_tickers()
    logger.info("Starting historical earnings backfill: %d tickers, up to 4 trailing quarters each", len(tickers))

    db = SessionLocal()
    try:
        summary = backfill_universe(db, tickers)
    except EarningsProviderAuthError:
        logger.exception(
            "Backfill stopped: provider auth error - fix EARNINGS_PROVIDER_API_KEY and re-run "
            "this same command to resume from where it left off"
        )
        return
    finally:
        db.close()

    logger.info("Backfill run complete: %s", summary)


if __name__ == "__main__":
    main()
