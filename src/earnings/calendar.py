"""
Earnings calendar source: yfinance's own earnings-date calendar - free
and already a project dependency, and deliberately a separate source
from api-ninjas.com, the transcript provider.
"""
import logging
from datetime import date
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# get_earnings_dates walks backward from today; a handful of rows is
# plenty to find the most recent reported one.
_EARNINGS_DATES_LOOKBACK = 8


def latest_reported_earnings_date(ticker: str) -> Optional[date]:
    """
    The most recent date yfinance shows an actual `Reported EPS` for
    `ticker` - i.e. the call has already happened, not merely estimated.
    Returns None if yfinance has no earnings-date data for this ticker,
    or nothing has been reported yet.

    Never raises: a delisted/unknown ticker or a yfinance/network hiccup
    degrades to "nothing to report" (None) rather than failing the whole
    discovery run over one bad ticker - same defensive posture as
    RoboAnalyst.analyze()'s own yfinance calls.
    """
    ticker = ticker.strip().upper()
    try:
        earnings_dates = yf.Ticker(ticker).get_earnings_dates(limit=_EARNINGS_DATES_LOOKBACK)
    except Exception:
        logger.warning("yfinance earnings-date lookup failed for %s", ticker, exc_info=True)
        return None

    if earnings_dates is None or earnings_dates.empty or "Reported EPS" not in earnings_dates.columns:
        return None

    reported = earnings_dates[earnings_dates["Reported EPS"].notna()]
    if reported.empty:
        return None

    return reported.index.max().date()
