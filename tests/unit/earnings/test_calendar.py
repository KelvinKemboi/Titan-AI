from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.earnings.calendar import latest_reported_earnings_date


def _earnings_dates_df(rows):
    """rows: list of (date_str, reported_eps_or_none), most recent last or
    first - order doesn't matter, latest_reported_earnings_date takes the
    max of the reported ones."""
    index = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"Reported EPS": [r[1] for r in rows]}, index=index)


def _mock_ticker(df):
    ticker = MagicMock()
    ticker.get_earnings_dates.return_value = df
    return ticker


def test_returns_the_most_recent_date_with_a_reported_eps():
    df = _earnings_dates_df([
        ("2026-10-29", None), # next estimated call - not yet reported
        ("2026-07-30", 2.02), # most recent actual report
        ("2026-04-30", 2.01),
    ])
    with patch("yfinance.Ticker", return_value=_mock_ticker(df)):
        result = latest_reported_earnings_date("AAPL")

    assert result == datetime(2026, 7, 30).date()


def test_ticker_is_normalized_before_being_passed_to_yfinance():
    df = _earnings_dates_df([("2026-07-30", 2.02)])
    mock_ticker_cls = MagicMock(return_value=_mock_ticker(df))
    with patch("yfinance.Ticker", mock_ticker_cls):
        latest_reported_earnings_date(" aapl ")

    mock_ticker_cls.assert_called_once_with("AAPL")


def test_no_reported_rows_yet_returns_none():
    df = _earnings_dates_df([("2026-10-29", None)])
    with patch("yfinance.Ticker", return_value=_mock_ticker(df)):
        result = latest_reported_earnings_date("AAPL")

    assert result is None


def test_empty_dataframe_returns_none():
    with patch("yfinance.Ticker", return_value=_mock_ticker(pd.DataFrame())):
        result = latest_reported_earnings_date("AAPL")

    assert result is None


def test_none_dataframe_returns_none():
    with patch("yfinance.Ticker", return_value=_mock_ticker(None)):
        result = latest_reported_earnings_date("AAPL")

    assert result is None


# a delisted/unknown ticker or any yfinance/network failure degrades to
# None rather than raising and killing the whole discovery run
def test_yfinance_exception_degrades_to_none_instead_of_raising():
    broken_ticker = MagicMock()
    broken_ticker.get_earnings_dates.side_effect = RuntimeError("symbol may be delisted")
    with patch("yfinance.Ticker", return_value=broken_ticker):
        result = latest_reported_earnings_date("ZZZZNOTREAL")

    assert result is None
