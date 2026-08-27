from types import SimpleNamespace
from unittest.mock import MagicMock

from src.analytics import scanner_service


def _fake_result(ticker):
    return SimpleNamespace(
        ticker=ticker, valid=True, score=80.0, rating="BUY", memo="",
        metrics={
            "Scores": [1, 2, 3, 4, 5], "Price": 1, "RSI": 1, "Trend": "Bullish",
            "Val_Metric": 1, "Val_Type": "PEG", "Margin": 0.1, "Debt": 1, "Beta": 1,
        },
        info={},
    )


def _mock_session():
    session = MagicMock()
    # _blocking_run's query chain: no scan currently running
    session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
    return session


def _patch_scan_internals(monkeypatch, results):
    session = _mock_session()
    monkeypatch.setattr(scanner_service, "SessionLocal", MagicMock(return_value=session))
    monkeypatch.setattr(scanner_service, "run_scan", MagicMock(return_value=results))
    monkeypatch.setattr(scanner_service, "_upsert_company", MagicMock())
    monkeypatch.setattr(scanner_service, "_save_factor_score", MagicMock())
    monkeypatch.setattr(scanner_service, "index_memos", MagicMock())
    return session


# scan completion invalidates the caches for exactly thetickers it just updated
def test_successful_scan_invalidates_caches_for_scanned_tickers(monkeypatch):
    _patch_scan_internals(monkeypatch, [_fake_result("AAPL"), _fake_result("MSFT")])
    mock_invalidate = MagicMock()
    monkeypatch.setattr(scanner_service, "invalidate_scan_caches", mock_invalidate)

    scanner_service.run_scan_and_persist(["AAPL", "MSFT"])

    mock_invalidate.assert_called_once_with(["AAPL", "MSFT"])


# a scan where every ticker fails has nothing new in Postgres
def test_fully_failed_scan_does_not_invalidate_caches(monkeypatch):
    _patch_scan_internals(monkeypatch, [])
    mock_invalidate = MagicMock()
    monkeypatch.setattr(scanner_service, "invalidate_scan_caches", mock_invalidate)

    scanner_service.run_scan_and_persist(["AAPL"])

    mock_invalidate.assert_not_called()


# a partial scan (some tickers failed) still invalidates for whichever tickers actually got fresh data
def test_partial_scan_invalidates_only_the_tickers_that_succeeded(monkeypatch):
    _patch_scan_internals(monkeypatch, [_fake_result("AAPL")])
    mock_invalidate = MagicMock()
    monkeypatch.setattr(scanner_service, "invalidate_scan_caches", mock_invalidate)

    scanner_service.run_scan_and_persist(["AAPL", "MSFT"])

    mock_invalidate.assert_called_once_with(["AAPL"])
