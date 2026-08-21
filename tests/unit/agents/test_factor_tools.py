from unittest.mock import MagicMock

import pytest

from src.agents.tools.factor_tools import compare_tickers
from src.data.models import FactorScore


def _db_with_single_ticker_row(ticker="MSFT", scan_run_id=1):
    row = FactorScore(
        ticker=ticker, scan_run_id=scan_run_id, value_score=50, momentum_score=50,
        quality_score=50, solvency_score=50, volatility_score=50, composite_score=50,
        rating="HOLD", raw_metrics={},
    )
    db = MagicMock()
    db.query.return_value.scalar.return_value = scan_run_id
    db.query.return_value.filter.return_value.all.return_value = [row]
    return db


# duplicate tickers (exact or case-insensitive) must not silently produce a degenerate
# "X vs X" comparison with all-zero deltas - they should collapse to one distinct
# ticker and hit the same "fewer than 2 distinct tickers" error the HTTP route enforces
@pytest.mark.parametrize("tickers", [["MSFT", "MSFT"], ["MSFT", "msft"], [" msft ", "MSFT"]])
def test_duplicate_tickers_raise_instead_of_comparing_a_ticker_to_itself(tickers):
    db = _db_with_single_ticker_row()

    with pytest.raises(ValueError, match="Fewer than 2"):
        compare_tickers(db, tickers)


def test_distinct_tickers_are_preserved_in_order():
    db = MagicMock()
    db.query.return_value.scalar.return_value = 1
    db.query.return_value.filter.return_value.all.return_value = [
        FactorScore(ticker="MSFT", scan_run_id=1, value_score=50, momentum_score=50, quality_score=50,
                    solvency_score=50, volatility_score=50, composite_score=50, rating="HOLD", raw_metrics={}),
        FactorScore(ticker="GOOGL", scan_run_id=1, value_score=60, momentum_score=60, quality_score=60,
                    solvency_score=60, volatility_score=60, composite_score=60, rating="BUY", raw_metrics={}),
    ]

    result = compare_tickers(db, ["MSFT", "GOOGL"])

    assert [t["ticker"] for t in result.data["tickers"]] == ["MSFT", "GOOGL"]
    assert len(result.sources) == 2
