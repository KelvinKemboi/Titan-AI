from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.tools.factor_tools import compare_tickers, get_factor_scores
from src.analytics.explain import FactorExplanation, FactorScoreExplanation
from src.data.models import FactorScore, ScanRun


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


# the source's `detail` carries the full composite/rating/factor breakdown that
# explain_ticker already computed, so a citation UI can verify the claim
# ("why is AAPL ranked so high") without re-fetching it
def test_get_factor_scores_source_detail_includes_composite_rating_and_factor_breakdown(monkeypatch):
    explanation = FactorScoreExplanation(
        ticker="AAPL",
        scan_run_id=7,
        computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        composite_score=82.5,
        rating="BUY",
        factors=[
            FactorExplanation(factor="value", score=70.0, weight=0.25, contribution=17.5, driver="PEG of 1.2"),
            FactorExplanation(factor="momentum", score=90.0, weight=0.30, contribution=27.0, driver="RSI 65"),
        ],
    )
    monkeypatch.setattr("src.agents.tools.factor_tools.explain_ticker", MagicMock(return_value=explanation))
    db = MagicMock()
    db.get.return_value = ScanRun(id=7, completed_at=datetime(2026, 1, 2, tzinfo=timezone.utc))

    result = get_factor_scores(db, "AAPL")

    assert len(result.sources) == 1
    detail = result.sources[0].detail
    assert detail["composite_score"] == 82.5
    assert detail["rating"] == "BUY"
    assert [f["factor"] for f in detail["factors"]] == ["value", "momentum"]
    assert detail["factors"][0] == {
        "factor": "value", "score": 70.0, "weight": 0.25, "contribution": 17.5, "driver": "PEG of 1.2",
    }


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


# each ticker's source carries its own factor/composite scores in `detail`, so a
# citation UI can verify the claim ("why is MSFT above GOOGL") without a second
# fetch - and never the *other* ticker's numbers under a mismatched source
def test_compare_tickers_source_detail_carries_that_tickers_own_scores():
    db = MagicMock()
    db.query.return_value.scalar.return_value = 1
    db.query.return_value.filter.return_value.all.return_value = [
        FactorScore(ticker="MSFT", scan_run_id=1, value_score=50, momentum_score=55, quality_score=60,
                    solvency_score=65, volatility_score=70, composite_score=58, rating="HOLD", raw_metrics={}),
        FactorScore(ticker="GOOGL", scan_run_id=1, value_score=80, momentum_score=81, quality_score=82,
                    solvency_score=83, volatility_score=84, composite_score=82, rating="BUY", raw_metrics={}),
    ]

    result = compare_tickers(db, ["MSFT", "GOOGL"])

    by_ticker = {s.ticker: s for s in result.sources}
    assert by_ticker["MSFT"].detail["composite_score"] == 58
    assert by_ticker["MSFT"].detail["value_score"] == 50
    assert "ticker" not in by_ticker["MSFT"].detail
    assert by_ticker["GOOGL"].detail["composite_score"] == 82
    assert by_ticker["GOOGL"].detail["value_score"] == 80
