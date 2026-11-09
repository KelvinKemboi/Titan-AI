from datetime import datetime, timedelta, timezone

import pandas as pd

from src.analytics.backtest import compute_backtest

T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _scan(scan_id, when, prices, scores):
    return [
        {"scan_run_id": scan_id, "started_at": when, "ticker": f"T{i}", "composite_score": s,
         "rating": "STRONG BUY" if s >= 80 else "SELL", "price": p}
        for i, (p, s) in enumerate(zip(prices, scores))
    ]


def _frame(future_return_for_score):
    n = 100
    scores = list(range(n))
    start = [100.0] * n
    end = [100.0 * (1 + future_return_for_score(s)) for s in scores]
    return pd.DataFrame(
        _scan(1, T0, start, scores) + _scan(2, T0 + timedelta(days=14), end, scores)
    )


def test_perfectly_predictive_score_has_ic_one_and_positive_spread():
    report = compute_backtest(_frame(lambda s: s / 1000), horizon_days=14)
    assert len(report.pairs) == 1
    assert abs(report.mean_ic - 1.0) < 1e-9
    assert report.mean_spread > 0
    assert report.mean_hit_rate == 1.0
    assert report.rating_returns["STRONG BUY"] > report.rating_returns["SELL"]


def test_inversely_predictive_score_has_negative_ic():
    report = compute_backtest(_frame(lambda s: -s / 1000), horizon_days=14)
    assert report.mean_ic < -0.99
    assert report.mean_spread < 0


def test_no_pair_when_next_scan_is_beyond_twice_the_horizon():
    df = _frame(lambda s: 0.0)
    df.loc[df.scan_run_id == 2, "started_at"] = T0 + timedelta(days=40)
    assert compute_backtest(df, horizon_days=14).pairs == []


def test_empty_frame_returns_empty_report():
    empty = pd.DataFrame(columns=["scan_run_id", "started_at", "ticker", "composite_score", "rating", "price"])
    report = compute_backtest(empty)
    assert report.pairs == [] and report.mean_ic is None
