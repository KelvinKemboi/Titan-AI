"""
Signal backtest: does a high composite score actually predict higher
forward returns? This is the metric that says whether the scanner is worth
anything - everything else (latency, uptime, chat accuracy) is secondary.

Uses only data Titan already stores: every factor_scores row carries the
ticker's price at scan time in raw_metrics["Price"], so the forward return
between two scans `horizon_days` apart is price_later / price_earlier - 1.
No extra market-data calls are needed.

Reported per scan pair, then averaged:
  - IC (information coefficient): Spearman rank correlation between
    composite_score and forward return. Rule of thumb for equity factors:
    mean IC of 0.02-0.05 is useful, >0.05 is strong, <=0 means no edge.
  - ICIR: mean IC / std IC - consistency of the edge.
  - Long-short spread: mean forward return of the top quintile minus the
    bottom quintile.
  - Top-quintile excess: top quintile vs the equal-weighted universe.
  - Hit rate: share of top-quintile names that beat the universe median.
  - Mean forward return per rating bucket (STRONG BUY ... SELL) - checks the
    ratings are monotonic.

Caveats, printed with the report: windows overlap (pairs aren't
independent), history is short, and the universe is today's S&P 500
(survivorship bias). Treat it as a sanity check / regression metric, not a
publishable backtest.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

# Scans with fewer rows than this are test/partial runs (e.g. a 2-ticker
# smoke scan) and would add noise rather than signal.
MIN_TICKERS_PER_SCAN = 300
QUANTILES = 5


@dataclass
class PairResult:
    from_scan_id: int
    to_scan_id: int
    from_date: str
    days: float
    n: int
    ic: float
    long_short_spread: float
    top_quintile_excess: float
    hit_rate: float


@dataclass
class BacktestReport:
    horizon_days: int
    pairs: List[PairResult] = field(default_factory=list)
    rating_returns: Dict[str, float] = field(default_factory=dict)

    @property
    def mean_ic(self) -> Optional[float]:
        return _mean([p.ic for p in self.pairs])

    @property
    def icir(self) -> Optional[float]:
        ics = pd.Series([p.ic for p in self.pairs], dtype=float)
        if len(ics) < 2 or ics.std() == 0:
            return None
        return float(ics.mean() / ics.std())

    @property
    def mean_spread(self) -> Optional[float]:
        return _mean([p.long_short_spread for p in self.pairs])

    @property
    def mean_top_excess(self) -> Optional[float]:
        return _mean([p.top_quintile_excess for p in self.pairs])

    @property
    def mean_hit_rate(self) -> Optional[float]:
        return _mean([p.hit_rate for p in self.pairs])

    @property
    def pct_positive_ic(self) -> Optional[float]:
        if not self.pairs:
            return None
        return sum(p.ic > 0 for p in self.pairs) / len(self.pairs)


def _mean(values) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _spearman(a: pd.Series, b: pd.Series) -> float:
    # Rank-then-Pearson: identical to Spearman, avoids a scipy dependency.
    return float(a.rank().corr(b.rank()))


def load_scan_prices(db: Session) -> pd.DataFrame:
    """One row per (scan, ticker) with the composite score, rating and the
    price at scan time, for scans large enough to be real full-universe runs."""
    rows = db.execute(text("""
        SELECT f.scan_run_id, s.started_at, f.ticker, f.composite_score, f.rating,
               (f.raw_metrics->>'Price')::float AS price
        FROM factor_scores f
        JOIN scan_runs s ON s.id = f.scan_run_id
        WHERE f.scan_run_id IN (
            SELECT scan_run_id FROM factor_scores
            GROUP BY scan_run_id HAVING count(*) >= :min_n
        )
    """), {"min_n": MIN_TICKERS_PER_SCAN}).mappings().all()
    return pd.DataFrame(rows, columns=["scan_run_id", "started_at", "ticker", "composite_score", "rating", "price"])


def compute_backtest(df: pd.DataFrame, horizon_days: int = 14) -> BacktestReport:
    """
    Pure function over load_scan_prices()'s frame. Keeps one scan per
    calendar day (the latest), then pairs each scan with the first later scan
    at least `horizon_days` away (and no more than 2x that, so a long gap in
    scanning doesn't silently become a much longer horizon).
    """
    report = BacktestReport(horizon_days=horizon_days)
    if df.empty:
        return report

    df = df.dropna(subset=["price", "composite_score"]).copy()
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    scans = (
        df.groupby("scan_run_id")["started_at"].first().sort_values().to_frame()
        .assign(day=lambda s: s["started_at"].dt.date)
        .groupby("day").tail(1)  # latest scan per day
    )

    horizon = pd.Timedelta(days=horizon_days)
    bucket_returns: List[pd.DataFrame] = []
    for scan_id, row in scans.iterrows():
        later = scans[(scans["started_at"] - row["started_at"] >= horizon)
                      & (scans["started_at"] - row["started_at"] <= 2 * horizon)]
        if later.empty:
            continue
        to_id = later.index[0]
        a = df[df.scan_run_id == scan_id].set_index("ticker")
        b = df[df.scan_run_id == to_id].set_index("ticker")
        joined = a[["composite_score", "rating", "price"]].join(b[["price"]], rsuffix="_fwd", how="inner")
        joined = joined[joined["price"] > 0]
        if len(joined) < QUANTILES * 10:
            continue
        joined["fwd_return"] = joined["price_fwd"] / joined["price"] - 1
        joined["quintile"] = pd.qcut(joined["composite_score"].rank(method="first"), QUANTILES, labels=False)

        top = joined[joined.quintile == QUANTILES - 1]["fwd_return"]
        bottom = joined[joined.quintile == 0]["fwd_return"]
        universe_mean = joined["fwd_return"].mean()
        report.pairs.append(PairResult(
            from_scan_id=int(scan_id),
            to_scan_id=int(to_id),
            from_date=row["started_at"].date().isoformat(),
            days=(later.iloc[0]["started_at"] - row["started_at"]).total_seconds() / 86400,
            n=len(joined),
            ic=_spearman(joined["composite_score"], joined["fwd_return"]),
            long_short_spread=float(top.mean() - bottom.mean()),
            top_quintile_excess=float(top.mean() - universe_mean),
            hit_rate=float((top > joined["fwd_return"].median()).mean()),
        ))
        # Demeaned so rating buckets are compared net of the market move.
        bucket_returns.append(joined.assign(excess=joined["fwd_return"] - universe_mean)[["rating", "excess"]])

    if bucket_returns:
        all_buckets = pd.concat(bucket_returns)
        report.rating_returns = all_buckets.groupby("rating")["excess"].mean().sort_values(ascending=False).to_dict()
    return report
