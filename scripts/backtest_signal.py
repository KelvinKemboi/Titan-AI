"""
Signal backtest: do higher composite scores predict higher forward returns?
Uses the per-scan prices already stored in factor_scores.raw_metrics - no
market-data calls. See src/analytics/backtest.py for metric definitions.

    python -m scripts.backtest_signal                 # 7, 14 and 28-day horizons
    python -m scripts.backtest_signal --horizon 14 --pairs
"""
import argparse

from dotenv import load_dotenv

load_dotenv()  # before src.* imports: src/data/db.py reads DATABASE_URL at import

from src.analytics.backtest import compute_backtest, load_scan_prices  # noqa: E402
from src.data.db import SessionLocal  # noqa: E402


def _pct(x):
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _num(x):
    return "n/a" if x is None else f"{x:+.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, action="append", help="Forward horizon in days (repeatable)")
    parser.add_argument("--pairs", action="store_true", help="Print every scan pair, not just the summary")
    args = parser.parse_args()
    horizons = args.horizon or [7, 14, 28]

    db = SessionLocal()
    try:
        df = load_scan_prices(db)
    finally:
        db.close()
    print(f"Loaded {df['scan_run_id'].nunique() if not df.empty else 0} full-universe scans, {len(df)} rows\n")

    for h in horizons:
        r = compute_backtest(df, horizon_days=h)
        print(f"=== {h}-day forward horizon: {len(r.pairs)} scan pairs ===")
        if not r.pairs:
            print("  not enough scan history at this horizon\n")
            continue
        print(f"  Mean IC (Spearman)        {_num(r.mean_ic)}   (>0.02 useful, >0.05 strong, <=0 no edge)")
        print(f"  ICIR                      {_num(r.icir)}")
        print(f"  % pairs with positive IC  {r.pct_positive_ic * 100:.0f}%")
        print(f"  Long-short (Q5 - Q1)      {_pct(r.mean_spread)}")
        print(f"  Top quintile vs universe  {_pct(r.mean_top_excess)}")
        print(f"  Top-quintile hit rate     {r.mean_hit_rate * 100:.0f}%  (50% = coin flip)")
        print("  Excess return by rating:")
        for rating, ret in r.rating_returns.items():
            print(f"    {rating:<12} {_pct(ret)}")
        if args.pairs:
            for p in r.pairs:
                print(f"    scan {p.from_scan_id:>3} -> {p.to_scan_id:<3} {p.from_date} ({p.days:4.1f}d, n={p.n}) "
                      f"IC {_num(p.ic)}  spread {_pct(p.long_short_spread)}")
        print()

    print("Caveats: overlapping windows (pairs not independent), short history, and a "
          "today's-S&P-500 universe (survivorship bias). A regression metric, not proof of alpha.")


if __name__ == "__main__":
    main()
