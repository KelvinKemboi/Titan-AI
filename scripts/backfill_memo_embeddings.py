"""
One-off backfill: embeds + indexes analyst memos for the latest completed
scan_run, for data that predates memo_embeddings existing. RoboAnalyst.memo
was never persisted (only the numbers it's derived from were), so this
regenerates each ticker's memo text from its already-persisted
factor_scores row (composite_score + raw_metrics) via the exact same
RoboAnalyst.generate_memo() a live scan uses - byte-identical output,
no re-fetch from yfinance needed.

Requires VOYAGE_API_KEY (see src/embeddings/service.py) and a completed
scan already in Postgres.

Run:
    python -m scripts.backfill_memo_embeddings
"""
from dotenv import load_dotenv

from src.analytics.memo_indexing import index_memos
from src.data.db import SessionLocal
from src.data.models import FactorScore, ScanRun
from titan.analyst import RoboAnalyst

load_dotenv()

# Columns generate_memo() reads off .metrics - a row missing any of these
# (e.g. legacy/malformed raw_metrics) can't be reconstructed and is skipped
# rather than aborting the whole backfill, matching run_scan_and_persist's
# own "one bad ticker doesn't fail the run" philosophy.
_REQUIRED_METRIC_KEYS = {"Price", "RSI", "Trend", "Val_Metric", "Val_Type", "Margin", "Beta"}


def _latest_completed_scan_run(session):
    return (
        session.query(ScanRun)
        .filter(ScanRun.status.in_(["complete", "partial"]))
        .order_by(ScanRun.started_at.desc())
        .first()
    )


def _reconstruct_analyst(factor_score: FactorScore) -> RoboAnalyst:
    """Rebuilds just enough of a RoboAnalyst to call generate_memo() -
    the method only reads .ticker/.score/.metrics, all persisted verbatim
    on `factor_score`. Raises ValueError if composite_score/raw_metrics
    are missing or incomplete (e.g. a legacy/malformed row)."""
    if factor_score.composite_score is None:
        raise ValueError(f"{factor_score.ticker}: composite_score is null")
    raw_metrics = factor_score.raw_metrics or {}
    missing = _REQUIRED_METRIC_KEYS - raw_metrics.keys()
    if missing:
        raise ValueError(f"{factor_score.ticker}: raw_metrics missing {sorted(missing)}")

    analyst = RoboAnalyst(factor_score.ticker)
    analyst.score = float(factor_score.composite_score)
    analyst.metrics = raw_metrics
    analyst.generate_memo()
    return analyst


def main():
    session = SessionLocal()
    try:
        scan_run = _latest_completed_scan_run(session)
        if scan_run is None:
            print("No completed scan_runs found - run a scan first (Initialize Market Scan, or the scheduler).")
            return

        factor_scores = session.query(FactorScore).filter(FactorScore.scan_run_id == scan_run.id).all()
        print(f"Backfilling memo embeddings for scan_run_id={scan_run.id} ({len(factor_scores)} tickers)...")

        results = []
        for fs in factor_scores:
            try:
                results.append(_reconstruct_analyst(fs))
            except ValueError as exc:
                print(f"  skipping: {exc}")

        index_memos(session, scan_run.id, results)
        session.commit()

        print(f"Indexed {len(results)}/{len(factor_scores)} memos for scan_run_id={scan_run.id}.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
