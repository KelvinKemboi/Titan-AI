"""
Combines a ticker's latest factor_scores row with its latest
earnings_insights row into one analyst-style memo.
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.routes.earnings import get_earnings_for_ticker as _get_earnings_for_ticker
from src.data.models import FactorScore
from titan.analyst import RoboAnalyst


class AnalystReport(BaseModel):
    ticker: str
    scan_run_id: int
    computed_at: Optional[datetime] = None
    composite_score: float
    rating: Optional[str] = None
    memo: str
    has_earnings_data: bool


def _earnings_insight_dict(insight: Optional[Any]) -> Optional[dict]:
    """Converts an earnings insight into the plain dict RoboAnalyst.generate_memo expects."""
    if insight is None:
        return None
    return {
        "summary": insight.summary,
        "guidance_direction": insight.guidance_direction,
        "risks": insight.risks or [],
    }


def generate_report(factor_score: FactorScore, earnings_insight: Optional[Any] = None) -> AnalystReport:
    """DB-free: reconstructs just enough of a RoboAnalyst from an
    already-persisted `factor_scores` row to call generate_memo()."""
    if factor_score.composite_score is None:
        raise ValueError(f"{factor_score.ticker}: composite_score is null")

    analyst = RoboAnalyst(factor_score.ticker)
    analyst.score = float(factor_score.composite_score)
    analyst.metrics = factor_score.raw_metrics or {}
    analyst.generate_memo(_earnings_insight_dict(earnings_insight))

    return AnalystReport(
        ticker=factor_score.ticker,
        scan_run_id=factor_score.scan_run_id,
        computed_at=factor_score.computed_at,
        composite_score=float(factor_score.composite_score),
        rating=analyst.rating,
        memo=analyst.memo,
        has_earnings_data=earnings_insight is not None,
    )


def generate_ticker_report(db: Session, ticker: str) -> AnalystReport:
    """
    DB-aware wrapper: latest factor_scores row + latest earnings_insights
    row (the ticker's most-recently-ingested transcript, if any) for
    `ticker`. Raises ValueError if the ticker has no factor_scores row at
    all - a ticker that's never been scanned has nothing to report on,
    regardless of earnings data.
    """
    ticker = ticker.strip().upper()
    latest_score = (
        db.query(FactorScore)
        .filter(FactorScore.ticker == ticker)
        .order_by(FactorScore.computed_at.desc())
        .first()
    )
    if latest_score is None:
        raise ValueError(f"No factor_scores found for ticker '{ticker}'")

    transcripts = _get_earnings_for_ticker(db, ticker) # most-recent fiscal_year/fiscal_quarter first
    insight = transcripts[0].insight if transcripts else None

    return generate_report(latest_score, insight)
