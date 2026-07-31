from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.data.models import FactorScore
from titan.config import WEIGHTS

# factor key -- (WEIGHTS key, FactorScore score column)
_FACTOR_SPEC = [
    ("value", "Value", "value_score"),
    ("momentum", "Momentum", "momentum_score"),
    ("quality", "Quality", "quality_score"),
    ("solvency", "Solvency", "solvency_score"),
    ("volatility", "Volatility", "volatility_score"),
]

# for the factor score explanation API response
class FactorExplanation(BaseModel):
    factor: str
    score: float
    weight: float
    contribution: float
    driver: str

# for the factor score explanation API response
class FactorScoreExplanation(BaseModel):
    ticker: str
    scan_run_id: int
    computed_at: datetime
    composite_score: float
    rating: Optional[str] = None
    factors: List[FactorExplanation]

# helper function to generate one-line explanations for each factor based on the raw_metrics dictionary
def _value_driver(raw_metrics: dict) -> str:
    val_type = raw_metrics.get("Val_Type", "Unknown")
    val_metric = raw_metrics.get("Val_Metric")
    if val_metric is None:
        return "Val_Metric unavailable"
    if val_type == "Unknown":
        return (
            f"No PEG or P/E available - defaulted to an assumed-expensive "
            f"{val_metric:.2f} (PEG < 1.0 is elite, PEG > 3.0 is poor)"
        )
    return f"{val_type} of {val_metric:.2f} (PEG < 1.0 is elite, PEG > 3.0 is poor)"


def _momentum_driver(raw_metrics: dict) -> str:
    rsi = raw_metrics.get("RSI")
    trend = raw_metrics.get("Trend", "Unknown")
    if rsi is None:
        return f"{trend} trend"
    return f"RSI {rsi:.1f}, {trend} trend (RSI 40-75 is the target range)"


def _quality_driver(raw_metrics: dict) -> str:
    margin = raw_metrics.get("Margin")
    if margin is None:
        return "Margin unavailable"
    return f"Net margins of {margin:.1%} (>20% margins is elite)"


def _solvency_driver(raw_metrics: dict) -> str:
    debt = raw_metrics.get("Debt")
    if debt is None:
        return "Debt/Equity unavailable"
    return f"Debt/Equity of {debt:.1f} (<50% is elite)"


def _volatility_driver(raw_metrics: dict) -> str:
    beta = raw_metrics.get("Beta")
    if beta is None:
        return "Beta unavailable"
    return f"Beta of {beta:.2f} (<1.0 is considered safe)"


_DRIVER_FUNCS = {
    "value": _value_driver,
    "momentum": _momentum_driver,
    "quality": _quality_driver,
    "solvency": _solvency_driver,
    "volatility": _volatility_driver,
}


def explain_factor_scores(factor_score: FactorScore) -> FactorScoreExplanation:
    """
    Factor Score Explanation Engine (technical-design.md §2): a structured,
    typed breakdown of a factor_scores row - per-factor score, its
    `WEIGHTS` value, its contribution to composite (`score * weight`), and
    a one-line description of what drove it, templated from the same
    thresholds `titan/analyst.py` scores against (e.g. "PEG < 1.0 is
    elite") rather than re-derived or guessed.

    Pure/deterministic - no DB or network access - so it's unit testable
    directly against hand-built `FactorScore` fixtures. Returns typed
    fields only; generating prose from this is the LLM's job downstream
    (technical-design.md §2's risk note: don't let the LLM restate
    numbers freely from memory, or it can round/hallucinate - it must
    quote these fields verbatim).
    """
    raw_metrics = factor_score.raw_metrics or {}

    factors = []
    for factor_key, weight_key, score_col in _FACTOR_SPEC:
        score = getattr(factor_score, score_col)
        score = float(score) if score is not None else 0.0
        weight = WEIGHTS[weight_key]
        factors.append(
            FactorExplanation(
                factor=factor_key,
                score=score,
                weight=weight,
                contribution=score * weight,
                driver=_DRIVER_FUNCS[factor_key](raw_metrics),
            )
        )

    composite = factor_score.composite_score
    return FactorScoreExplanation(
        ticker=factor_score.ticker,
        scan_run_id=factor_score.scan_run_id,
        computed_at=factor_score.computed_at,
        composite_score=float(composite) if composite is not None else 0.0,
        rating=factor_score.rating,
        factors=factors,
    )


def explain_ticker(db: Session, ticker: str) -> FactorScoreExplanation:
    """
    DB-aware convenience wrapper: explains a ticker's latest factor_scores
    row. Raises ValueError if the ticker has no factor_scores row at all.
    """
    ticker = ticker.strip().upper()
    latest = (
        db.query(FactorScore)
        .filter(FactorScore.ticker == ticker)
        .order_by(FactorScore.computed_at.desc())
        .first()
    )
    if latest is None:
        raise ValueError(f"No factor_scores found for ticker '{ticker}'")
    return explain_factor_scores(latest)
