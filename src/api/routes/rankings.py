from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.data.models import FactorScore

router = APIRouter()

_FACTOR_COLUMNS = {
    "value": FactorScore.value_score,
    "momentum": FactorScore.momentum_score,
    "quality": FactorScore.quality_score,
    "solvency": FactorScore.solvency_score,
    "volatility": FactorScore.volatility_score,
    "composite": FactorScore.composite_score,
}


class RankingItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticker: str
    scan_run_id: int
    computed_at: datetime
    composite_score: Optional[float] = None
    value_score: Optional[float] = None
    momentum_score: Optional[float] = None
    quality_score: Optional[float] = None
    solvency_score: Optional[float] = None
    volatility_score: Optional[float] = None
    rating: Optional[str] = None


@router.get("/rankings", response_model=List[RankingItem])
def get_rankings(
    factor: Optional[str] = Query(
        default=None,
        description="Sort by this factor's score instead of composite_score: "
        + ", ".join(sorted(_FACTOR_COLUMNS)),
    ),
    db: Session = Depends(get_db),
):
    """Latest scan's factor scores, sorted by composite_score (or `factor`) desc."""
    sort_column = FactorScore.composite_score
    if factor is not None:
        key = factor.strip().lower()
        if key not in _FACTOR_COLUMNS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown factor '{factor}'. Must be one of: {', '.join(sorted(_FACTOR_COLUMNS))}",
            )
        sort_column = _FACTOR_COLUMNS[key]

    # The scan_run with the most recent factor_scores rows — not just the
    # most recently *started* scan_run, which may still be `running` (no
    # rows yet) or have `failed` (zero rows) while an earlier run has data.
    latest_scan_run_id = db.query(func.max(FactorScore.scan_run_id)).scalar()
    if latest_scan_run_id is None:
        return []

    return (
        db.query(FactorScore)
        .filter(FactorScore.scan_run_id == latest_scan_run_id)
        .order_by(sort_column.desc(), FactorScore.ticker.asc())
        .all()
    )
