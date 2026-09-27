from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, TypeAdapter
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.data.cache import cache_get, cache_set, rankings_cache_key
from src.data.models import FactorScore

router = APIRouter()

# Maps a factor name to the FactorScore column to sort by.
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

_rankings_adapter = TypeAdapter(List[RankingItem])


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

    cache_key = rankings_cache_key(factor)
    cached = cache_get(cache_key)
    if cached is not None:
        return _rankings_adapter.validate_json(cached)

    latest_scan_run_id = db.query(func.max(FactorScore.scan_run_id)).scalar()
    if latest_scan_run_id is None:
        return []
    items = [
        RankingItem.model_validate(row)
        for row in (
            db.query(FactorScore)
            .filter(FactorScore.scan_run_id == latest_scan_run_id)
            .order_by(sort_column.desc(), FactorScore.ticker.asc())
            .all()
        )
    ]
    cache_set(cache_key, _rankings_adapter.dump_json(items).decode())
    return items
