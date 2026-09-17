from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from src.analytics.report import AnalystReport, generate_ticker_report
from src.api.deps import get_db
from src.data.cache import cache_get, cache_set, company_cache_key
from src.data.models import Company, FactorScore

router = APIRouter()


# Pydantic model for the API response
class CompanyDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    scan_run_id: Optional[int] = None
    computed_at: Optional[datetime] = None
    composite_score: Optional[float] = None
    value_score: Optional[float] = None
    momentum_score: Optional[float] = None
    quality_score: Optional[float] = None
    solvency_score: Optional[float] = None
    volatility_score: Optional[float] = None
    rating: Optional[str] = None
    raw_metrics: Optional[Dict[str, Any]] = None



@router.get("/company/{ticker}", response_model=CompanyDetail)
# retrieves the company profile and its latest factor scores based on the provided ticker symbol
def get_company(ticker: str, db: Session = Depends(get_db)):
    """Company profile joined with its latest factor_scores row."""
    ticker = ticker.strip().upper()

    # A cache hit will return a CompanyDetail object from Redis
    cache_key = company_cache_key(ticker)
    cached = cache_get(cache_key)
    if cached is not None:
        return CompanyDetail.model_validate_json(cached)

    company = db.get(Company, ticker) # fetches the company record from the database using the provided ticker symbol
    if company is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'")
    # fetches the latest factor score by querying the FactorScore table, filtering by the ticker, ordering by the computed_at timestamp in descending order, and retrieving the first result
    latest_score = (
        db.query(FactorScore)
        .filter(FactorScore.ticker == ticker)
        .order_by(FactorScore.computed_at.desc())
        .first()
    )
    # returns a CompanyDetail object populated with the company profile and its latest factor scores, or None for the score fields if no factor score is found
    detail = CompanyDetail(
        ticker=company.ticker,
        name=company.name,
        sector=company.sector,
        industry=company.industry,
        description=company.description,
        scan_run_id=latest_score.scan_run_id if latest_score else None,
        computed_at=latest_score.computed_at if latest_score else None,
        composite_score=latest_score.composite_score if latest_score else None,
        value_score=latest_score.value_score if latest_score else None,
        momentum_score=latest_score.momentum_score if latest_score else None,
        quality_score=latest_score.quality_score if latest_score else None,
        solvency_score=latest_score.solvency_score if latest_score else None,
        volatility_score=latest_score.volatility_score if latest_score else None,
        rating=latest_score.rating if latest_score else None,
        raw_metrics=latest_score.raw_metrics if latest_score else None,
    )
    cache_set(cache_key, detail.model_dump_json())
    return detail


@router.get("/company/{ticker}/report", response_model=AnalystReport)
def get_company_report(ticker: str, db: Session = Depends(get_db)):
    """Analyst-style memo: latest factor scores
    + latest earnings insight, combined by src.analytics.report"""
    ticker = ticker.strip().upper()
    if db.get(Company, ticker) is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'")
    try:
        return generate_ticker_report(db, ticker)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"No factor scores found for ticker '{ticker}'")
