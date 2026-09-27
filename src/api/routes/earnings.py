"""
GET /earnings/{ticker}: every ingested transcript and its insights for
one ticker. get_earnings_for_ticker is the shared retrieval function
this route and the get_earnings_insight tool both call.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.data.models import Company, EarningsInsight, EarningsTranscript

router = APIRouter()


class EarningsInsightDetail(BaseModel):
    summary: Optional[str] = None
    guidance_direction: Optional[str] = None
    guidance_quote: Optional[str] = None
    sentiment_score: Optional[float] = None
    risks: List[Dict[str, str]] = []
    qoq_changes: Optional[Dict[str, Any]] = None
    generated_at: Optional[datetime] = None


class EarningsTranscriptDetail(BaseModel):
    transcript_id: int
    fiscal_year: int
    fiscal_quarter: str
    source_url: Optional[str] = None
    ingested_at: Optional[datetime] = None
    insight: Optional[EarningsInsightDetail] = None


class EarningsResponse(BaseModel):
    ticker: str
    transcripts: List[EarningsTranscriptDetail]


def get_earnings_for_ticker(db: Session, ticker: str) -> List[EarningsTranscriptDetail]:
    """
    Every earnings_transcripts row for `ticker`, most recent fiscal_year/
    fiscal_quarter first, each paired with its earnings_insights row if
    one has been generated (None otherwise - each extraction pass runs
    independently at ingestion time and any one of them could have failed
    or not run yet; see src/earnings/ingestion.py).
    """
    ticker = ticker.strip().upper()
    transcripts = (
        db.query(EarningsTranscript)
        .filter(EarningsTranscript.ticker == ticker)
        .order_by(EarningsTranscript.fiscal_year.desc(), EarningsTranscript.fiscal_quarter.desc())
        .all()
    )

    details = []
    for transcript in transcripts:
        insight = (
            db.query(EarningsInsight)
            .filter(EarningsInsight.transcript_id == transcript.id)
            .one_or_none()
        )
        details.append(
            EarningsTranscriptDetail(
                transcript_id=transcript.id,
                fiscal_year=transcript.fiscal_year,
                fiscal_quarter=transcript.fiscal_quarter,
                source_url=transcript.source_url,
                ingested_at=transcript.ingested_at,
                insight=EarningsInsightDetail(
                    summary=insight.summary,
                    guidance_direction=insight.guidance_direction,
                    guidance_quote=insight.guidance_quote,
                    sentiment_score=float(insight.sentiment_score) if insight.sentiment_score is not None else None,
                    risks=insight.risks or [],
                    qoq_changes=insight.qoq_changes,
                    generated_at=insight.generated_at,
                )
                if insight is not None
                else None,
            )
        )
    return details


@router.get("/earnings/{ticker}", response_model=EarningsResponse)
def get_earnings(ticker: str, db: Session = Depends(get_db)):
    """List of ingested transcripts + insights for a ticker; 404 for an
    unknown ticker, empty `transcripts` for a real ticker with none
    ingested yet."""
    ticker = ticker.strip().upper()
    if db.get(Company, ticker) is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker '{ticker}'")
    return EarningsResponse(ticker=ticker, transcripts=get_earnings_for_ticker(db, ticker))
