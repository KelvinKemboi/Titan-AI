from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.data.models import FactorScore

router = APIRouter()

# Factor names in the order they should be displayed in the UI and in the deltas table
_FACTORS = ["value", "momentum", "quality", "solvency", "volatility"]


# Ticker scores Pydantic model for the API response
class TickerScores(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticker: str
    value_score: Optional[float] = None
    momentum_score: Optional[float] = None
    quality_score: Optional[float] = None
    solvency_score: Optional[float] = None
    volatility_score: Optional[float] = None
    composite_score: Optional[float] = None
    rating: Optional[str] = None

# Pydantic model for the factor deltas between two tickers
class FactorDelta(BaseModel):
    factor: str
    a: Optional[float] = None
    b: Optional[float] = None
    delta: Optional[float] = None

# Pydantic model for the API response of the compare endpoint
class CompareResult(BaseModel):
    scan_run_id: int
    tickers: List[TickerScores]
    missing_tickers: List[str] = []
    delta_between: Optional[List[str]] = None
    deltas: List[FactorDelta] = []


def compare_tickers(db: Session, tickers: List[str]) -> CompareResult:
    """
    for `tickers`, all pinned to the same (latest) scan_run_id so a stale
    row never gets silently compared against a fresh one. Also returns
    the factor deltas between the first two found tickers, sorted by
    abs(delta) desc - the algorithm technical-design.md §3 specifies is
    inherently pairwise (`scores[a] - scores[b]`); extra tickers still
    appear in the aligned `tickers` table but aren't part of `deltas`.

    Raises ValueError if fewer than 2 of the requested tickers have a row
    in the latest scan run (whether because they were never scanned, or
    just not part of the most recent run).
    """
    latest_scan_run_id = db.query(func.max(FactorScore.scan_run_id)).scalar() # retrieves the latest scan_run_id from the FactorScore table in the database

    rows = []
    if latest_scan_run_id is not None:
        # fetches all FactorScore rows for the latest scan run that match the provided tickers
        rows = (
            db.query(FactorScore)
            .filter(
                FactorScore.scan_run_id == latest_scan_run_id,
                FactorScore.ticker.in_(tickers),
            )
            .all()
        )
    by_ticker = {row.ticker: row for row in rows} # dictionary mapping each ticker to its corresponding FactorScore row for O(1) lookup

    found = [t for t in tickers if t in by_ticker] # list of tickers that were found in the latest scan run
    missing = [t for t in tickers if t not in by_ticker] # list of tickers that were not found in the latest scan run

    if len(found) < 2:
        raise ValueError(
            f"Fewer than 2 of the requested tickers have data in the latest "
            f"scan (scan_run_id={latest_scan_run_id}): found={found or []}, "
            f"missing={missing}"
        )

    ticker_scores = [TickerScores.model_validate(by_ticker[t]) for t in found] # list of TickerScores objects for the found tickers

    ticker_a, ticker_b = found[0], found[1] # the first two tickers found in the latest scan run
    fs_a, fs_b = by_ticker[ticker_a], by_ticker[ticker_b] # the corresponding FactorScore rows for the first two tickers
    deltas = []
    # compute the deltas for each factor between the two tickers, handling None values appropriately
    for factor in _FACTORS:
        col = f"{factor}_score"
        a_val = getattr(fs_a, col)
        b_val = getattr(fs_b, col)
        a_val = float(a_val) if a_val is not None else None
        b_val = float(b_val) if b_val is not None else None
        delta = (a_val - b_val) if a_val is not None and b_val is not None else None
        deltas.append(FactorDelta(factor=factor, a=a_val, b=b_val, delta=delta)) # list of FactorDelta objects representing the difference in scores for each factor between the two tickers
    deltas.sort(key=lambda d: abs(d.delta) if d.delta is not None else -1.0, reverse=True)

    # return the CompareResult object containing the latest scan_run_id, the aligned ticker scores, any missing tickers, the first two tickers compared, and the computed deltas for each factor
    return CompareResult(
        scan_run_id=latest_scan_run_id,
        tickers=ticker_scores,
        missing_tickers=missing,
        delta_between=[ticker_a, ticker_b],
        deltas=deltas,
    )


@router.get("/compare", response_model=CompareResult)
# API endpoint for comparing multiple tickers based on their factor scores
def get_compare(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. MSFT,GOOGL"),
    db: Session = Depends(get_db), # database session dependency for querying the database
):
    parsed = []
    seen = set()
    # parse the comma-separated tickers, normalize them to uppercase, and ensure they are distinct
    for raw in tickers.split(","):
        t = raw.strip().upper()
        if t and t not in seen:
            seen.add(t)
            parsed.append(t)

    if len(parsed) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 distinct tickers are required (e.g. ?tickers=MSFT,GOOGL)",
        )

    try:
        return compare_tickers(db, parsed) # calls the compare_tickers function to perform the comparison and return the results as a CompareResult object
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
