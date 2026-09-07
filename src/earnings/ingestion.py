"""
Earnings transcript ingestion job: given one
ticker + fiscal quarter, fetches the raw transcript via
src/earnings/provider_client.py and persists it to earnings_transcripts
(migration 4aee5f48ae19).
"""
import logging
from typing import Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.models import EarningsTranscript
from src.earnings.provider_client import EarningsProviderAuthError, EarningsProviderError, get_transcript

logger = logging.getLogger(__name__)


def ingest_transcript(db: Session, ticker: str, year: int, quarter: int) -> Optional[EarningsTranscript]:
    """
    Fetches the transcript for (ticker, year, quarter) and persists it as
    one earnings_transcripts row (raw text + source URL), returning it.

    Returns None when the provider has no transcript for
    this ticker/quarter
    """
    ticker = ticker.strip().upper()
    fiscal_quarter = f"Q{quarter}"

    existing = (
        db.query(EarningsTranscript)
        .filter(
            EarningsTranscript.ticker == ticker,
            EarningsTranscript.fiscal_year == year,
            EarningsTranscript.fiscal_quarter == fiscal_quarter,
        )
        .one_or_none()
    )
    if existing is not None:
        logger.info(
            "earnings_transcripts row already exists for %s %s %s; skipping provider call",
            ticker, fiscal_quarter, year,
        )
        return existing

    try:
        transcript = get_transcript(ticker, year=year, quarter=quarter)
    except EarningsProviderAuthError:
        raise
    except EarningsProviderError:
        logger.warning(
            "Provider call failed for %s %s %s; skipping this ticker/quarter", ticker, fiscal_quarter, year,
            exc_info=True,
        )
        return None

    if transcript is None:
        logger.warning("No transcript available for %s %s %s", ticker, fiscal_quarter, year)
        return None

    stmt = (
        pg_insert(EarningsTranscript)
        .values(
            ticker=transcript.ticker,
            fiscal_year=transcript.fiscal_year,
            fiscal_quarter=transcript.fiscal_quarter,
            raw_text=transcript.raw_text,
            source_url=transcript.source_url,
        )
        .on_conflict_do_nothing(index_elements=["ticker", "fiscal_year", "fiscal_quarter"])
        .returning(EarningsTranscript.id)
    )
    inserted_id = db.execute(stmt).scalar_one_or_none()
    if inserted_id is None:
        # Lost a race to a concurrent insert for the same (ticker, fiscal_year,
        # fiscal_quarter) between the check above and this insert 
        return (
            db.query(EarningsTranscript)
            .filter(
                EarningsTranscript.ticker == transcript.ticker,
                EarningsTranscript.fiscal_year == transcript.fiscal_year,
                EarningsTranscript.fiscal_quarter == transcript.fiscal_quarter,
            )
            .one()
        )
    return db.get(EarningsTranscript, inserted_id)
