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
from src.earnings.chunk_indexing import index_transcript_chunks
from src.earnings.guidance import generate_guidance
from src.earnings.provider_client import EarningsProviderAuthError, EarningsProviderError, get_transcript
from src.earnings.qoq import generate_qoq_changes
from src.earnings.risk import generate_risks
from src.earnings.sentiment import generate_sentiment
from src.earnings.summary import generate_summary

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
        # fiscal_quarter) between the check above and this insert - whichever
        # call won already indexed this transcript's chunks, so don't repeat it
        return (
            db.query(EarningsTranscript)
            .filter(
                EarningsTranscript.ticker == transcript.ticker,
                EarningsTranscript.fiscal_year == transcript.fiscal_year,
                EarningsTranscript.fiscal_quarter == transcript.fiscal_quarter,
            )
            .one()
        )

    # Chunks, summary, guidance, sentiment, and risks, each swallowing its
    # own failures since none of this is on a user-facing request path.
    # QoQ runs last since it diffs against the other four.
    index_transcript_chunks(db, inserted_id, transcript.raw_text)
    generate_summary(db, inserted_id, transcript.raw_text)
    generate_guidance(db, inserted_id, transcript.raw_text)
    generate_sentiment(db, inserted_id, transcript.raw_text)
    generate_risks(db, inserted_id, transcript.raw_text)
    generate_qoq_changes(db, inserted_id, transcript.ticker)
    return db.get(EarningsTranscript, inserted_id)
