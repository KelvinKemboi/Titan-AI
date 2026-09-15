"""
Semantic search over earnings_chunks 
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.data.models import EarningsChunk, EarningsTranscript
from src.embeddings.service import embed_text

DEFAULT_TOP_K = 5


class EarningsChunkHit(BaseModel):
    """One earnings_chunks match: which transcript it came from, the
    chunk's own text and section, and when Titan ingested that transcript
    (source-attribution metadata, mirroring memo_search.py's MemoHit)."""

    ticker: str
    transcript_id: int
    fiscal_year: int
    fiscal_quarter: str
    chunk_type: str
    chunk_text: str
    as_of: Optional[datetime] = None


def search_earnings(db: Session, query: str, top_k: int = DEFAULT_TOP_K) -> List[EarningsChunkHit]:
    """
    Embeds `query` and returns the `top_k` earnings_chunks rows nearest by cosine distance,
    closest first, across every ingested ticker/quarter.
    """
    query_vector = embed_text(query, input_type="query")
    rows = (
        db.query(EarningsChunk, EarningsTranscript)
        .join(EarningsTranscript, EarningsChunk.transcript_id == EarningsTranscript.id)
        .order_by(EarningsChunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
        .all()
    )
    return [
        EarningsChunkHit(
            ticker=transcript.ticker,
            transcript_id=transcript.id,
            fiscal_year=transcript.fiscal_year,
            fiscal_quarter=transcript.fiscal_quarter,
            chunk_type=chunk.chunk_type,
            chunk_text=chunk.chunk_text,
            as_of=transcript.ingested_at,
        )
        for chunk, transcript in rows
    ]
