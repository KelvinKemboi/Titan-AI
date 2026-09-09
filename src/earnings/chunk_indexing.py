"""
Embeds and persists one transcript's chunks (src/earnings/chunking.py)
into earnings_chunks
Called by src/earnings/ingestion.py right after a new transcript row is
persisted, mirroring how scanner_service calls index_memos right after a
scan's factor_scores are persisted.
"""
import logging
from typing import List

from sqlalchemy.orm import Session

from src.data.models import EarningsChunk
from src.earnings.chunking import chunk_transcript
from src.embeddings.service import embed_texts

logger = logging.getLogger(__name__)


def index_transcript_chunks(db: Session, transcript_id: int, raw_text: str) -> List[EarningsChunk]:
    """
    Splits `raw_text` into chunks (chunk_transcript), embeds every chunk's
    text in one batched embed_texts call for
    cost/latency

    Returns [] without calling the embeddings API at all for a transcript
    with no chunks (e.g. blank raw_text). Embedding failures are logged
    """
    chunks = chunk_transcript(raw_text)
    if not chunks:
        return []

    try:
        vectors = embed_texts([c.chunk_text for c in chunks])
    except Exception:
        logger.exception(
            "Chunk embedding failed for transcript_id=%s; transcript is still persisted", transcript_id,
        )
        return []

    rows = [
        EarningsChunk(
            transcript_id=transcript_id,
            chunk_type=chunk.chunk_type,
            chunk_text=chunk.chunk_text,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    db.add_all(rows)
    db.flush()
    return rows
