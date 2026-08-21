"""
Semantic search over memo_embeddings - Titan's first qualitative
retrieval source (docs/architecture.md #6). Backs the search_memos tool
(src/agents/tools/memo_tools.py).
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.data.models import MemoEmbedding, ScanRun
from src.embeddings.service import embed_text

DEFAULT_TOP_K = 5


class MemoHit(BaseModel):
    """One memo match: the ticker/scan it came from, its memo text, and
    when that scan completed (source-attribution metadata, mirroring
    factor_tools.py's `_scan_run_as_of`)."""

    ticker: str
    scan_run_id: int
    memo_text: str
    as_of: Optional[datetime] = None


def search_memos(db: Session, query: str, top_k: int = DEFAULT_TOP_K) -> List[MemoHit]:
    """
    Embeds `query` (input_type="query" - see src/embeddings/service.py) and
    returns the `top_k` memo_embeddings rows nearest by cosine distance,
    closest first.
    """
    query_vector = embed_text(query, input_type="query")
    rows = (
        db.query(MemoEmbedding, ScanRun)
        .join(ScanRun, MemoEmbedding.scan_run_id == ScanRun.id)
        .order_by(MemoEmbedding.embedding.cosine_distance(query_vector))
        .limit(top_k)
        .all()
    )
    return [
        MemoHit(
            ticker=memo.ticker,
            scan_run_id=memo.scan_run_id,
            memo_text=memo.memo_text,
            as_of=scan_run.completed_at or scan_run.started_at,
        )
        for memo, scan_run in rows
    ]
