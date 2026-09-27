"""
Embeds and persists each scan's analyst memos into memo_embeddings, the
qualitative retrieval source behind the search_memos tool. Called by
scanner_service right after a scan's factor_scores are persisted, so
search_memos has fresh, embedded memos as soon as a scan completes.
"""
import logging
from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.models import MemoEmbedding
from src.embeddings.service import embed_texts

logger = logging.getLogger(__name__)


def index_memos(session: Session, scan_run_id: int, results: List) -> None:
    """
    Embeds every `results` entry's non-blank `.memo` in one batched
    `embed_texts` call, then upserts one memo_embeddings row per ticker for
    this scan_run_id (keyed on the (ticker, scan_run_id) unique index, so
    re-running indexing for the same scan - e.g. a backfill re-run - is
    safe rather than duplicating rows).

    Embedding failures (e.g. the Voyage API being unreachable) are logged
    and swallowed here, not raised: a scan's core factor-score results
    must still count as persisted even if memo indexing can't reach the
    embeddings API this run.
    """
    memo_results = [r for r in results if r.memo]
    if not memo_results:
        return

    try:
        vectors = embed_texts([r.memo for r in memo_results])
    except Exception:
        logger.exception(
            "Memo embedding failed for scan_run_id=%s; factor scores are still persisted", scan_run_id
        )
        return

    for result, vector in zip(memo_results, vectors):
        stmt = pg_insert(MemoEmbedding).values(
            ticker=result.ticker,
            scan_run_id=scan_run_id,
            memo_text=result.memo,
            embedding=vector,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "scan_run_id"],
            set_={"memo_text": stmt.excluded.memo_text, "embedding": stmt.excluded.embedding},
        )
        session.execute(stmt)
