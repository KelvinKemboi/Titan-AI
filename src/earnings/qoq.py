"""
Quarter-over-quarter diffing: a deterministic comparison of one
transcript's structured insights against the ticker's immediately prior
ingested quarter, no LLM call involved.
"""
import logging
from typing import Any, Dict, Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.api.routes.earnings import get_earnings_for_ticker as _get_earnings_for_ticker
from src.data.models import EarningsInsight

logger = logging.getLogger(__name__)


def _risk_key(risk: Dict[str, str]) -> str:
    """Normalizes a risk's own text for set-difference matching across
    quarters"""
    return risk["risk"].strip().lower()


def _diff_risks(prior_risks, current_risks):
    """(new_risks, resolved_risks): new = in current but not prior, resolved
    = in prior but not current, keyed on _risk_key. Preserves each list's
    own original order (not set iteration order)."""
    prior_keys = {_risk_key(r) for r in prior_risks}
    current_keys = {_risk_key(r) for r in current_risks}
    new_risks = [r for r in current_risks if _risk_key(r) not in prior_keys]
    resolved_risks = [r for r in prior_risks if _risk_key(r) not in current_keys]
    return new_risks, resolved_risks


def compute_qoq_changes(db: Session, ticker: str, transcript_id: int) -> Dict[str, Any]:
    """
    Computes `transcript_id`'s QoQ diff against `ticker`'s immediately
    prior ingested quarter. Returns {"status": "insufficient_history"}
    when there's no prior quarter to diff against, or it has no
    earnings_insights yet. Raises ValueError if `transcript_id` isn't an
    ingested transcript for `ticker`.
    """
    ticker = ticker.strip().upper()
    transcripts = _get_earnings_for_ticker(db, ticker)  # most-recent fiscal_year/fiscal_quarter first
    index = next((i for i, t in enumerate(transcripts) if t.transcript_id == transcript_id), None)
    if index is None:
        raise ValueError(f"transcript_id {transcript_id} is not an ingested transcript for ticker '{ticker}'")

    current = transcripts[index]
    prior = transcripts[index + 1] if index + 1 < len(transcripts) else None
    if prior is None or prior.insight is None or current.insight is None:
        return {"status": "insufficient_history"}

    current_insight, prior_insight = current.insight, prior.insight

    sentiment_delta = None
    if current_insight.sentiment_score is not None and prior_insight.sentiment_score is not None:
        sentiment_delta = current_insight.sentiment_score - prior_insight.sentiment_score

    new_risks, resolved_risks = _diff_risks(prior_insight.risks, current_insight.risks)

    return {
        "status": "ok",
        "prior_transcript_id": prior.transcript_id,
        "prior_fiscal_year": prior.fiscal_year,
        "prior_fiscal_quarter": prior.fiscal_quarter,
        "guidance_direction": {
            "prior": prior_insight.guidance_direction,
            "current": current_insight.guidance_direction,
            "changed": prior_insight.guidance_direction != current_insight.guidance_direction,
        },
        "sentiment_delta": sentiment_delta,
        "new_risks": new_risks,
        "resolved_risks": resolved_risks,
    }


def generate_qoq_changes(db: Session, transcript_id: int, ticker: str) -> Optional[EarningsInsight]:
    """Computes and persists `transcript_id`'s qoq_changes."""
    try:
        qoq_changes = compute_qoq_changes(db, ticker, transcript_id)
    except Exception:
        logger.exception("QoQ diff failed for transcript_id=%s; transcript is still persisted", transcript_id)
        return None

    stmt = (
        pg_insert(EarningsInsight)
        .values(transcript_id=transcript_id, qoq_changes=qoq_changes)
        .on_conflict_do_update(index_elements=["transcript_id"], set_={"qoq_changes": qoq_changes})
        .returning(EarningsInsight.id)
    )
    insight_id = db.execute(stmt).scalar_one()
    return db.get(EarningsInsight, insight_id)
