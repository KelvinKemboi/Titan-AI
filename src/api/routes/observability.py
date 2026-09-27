"""
LLM spend and request-tracing endpoints, backed by
src/observability/spend.py. Gated behind the same auth as /chat, since
there's no separate admin role in this system yet.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.auth import get_current_user
from src.api.deps import get_db
from src.observability.spend import (
    RequestTrace,
    SpendSummary,
    get_aggregate_spend,
    get_request_trace,
    get_session_spend,
    get_top_sessions_by_spend,
)

router = APIRouter(prefix="/observability")


@router.get("/spend", response_model=SpendSummary)
def get_aggregate_spend_route(
    since_days: Optional[int] = Query(None, description="Restrict to calls in the last N days"),
    db: Session = Depends(get_db),
    _user_id: str = Depends(get_current_user),
):
    """Aggregate LLM spend across every traced call."""
    since = datetime.now(timezone.utc) - timedelta(days=since_days) if since_days else None
    return get_aggregate_spend(db, since=since)


@router.get("/spend/top-sessions")
def get_top_sessions_route(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    _user_id: str = Depends(get_current_user),
):
    """The `limit` sessions with the highest traced LLM cost, highest first."""
    return get_top_sessions_by_spend(db, limit=limit)


@router.get("/spend/{session_id}", response_model=SpendSummary)
def get_session_spend_route(
    session_id: str, db: Session = Depends(get_db), _user_id: str = Depends(get_current_user),
):
    """LLM spend for one chat session. 404 for a malformed (non-UUID)
    session_id; a real session with nothing traced yet returns an
    all-zero summary, not a 404 - it exists, it just hasn't spent anything."""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Invalid session_id '{session_id}'")
    return get_session_spend(db, session_uuid)


@router.get("/trace/{request_id}", response_model=RequestTrace)
def get_request_trace_route(
    request_id: str, db: Session = Depends(get_db), _user_id: str = Depends(get_current_user),
):
    """Every LLM call and tool call made while answering one /chat
    request, so a bad answer can be traced back to what produced it.
    404 when nothing was ever traced under this request_id."""
    trace = get_request_trace(db, request_id)
    if not trace.llm_calls and not trace.tool_calls:
        raise HTTPException(status_code=404, detail=f"No trace found for request_id '{request_id}'")
    return trace
